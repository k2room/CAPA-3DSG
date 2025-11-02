import abc
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import copy
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from natsort import natsorted
from scipy.spatial.transform import Rotation as R

from dataloader.colmap import read_model
from slam import datautils
from slam.geometryutils import relative_transformation

from utils.general_utils import measure_time, to_scalar
from conceptgraph.dataset.datasets_common import (
    as_intrinsics_matrix, readEXR_onlydepth, load_dataset_config, update_recursive, common_dataset_to_batch,
    ICLDataset, ReplicaDataset, ScannetDataset, Ai2thorDataset, 
    AzureKinectDataset, MultiscanDataset, RealsenseDataset, Record3DDataset, Hm3dDataset
)

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def as_intrinsics_matrix_rotated(intrinsics, ori_height):
    """
    Get matrix representation of intrinsics.
    [self.fx, self.fy, self.cx, self.cy]

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[1]
    K[1, 1] = intrinsics[0]
    K[0, 2] = ori_height - intrinsics[3]
    K[1, 2] = intrinsics[2]
    return K


def from_intrinsics_matrix(K: torch.Tensor) -> tuple[float, float, float, float]:
    '''
    Get fx, fy, cx, cy from the intrinsics matrix
    
    return 4 scalars
    '''
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy


def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header["dataWindow"]
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header["channels"]:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if "Y" not in header["channels"] else channelData["Y"]

    return Y


class GradSLAMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_dict,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cuda:0",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = True, # If True, the pose is relative to the first frame
        **kwargs,
    ):
        super().__init__()
        self.name = config_dict["dataset_name"]
        self.device = device
        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]

        self.orig_height = config_dict["camera_params"]["image_height"]
        self.orig_width = config_dict["camera_params"]["image_width"]
        self.fx = config_dict["camera_params"]["fx"]
        self.fy = config_dict["camera_params"]["fy"]
        self.cx = config_dict["camera_params"]["cx"]
        self.cy = config_dict["camera_params"]["cy"]

        self.dtype = dtype

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.load_embeddings = load_embeddings
        self.embedding_dir = embedding_dir
        self.embedding_dim = embedding_dim
        self.relative_pose = relative_pose

        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end == -1 or end > start):
            raise ValueError(
                "end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start)
            )

        self.distortion = (
            np.array(config_dict["camera_params"]["distortion"])
            if "distortion" in config_dict["camera_params"]
            else None
        )
        self.crop_size = (
            config_dict["camera_params"]["crop_size"]
            if "crop_size" in config_dict["camera_params"]
            else None
        )

        self.crop_edge = None
        if "crop_edge" in config_dict["camera_params"].keys():
            self.crop_edge = config_dict["camera_params"]["crop_edge"]

        self.color_paths, self.depth_paths, self.embedding_paths = self.get_filepaths()
        if len(self.color_paths) != len(self.depth_paths):
            print(len(self.color_paths))
            print(len(self.depth_paths))
            raise ValueError("Number of color and depth images must be the same.")
        if self.load_embeddings:
            if len(self.color_paths) != len(self.embedding_paths):
                raise ValueError(
                    "Mismatch between number of color images and number of embedding files."
                )
        self.num_imgs = len(self.color_paths)
        self.poses = self.load_poses()
        
        if self.end == -1:
            self.end = self.num_imgs

        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.depth_paths = self.depth_paths[self.start : self.end : stride]
        if self.load_embeddings:
            self.embedding_paths = self.embedding_paths[self.start : self.end : stride]
        self.poses = self.poses[self.start : self.end : stride]
        # Tensor of retained indices (indices of frames and poses that were retained)
        self.retained_inds = torch.arange(self.num_imgs)[self.start : self.end : stride]
        # Update self.num_images after subsampling the dataset
        self.num_imgs = len(self.color_paths)

        # self.transformed_poses = datautils.poses_to_transforms(self.poses)
        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

    def __len__(self):
        return self.num_imgs

    def get_filepaths(self):
        """Return paths to color images, depth images. Implement in subclass."""
        raise NotImplementedError

    def load_poses(self):
        """Load camera poses. Implement in subclass."""
        raise NotImplementedError

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.camera_axis == 'Left':
            color = cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE)
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        if self.camera_axis == 'Left':
            depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
        depth = np.expand_dims(depth, -1) 
        if self.channels_first:
            depth = datautils.channels_first(depth)
        return depth / self.png_depth_scale
    
    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )
        
    def get_cam_K(self):
        '''
        Return camera intrinsics matrix K
        
        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        '''
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        return K
    
    def read_embedding_from_file(self, embedding_path: str):
        '''
        Read embedding from file and process it. To be implemented in subclass for each dataset separately.
        '''
        raise NotImplementedError

    def __getitem__(self, index):
        if not hasattr(self, 'camera_axis'):
            self.camera_axis = 'Up'
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)
        if ".png" in depth_path:
            # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        elif ".exr" in depth_path:
            depth = readEXR_onlydepth(depth_path)
        elif ".npy" in depth_path:
            depth = np.load(depth_path)
        else:
            raise NotImplementedError

        if self.camera_axis == 'Left':
            K = as_intrinsics_matrix_rotated([self.fx, self.fy, self.cx, self.cy], self.desired_height)
        elif self.camera_axis == 'Up':
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        if self.camera_axis == 'Left':
            K = datautils.scale_intrinsics(
                K, self.width_downsample_ratio, self.height_downsample_ratio
            )
        elif self.camera_axis == 'Up':
            K = datautils.scale_intrinsics(
                K, self.height_downsample_ratio, self.width_downsample_ratio
            )
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K
        # print(intrinsics)

        pose = self.transformed_poses[index]
        # print(pose)

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),  # Allow embedding to be another dtype
                # self.retained_inds[index].item(),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            # self.retained_inds[index].item(),
        )

        
class FunGraph3DDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        cameras, images = read_model(self.input_folder, ".txt")
        for color_p in self.color_paths:
            for image_id, image in images.items():
                image_name = image.name.split('/')[-1]
                if image_name in color_p:
                    world_to_camera = image.world_to_camera
                    c2w = np.linalg.inv(world_to_camera)
                    c2w = torch.from_numpy(c2w).float()
                    poses.append(c2w)
                    break
        return poses
    
    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)

class SceneFun3DDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        resolution: Optional[str] = "high",  # "high" or "low"
        **kwargs,
    ):
        # Root path to the sequence
        self.input_folder = os.path.join(basedir, sequence)
        self.resolution = resolution  # must be set before using it

        if self.resolution == "high":
            _color_paths = natsorted(glob.glob(f"{self.input_folder}/hires_wide/*.jpg"))
            _depth_paths = natsorted(glob.glob(f"{self.input_folder}/hires_depth/*.png"))
            self.pose_path = os.path.join(self.input_folder, "hires_poses.traj")
        elif self.resolution == "low":
            _color_paths = natsorted(glob.glob(f"{self.input_folder}/lowres_wide/*.jpg"))
            _depth_paths = natsorted(glob.glob(f"{self.input_folder}/lowres_depth/*.png"))
            self.pose_path = os.path.join(self.input_folder, "lowres_wide.traj")
        else:
            raise ValueError(f"Unsupported resolution: {self.resolution}")

        if len(_color_paths) == 0 or len(_depth_paths) == 0:
            raise FileNotFoundError(f"No images found under {self.input_folder} for resolution={self.resolution}")

        _sample = cv2.imread(_color_paths[0], cv2.IMREAD_COLOR)
        if _sample is None:
            raise RuntimeError(f"Failed to read image: {_color_paths[0]}")
        H_file, W_file = _sample.shape[:2]

        cfg = copy.deepcopy(config_dict)
        cam = cfg["camera_params"]
        H_cfg, W_cfg = cam["image_height"], cam["image_width"]
        if (H_cfg != H_file) or (W_cfg != W_file):
            scale_h = float(H_file) / float(H_cfg)
            scale_w = float(W_file) / float(W_cfg)
            cam["fx"] *= scale_w
            cam["fy"] *= scale_h
            cam["cx"] *= scale_w
            cam["cy"] *= scale_h
            cam["image_height"] = H_file
            cam["image_width"] = W_file

        desired_height = H_file
        desired_width = W_file
        self.camera_axis = "Up"

        # Call parent __init__ (this will call get_filepaths() and load_poses())
        super().__init__(
            cfg,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    # ----------------------- file listing -----------------------
    def get_filepaths(self):
        if self.resolution == "high":
            color_paths = natsorted(glob.glob(f"{self.input_folder}/hires_wide/*.jpg"))
            depth_paths = natsorted(glob.glob(f"{self.input_folder}/hires_depth/*.png"))
        else:  # "low"
            color_paths = natsorted(glob.glob(f"{self.input_folder}/lowres_wide/*.jpg"))
            depth_paths = natsorted(glob.glob(f"{self.input_folder}/lowres_depth/*.png"))

        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    # ----------------------- pose loading -----------------------
    def load_poses(self):
        poses = []
        poses_from_traj = {}
        with open(self.pose_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            traj_timestamp = line.split(" ")[0]
            poses_from_traj[f"{round(float(traj_timestamp), 3):.3f}"] = np.array(self._traj_string_to_matrix(line)[1].tolist())

        color_paths, depth_paths, embedding_paths = self.get_filepaths()

        new_color_paths, new_depth_paths, new_embedding_paths = [], [], []
        for idx, color_file in enumerate(color_paths):
            frame_id = os.path.basename(color_file).split(".jpg")[0].split("_")[1]
            c2w = self.get_nearest_pose(frame_id, poses_from_traj, use_interpolation=True, time_distance_threshold=0.2)

            if c2w is not None:
                if self.camera_axis == 'Left':
                    R_z_90 = np.array([[0, 1, 0],
                                       [-1, 0, 0],
                                       [0, 0, 1]])
                    R_c2w = c2w[:3, :3]
                    t_c2w = c2w[:3, 3]
                    R_c2w_prime = np.dot(R_c2w, R_z_90)
                    adjusted = np.eye(4)
                    adjusted[:3, :3] = R_c2w_prime
                    adjusted[:3, 3] = t_c2w
                    c2w = adjusted

                poses.append(torch.from_numpy(c2w).float())
                new_color_paths.append(color_file)
                new_depth_paths.append(depth_paths[idx])
                if self.load_embeddings and embedding_paths is not None:
                    new_embedding_paths.append(embedding_paths[idx])

        self.color_paths, self.depth_paths, self.embedding_paths = new_color_paths, new_depth_paths, new_embedding_paths
        return poses

    # ----------------------- project API compatibility -----------------------
    def read_embedding_from_file(self, embedding_file_path):
        emb = torch.load(embedding_file_path)
        return emb.permute(0, 2, 3, 1)  # (1, H, W, D)

    # ----------------------- helpers (embedded from official parser) -----------------------
    @staticmethod
    def _frame_timestamp_from_path(path: str) -> str:
        """Extract timestamp string from filename 'xxx_<ts>.jpg|png'."""
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        ts = stem.split("_")[-1]
        return ts

    @staticmethod
    def _convert_angle_axis_to_matrix3(angle_axis: np.ndarray) -> np.ndarray:
        """Angle-axis (Rodrigues) -> 3x3 rotation matrix. (OpenCV)"""
        angle_axis = np.asarray(angle_axis, dtype=float).reshape(3,)
        R, _ = cv2.Rodrigues(angle_axis)
        return R

    def _traj_string_to_matrix(self, traj_str: str) -> tuple[str, np.ndarray]:
        """Line 'ts rx ry rz tx ty tz' -> (timestamp, 4x4 pose) in world frame."""
        toks = traj_str.split()
        assert len(toks) == 7, "Trajectory line must have 7 columns"
        ts = toks[0]
        angle_axis = np.array([float(toks[1]), float(toks[2]), float(toks[3])], dtype=float)
        R_w_p = self._convert_angle_axis_to_matrix3(angle_axis)
        t_w_p = np.array([float(toks[4]), float(toks[5]), float(toks[6])], dtype=float)
        extr = np.eye(4, dtype=float)
        extr[:3, :3] = R_w_p
        extr[:3, 3] = t_w_p
        Rt = np.linalg.inv(extr)  # pose of camera in world (c2w)
        return ts, Rt

    def _read_traj_to_dict(self, traj_path: str) -> dict[str, np.ndarray]:
        """Read *.traj file -> dict[str->4x4]. High-res keeps full precision; low-res rounds to 3 decimals."""
        with open(traj_path, "r") as f:
            lines = f.readlines()

        poses = {}
        round3 = (self.resolution == "low")
        for line in lines:
            ts, Rt = self._traj_string_to_matrix(line)
            key = f"{float(ts):.3f}" if round3 else ts
            poses[key] = Rt
        return poses

    @staticmethod
    def _decide_pose(pose_4x4: np.ndarray) -> int:
        """Return 0:upright, 1:left, 2:upside-down, 3:right by comparing z-axis. (Official logic)"""
        z_vec = pose_4x4[2, :3]
        z_orien = np.array([
            [0.0, -1.0, 0.0],  # upright
            [-1.0, 0.0, 0.0],  # left
            [0.0,  1.0, 0.0],  # upside-down
            [1.0,  0.0, 0.0],  # right
        ], dtype=float)
        corr = z_orien @ z_vec
        return int(np.argmax(corr))

    @staticmethod
    def _peek_first_pose(traj_path: str) -> np.ndarray:
        """Read first pose from a trajectory file."""
        with open(traj_path, "r") as f:
            first = f.readline()
        # Minimal parser to avoid class context here:
        toks = first.split()
        angle_axis = np.array([float(toks[1]), float(toks[2]), float(toks[3])], dtype=float)
        R, _ = cv2.Rodrigues(angle_axis)
        t = np.array([float(toks[4]), float(toks[5]), float(toks[6])], dtype=float)
        extr = np.eye(4, dtype=float); extr[:3,:3]=R; extr[:3,3]=t
        return np.linalg.inv(extr)

    @staticmethod
    def _trans(H: np.ndarray) -> np.ndarray:
        """Extract translation (for frame-distance checks if you later enable them)."""
        return H[:3, 3]

    def _interp_split(self, t_des: float, H0: np.ndarray, t0: float, H1: np.ndarray, t1: float) -> np.ndarray:
        """Rigid interp in SO(3) x R^3 (slerp + linear)."""
        # times in seconds
        alpha = (t_des - t0) / (t1 - t0 + 1e-8)
        # rotations -> use cv2.Rodrigues / log/exp fallback (simple slerp via quaternions)
        from scipy.spatial.transform import Rotation as R
        R0 = R.from_matrix(H0[:3, :3])
        R1 = R.from_matrix(H1[:3, :3])
        Rm = R.slerp(0, 1, [R0, R1])(alpha).as_matrix()
        tm = (1 - alpha) * H0[:3, 3] + alpha * H1[:3, 3]
        Hm = np.eye(4, dtype=float); Hm[:3,:3]=Rm; Hm[:3,3]=tm
        return Hm

    # --------------- API kept for compatibility with existing code ---------------
    def get_nearest_pose(
        self, 
        desired_timestamp,
        poses_from_traj, 
        time_distance_threshold = np.inf,
        use_interpolation = False,
        interpolation_method = 'split',
        frame_distance_threshold = np.inf,
    ):
        """Project API-compatible method. If use_interpolation=True, we do split-interp."""
        # bounds check
        keys = list(poses_from_traj.keys())
        fkeys = [float(k) for k in keys]
        fmin, fmax = min(fkeys), max(fkeys)
        fdes = float(desired_timestamp)
        if fdes < fmin or fdes > fmax:
            return None

        if desired_timestamp in poses_from_traj:
            return poses_from_traj[desired_timestamp]

        # nearest keys around desired ts
        greater = [k for k in fkeys if k > fdes]
        smaller = [k for k in fkeys if k < fdes]
        if not greater or not smaller:
            # fallback to the single closest sample
            closest = min(keys, key=lambda x: abs(float(x) - fdes))
            if abs(float(closest) - fdes) > time_distance_threshold:
                return None
            return poses_from_traj[closest]

        g_cl = min(greater, key=lambda x: abs(x - fdes))
        s_cl = min(smaller, key=lambda x: abs(x - fdes))
        if (abs(g_cl - fdes) > time_distance_threshold) or (abs(s_cl - fdes) > time_distance_threshold):
            return None

        if use_interpolation:
            H0 = poses_from_traj[f"{s_cl:.3f}" if isinstance(next(iter(poses_from_traj)), str) else s_cl]
            H1 = poses_from_traj[f"{g_cl:.3f}" if isinstance(next(iter(poses_from_traj)), str) else g_cl]
            # optional frame-distance check (disabled by default)
            if np.linalg.norm(self._trans(H0) - self._trans(H1)) > frame_distance_threshold:
                return None
            if interpolation_method != "split":
                # Geodesic path could be implemented similarly; we stick to split to keep dependencies small.
                interpolation_method = "split"
            return self._interp_split(fdes, H0, s_cl, H1, g_cl)
        else:
            closest = min(keys, key=lambda x: abs(float(x) - fdes))
            if abs(float(closest) - fdes) > time_distance_threshold:
                return None
            return poses_from_traj[closest]


def load_dataset_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from file itself
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = cfg_special.get("inherit_from")

    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_dataset_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def common_dataset_to_batch(dataset):
    colors, depths, poses = [], [], []
    intrinsics, embeddings = None, None
    for idx in range(len(dataset)):
        _color, _depth, intrinsics, _pose, _embedding = dataset[idx]
        colors.append(_color)
        depths.append(_depth)
        poses.append(_pose)
        if _embedding is not None:
            if embeddings is None:
                embeddings = [_embedding]
            else:
                embeddings.append(_embedding)
    colors = torch.stack(colors)
    depths = torch.stack(depths)
    poses = torch.stack(poses)
    if embeddings is not None:
        embeddings = torch.stack(embeddings, dim=1)
        # # (1, NUM_IMG, DIM_EMBED, H, W) -> (1, NUM_IMG, H, W, DIM_EMBED)
        # embeddings = embeddings.permute(0, 1, 3, 4, 2)
    colors = colors.unsqueeze(0)
    depths = depths.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
    poses = poses.unsqueeze(0)
    colors = colors.float()
    depths = depths.float()
    intrinsics = intrinsics.float()
    poses = poses.float()
    if embeddings is not None:
        embeddings = embeddings.float()
    return colors, depths, intrinsics, poses, embeddings



@measure_time
def get_dataset(dataconfig, basedir, sequence, **kwargs):
    config_dict = load_dataset_config(dataconfig)

    if config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["fungraph3d"]:
        return FunGraph3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict['dataset_name'].lower() in ['scenefun3d']:
        return SceneFun3DDataset(config_dict, basedir, sequence, **kwargs)

    # TODO: add more annotated datasets     
    elif config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["multiscan"]:
        return MultiscanDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict['dataset_name'].lower() in ['hm3d']:
        return Hm3dDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


if __name__ == "__main__":
    cfg = load_dataset_config(
        "/home/replica.yaml"
    )
    dataset = ReplicaDataset(
        config_dict=cfg,
        basedir="/home/Datasets/Replica",
        sequence="office0",
        start=0,
        end=1900,
        stride=100,
        # desired_height=680,
        # desired_width=1200,
        desired_height=240,
        desired_width=320,
    )

    colors, depths, poses = [], [], []
    intrinsics = None
    for idx in range(len(dataset)):
        _color, _depth, intrinsics, _pose = dataset[idx]
        colors.append(_color)
        depths.append(_depth)
        poses.append(_pose)
    colors = torch.stack(colors)
    depths = torch.stack(depths)
    poses = torch.stack(poses)
    colors = colors.unsqueeze(0)
    depths = depths.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
    poses = poses.unsqueeze(0)
    colors = colors.float()
    depths = depths.float()
    intrinsics = intrinsics.float()
    poses = poses.float()

    # create rgbdimages object
    rgbdimages = RGBDImages(
        colors,
        depths,
        intrinsics,
        poses,
        channels_first=False,
        has_embeddings=False, 
    )
