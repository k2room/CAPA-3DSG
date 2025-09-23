import atexit
import bisect
import multiprocessing as mp
from collections import deque, OrderedDict
import cv2
import torch
import numpy as np
from torch.nn import functional as F

from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode

from vlpart.modeling.text_encoder.text_encoder import build_text_encoder
from .vlp_visualizer import CustomVisualizer

from pathlib import Path
_VLPART_DIR = Path(__file__).resolve().parents[1] / "thirdparty" / "vlpart"
_METADATA_DIR = _VLPART_DIR / "datasets" / "metadata"

_TEXT_ENCODER = None              # Singleton text encoder
_PHRASE_CACHE = OrderedDict()     # phrase(str) -> torch.Tensor(D,)
_CACHE_CAP = 4096                 # LRU capacity; tune if needed

_BUILDIN_CLASSIFIER = {
    "lvis":         str((_METADATA_DIR / "lvis_v1_clip_RN50_a+cname.npy").resolve()),
    "paco":         str((_METADATA_DIR / "paco_clip_RN50_a+cname.npy").resolve()),
    "coco":         str((_METADATA_DIR / "coco_clip_RN50_a+cname.npy").resolve()),
    "voc":          str((_METADATA_DIR / "voc_clip_RN50_a+cname.npy").resolve()),
    "pascal_part":  str((_METADATA_DIR / "pascal_part_clip_RN50_a+cname.npy").resolve()),
    "partimagenet": str((_METADATA_DIR / "partimagenet_clip_RN50_a+cname.npy").resolve()),
}

_BUILDIN_METADATA_PATH = {
    'pascal_part': 'pascal_part_val',
    'partimagenet': 'partimagenet_val',
    'paco': 'paco_lvis_v1_val',
    'lvis': 'lvis_v1_val',
    'coco': 'coco_2017_val',
    'voc': 'voc_2007_val',
}

def _resolve_metadata_path(s: str) -> str:
    """Convert relative path to absolute path under _METADATA_DIR for detectron2 metadata"""
    p = Path(s)
    if p.is_absolute() and p.exists():
        return str(p)
    q = (_METADATA_DIR / p.name).resolve()
    return str(q)

def _rewrite_metadata_paths_in_cfg(node):
    """Convert relative path to absolute path under _METADATA_DIR for detectron2 cfg (yacs.CfgNod)."""
    for k, v in node.items():
        try:
            if hasattr(v, "items"):
                _rewrite_metadata_paths_in_cfg(v)
                continue
        except Exception:
            pass

        if isinstance(v, list):
            changed = False
            new_list = []
            for x in v:
                if isinstance(x, str) and ("datasets/metadata/" in x or x.endswith("_clip_RN50_a+cname.npy")):
                    new_list.append(_resolve_metadata_path(x))
                    changed = True
                else:
                    new_list.append(x)
            if changed:
                node[k] = new_list

        elif isinstance(v, str) and ("datasets/metadata/" in v or v.endswith("_clip_RN50_a+cname.npy")):
            node[k] = _resolve_metadata_path(v)


def reset_cls_test(model, cls_path):
    # model.roi_heads.num_classes = num_classes
    if type(cls_path) == str:
        print('Resetting zs_weight', cls_path)
        if cls_path.endswith('npy'):
            zs_weight = np.load(cls_path)
            zs_weight = torch.tensor(
                zs_weight, dtype=torch.float32).permute(1, 0).contiguous()  # dim x C
        elif cls_path.endswith('pth'):
            zs_weight = torch.load(cls_path, map_location='cpu')
            zs_weight = zs_weight.clone().detach().permute(1, 0).contiguous()  # dim x C
        else:
            raise NotImplementedError
        # zs_weight = torch.tensor(
        #     np.load(cls_path),
        #     dtype=torch.float32).permute(1, 0).contiguous() # D x C
    else:
        zs_weight = cls_path
    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1))],
        dim=1) # D x (C + 1)
    # if model.roi_heads.box_predictor[0].cls_score.norm_weight:
    #     zs_weight = F.normalize(zs_weight, p=2, dim=0)
    # zs_weight = zs_weight.to(model.device)
    # for k in range(len(model.roi_heads.box_predictor)):
    #     del model.roi_heads.box_predictor[k].cls_score.zs_weight
    #     model.roi_heads.box_predictor[k].cls_score.zs_weight = zs_weight
    zs_weight = F.normalize(zs_weight, p=2, dim=0)
    zs_weight = zs_weight.to(model.device)

    if isinstance(model.roi_heads.box_predictor, torch.nn.ModuleList):
        for idx in range(len(model.roi_heads.box_predictor)):
            model.roi_heads.box_predictor[idx].cls_score.zs_weight_inference = zs_weight
    else:
        model.roi_heads.box_predictor.cls_score.zs_weight_inference = zs_weight


# def get_clip_embeddings(vocabulary, prompt='a '):
#     text_encoder = build_text_encoder(pretrain=True)
#     text_encoder.eval()
#     texts = [prompt + x.lower().replace(':', ' ') for x in vocabulary]
#     emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
#     return emb

def _get_text_encoder():
    """Build the internal text encoder once and reuse."""
    global _TEXT_ENCODER
    if _TEXT_ENCODER is None:
        enc = build_text_encoder(pretrain=True)  # original builder from VLPart
        enc.eval()                               # disable dropout for determinism
        _TEXT_ENCODER = enc
    return _TEXT_ENCODER

def _canon(s: str) -> str:
    """Canonicalize phrase for cache keys (lowercase + whitespace squeeze)."""
    return " ".join(s.split()).strip().lower().replace(":", " ")

@torch.no_grad()
def get_clip_embeddings(vocabulary, prompt='a '):
    """
    Return zs-weight matrix (D, C) on CPU for the given vocabulary.
    This uses a persistent internal text encoder and a small phrase-level LRU cache
    so that only unseen phrases are encoded each time.
    """
    enc = _get_text_encoder()

    # 1) Find cache misses and prepare normalized texts
    keys = []
    miss_idx, miss_texts = [], []
    for i, term in enumerate(vocabulary):
        k = _canon(term)
        keys.append(k)
        if k not in _PHRASE_CACHE:
            miss_idx.append(i)
            miss_texts.append(prompt + term.lower().replace(":", " "))

    # 2) Encode only the missing phrases
    if len(miss_texts) > 0:
        vec = enc(miss_texts)                               # (M, D)
        vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-6) # L2 norm for stability
        vec = vec.to(dtype=torch.float32, device="cpu")
        for j, i in enumerate(miss_idx):
            k = keys[i]
            _PHRASE_CACHE[k] = vec[j]
            # LRU eviction
            if len(_PHRASE_CACHE) > _CACHE_CAP:
                _PHRASE_CACHE.popitem(last=False)

    # 3) Assemble zs-weight (D, C) following current vocabulary order
    cols = [ _PHRASE_CACHE[k].unsqueeze(1) for k in keys ]  # each (D,1)
    mat = torch.cat(cols, dim=1)                            # (D, C)
    return mat

class VisualizationDemo(object):
    def __init__(self, cfg, args=None, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
        """
        # --- Dynamic custom-vocabulary mode for CAPA ---
        # Use '__unused' metadata and do NOT set thing_classes here.
        if args is not None and getattr(args, "vocabulary", "") == "custom_dynamic":
            self.metadata = MetadataCatalog.get("__unused")
            classifier = None  # Zero-shot weights will be injected at runtime (per frame).

        # --- Existing branches unchanged below ---
        elif args is None:
            self.metadata = MetadataCatalog.get(_BUILDIN_METADATA_PATH['pascal_part'])
            classifier = _BUILDIN_CLASSIFIER['pascal_part']

        elif args.vocabulary == 'custom':
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = args.custom_vocabulary.split(',')
            classifier = get_clip_embeddings(self.metadata.thing_classes)

        elif args.vocabulary == 'pascal_part_voc':
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = args.custom_vocabulary.split(',')
            classifier = get_clip_embeddings(self.metadata.thing_classes)

        elif args.vocabulary == 'lvis_paco':
            self.metadata = MetadataCatalog.get("__unused")
            lvis_thing_classes = MetadataCatalog.get(_BUILDIN_METADATA_PATH['lvis']).thing_classes
            paco_thing_classes = MetadataCatalog.get(_BUILDIN_METADATA_PATH['paco']).thing_classes[75:]
            self.metadata.thing_classes = lvis_thing_classes + paco_thing_classes
            classifier = get_clip_embeddings(self.metadata.thing_classes)

        else:
            self.metadata = MetadataCatalog.get(_BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = _BUILDIN_CLASSIFIER[args.vocabulary]

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        cfg = cfg.clone()
        cfg.defrost()
        _rewrite_metadata_paths_in_cfg(cfg)
        cfg.freeze()

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)
        self.cfg = cfg
        
        # Only apply classifier for static modes (dynamic mode injects per frame)
        if classifier is not None:
            reset_cls_test(self.predictor.model, classifier)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = CustomVisualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances, args=self.cfg)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
