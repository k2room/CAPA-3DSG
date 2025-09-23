import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional
from PIL import Image
import cv2
from types import SimpleNamespace

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode

from vlpart.config import add_vlpart_config
from .vlp_predictor import VisualizationDemo, reset_cls_test, get_clip_embeddings

class _DetResult:
    """Container for detections and optional embeddings."""
    __slots__ = ("xyxy", "confidence", "class_id", "mask",
                 "image_crops", "image_feats", "text_feats")

    def __init__(self,
                 xyxy: np.ndarray,
                 confidence: np.ndarray,
                 class_id: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 image_crops: Optional[List[np.ndarray]] = None,
                 image_feats: Optional[np.ndarray] = None,
                 text_feats: Optional[np.ndarray] = None):
        self.xyxy = xyxy                    # (N, 4) float32, XYXY
        self.confidence = confidence        # (N,) float32
        self.class_id = class_id            # (N,) int (index into `classes`)
        self.mask = mask                    # (N, H, W) bool or None
        self.image_crops = image_crops      # list of RGB crops or None
        self.image_feats = image_feats      # (N, D) float32 or None
        self.text_feats = text_feats        # (C, D) float32 or None


class VLPart(object):
    """
    VLPart wrapper that:
      - Refreshes zero-shot classifier weights from internal CLIP text encoder.
      - Runs inference and returns boxes/scores/classes/masks.
      - Captures per-ROI features from the last box classifier input as image embeddings.
      - Returns text embeddings in the same space as zs-weight for direct similarity.
    """
    def __init__(
        self,
        cfg=None,
        instance_mode: ColorMode = ColorMode.IMAGE,
        parallel: bool = False,                     # keep false (cause by LRU cache); To-do: support multi-gpu
        model_path: Optional[str] = None,
        config_file: Optional[str] = None,
        device: Optional[str] = None,
        score_thresh: Optional[float] = None,
    ):
        # ---- Resolve arguments (Hydra cfg fallback) ----
        self.model_path = model_path or (getattr(cfg, "vlpart_ckpt_path", None) if cfg is not None else None)
        self.config_file = config_file or (getattr(cfg, "vlpart_cfg_path", None) if cfg is not None else None)
        self.device = device or (getattr(cfg, "device", "cuda"))
        self.score_thresh = float(score_thresh if score_thresh is not None else getattr(cfg, "score_thresh", 0.2))
        if not self.model_path:
            raise ValueError("[VLPart] Unknown model_path (vlpart_ckpt_path)")
        if not self.config_file:
            raise ValueError("[VLPart] Unknown config_file (vlpart_cfg_path)")

        # ---- Build Detectron2 cfg ----
        d2cfg = get_cfg()
        add_vlpart_config(d2cfg)
        d2cfg.merge_from_file(self.config_file)
        d2cfg.MODEL.WEIGHTS = self.model_path
        d2cfg.MODEL.DEVICE = self.device
        d2cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_thresh
        d2cfg.freeze()

        # ---- Build predictor (VisualizationDemo wraps DefaultPredictor) ----
        dyn_args = SimpleNamespace(vocabulary="custom_dynamic")
        self.demo = VisualizationDemo(d2cfg, args=dyn_args, instance_mode=instance_mode, parallel=parallel)
        self.cpu = torch.device("cpu")

        # ---- Feature capture (ROI features right before cls_score) ----
        self._roi_feats_last = None
        self._feature_hooks = []
        self._hook_mode = "cls_score_pre"  # or "zs_forward"
        self._align_warn_threshold = 0.8  # warn when match ratio < 80%
        self._align_warn_limit = 3         # print at most 3 warnings to avoid log spam
        self._align_warn_count = 0
        self._install_feature_hooks()

    def _install_feature_hooks(self):
        """
        Register forward pre-hooks on the box predictor's cls_score modules.
        This captures per-ROI features right before classification (shared D-dim space).
        """
        model = self.demo.predictor.model
        if not hasattr(model, "roi_heads"):
            return
        bp = getattr(model.roi_heads, "box_predictor", None)
        if bp is None:
            return

        def _save_feats(module, args):
            # args[0] is the input tensor to cls_score: shape (N, D)
            x = args[0]
            if isinstance(x, (list, tuple)):
                x = x[0]
            if torch.is_tensor(x):
                self._roi_feats_last = x.detach()

        # Cascade may use ModuleList of predictors; single-stage is a single module.
        if isinstance(bp, torch.nn.ModuleList):
            for m in bp:
                if hasattr(m, "cls_score"):
                    self._feature_hooks.append(m.cls_score.register_forward_pre_hook(_save_feats))
        else:
            if hasattr(bp, "cls_score"):
                self._feature_hooks.append(bp.cls_score.register_forward_pre_hook(_save_feats))

    def _ensure_embedding_alignment(self, image_feats: torch.Tensor,
                                    text_feats: torch.Tensor,
                                    clsid_np: np.ndarray) -> torch.Tensor:
        """Normalize, measure top-1 agreement vs gt class IDs, warn if low."""
        i = F.normalize(image_feats, p=2, dim=-1)   # (N, D)
        t = F.normalize(text_feats, p=2, dim=-1)    # (C, D)
        sim = i @ t.T                               
        pred = sim.argmax(dim=1).cpu().numpy()
        match = np.mean((pred == clsid_np)) if clsid_np.size > 0 else 1.0
        if (match < self._align_warn_threshold) and (self._align_warn_count < self._align_warn_limit):
            n = int(clsid_np.size); k = min(5, n)
            print(f"[VLPart][Alignment WARN] low image/text alignment: match={match:.3f} (N={n})")
            print(f"[VLPart][Alignment INFO] hook_mode={self._hook_mode}, top{k} pred={pred[:k]} vs gt={clsid_np[:k]}")
            self._align_warn_count += 1
        return i

    @torch.no_grad()
    def predict_with_classes(self, image: np.ndarray, classes: List[str]) -> _DetResult:
        """
        Build zs-weight from the *current* vocabulary, run inference, and return
        detections + (N,D) image_feats + (C,D) text_feats (all L2 normalized).
        """
        # Always build zs-weight for current tags (CAPA vocab changes per frame)
        zs_weight_dc = get_clip_embeddings(classes)   # (D, C) on CPU
        reset_cls_test(self.demo.predictor.model, zs_weight_dc)


        # Inference
        self._roi_feats_last = None
        out = self.demo.predictor(image)
        if "instances" not in out:
            return _DetResult(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros((0,), dtype=np.float32),
                class_id=np.zeros((0,), dtype=np.int32),
                mask=None, image_crops=None, image_feats=None, text_feats=None
            )

        inst = out["instances"].to(self.cpu)
        if not hasattr(inst, "pred_boxes") or len(inst) == 0:
            return _DetResult(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros((0,), dtype=np.float32),
                class_id=np.zeros((0,), dtype=np.int32),
                mask=None, image_crops=None, image_feats=None, text_feats=None
            )

        # Detections
        xyxy = inst.pred_boxes.tensor.numpy().astype(np.float32)
        scores = inst.scores.numpy().astype(np.float32) if hasattr(inst, "scores") else np.ones((xyxy.shape[0],), np.float32)
        clsid  = inst.pred_classes.numpy().astype(np.int32) if hasattr(inst, "pred_classes") else np.zeros((xyxy.shape[0],), np.int32)
        mask = inst.pred_masks.numpy().astype(bool) if hasattr(inst, "pred_masks") else None

        # Crops (masked if available)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crops = []
        N = xyxy.shape[0]
        for i in range(N):
            mi = mask[i] if (mask is not None and mask.shape[0] == N) else None
            # zero background in the crop if mask is present
            H, W = rgb.shape[:2]
            x1, y1, x2, y2 = xyxy[i].astype(int)
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(W - 1, x2); y2 = min(H - 1, y2)
            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((1,1,3), dtype=np.uint8))
            else:
                crop = rgb[y1:y2, x1:x2].copy()
                if mi is not None:
                    m_crop = mi[y1:y2, x1:x2].astype(np.uint8)
                    if m_crop.size > 0:
                        crop[m_crop == 0] = 0
                crops.append(crop)

        # Text embeddings (C, D): transpose & L2-norm
        if isinstance(zs_weight_dc, torch.Tensor):
            tfeat = zs_weight_dc.permute(1, 0).contiguous()  # (C, D)
        else:
            tfeat = torch.from_numpy(np.asarray(zs_weight_dc)).permute(1, 0).contiguous()
        tfeat = F.normalize(tfeat.to(torch.float32), p=2, dim=-1)

        # Image embeddings from the zero-shot stream
        if self._roi_feats_last is not None and torch.is_tensor(self._roi_feats_last):
            img_feats = self._ensure_embedding_alignment(
                image_feats=self._roi_feats_last.to(self.cpu).to(torch.float32),
                text_feats=tfeat,
                clsid_np=clsid
            )
            image_feats = img_feats.cpu().numpy()
        else:
            image_feats = None

        return _DetResult(
            xyxy=xyxy, confidence=scores, class_id=clsid,
            mask=mask, image_crops=crops,
            image_feats=image_feats, text_feats=tfeat.cpu().numpy()
        )