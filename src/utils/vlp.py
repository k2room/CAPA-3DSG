import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode

from vlpart.config import add_vlpart_config 
from .vlp_predictor import VisualizationDemo, reset_cls_test, get_clip_embeddings

class _DetResult:
    __slots__ = ("xyxy", "confidence", "class_id")
    def __init__(self, xyxy: np.ndarray, confidence: np.ndarray, class_id: np.ndarray):
        self.xyxy = xyxy          # (N,4) float32, xyxy
        self.confidence = confidence  # (N,) float32
        self.class_id = class_id      # (N,) int (classes index)


class VLPart(object):
    """
    VLPart Detectron2 파이프라인 래퍼.
    - 매 호출시 custom vocabulary(CLIP 임베딩)로 분류기(zs_weight_inference) 갱신
    - 예측 결과를 _DetResult(xyxy/confidence/class_id)로 변환
    """
    def __init__(
        self,
        cfg=None,                             # 사용자가 제시한 시그니처 유지
        instance_mode: ColorMode = ColorMode.IMAGE,
        parallel: bool = False,
        # ---- 유연한 생성자: main.py의 CAPA 분기 호환을 위해 키워드 인자도 허용 ----
        model_path: str = None,
        config_file: str = None,
        device: str = None,
        score_thresh: float = None,
    ):
        """
        인자 우선순위:
        1) 명시 인자(model_path/config_file/device/score_thresh)
        2) hydra cfg 객체의 필드(vlpart_ckpt_path, vlpart_cfg_path, device, box_thresh)
        """
        # 1) 소스 파라미터 정리
        self.model_path = model_path or (getattr(cfg, "vlpart_ckpt_path", None) if cfg is not None else None)
        self.config_file = config_file or (getattr(cfg, "vlpart_cfg_path", None) if cfg is not None else None)
        self.device = device or (getattr(cfg, "device", "cuda"))
        self.score_thresh = float(score_thresh if score_thresh is not None else getattr(cfg, "box_thresh", 0.5))

        if not self.model_path:
            raise ValueError("[VLPart] Unknown model_path(vlpart_ckpt_path)")
        if not self.config_file:
            raise ValueError("[VLPart] Unknown config_file(vlpart_cfg_path)")

        d2cfg = get_cfg()
        # add_vlpart_config 는 VisualizationDemo 내부에서 호출되는 설정 확장을 전제로 함
        # (VisualizationDemo(DefaultPredictor) 사용 시 cfg가 완비되어야 함)
        add_vlpart_config(d2cfg)
        d2cfg.merge_from_file(self.config_file)
        d2cfg.MODEL.WEIGHTS = self.model_path
        d2cfg.MODEL.DEVICE = self.device
        d2cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.score_thresh
        d2cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_thresh
        d2cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.score_thresh
        d2cfg.freeze()

        # 3) VisualizationDemo(DefaultPredictor) 구성
        #    - args는 None으로 두되, 매 호출시 분류기/메타데이터를 갱신하므로 문제 없음
        self.demo = VisualizationDemo(d2cfg, args=None, instance_mode=instance_mode, parallel=parallel)
        self.cpu = torch.device("cpu")

    @torch.no_grad()
    def predict_with_classes(
        self,
        image: np.ndarray,                 # BGR uint8 (cv2.imread 결과)
        classes: list,                     # ["chair","table", ...] 등 커스텀 어휘
        box_threshold: float = 0.5,        # GDINO 호환 시그니처(내부선 초기화된 값 사용)
        text_threshold: float = 0.5,       # VLPart 경로에서 미사용(호환성 유지용)
    ) -> _DetResult:
        """
        Returns:
            _DetResult: xyxy (N,4) float32, confidence (N,), class_id (N,)  *class_id는 classes 인덱스
        """
        # (1) 현재 호출의 vocabulary에 맞추어 CLIP 임베딩 생성 + 분류기 재설정
        #     predictor.model.roi_heads.box_predictor.* 에 zs_weight_inference 주입
        emb = get_clip_embeddings(classes)  # (dim, C)
        reset_cls_test(self.demo.predictor.model, emb)

        # 메타데이터 thing_classes도 동기화 (pred_classes 인덱스와 대응)
        if getattr(self.demo, "metadata", None) is not None:
            self.demo.metadata.thing_classes = list(classes)

        # (2) 예측 실행 (시각화는 불필요하므로 predictor 직접 호출)
        out = self.demo.predictor(image)  # dict with "instances" | "sem_seg" | "panoptic_seg"
        if "instances" not in out:
            # 빈 결과 반환
            return _DetResult(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros((0,), dtype=np.float32),
                class_id=np.zeros((0,), dtype=np.int32),
            )

        inst = out["instances"].to(self.cpu)
        if not hasattr(inst, "pred_boxes") or len(inst) == 0:
            return _DetResult(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros((0,), dtype=np.float32),
                class_id=np.zeros((0,), dtype=np.int32),
            )

        # (3) Detectron2 -> numpy 변환
        xyxy = inst.pred_boxes.tensor.numpy().astype(np.float32)            # (N,4)
        scores = inst.scores.numpy().astype(np.float32) if hasattr(inst, "scores") else np.ones((xyxy.shape[0],), np.float32)
        clsid = inst.pred_classes.numpy().astype(np.int32) if hasattr(inst, "pred_classes") else np.zeros((xyxy.shape[0],), np.int32)

        # (4) run_ram_branch 호환 포맷 반환
        return _DetResult(xyxy=xyxy, confidence=scores, class_id=clsid)
