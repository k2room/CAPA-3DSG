"""
Fast, training-free color features for 2D masks.
- Lab a*b* 2D histogram (Hab)
- Opponent o1*o2 2D histogram (Hopp, optional)
- normalized r,g 2D histogram (Hrg, optional)
- optional 11-bin Color Names histogram (Hcn)
- robust medians for a*, b*, r, g (optional)
- HSV-based gating to drop low-chroma / specular pixels

Usage:
    feats = extract_color_features(cfg, img_rgb_u8=image_rgb, masks=masks_bool, params=dict(...))

Notes:
    - image_rgb must be uint8 RGB (0..255)
    - masks: (N, H, W) bool/uint8
    - returns: list of dicts per mask: (subset of) {Hab, Hopp, Hrg, Hcn, med}
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
import torch
from omegaconf import DictConfig 

def _ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img, 0, 255)
    if img.max() <= 1.0:
        img = (img * 255.0).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img

def _gray_world_wb_u8(img_rgb_u8: np.ndarray) -> np.ndarray:
    img = img_rgb_u8.astype(np.float32) / 255.0
    mean = img.reshape(-1, 3).mean(axis=0) + 1e-6
    gain = mean.mean() / mean
    out = np.clip(img * gain[None, None, :], 0, 1)
    return (out * 255.0 + 0.5).astype(np.uint8)

def _retinex_ssr_u8(img_rgb_u8: np.ndarray, radius: int) -> np.ndarray:
    # Single-Scale Retinex (per channel): log(I) - log(blur(I))
    f = img_rgb_u8.astype(np.float32) / 255.0
    blur = cv2.boxFilter(f, ddepth=-1, ksize=(radius, radius), normalize=True)
    eps = 1e-6
    r = np.log(f + eps) - np.log(blur + eps)
    # normalize to 0..1 per channel
    r -= r.min(axis=(0, 1), keepdims=True)
    r /= (r.max(axis=(0, 1), keepdims=True) + 1e-6)
    return (r * 255.0 + 0.5).astype(np.uint8)

def _rgb2hsv_float(img_rgb_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2HSV)  # H:0..179, S:0..255, V:0..255
    hsv = hsv.astype(np.float32)
    h = hsv[..., 0] * 2.0
    s = hsv[..., 1] / 255.0  # 0..1
    v = hsv[..., 2] / 255.0  # 0..1
    return h, s, v

def _rgb2ab_float(img_rgb_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2LAB)  # uint8 L:0..255, a/b:0..255 (128 is zero)
    a = lab[..., 1].astype(np.float32) - 128.0
    b = lab[..., 2].astype(np.float32) - 128.0
    return a, b

def _rgb2opp_float(img_rgb_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Opponent: o1=(R-G)/sqrt(2), o2=(R+G-2B)/sqrt(6)  (o3 intensity not used)
    f = img_rgb_u8.astype(np.float32) / 255.0
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    o1 = (R - G) / np.sqrt(2.0)
    o2 = (R + G - 2.0 * B) / np.sqrt(6.0)
    return o1, o2

def _hist2d(values_x: np.ndarray, values_y: np.ndarray, bins: int,
            range_: Tuple[Tuple[float, float], Tuple[float, float]]) -> np.ndarray:
    H, _, _ = np.histogram2d(values_x, values_y, bins=bins, range=range_)
    H = H.astype(np.float32)
    H /= (H.sum() + 1e-12)
    return H

def _color_names_idx(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    # 11 bins: 0:black,1:white,2:gray,3:red,4:orange,5:yellow,6:green,7:cyan,8:blue,9:purple,10:brown
    idx = np.full(h.shape, -1, dtype=np.int16)
    idx[v < 0.2] = 0
    idx[(s < 0.1) & (v >= 0.8)] = 1
    idx[(s < 0.1) & (idx < 0)] = 2
    chroma = (s >= 0.1) & (idx < 0)
    hh = h % 360.0

    def in_range(lo, hi):
        if lo <= hi:
            return (hh >= lo) & (hh < hi)
        else:
            return (hh >= lo) | (hh < hi)

    red = in_range(345, 360) | in_range(0, 15)
    orange = in_range(15, 45)
    yellow = in_range(45, 75)
    green = in_range(75, 165)
    cyan = in_range(165, 195)
    blue = in_range(195, 255)
    purple = in_range(255, 285)
    pink = in_range(285, 330)
    brown = orange & (v < 0.5)

    idx[chroma & red] = 3
    idx[chroma & orange & ~brown] = 4
    idx[chroma & yellow] = 5
    idx[chroma & green] = 6
    idx[chroma & cyan] = 7
    idx[chroma & blue] = 8
    idx[chroma & purple] = 9
    # merge pink into red (keep 11 bins)
    idx[chroma & pink] = 3
    idx[chroma & brown] = 10
    idx[(idx < 0) & chroma] = 3  # fallback to red
    return idx

def _compute_rg(img_rgb_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    f = img_rgb_u8.astype(np.float32) / 255.0
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    S = R + G + B + 1e-6
    r = R / S
    g = G / S
    return r, g

def _mask_and_gate(mask: np.ndarray, h: np.ndarray, s: np.ndarray, v: np.ndarray,
                   s_th: float, v_spec: float) -> np.ndarray:
    # drop low-chroma; drop specular: (V > v_spec) & (S < 0.1)
    gate = (s >= s_th) & ~((v > v_spec) & (s < 0.1))
    return (mask.astype(bool) & gate)

# ---------------------------- public API ----------------------------

def extract_color_feature_for_mask(
    cfg: Optional["DictConfig"],
    img_rgb_u8: np.ndarray,
    mask_bool: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extract color feature for a single mask.
    Returns None if no valid pixels after gating.
    - cfg.color_feat_params must define all needed keys.
    - params (if provided) overrides cfg.color_feat_params.
    """
    P = dict(cfg.color_feat_params) if cfg is not None else {}
    if params:
        P.update(params)

    img = _ensure_uint8_rgb(img_rgb_u8)

    if P["use_wb"]:
        img = _gray_world_wb_u8(img)
    if P["use_retinex"]:
        img = _retinex_ssr_u8(img, int(P["retinex_radius"]))

    h, s, v = _rgb2hsv_float(img)
    gate = _mask_and_gate(mask_bool, h, s, v, float(P["s_threshold"]), float(P["v_spec_threshold"]))
    if gate.sum() < 1:
        return None

    a, b = _rgb2ab_float(img)
    clip_ab = float(P["clip_ab"])
    a = np.clip(a, -clip_ab, clip_ab)
    b = np.clip(b, -clip_ab, clip_ab)

    # optional channels
    r = g = None
    if P["compute_rg"]:
        r, g = _compute_rg(img)
    o1 = o2 = None
    if P["compute_opp"]:
        o1, o2 = _rgb2opp_float(img)

    aa = a[gate]
    bb = b[gate]

    bins = int(P["bins"])
    out: Dict[str, Any] = {}

    # Lab a*b* hist
    Hab = _hist2d(aa, bb, bins=bins, range_=((-clip_ab, clip_ab), (-clip_ab, clip_ab)))
    out["Hab"] = Hab

    # rg hist (optional)
    if r is not None:
        rr = r[gate]; gg = g[gate]
        Hrg = _hist2d(rr, gg, bins=bins, range_=((0.0, 1.0), (0.0, 1.0)))
        out["Hrg"] = Hrg
    else:
        out["Hrg"] = None

    # opponent hist (optional)
    if o1 is not None:
        Hopp = _hist2d(o1[gate], o2[gate], bins=bins, range_=((-1.0, 1.0), (-1.0, 1.0)))
        out["Hopp"] = Hopp
    else:
        out["Hopp"] = None

    # medians (optional)
    if P["compute_med"]:
        med_a = np.median(aa).astype(np.float32)
        med_b = np.median(bb).astype(np.float32)
        if r is not None:
            rr = r[gate]; gg = g[gate]
            med_r = np.median(rr).astype(np.float32)
            med_g = np.median(gg).astype(np.float32)
        else:
            med_r = med_g = 0.0
        out["med"] = np.array([med_a, med_b, med_r, med_g], dtype=np.float32)
    else:
        out["med"] = None

    # Color-Names (optional)
    if P["compute_cn"]:
        idx = _color_names_idx(h, s, v)[gate]
        Hcn = np.bincount(np.clip(idx, 0, 10), minlength=11).astype(np.float32)
        Hcn /= (Hcn.sum() + 1e-12)
        out["Hcn"] = Hcn
    else:
        out["Hcn"] = None

    return out

def extract_color_features(
    cfg: Optional["DictConfig"],
    img_rgb_u8: np.ndarray,
    masks: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> List[Optional[Dict[str, Any]]]:
    """
    Extract color features for all masks.
    - cfg.color_feat_params must define all needed keys.
    - params (if provided) overrides cfg.color_feat_params.
    - img_rgb_u8: HxWx3 uint8 RGB
    - masks: (N,H,W) bool/uint8
    Returns list of length N (feature dict or None).
    """
    if masks is None or len(masks) == 0:
        return []

    # merge params (without defaults)
    P = dict(cfg.color_feat_params) if cfg is not None else {}
    if params:
        P.update(params)

    img = _ensure_uint8_rgb(img_rgb_u8)
    masks_bool = masks.astype(bool)

    if P["use_wb"]:
        img = _gray_world_wb_u8(img)
    if P["use_retinex"]:
        img = _retinex_ssr_u8(img, int(P["retinex_radius"]))

    # shared per-frame conversions
    h, s, v = _rgb2hsv_float(img)
    a, b = _rgb2ab_float(img)
    clip_ab = float(P["clip_ab"])
    a = np.clip(a, -clip_ab, clip_ab)
    b = np.clip(b, -clip_ab, clip_ab)

    r = g = None
    if P["compute_rg"]:
        r, g = _compute_rg(img)
    o1 = o2 = None
    if P["compute_opp"]:
        o1, o2 = _rgb2opp_float(img)

    bins = int(P["bins"])
    HCN_IDX_ALL = _color_names_idx(h, s, v) if P["compute_cn"] else None

    feats: List[Optional[Dict[str, Any]]] = [] # list of dicts per mask
    for i in range(masks_bool.shape[0]):
        mask = masks_bool[i]
        gate = _mask_and_gate(mask, h, s, v, float(P["s_threshold"]), float(P["v_spec_threshold"]))
        nvalid = int(gate.sum())
        if nvalid < 1:
            feats.append(None)
            continue

        aa = a[gate]; bb = b[gate]
        out: Dict[str, Any] = {}

        # Lab a*b* hist
        Hab = _hist2d(aa, bb, bins=bins, range_=((-clip_ab, clip_ab), (-clip_ab, clip_ab)))
        out["Hab"] = Hab

        # rg hist (optional)
        if r is not None:
            rr = r[gate]; gg = g[gate]
            Hrg = _hist2d(rr, gg, bins=bins, range_=((0.0, 1.0), (0.0, 1.0)))
            out["Hrg"] = Hrg
        else:
            out["Hrg"] = None

        # opponent hist (optional)
        if o1 is not None:
            Hopp = _hist2d(o1[gate], o2[gate], bins=bins, range_=((-1.0, 1.0), (-1.0, 1.0)))
            out["Hopp"] = Hopp
        else:
            out["Hopp"] = None

        # medians (optional)
        if P["compute_med"]:
            if r is not None:
                rr = r[gate]; gg = g[gate]
                med_rg = [np.median(rr), np.median(gg)]
            else:
                med_rg = [0.0, 0.0]
            out["med"] = np.array([np.median(aa), np.median(bb), *med_rg], dtype=np.float32)
        else:
            out["med"] = None

        # Color-Names (optional)
        if HCN_IDX_ALL is not None:
            idx = HCN_IDX_ALL[gate]
            Hcn = np.bincount(np.clip(idx, 0, 10), minlength=11).astype(np.float32)
            Hcn /= (Hcn.sum() + 1e-12)
            out["Hcn"] = Hcn
        else:
            out["Hcn"] = None

        feats.append(out)

    return feats

# ---------------------------- distances (optional) ----------------------------

def chi2(H1: np.ndarray, H2: np.ndarray) -> float:
    denom = (H1 + H2 + 1e-12)
    return 0.5 * float(np.sum((H1 - H2) ** 2 / denom))

def color_distance(feat_obs: Dict[str, Any], feat_ref: Dict[str, Any], w=None) -> float:
    """
    Weighted sum of available terms. w can be dict keys {'ab','opp','rg','cn','med'}.
    If w is None, use a reasonable default for (ab,opp,cn) only.
    """
    if w is None:
        w = {"ab": 0.4, "opp": 0.4, "cn": 0.2, "rg": 0.0, "med": 0.0}
    if not isinstance(w, dict):
        w = {"ab": w[0], "rg": w[1], "cn": w[2], "med": w[3], "opp": 0.0}

    d_ab  = chi2(feat_obs.get("Hab"),  feat_ref.get("Hab"))  if (feat_obs.get("Hab")  is not None and feat_ref.get("Hab")  is not None) else 0.0
    d_opp = chi2(feat_obs.get("Hopp"), feat_ref.get("Hopp")) if (feat_obs.get("Hopp") is not None and feat_ref.get("Hopp") is not None) else 0.0
    d_rg  = chi2(feat_obs.get("Hrg"),  feat_ref.get("Hrg"))  if (feat_obs.get("Hrg")  is not None and feat_ref.get("Hrg")  is not None) else 0.0
    d_cn  = chi2(feat_obs.get("Hcn"),  feat_ref.get("Hcn"))  if (feat_obs.get("Hcn")  is not None and feat_ref.get("Hcn")  is not None) else 0.0
    d_md  = float(np.linalg.norm(feat_obs["med"] - feat_ref["med"])) if (feat_obs.get("med") is not None and feat_ref.get("med") is not None) else 0.0

    return (w.get("ab", 0.0)  * d_ab  +
            w.get("opp", 0.0) * d_opp +
            w.get("rg", 0.0)  * d_rg  +
            w.get("cn", 0.0)  * d_cn  +
            w.get("med", 0.0) * d_md)

# ---------------------------- texture similarity (pairwise) ----------------------------
def _dist_to_sim(d: float, mapping: str = "inv", gamma: float = 3.0) -> float:
    """
    Convert a non-negative distance to similarity in [0,1].
    - "inv":    sim = 1/(1+d)            (scale-free, robust default)
    - "exp":    sim = exp(-gamma * d)    (requires tuning gamma)
    """
    if mapping == "exp":
        return float(np.exp(-gamma * max(d, 0.0)))
    return float(1.0 / (1.0 + max(d, 0.0)))

def compute_texture_sim(
    det_color_feats: list,
    obj_color_feats: list,
    weights=None,
    mapping: str = "inv",
    gamma: float = 3.0,
):
    """
    Compute an MxN texture-similarity matrix from two python lists of color feature dicts.
    Each feature dict should be output of extract_color_feature(s).
    Returns a torch.FloatTensor (M,N) if torch is available, else a numpy array.
    """
    M, N = len(det_color_feats), len(obj_color_feats)
    S = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        fi = det_color_feats[i]
        if fi is None:
            continue
        for j in range(N):
            fj = obj_color_feats[j]
            if fj is None:
                continue
            d = color_distance(fi, fj, w=weights)
            S[i, j] = _dist_to_sim(d, mapping=mapping, gamma=gamma)
    if torch is not None:
        return torch.from_numpy(S)
    return S

# ---------------------------- EMA utilities ----------------------------

def ema_update_color_feat(model_feat: Dict[str, Any], obs_feat: Dict[str, Any], alpha: float = 0.3) -> Dict[str, Any]:
    """
    Exponentially weighted moving average update for a single object's color feature.
    - Histograms are L1-renormalized after blending.
    - Medians are updated linearly (if present).
    """
    if model_feat is None:
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs_feat.items()}

    out: Dict[str, Any] = {}
    # blend histograms that exist
    for key in ("Hab", "Hrg", "Hopp"):
        Hm = model_feat.get(key)
        Ho = obs_feat.get(key)
        if Hm is None and Ho is None:
            continue
        if Hm is None and Ho is not None:
            out[key] = Ho.astype(np.float32)
            continue
        if Ho is None and Hm is not None:
            out[key] = Hm.astype(np.float32)
            continue
        Hm = Hm.astype(np.float32); Ho = Ho.astype(np.float32)
        H = (1 - alpha) * Hm + alpha * Ho
        H /= (H.sum() + 1e-12)
        out[key] = H

    # optional CN
    if (model_feat.get("Hcn") is not None) and (obs_feat.get("Hcn") is not None):
        Hm = model_feat["Hcn"].astype(np.float32)
        Ho = obs_feat["Hcn"].astype(np.float32)
        H = (1 - alpha) * Hm + alpha * Ho
        H /= (H.sum() + 1e-12)
        out["Hcn"] = H
    else:
        out["Hcn"] = model_feat.get("Hcn", None)

    # medians (optional)
    if (model_feat.get("med") is not None) and (obs_feat.get("med") is not None):
        out["med"] = (1 - alpha) * model_feat["med"].astype(np.float32) + alpha * obs_feat["med"].astype(np.float32)
    else:
        out["med"] = model_feat.get("med", None)
    return out
