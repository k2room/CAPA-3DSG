# file: part_shape_from_pklgz.py
import gzip, pickle, numpy as np, pandas as pd
import argparse, sys

def is_numeric_array(x):
    try:
        return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number)
    except Exception:
        return False

def gather_pointclouds(obj, path="root", max_items=5000):
    """
    재귀 탐색으로 (경로, Nx3 float32, N) 목록을 수집.
    - shape (N,3) 또는 (N,>=3) 배열만 채택, N>=10
    - 너무 큰 배열은 나중에 필터링
    """
    out, stack, seen = [], [(obj, path)], set()
    while stack:
        node, p = stack.pop()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        try:
            if isinstance(node, dict):
                for k, v in node.items():
                    stack.append((v, f"{p}/{k}"))
            elif isinstance(node, (list, tuple)):
                for i, v in enumerate(node):
                    stack.append((v, f"{p}[{i}]"))
            elif is_numeric_array(node):
                arr = node
                if arr.ndim == 2 and arr.shape[0] >= 10 and arr.shape[1] >= 3:
                    pts = np.ascontiguousarray(arr[:, :3]).astype(np.float32, copy=False)
                    out.append((p, pts, int(arr.shape[0])))
            elif hasattr(node, "__dict__"):
                stack.append((node.__dict__, f"{p}/__dict__"))
        except Exception:
            pass
        if len(out) >= max_items:
            break
    return out

def pca_features(pts, max_used=50000):
    n = pts.shape[0]
    if n > max_used:
        sel = np.random.choice(n, size=max_used, replace=False)
        X = pts[sel]
    else:
        X = pts
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(1, Xc.shape[0]-1)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    l1, l2, l3 = float(vals[0]), float(vals[1]), float(vals[2])
    eps = 1e-12
    linearity  = (l1 - l2) / (l1 + eps)
    planarity  = (l2 - l3) / (l1 + eps)
    sphericity =  l3 / (l1 + eps)
    anisotropy = (l1 - l3) / (l1 + eps)
    proj = Xc @ vecs
    L1, L2, L3 = float(proj[:,0].ptp()), float(proj[:,1].ptp()), float(proj[:,2].ptp())
    R12, R23 = L1/max(L2, eps), L2/max(L3, eps)
    return dict(
        l1=l1,l2=l2,l3=l3, linearity=float(linearity),
        planarity=float(planarity), sphericity=float(sphericity),
        anisotropy=float(anisotropy), L1=L1,L2=L2,L3=L3,
        R12=float(R12), R23=float(R23), n_used=int(X.shape[0])
    )

def classify_shape(feat):
    # 권장 초기 임계값 (데이터로 미세튜닝 권장)
    if feat["sphericity"] > 0.30 and feat["R12"] < 1.5 and feat["R23"] < 1.5:
        return "roundish"   # knob/button 계열
    if feat["linearity"] > 0.60 and feat["R12"] > 2.0 and feat["R23"] < 2.0:
        return "elongated"  # handle/bar/lever 계열
    if feat["planarity"] > 0.60 and feat["R23"] > 2.0:
        return "flat_plate" # 얇은 플랩/레버판
    return "ambiguous"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to .pkl.gz")
    ap.add_argument("--output", default="part_shape_summary.csv", help="output CSV")
    ap.add_argument("--strict_names", action="store_true",
                    help="경로명에 part/handle/knob/button/lever/component/instance 포함된 배열만")
    ap.add_argument("--max_candidates", type=int, default=500, help="최대 후보 개수")
    args = ap.parse_args()

    # 1) 로드
    with gzip.open(args.input, "rb") as f:
        data = pickle.load(f)

    # 2) 후보 추출
    cands = gather_pointclouds(data, path="root", max_items=5000)

    # 3) part-ish 필터
    def is_partish(path: str):
        pl = path.lower()
        keys = ["part","handle","knob","button","lever","component","instance"]
        return any(k in pl for k in keys)

    scene_scale = max((N for _,_,N in cands), default=0)
    size_thr = max(5000, int(scene_scale * 0.25))  # 파트는 보통 씬의 25% 미만
    selected, fallback = [], []
    for p, pts, N in cands:
        if (not args.strict_names or is_partish(p)) and N <= max(250_000, size_thr):
            selected.append((p, pts, N))
        else:
            if N <= 150_000 and len(fallback) < 100:
                fallback.append((p, pts, N))
    if not selected and fallback:
        selected = fallback
    selected = selected[:args.max_candidates]

    # 4) 피처 및 라벨
    rows = []
    for p, pts, N in selected:
        try:
            feat = pca_features(pts)
            label = classify_shape(feat)
            rows.append({
                "id_path": p, "n_points": N,
                "L1": feat["L1"], "L2": feat["L2"], "L3": feat["L3"],
                "R12": feat["R12"], "R23": feat["R23"],
                "linearity": feat["linearity"], "planarity": feat["planarity"],
                "sphericity": feat["sphericity"], "anisotropy": feat["anisotropy"],
                "n_used_for_pca": feat["n_used"], "shape_label": label
            })
        except Exception as e:
            rows.append({"id_path": p, "n_points": N, "error": str(e)})

    df = pd.DataFrame(rows).sort_values(
        by=["n_points","shape_label"], ascending=[True, True]
    ).reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print(f"[done] candidates_in={len(cands)}, selected={len(selected)}, saved={args.output}")

if __name__ == "__main__":
    main()
