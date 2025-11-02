import json
from collections import defaultdict
import numpy as np
from pathlib import Path


def aabb_from_obb(obb):
    pts = np.asarray(obb.get_box_points())
    mn = pts.min(axis=0); mx = pts.max(axis=0)
    return mn, mx

def overlap_1d(a1, a2):
    lo = max(a1[0], a2[0]); hi = min(a1[1], a2[1])
    return max(0.0, hi - lo)

def area_xy(mn, mx):
    return max(0.0, (mx[0] - mn[0])) * max(0.0, (mx[1] - mn[1]))

def overlap_xy_area(mnA, mxA, mnB, mxB):
    w = overlap_1d((mnA[0], mxA[0]), (mnB[0], mxB[0]))
    h = overlap_1d((mnA[1], mxA[1]), (mnB[1], mxB[1]))
    return w * h

def overlap_xz_area(mnA, mxA, mnB, mxB):
    w = overlap_1d((mnA[0], mxA[0]), (mnB[0], mxB[0]))
    d = overlap_1d((mnA[2], mxA[2]), (mnB[2], mxB[2]))
    return w * d

def overlap_yz_area(mnA, mxA, mnB, mxB):
    h = overlap_1d((mnA[1], mxA[1]), (mnB[1], mxB[1]))
    d = overlap_1d((mnA[2], mxA[2]), (mnB[2], mxB[2]))
    return h * d