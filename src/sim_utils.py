import math
from math import hypot


# -----------------------------
# Utils
# -----------------------------
def safe_float(x, default=None):
    """Ép float an toàn; nếu NaN/Inf/None/lỗi -> default."""
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def dist(a, b):
    """Khoảng cách Euclid giữa 2 điểm 2D a=(x,y), b=(x,y)."""
    return float(hypot(a[0] - b[0], a[1] - b[1]))
