import sys
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QRadioButton, QLabel, QSpinBox, QDoubleSpinBox,
    QButtonGroup, QFrame, QScrollArea, QFileDialog,
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QFont


# ── Coordinate system ─────────────────────────────────────────────────────────
# Drawing space: x ∈ [15, 35], y ∈ [15, 35] (y increases upward)

COORD_MIN   = 15.0
COORD_MAX   = 35.0
COORD_RANGE = COORD_MAX - COORD_MIN   # 20 units

PX_PER_UNIT = 30                           # pixels per coordinate unit
PLOT_W = int(COORD_RANGE * PX_PER_UNIT)   # 600 px
PLOT_H = int(COORD_RANGE * PX_PER_UNIT)   # 600 px

ML = 52   # left margin  (y-axis labels)
MR = 14   # right margin
MT = 14   # top margin
MB = 48   # bottom margin (x-axis labels)

CANVAS_W = ML + PLOT_W + MR   # 666
CANVAS_H = MT + PLOT_H + MB   # 662


def to_px(cx: float, cy: float) -> tuple[float, float]:
    """Coordinate space → canvas pixel (coord y increases upward)."""
    return (
        ML + (cx - COORD_MIN) / COORD_RANGE * PLOT_W,
        MT + (1.0 - (cy - COORD_MIN) / COORD_RANGE) * PLOT_H,
    )


def to_coord(px: float, py: float) -> tuple[float, float]:
    """Canvas pixel → coordinate space."""
    return (
        COORD_MIN + (px - ML) / PLOT_W * COORD_RANGE,
        COORD_MIN + (1.0 - (py - MT) / PLOT_H) * COORD_RANGE,
    )


# ── Canvas ────────────────────────────────────────────────────────────────────

class Canvas(QWidget):
    shape_added = pyqtSignal(str, int)

    def __init__(self):
        super().__init__()
        self.shapes: list[dict] = []
        self.current: dict | None = None
        self.tool = "line"
        self.dist_pts: list[tuple[float, float]] = []
        self._preview_pos: tuple[float, float] | None = None  # coord space

        self.setFixedSize(CANVAS_W, CANVAS_H)
        self.setMouseTracking(True)

    def set_tool(self, tool: str):
        if tool != self.tool and self.current:
            self.current = None
        self.tool = tool
        self._preview_pos = None
        self.update()

    # ── Mouse events (pixel → coordinate on entry) ────────────────────────────

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        cx, cy = to_coord(float(event.x()), float(event.y()))

        if self.tool == "line":
            self.current = {"type": "line", "p1": (cx, cy), "p2": (cx, cy)}

        elif self.tool == "bezier":
            if self.current is None:
                self.current = {"type": "bezier", "anchors": []}
            self.current["anchors"].append({"pos": (cx, cy), "h_in": None, "h_out": None})

        elif self.tool == "circle":
            self.current = {"type": "circle", "cx": cx, "cy": cy, "r": 0.0}

        self.update()

    def mouseMoveEvent(self, event):
        cx, cy = to_coord(float(event.x()), float(event.y()))

        if self.tool == "bezier":
            if event.buttons() & Qt.LeftButton:
                if self.current and self.current["anchors"]:
                    anchor = self.current["anchors"][-1]
                    ax, ay = anchor["pos"]
                    dx, dy = cx - ax, cy - ay
                    if math.hypot(dx, dy) > 0.1:           # ~3px threshold in coord
                        anchor["h_out"] = (cx, cy)
                        anchor["h_in"]  = (ax - dx, ay - dy)   # mirror
            else:
                self._preview_pos = (cx, cy)
            self.update()
            return

        if not self.current or not (event.buttons() & Qt.LeftButton):
            return
        t = self.current["type"]
        if t == "line":
            self.current["p2"] = (cx, cy)
        elif t == "circle":
            self.current["r"] = math.hypot(cx - self.current["cx"],
                                            cy - self.current["cy"])
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        cx, cy = to_coord(float(event.x()), float(event.y()))

        if self.tool == "bezier":
            self.update()
            return

        if not self.current:
            return
        t = self.current["type"]
        if t == "line":
            self.current["p2"] = (cx, cy)
            valid = math.hypot(cx - self.current["p1"][0],
                               cy - self.current["p1"][1]) > 0.15
        elif t == "circle":
            valid = self.current["r"] > 0.15
        else:
            valid = False

        if valid:
            self.shapes.append(self.current)
            self.shape_added.emit(t, len(self.shapes))
        self.current = None
        self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        if self.tool == "bezier" and self.current:
            anchors = self.current["anchors"]
            if anchors:
                anchors.pop()   # remove anchor added by double-click's press
            if len(anchors) >= 2:
                self.shapes.append(self.current)
                self.shape_added.emit("bezier", len(self.shapes))
            self.current = None
            self._preview_pos = None
            self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.tool == "bezier":
            self.current = None
            self._preview_pos = None
            self.update()

    # ── Public API ────────────────────────────────────────────────────────────

    def undo(self) -> str | None:
        """Remove the last committed shape; returns its type, or None if empty."""
        if not self.shapes:
            return None
        removed = self.shapes.pop()
        self.dist_pts.clear()
        self.update()
        return removed["type"]

    def clear(self):
        self.shapes.clear()
        self.dist_pts.clear()
        self.current = None
        self._preview_pos = None
        self.update()

    def distribute(self, n: int, min_dist: float) -> int:
        self.dist_pts = _place_points(self.shapes, n, min_dist)
        self.update()
        return len(self.dist_pts)

    # ── Painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Canvas background outside plot
        p.fillRect(self.rect(), QColor("#f8f8fa"))

        _paint_grid(p)

        # Clip drawing to plot area so shapes don't overflow labels
        p.setClipRect(QRectF(ML, MT, PLOT_W, PLOT_H))

        # Finalized shapes
        p.setBrush(Qt.NoBrush)
        for shape in self.shapes:
            p.setPen(QPen(QColor("#1e1e2e"), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            _paint_shape(p, shape)

        # In-progress bezier
        if self.current and self.current["type"] == "bezier":
            anchors = self.current["anchors"]
            if len(anchors) >= 2:
                p.setPen(QPen(QColor("#89b4fa"), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                p.setBrush(Qt.NoBrush)
                _paint_bezier_path(p, anchors)
            _paint_bezier_controls(p, anchors)
            if anchors and self._preview_pos:
                last = anchors[-1]
                ppx, ppy = to_px(*self._preview_pos)
                ax, ay = to_px(*last["pos"])
                h_out = last.get("h_out")
                p.setPen(QPen(QColor("#89b4fa"), 1.2, Qt.DashLine))
                if h_out:
                    hpx, hpy = to_px(*h_out)
                    path = QPainterPath()
                    path.moveTo(ax, ay)
                    path.cubicTo(hpx, hpy, ppx, ppy, ppx, ppy)
                    p.drawPath(path)
                else:
                    p.drawLine(QPointF(ax, ay), QPointF(ppx, ppy))

        # In-progress line / circle
        elif self.current:
            p.setPen(QPen(QColor("#89b4fa"), 2.0, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            _paint_shape(p, self.current)

        # Distributed points (coordinate space → pixel)
        if self.dist_pts:
            p.setPen(QPen(QColor("#d20f39"), 1))
            p.setBrush(QBrush(QColor("#f38ba8")))
            for cx, cy in self.dist_pts:
                px, py = to_px(cx, cy)
                p.drawEllipse(QRectF(px - 3.5, py - 3.5, 7.0, 7.0))

        p.setClipping(False)


# ── Grid / axes ───────────────────────────────────────────────────────────────

def _paint_grid(p: QPainter):
    # White plot background
    p.fillRect(QRectF(ML, MT, PLOT_W, PLOT_H), QColor("white"))

    # Minor grid lines at every 0.5 unit (faint)
    p.setPen(QPen(QColor("#ebebf0"), 0.6))
    v = COORD_MIN + 0.5
    while v < COORD_MAX:
        gx = ML + (v - COORD_MIN) / COORD_RANGE * PLOT_W
        gy = MT + (1.0 - (v - COORD_MIN) / COORD_RANGE) * PLOT_H
        p.drawLine(QPointF(gx, MT), QPointF(gx, MT + PLOT_H))
        p.drawLine(QPointF(ML, gy), QPointF(ML + PLOT_W, gy))
        v += 1.0

    # Major grid lines at every integer (light); red at 20 and 30
    for i in range(int(COORD_MIN), int(COORD_MAX) + 1):
        gx = ML + (i - COORD_MIN) / COORD_RANGE * PLOT_W
        gy = MT + (1.0 - (i - COORD_MIN) / COORD_RANGE) * PLOT_H
        if i in (20, 30):
            p.setPen(QPen(QColor("#ef4444"), 0.9))
        else:
            p.setPen(QPen(QColor("#d4d4d8"), 0.8))
        p.drawLine(QPointF(gx, MT), QPointF(gx, MT + PLOT_H))
        p.drawLine(QPointF(ML, gy), QPointF(ML + PLOT_W, gy))

    # Plot border
    p.setPen(QPen(QColor("#52525b"), 1.2))
    p.drawRect(QRectF(ML, MT, PLOT_W, PLOT_H))

    # Tick marks + labels every 5 units; small ticks at every integer
    label_font = QFont("Segoe UI", 8)
    p.setFont(label_font)
    p.setPen(QPen(QColor("#3f3f46")))

    for i in range(int(COORD_MIN), int(COORD_MAX) + 1):
        gx = ML + (i - COORD_MIN) / COORD_RANGE * PLOT_W
        gy = MT + (1.0 - (i - COORD_MIN) / COORD_RANGE) * PLOT_H
        tick = 6 if i % 5 == 0 else 3

        # Bottom tick
        p.setPen(QPen(QColor("#52525b"), 1.0))
        p.drawLine(QPointF(gx, MT + PLOT_H), QPointF(gx, MT + PLOT_H + tick))
        # Left tick
        p.drawLine(QPointF(ML - tick, gy), QPointF(ML, gy))

        if i % 5 == 0:
            p.setPen(QPen(QColor("#3f3f46")))
            # X label
            p.drawText(QRectF(gx - 20, MT + PLOT_H + 8, 40, 18),
                       Qt.AlignCenter, str(i))
            # Y label
            p.drawText(QRectF(0, gy - 10, ML - 8, 20),
                       Qt.AlignRight | Qt.AlignVCenter, str(i))

    # Axis name labels
    axis_font = QFont("Segoe UI", 9, QFont.Bold)
    p.setFont(axis_font)
    p.setPen(QPen(QColor("#27272a")))
    # X axis name — centred below plot
    p.drawText(QRectF(ML, MT + PLOT_H + 30, PLOT_W, 16), Qt.AlignCenter, "x")
    # Y axis name — rotated, centred left of plot
    p.save()
    p.translate(10, MT + PLOT_H / 2)
    p.rotate(-90)
    p.drawText(QRectF(-30, -8, 60, 16), Qt.AlignCenter, "y")
    p.restore()


# ── Shape rendering (all coordinates in coord space, converted via to_px) ─────

def _paint_shape(p: QPainter, shape: dict):
    t = shape["type"]
    if t == "line":
        x1, y1 = to_px(*shape["p1"])
        x2, y2 = to_px(*shape["p2"])
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    elif t == "circle":
        cx, cy = to_px(shape["cx"], shape["cy"])
        r_px = shape["r"] * PX_PER_UNIT           # uniform scale (square plot)
        p.drawEllipse(QRectF(cx - r_px, cy - r_px, r_px * 2, r_px * 2))

    elif t == "bezier":
        if len(shape["anchors"]) >= 2:
            _paint_bezier_path(p, shape["anchors"])


def _paint_bezier_path(p: QPainter, anchors: list[dict]):
    path = QPainterPath()
    ax0, ay0 = to_px(*anchors[0]["pos"])
    path.moveTo(ax0, ay0)
    for i in range(1, len(anchors)):
        a1, a2 = anchors[i - 1], anchors[i]
        cp1 = to_px(*(a1["h_out"] if a1["h_out"] else a1["pos"]))
        cp2 = to_px(*(a2["h_in"]  if a2["h_in"]  else a2["pos"]))
        ep  = to_px(*a2["pos"])
        path.cubicTo(cp1[0], cp1[1], cp2[0], cp2[1], ep[0], ep[1])
    p.drawPath(path)


def _paint_bezier_controls(p: QPainter, anchors: list[dict]):
    for anchor in anchors:
        ax, ay = to_px(*anchor["pos"])
        for key, fill in [("h_in", QColor("#f5c2e7")), ("h_out", QColor("#cba6f7"))]:
            h = anchor.get(key)
            if h:
                hx, hy = to_px(*h)
                p.setPen(QPen(QColor("#7f849c"), 1))
                p.setBrush(Qt.NoBrush)
                p.drawLine(QPointF(ax, ay), QPointF(hx, hy))
                p.setBrush(QBrush(fill))
                p.drawEllipse(QRectF(hx - 4, hy - 4, 8, 8))
        # Anchor square
        p.setPen(QPen(QColor("#89b4fa"), 1.5))
        p.setBrush(QBrush(QColor("#313244")))
        p.drawRect(QRectF(ax - 4.5, ay - 4.5, 9, 9))


# ── Geometry / point distribution ─────────────────────────────────────────────

_BEZIER_SAMPLES = 60


def _cubic_pt(p0, p1, p2, p3, t):
    mt = 1.0 - t
    return (
        mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0],
        mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1],
    )


def _bezier_dense(anchors: list[dict]) -> list[tuple[float, float]]:
    pts = [anchors[0]["pos"]]
    for i in range(1, len(anchors)):
        a1, a2 = anchors[i - 1], anchors[i]
        p0 = a1["pos"]
        p1 = a1["h_out"] if a1["h_out"] else a1["pos"]
        p2 = a2["h_in"]  if a2["h_in"]  else a2["pos"]
        p3 = a2["pos"]
        for j in range(1, _BEZIER_SAMPLES + 1):
            pts.append(_cubic_pt(p0, p1, p2, p3, j / _BEZIER_SAMPLES))
    return pts


def _arc_length(shape: dict) -> float:
    t = shape["type"]
    if t == "line":
        return math.hypot(shape["p2"][0] - shape["p1"][0],
                          shape["p2"][1] - shape["p1"][1])
    if t == "circle":
        return 2.0 * math.pi * shape["r"]
    if t == "bezier":
        pts = _bezier_dense(shape["anchors"])
        return sum(math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
                   for i in range(len(pts)-1))
    return 0.0


def _sample_shape(shape: dict, n: int) -> list[tuple[float, float]]:
    if n <= 0:
        return []
    t = shape["type"]

    if t == "line":
        x1, y1 = shape["p1"]; x2, y2 = shape["p2"]
        ts = [i / (n - 1) for i in range(n)] if n > 1 else [0.5]
        return [(x1 + s*(x2-x1), y1 + s*(y2-y1)) for s in ts]

    if t == "circle":
        cx, cy, r = shape["cx"], shape["cy"], shape["r"]
        return [(cx + r*math.cos(2*math.pi*i/n),
                 cy + r*math.sin(2*math.pi*i/n)) for i in range(n)]

    if t == "bezier":
        pts = _bezier_dense(shape["anchors"])
        cum = [0.0]
        for i in range(len(pts)-1):
            cum.append(cum[-1] + math.hypot(pts[i+1][0]-pts[i][0],
                                             pts[i+1][1]-pts[i][1]))
        total = cum[-1]
        if total == 0:
            return [pts[0]] * n
        result = []
        for i in range(n):
            target = total * (i / (n-1) if n > 1 else 0.5)
            lo, hi = 0, len(cum) - 2
            while lo < hi:
                mid = (lo + hi) // 2
                if cum[mid + 1] < target:
                    lo = mid + 1
                else:
                    hi = mid
            j = lo
            seg = cum[j+1] - cum[j]
            alpha = (target - cum[j]) / seg if seg else 0.0
            result.append((pts[j][0] + alpha*(pts[j+1][0]-pts[j][0]),
                           pts[j][1] + alpha*(pts[j+1][1]-pts[j][1])))
        return result

    return []


def _place_points(shapes: list[dict], n: int, min_dist: float) -> list[tuple[float, float]]:
    """Progressive multi-pass placement.

    Pass 1 uses a coarse exclusion radius (d_start ≈ total_length / 10% of N) so
    every shape receives a sparse but complete first coating of points.  Each
    subsequent pass multiplies the radius by 0.8, filling gaps left by the
    previous pass — identical to the user-specified "try 1.0, then 0.8, …"
    strategy.  Passes continue until the radius reaches min_dist.  Only AFTER
    all passes is the result trimmed to n (by uniform subsampling), so no shape
    is ever starved of its initial coverage.
    """
    lengths = [_arc_length(s) for s in shapes]
    total   = sum(lengths)
    if total == 0 or n == 0:
        return []

    # ── Dense candidates (per shape, proportional to arc length) ─────────────
    # Use enough candidates so that every possible min_dist gap has a candidate
    # inside it, but cap memory at 20× N.
    nc = n * 8
    if min_dist > 0:
        nc = max(nc, int(total / (min_dist * 0.4)))
    nc = min(nc, n * 20)

    exact  = [nc * l / total for l in lengths]
    floors = [int(e) for e in exact]
    deficit = nc - sum(floors)
    order = sorted(range(len(exact)), key=lambda i: -(exact[i] - floors[i]))
    for i in order[:deficit]:
        floors[i] += 1

    cands: list[tuple[float, float]] = []
    for shape, count in zip(shapes, floors):
        if count > 0:
            cands.extend(_sample_shape(shape, count))

    # ── Uniform case ─────────────────────────────────────────────────────────
    if min_dist <= 0:
        if len(cands) <= n:
            return cands
        step = len(cands) / n
        return [cands[int(i * step)] for i in range(n)]

    # ── Build the pass schedule (d_start → … → min_dist) ─────────────────────
    FACTOR  = 0.8
    d_start = max(total / max(n * 0.10, 1.0), min_dist)

    passes: list[float] = []
    d = d_start
    while d > min_dist * 1.001:
        passes.append(d)
        d *= FACTOR
    passes.append(min_dist)

    # ── Grid-accelerated greedy placement (no early stopping) ────────────────
    cell = min_dist
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    placed: list[tuple[float, float]] = []

    def _blocked(x: float, y: float, d_check: float) -> bool:
        gx, gy = int(x / cell), int(y / cell)
        r = max(1, math.ceil(d_check / cell))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for px, py in grid.get((gx + dx, gy + dy), []):
                    if math.hypot(x - px, y - py) < d_check:
                        return True
        return False

    def _add(x: float, y: float) -> None:
        placed.append((x, y))
        grid.setdefault((int(x / cell), int(y / cell)), []).append((x, y))

    for d_pass in passes:
        for x, y in cands:
            if not _blocked(x, y, d_pass):
                _add(x, y)

    # ── Trim to n by uniform subsampling (preserves coverage across shapes) ──
    if len(placed) <= n:
        return placed
    step = len(placed) / n
    return [placed[int(i * step)] for i in range(n)]


# ── Main window ───────────────────────────────────────────────────────────────

SB_BG   = "#1e1e2e"
SB_SEP  = "#313244"
TXT_DIM = "#7f849c"
TXT_BRT = "#cdd6f4"
BLUE    = "#89b4fa"
BTN_MID = "#45475a"
BTN_OK  = "#a6e3a1"
BTN_OK_FG = "#1e1e2e"
STS_BG  = "#181825"
STS_FG  = "#a6adc8"


def _lbl(parent, text, *, dim=False, bold=False, size=10) -> QLabel:
    w = QLabel(text, parent)
    w.setStyleSheet(
        f"color:{TXT_DIM if dim else TXT_BRT}; font-size:{size}px;"
        f" font-weight:{'bold' if bold else 'normal'};"
        " background:transparent; padding-left:16px;"
    )
    return w


def _sep(parent) -> QFrame:
    f = QFrame(parent)
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color:{SB_SEP}; margin:6px 12px;")
    return f


def _spin_style():
    return f"""
        QSpinBox, QDoubleSpinBox {{
            background:#313244; color:{TXT_BRT}; border:none; border-radius:4px;
            padding:5px 8px; font-size:12px; margin:0 14px;
        }}
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            background:{BTN_MID}; width:20px;
        }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover,
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
            background:#585b70;
        }}
    """


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shape Drawing — Point Distributor")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        # Wrap canvas in a scroll area so it stays fixed size
        self.canvas = Canvas()
        self.canvas.shape_added.connect(self._on_shape_added)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        scroll.setStyleSheet("background:#e4e4e7; border:none;")
        rl.addWidget(scroll, stretch=1)
        rl.addWidget(self._build_status_bar())

        root.addWidget(right, stretch=1)

        # Size window to fit canvas + sidebar comfortably
        self.resize(185 + CANVAS_W + 20, CANVAS_H + 28 + 20)

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        sb = QWidget()
        sb.setFixedWidth(185)
        sb.setStyleSheet(f"background:{SB_BG};")
        lo = QVBoxLayout(sb)
        lo.setContentsMargins(0, 14, 0, 14)
        lo.setSpacing(2)

        lo.addWidget(_lbl(sb, "DRAW TOOL", dim=True, bold=True))
        lo.addSpacing(6)

        self._tool_group = QButtonGroup(sb)
        self._rb: dict[str, QRadioButton] = {}
        for label, val in [("  Line", "line"), ("  Bezier Curve", "bezier"), ("  Circle", "circle")]:
            rb = QRadioButton(label, sb)
            rb.setStyleSheet(f"""
                QRadioButton {{
                    color:{TXT_BRT}; font-size:13px;
                    padding:5px 20px; background:transparent;
                }}
                QRadioButton::indicator {{
                    width:13px; height:13px; border-radius:7px;
                    border:2px solid {BTN_MID}; background:transparent;
                }}
                QRadioButton::indicator:checked {{
                    background:{BLUE}; border-color:{BLUE};
                }}
                QRadioButton:hover {{ color:{BLUE}; }}
            """)
            self._tool_group.addButton(rb)
            self._rb[val] = rb
            lo.addWidget(rb)

        self._rb["line"].setChecked(True)
        self._tool_group.buttonClicked.connect(self._on_tool_changed)

        lo.addWidget(_sep(sb))
        lo.addWidget(_lbl(sb, "POINT COUNT", dim=True, bold=True))
        lo.addSpacing(4)
        self._spin_n = QSpinBox(sb)
        self._spin_n.setRange(2, 50000)
        self._spin_n.setValue(100)
        self._spin_n.setStyleSheet(_spin_style())
        lo.addWidget(self._spin_n)

        lo.addSpacing(8)
        lo.addWidget(_lbl(sb, "MIN DISTANCE", dim=True, bold=True))
        lo.addSpacing(4)
        self._spin_d = QDoubleSpinBox(sb)
        self._spin_d.setRange(0.0, 10.0)
        self._spin_d.setValue(0.0)
        self._spin_d.setSingleStep(0.05)
        self._spin_d.setDecimals(2)
        self._spin_d.setStyleSheet(_spin_style())
        lo.addWidget(self._spin_d)

        lo.addWidget(_sep(sb))

        btn_undo = self._btn(sb, "Undo", BTN_MID, TXT_BRT, "#585b70")
        btn_undo.clicked.connect(self._on_undo)
        lo.addWidget(btn_undo)
        lo.addSpacing(4)

        btn_clear = self._btn(sb, "Clear", BTN_MID, TXT_BRT, "#585b70")
        btn_clear.clicked.connect(self._on_clear)
        lo.addWidget(btn_clear)
        lo.addSpacing(6)

        btn_done = self._btn(sb, "DONE  ✓", BTN_OK, BTN_OK_FG, "#94d39b",
                             bold=True, fsize=14, pady=11)
        btn_done.clicked.connect(self._on_done)
        lo.addWidget(btn_done)
        lo.addSpacing(4)

        btn_csv = self._btn(sb, "Save CSV", "#89b4fa", "#1e1e2e", "#74a8f0")
        btn_csv.clicked.connect(self._on_save_csv)
        lo.addWidget(btn_csv)

        lo.addWidget(_sep(sb))
        lo.addWidget(_lbl(sb, "HINTS", dim=True, bold=True))
        lo.addSpacing(2)
        self._hint = QLabel("", sb)
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(
            f"color:#a6adc8; font-size:11px; background:transparent; padding:0 14px;"
        )
        lo.addWidget(self._hint)
        self._update_hint("line")

        lo.addStretch()
        return sb

    @staticmethod
    def _btn(parent, text, bg, fg, hover, bold=False, fsize=13, pady=8) -> QPushButton:
        b = QPushButton(text, parent)
        b.setCursor(Qt.PointingHandCursor)
        b.setStyleSheet(f"""
            QPushButton {{
                background:{bg}; color:{fg}; border:none; border-radius:4px;
                padding:{pady}px; font-size:{fsize}px;
                font-weight:{'bold' if bold else 'normal'}; margin:0 12px;
            }}
            QPushButton:hover {{ background:{hover}; }}
        """)
        return b

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(28)
        bar.setStyleSheet(f"background:{STS_BG};")
        lo = QHBoxLayout(bar)
        lo.setContentsMargins(12, 0, 12, 0)
        self._status = QLabel("Select a tool and draw on the canvas.")
        self._status.setStyleSheet(f"color:{STS_FG}; font-size:12px; background:transparent;")
        lo.addWidget(self._status)
        return bar

    def _set_status(self, text: str):
        self._status.setText(text)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_tool_changed(self, btn: QRadioButton):
        for val, rb in self._rb.items():
            if rb is btn:
                self.canvas.set_tool(val)
                self._update_hint(val)
                break

    def _update_hint(self, tool: str):
        hints = {
            "line":   "Click and drag to draw a straight line.",
            "bezier": "Click to place anchors.\nDrag to pull smooth handles.\n"
                      "Double-click to finish.\nEsc to cancel.",
            "circle": "Click a centre, drag outward to set the radius.",
        }
        self._hint.setText(hints.get(tool, ""))

    def _on_shape_added(self, shape_type: str, total: int):
        self._set_status(
            f"{shape_type.capitalize()} added — {total} shape{'s' if total != 1 else ''} "
            "total.  Click DONE to distribute points."
        )

    def _on_undo(self):
        t = self.canvas.undo()
        if t:
            n = len(self.canvas.shapes)
            self._set_status(
                f"Undid last {t}. "
                + (f"{n} shape{'s' if n != 1 else ''} remaining." if n else "Canvas is now empty.")
            )
        else:
            self._set_status("Nothing to undo.")

    def _on_clear(self):
        self.canvas.clear()
        self._set_status("Canvas cleared.")

    def _on_done(self):
        s = self.canvas.current
        if s:
            t = s["type"]
            valid = (
                (t == "line"   and math.hypot(s["p2"][0]-s["p1"][0],
                                               s["p2"][1]-s["p1"][1]) > 0.15) or
                (t == "bezier" and len(s["anchors"]) >= 2) or
                (t == "circle" and s["r"] > 0.15)
            )
            if valid:
                self.canvas.shapes.append(s)
            self.canvas.current = None

        if not self.canvas.shapes:
            self._set_status("Draw some shapes first, then click DONE.")
            return

        n       = self._spin_n.value()
        min_d   = self._spin_d.value()
        placed  = self.canvas.distribute(n, min_d)
        ns      = len(self.canvas.shapes)

        msg = f"Distributed {placed} points across {ns} shape{'s' if ns != 1 else ''}"
        if min_d > 0 and placed < n:
            msg += f"  ({n - placed} removed — min dist {min_d:.2f})"
        self._set_status(msg + ".")

    def _on_save_csv(self):
        if not self.canvas.dist_pts:
            self._set_status("No points to save — click DONE first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save points as CSV", "points.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "y"])
            w.writerows((round(x, 2), round(y, 2)) for x, y in self.canvas.dist_pts)
        self._set_status(f"Saved {len(self.canvas.dist_pts)} points to {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
