#!/usr/bin/env python3
"""
Terrain Forge v12 — procedural terrain generator / editor (PySide6 + NumPy)

Complete feature set
GENERATION   Perlin / Value / Worley noise · fBm / Ridged / Billow patterns ·
             domain warp · seamless tiling · island masks (round / square /
             archipelago / volcano) · gamma · terraces · smoothing
EROSION      Flow (pipe model, float32) · classic droplets (numba-JIT, parallel
             or pure-python) · thermal talus slippage · LIVE preview streaming ·
             Apply-Erosion-to-Map on imported/sculpted/blended maps
LAYERS       Two-layer blend (A × mask + B) with noise/radial/gradient/height/
             PAINTED masks · splat layers (procedural + hand-painted R/G/B/A)
TOOLS        Sculpt brushes (raise/lower/smooth/flatten/paint splat/paint
             blend) with stylus pressure · undo/redo · heightmap import
             (PNG/JPG/BMP/raw/f32) · auto-contrast · lake detection ·
             autosave + crash recovery
RENDER       Shaded (hillshade, AO, rivers, contours) · Heightmap · Normals ·
             Slope · Splat · Blend mask · Wetness · Deposition · Curvature ·
             Walkable · Strata · biome coloring · 7 palettes + Biomes ·
             OpenGL 3D preview (textured, sea plane, lakes) with software
             fallback · cross-section · histogram · statistics
EXPORT       Unity .unitypackage (importer + water) · UE5 bundle + material
             script · Godot tscn/glb/f32/EXR · Blender self-contained script ·
             glTF/GLB (textured, water) · OBJ (+MTL/albedo) · PLY · watertight
             STL · chunked tiles (LODs, ZIP) · data masks · 16-bit PNG /
             float32 heightmaps · normal maps · metadata.json · presets
GUI          Dark Fusion theme · dock layout (tools palette, tabbed settings,
             analysis) · toolbar · View menu · layout persistence · F1 help
CLI          headless generation + export + batch (--export-many "seed=1..10")

Dependencies:  pip install PySide6 numpy        (optional: numba)
"""

import sys, os, io, json, math, re, time, random, traceback, struct, zlib, base64
import hashlib, tarfile, zipfile

import numpy as np
from PySide6.QtCore import (Qt, Signal, QObject, QThread, QTimer, QRectF,
                            QSettings, QBuffer, QUrl, QEvent, QStandardPaths,
                            QSize)
from PySide6.QtGui import (QImage, QPixmap, QPainter, QColor, QPen, QPolygonF,
                           QPointF, QShortcut, QKeySequence, QPainterPath,
                           QVector3D, QMatrix4x4, QDesktopServices,
                           QOpenGLTexture, QPalette)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QScrollArea, QVBoxLayout,
    QHBoxLayout, QGridLayout, QGroupBox, QLabel, QSlider, QSpinBox, QCheckBox,
    QComboBox, QPushButton, QColorDialog, QFileDialog, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QProgressBar, QMessageBox, QTabWidget,
    QDialog, QListWidget, QDoubleSpinBox, QFormLayout, QMenu, QLineEdit,
    QDockWidget, QToolBar, QToolButton, QButtonGroup, QFrame,
)

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

HAVE_GL = False
try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from PySide6.QtOpenGL import (QOpenGLShaderProgram, QOpenGLShader,
                                  QOpenGLBuffer)
    from shiboken6 import VoidPtr
    HAVE_GL = True
except Exception:
    HAVE_GL = False

GL_DEPTH_TEST = 0x0B71
GL_FLOAT = 0x1406
GL_TRIANGLES = 0x0004
GL_UNSIGNED_INT = 0x1405
GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_BLEND = 0x0BE2
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303

# ----------------------------------------------------------------------------
# Help text & theme
# ----------------------------------------------------------------------------

HELP_TEXT = """
<b>Shortcuts</b>
Ctrl+G Generate · Ctrl+N new seed · Ctrl+T toggle 2D/3D · Ctrl+F fit view
Ctrl+I import heightmap · Ctrl+S render PNG · Ctrl+Shift+S heightmap PNG
Ctrl+E OBJ · Ctrl+Shift+E engine export · Ctrl+Z undo stroke · Ctrl+Y redo · F1 help

<b>Brushes</b> (Tools dock; left-drag on the 2D view)
Raise / Lower / Smooth / Flatten edit heights. <b>Paint Splat</b> paints the
R/G/B/A layer weights (preview in the 'Splat' render mode; exported to Unity /
UE). <b>Paint Blend</b> paints the Layer-A blend mask by hand. A stylus works
with pressure — press harder for stronger strokes. Ctrl+Z / Ctrl+Y walk the
stroke history.

<b>Apply Erosion to Map</b> (Erosion tab) runs the configured hydraulic /
thermal erosion on the CURRENT map — imported, sculpted or blended — without
regenerating from noise.

<b>Layer Blend</b>: Capture current as Layer A → change settings → Generate.
Masks: Noise / Radial / Gradient / Height ramp / Painted. Preview in
'Blend mask' render mode (orange = A, teal = B).

<b>Data views & masks</b>: Wetness, Deposition, Curvature, Walkable, Strata
render modes; the same fields export as grayscale masks via
Export for Engine → 'Data masks'.

<b>CLI</b> (headless):
&nbsp;&nbsp;terrain_forge.py --cli --style "Alpine Peaks" --seed 42 --export bundle
&nbsp;&nbsp;terrain_forge.py --cli --export-many "seed=100..110" --out run1

<b>Safety net</b>: the map autosaves to app-data every 2 minutes and after
strokes/imports — offered for recovery on the next launch.
"""

ACCENT = '#4fc3f7'

DARK_QSS = f"""
QWidget {{ background-color: #23262b; color: #d6dae1; font-size: 12px; }}
QMainWindow::separator {{ background: #3a3f46; width: 3px; height: 3px; }}
QToolTip {{ background-color: #17191d; color: #d6dae1; border: 1px solid {ACCENT}; padding: 4px; }}
QToolBar {{ background: #262a30; border: none; border-bottom: 1px solid #17191d;
            padding: 3px 6px; spacing: 4px; }}
QToolBar::separator {{ background: #3a3f46; width: 1px; margin: 4px 6px; }}
QToolButton {{ background: transparent; color: #c9ced6; border: 1px solid transparent;
               border-radius: 4px; padding: 4px 9px; }}
QToolButton:hover {{ background: #333941; border-color: #454c55; }}
QToolButton:pressed {{ background: #2b313a; }}
QToolButton:disabled {{ color: #5b626c; }}
QToolButton:checked {{ background: #0e7490; border-color: #12a5cf; color: #ffffff; }}
QPushButton#genBtn {{ background: #0e7490; color: #ffffff; font-weight: 600;
                      border: 1px solid #0b5c73; padding: 4px 16px; }}
QPushButton#genBtn:hover {{ background: #1287a8; }}
QPushButton#genBtn:disabled {{ background: #23404a; color: #7fa3ad; }}
QPushButton#stopBtn {{ background: #7a3328; border: 1px solid #93402f;
                       color: #ffd9d2; padding: 4px 12px; }}
QPushButton#stopBtn:hover {{ background: #8f3f31; }}
QPushButton#stopBtn:disabled {{ background: #3a2a26; color: #6b5f5c; }}
QDockWidget {{ color: #aab2bd; font-weight: 600; }}
QDockWidget::title {{ background: #262a30; border-bottom: 1px solid #17191d;
                      padding: 5px 8px; }}
QTabWidget::pane {{ border: 1px solid #3a3f46; border-radius: 4px; top: -1px; }}
QTabBar::tab {{ background: transparent; color: #9aa2ad; padding: 6px 14px;
                border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: #d6dae1; }}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QGroupBox {{ font-weight: 600; color: #bfc7d1; border: 1px solid #343941;
             border-radius: 6px; margin-top: 12px; padding-top: 4px;
             background: #26292f; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px;
                    color: {ACCENT}; }}
QGroupBox::indicator {{ width: 11px; height: 11px; }}
QPushButton {{ background: #333941; border: 1px solid #454c55; border-radius: 4px;
               padding: 5px 12px; color: #d6dae1; }}
QPushButton:hover {{ background: #3d4450; border-color: #566072; }}
QPushButton:pressed {{ background: #2b313a; }}
QPushButton:disabled {{ color: #5b626c; background: #2c3037; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: #2a2e34;
    border: 1px solid #3f4550; border-radius: 4px; padding: 3px 6px;
    selection-background-color: #0e7490; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus
    {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: #2a2e34; border: 1px solid #3f4550;
    selection-background-color: #0e7490; }}
QSlider::groove:horizontal {{ height: 4px; background: #3a3f46; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: #2c6e86; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 13px; margin: -5px 0; border-radius: 7px;
                              background: {ACCENT}; }}
QSlider::handle:horizontal:hover {{ background: #7fd8ff; }}
QCheckBox::indicator, QGroupBox::indicator {{ width: 13px; height: 13px;
    border-radius: 3px; border: 1px solid #59616c; background: #2a2e34; }}
QCheckBox::indicator:checked, QGroupBox::indicator:checked
    {{ background: {ACCENT}; border-color: {ACCENT}; }}
QProgressBar {{ background: #2a2e34; border: 1px solid #3f4550; border-radius: 4px;
                text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QStatusBar {{ background: #262a30; border-top: 1px solid #17191d; }}
QStatusBar QLabel {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #3f4550; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #4d5563; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: #3f4550; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QListWidget {{ background: #26292f; border: 1px solid #3a3f46; border-radius: 4px; }}
QListWidget::item {{ padding: 4px; }}
QListWidget::item:selected {{ background: #0e7490; color: #ffffff; }}
QMenu {{ background: #26292f; border: 1px solid #3f4550; }}
QMenu::item {{ padding: 5px 24px; }}
QMenu::item:selected {{ background: #0e7490; }}
QMenu::separator {{ height: 1px; background: #3a3f46; margin: 4px 8px; }}
"""


def apply_theme(app):
    app.setStyle('Fusion')
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor('#23262b'))
    pal.setColor(QPalette.ColorRole.WindowText, QColor('#d6dae1'))
    pal.setColor(QPalette.ColorRole.Base, QColor('#1e2126'))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor('#26292f'))
    pal.setColor(QPalette.ColorRole.Text, QColor('#d6dae1'))
    pal.setColor(QPalette.ColorRole.Button, QColor('#333941'))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor('#d6dae1'))
    pal.setColor(QPalette.ColorRole.Highlight, QColor('#0e7490'))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor('#ffffff'))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor('#17191d'))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor('#d6dae1'))
    try:
        pal.setColor(QPalette.ColorRole.PlaceholderText, QColor('#6b7280'))
    except Exception:
        pass
    app.setPalette(pal)
    app.setStyleSheet(DARK_QSS)

# ----------------------------------------------------------------------------
# Noise core
# ----------------------------------------------------------------------------

class GenCancelled(Exception):
    pass


def _hash(ix, iy, seed):
    h = ix.astype(np.int64) * np.int64(374761393)
    h ^= iy.astype(np.int64) * np.int64(668265263)
    h ^= np.int64(seed) * np.int64(1442695041)
    h &= np.int64(0xFFFFFFFF)
    h ^= h >> np.int64(13)
    h = (h * np.int64(1274126177)) & np.int64(0xFFFFFFFF)
    h ^= h >> np.int64(16)
    return h.astype(np.float64) / 4294967296.0


_TABLES = {}


def _tables(seed):
    t = _TABLES.get(seed)
    if t is None:
        idx = np.arange(65536, dtype=np.int64)
        x = idx & 255; y = idx >> 8
        grad = (_hash(x, y, seed) * 8.0).astype(np.uint8)
        valt = _hash(x, y, seed ^ 0x9E3779B9).astype(np.float32)
        grad = np.concatenate([grad, grad[:257]])
        valt = np.concatenate([valt, valt[:257]])
        if len(_TABLES) > 24:
            _TABLES.pop(next(iter(_TABLES)))
        _TABLES[seed] = t = (grad, valt)
    return t


_GRADX = np.array([1, -1, 0, 0, 0.7071, -0.7071, 0.7071, -0.7071])
_GRADY = np.array([0, 0, 1, -1, 0.7071, 0.7071, -0.7071, -0.7071])


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin2d(x, y, seed):
    grad, _ = _tables(seed)
    x0 = np.floor(x); y0 = np.floor(y)
    fx = x - x0; fy = y - y0
    u = _fade(fx); v = _fade(fy)
    base = (((y0.astype(np.int64)) & 255) << 8) | ((x0.astype(np.int64)) & 255)
    i00 = grad[base]; i10 = grad[base + 1]
    i01 = grad[base + 256]; i11 = grad[base + 257]
    d00 = _GRADX[i00] * fx + _GRADY[i00] * fy
    d10 = _GRADX[i10] * (fx - 1) + _GRADY[i10] * fy
    d01 = _GRADX[i01] * fx + _GRADY[i01] * (fy - 1)
    d11 = _GRADX[i11] * (fx - 1) + _GRADY[i11] * (fy - 1)
    nx0 = d00 + (d10 - d00) * u
    nx1 = d01 + (d11 - d01) * u
    return (nx0 + (nx1 - nx0) * v) * 1.4


def value2d(x, y, seed):
    _, valt = _tables(seed)
    x0 = np.floor(x); y0 = np.floor(y)
    u = _fade(x - x0); v = _fade(y - y0)
    base = (((y0.astype(np.int64)) & 255) << 8) | ((x0.astype(np.int64)) & 255)
    v00 = valt[base]; v10 = valt[base + 1]
    v01 = valt[base + 256]; v11 = valt[base + 257]
    a = v00 + (v10 - v00) * u
    b = v01 + (v11 - v01) * u
    return (a + (b - a) * v) * 2.0 - 1.0


def perlin2d_tiled(x, y, seed, period):
    grad, _ = _tables(seed)
    x0 = np.floor(x); y0 = np.floor(y)
    fx = x - x0; fy = y - y0
    u = _fade(fx); v = _fade(fy)
    ix0 = np.mod(x0.astype(np.int64), period)
    iy0 = np.mod(y0.astype(np.int64), period)
    ix1 = (ix0 + 1) % period; iy1 = (iy0 + 1) % period
    i00 = grad[(iy0 << 8) | ix0]; i10 = grad[(iy0 << 8) | ix1]
    i01 = grad[(iy1 << 8) | ix0]; i11 = grad[(iy1 << 8) | ix1]
    d00 = _GRADX[i00] * fx + _GRADY[i00] * fy
    d10 = _GRADX[i10] * (fx - 1) + _GRADY[i10] * fy
    d01 = _GRADX[i01] * fx + _GRADY[i01] * (fy - 1)
    d11 = _GRADX[i11] * (fx - 1) + _GRADY[i11] * (fy - 1)
    nx0 = d00 + (d10 - d00) * u
    nx1 = d01 + (d11 - d01) * u
    return (nx0 + (nx1 - nx0) * v) * 1.4


def value2d_tiled(x, y, seed, period):
    _, valt = _tables(seed)
    x0 = np.floor(x); y0 = np.floor(y)
    u = _fade(x - x0); v = _fade(y - y0)
    ix0 = np.mod(x0.astype(np.int64), period)
    iy0 = np.mod(y0.astype(np.int64), period)
    ix1 = (ix0 + 1) % period; iy1 = (iy0 + 1) % period
    v00 = valt[(iy0 << 8) | ix0]; v10 = valt[(iy0 << 8) | ix1]
    v01 = valt[(iy1 << 8) | ix0]; v11 = valt[(iy1 << 8) | ix1]
    a = v00 + (v10 - v00) * u
    b = v01 + (v11 - v01) * u
    return (a + (b - a) * v) * 2.0 - 1.0


def worley2d(x, y, seed):
    """F1 cellular noise, ~[-1, 1]."""
    x0 = np.floor(x); y0 = np.floor(y)
    best = np.full(x.shape, 1e18)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx = x0 + dx; cy = y0 + dy
            px = cx + _hash(cx, cy, seed)
            py = cy + _hash(cx, cy, seed ^ 0x51F3)
            d = (px - x) ** 2 + (py - y) ** 2
            np.minimum(best, d, out=best)
    return np.sqrt(best) * 2.0 - 1.0


def worley2d_tiled(x, y, seed, period):
    x0 = np.floor(x); y0 = np.floor(y)
    best = np.full(x.shape, 1e18)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ax = x0 + dx; ay = y0 + dy
            cx = np.mod(ax, period); cy = np.mod(ay, period)
            px = ax + _hash(cx, cy, seed)
            py = ay + _hash(cx, cy, seed ^ 0x51F3)
            d = (px - x) ** 2 + (py - y) ** 2
            np.minimum(best, d, out=best)
    return np.sqrt(best) * 2.0 - 1.0


def _noise(kind, x, y, seed, period=None):
    if kind == 'Worley':
        return (worley2d_tiled(x, y, seed, period) if period is not None
                else worley2d(x, y, seed))
    if period is not None:
        return (perlin2d_tiled(x, y, seed, period) if kind == 'Perlin'
                else value2d_tiled(x, y, seed, period))
    return perlin2d(x, y, seed) if kind == 'Perlin' else value2d(x, y, seed)


def fbm(x, y, seed, octaves, persistence, lacunarity, kind='Perlin',
        ridged=False, billow=False, cancel=None, progress=None, period=None):
    total = np.zeros(x.shape); amp = 1.0; freq = 1.0
    norm = 0.0; weight = 1.0
    for o in range(octaves):
        if cancel is not None and cancel():
            raise GenCancelled()
        po = None
        if period is not None:
            freq = 2.0 ** o
            po = int(min(period << o, 256))
        n = _noise(kind, x * freq, y * freq, seed + o * 131, po)
        if billow:
            n = np.abs(n) * 2.0 - 1.0
        if ridged:
            n = 1.0 - np.abs(n)
            n = n * n * weight
            weight = np.clip(n * 2.0, 0.0, 1.0)
        total += n * amp
        norm += amp
        amp *= persistence
        freq *= lacunarity
        if progress is not None:
            progress((o + 1) / octaves)
    return total / norm

# ----------------------------------------------------------------------------
# Shaping helpers
# ----------------------------------------------------------------------------

def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def box_blur(h, iterations, wrap=False):
    mode = 'wrap' if wrap else 'edge'
    for _ in range(iterations):
        p = np.pad(h, 1, mode=mode)
        h = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
             p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
             p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0
    return h


def terrace(h, steps):
    s = h * steps
    fl = np.floor(s); fr = s - fl
    fr = np.clip((fr - 0.5) * 3.0 + 0.5, 0.0, 1.0)
    fr = fr * fr * (3.0 - 2.0 * fr)
    return (fl + fr) / steps


def _down(a, max_side=200):
    ds = max(1, math.ceil(max(a.shape) / max_side))
    return np.clip(a[::ds, ::ds], 0.0, 1.0).astype(np.float32).copy()


def blend_mask_field(p, shape, seed, b_hmap, painted=None):
    """Mask deciding where Layer A contributes (0 = B, 1 = A)."""
    H, W = shape
    kind = p.get('blend_mask', 'Noise')
    if kind == 'Painted' and painted is not None:
        if painted.shape != (H, W):
            return np.clip(resize2d(painted, W, H), 0.0, 1.0)
        return np.clip(painted, 0.0, 1.0)
    if kind == 'Noise':
        f = float(p.get('blend_freq', 3.0))
        xs = np.arange(W, dtype=np.float64) / max(W - 1, 1)
        ys = np.arange(H, dtype=np.float64) / max(H - 1, 1)
        X, Y = np.meshgrid(xs, ys)
        n = value2d(X * f + 3.7, Y * f + 9.1, seed ^ 0xB1E)
        m = np.clip(n * 0.5 + 0.5, 0.0, 1.0)
    elif kind == 'Radial':
        nx = np.linspace(-1, 1, W)[None, :]
        ny = np.linspace(-1, 1, H)[:, None]
        r = np.sqrt(nx * nx + ny * ny) / math.sqrt(2.0)
        m = 1.0 - smoothstep(0.15, 0.95, r)
    elif kind == 'Gradient X':
        m = np.tile(np.linspace(0.0, 1.0, W)[None, :], (H, 1))
    elif kind == 'Gradient Y':
        m = np.tile(np.linspace(0.0, 1.0, H)[:, None], (1, W))
    else:
        m = np.clip(b_hmap, 0.0, 1.0)
    if p.get('blend_invert'):
        m = 1.0 - m
    return m


def apply_island_mask(h, p, seed, tiled=False):
    kind = p['mask_type']
    if kind == 'None':
        return h
    strength = float(p['mask_strength'])
    size = float(p['mask_size'])
    if strength < 0.01:
        return h
    H, W = h.shape
    nx = np.linspace(-1, 1, W)[None, :]
    ny = np.linspace(-1, 1, H)[:, None]
    r = np.sqrt(nx * nx + ny * ny) / math.sqrt(2.0)
    if kind == 'Round island':
        m = 1.0 - smoothstep(size * 0.35, size * 1.02, r)
    elif kind == 'Square island':
        d = np.maximum(np.abs(nx), np.abs(ny)) * 1.45
        m = 1.0 - smoothstep(size * 0.35, size * 1.05, d)
    elif kind == 'Archipelago (noise)':
        K = max(1, round(2.3 * (size + 0.5) * W / 512.0))
        if tiled:
            n = fbm(nx * K, ny * K, seed + 777, 4, 0.5, 2.0, p['noise_type'],
                    period=K)
        else:
            n = fbm(nx * 2.3 * (size + 0.5) * W / 512.0,
                    ny * 2.3 * (size + 0.5) * W / 512.0, seed + 777, 4, 0.5,
                    2.0, p['noise_type'])
        m = smoothstep(0.42, 0.60, n * 0.5 + 0.5)
        m *= 1.0 - smoothstep(0.72, 1.0, r)
    elif kind == 'Volcano':
        cone = np.clip((size - r * 1.35) / max(size, 1e-6), 0.0, 1.0)
        cone = cone * cone * (3.0 - 2.0 * cone)
        crater = 1.0 - 0.45 * np.exp(-(r / 0.07) ** 2)
        mvol = np.clip(cone * crater, 0.0, 1.0)
        h2 = np.clip(h * 0.35 + mvol * 0.85 + h * mvol * 0.25, 0.0, 1.0)
        return h * (1.0 - strength) + h2 * strength
    else:
        return h
    return h * (1.0 - strength) + (h * np.clip(m, 0, 1)) * strength

# ----------------------------------------------------------------------------
# Erosion
# ----------------------------------------------------------------------------

def thermal_erode(heights, iters, talus, rate, progress, cancelled):
    H, W = heights.shape
    h = heights.copy()
    pad = np.empty((H + 2, W + 2))
    padI = pad[1:-1, 1:-1]
    acc = np.empty((H, W)); tmp = np.empty((H, W))
    c = rate * 0.25
    nbrs = (pad[1:-1, :-2], pad[1:-1, 2:], pad[:-2, 1:-1], pad[2:, 1:-1])
    for i in range(iters):
        if cancelled():
            raise GenCancelled()
        if i % 10 == 0:
            progress(i / iters)
        padI[:] = h
        pad[0, 1:-1] = h[0]; pad[-1, 1:-1] = h[-1]
        pad[:, 0] = pad[:, 1]; pad[:, -1] = pad[:, -2]
        first = True
        for sl in nbrs:
            np.subtract(h, sl, out=tmp)
            tmp -= talus
            np.maximum(tmp, 0.0, out=tmp)
            if first:
                np.copyto(acc, tmp); first = False
            else:
                acc += tmp
        acc *= c
        h -= acc
    return h


_BRUSH = ((0, 0, 0.10), (0, 1, 0.20), (0, -1, 0.20), (1, 0, 0.20), (-1, 0, 0.20),
          (1, 1, 0.05), (1, -1, 0.05), (-1, 1, 0.05), (-1, -1, 0.05))


def hydraulic_erode(heights, p, seed, progress, cancelled, snapshot=None):
    H, W = heights.shape
    h = heights.astype(np.float64, copy=True)
    rng = np.random.default_rng(seed)
    n_drops = int(p['droplets']); lifetime = int(p['lifetime'])
    inertia = float(p['inertia']); cap_f = float(p['capacity'])
    erode_s = float(p['erode']); deposit_s = float(p['deposit'])
    evap = float(p['evaporate'])
    sqrt = math.sqrt; hyp = math.hypot
    brush = _BRUSH
    min_slope = 0.001; gravity = 10.0
    snap_every = max(1, n_drops // 60)
    xy = rng.random((n_drops, 2))
    xs0 = 1.0 + xy[:, 0] * (W - 4.0)
    ys0 = 1.0 + xy[:, 1] * (H - 4.0)
    for i in range(n_drops):
        if cancelled():
            raise GenCancelled()
        if i % 256 == 0:
            progress(i / n_drops)
        if snapshot is not None and i % snap_every == 0:
            snapshot(_down(h))
        x = xs0[i]; y = ys0[i]
        dx = dy = 0.0; speed = 1.0; water = 1.0; sed = 0.0
        for _ in range(lifetime):
            ix = int(x); iy = int(y)
            if ix < 1 or iy < 1 or ix > W - 4 or iy > H - 4:
                break
            fx = x - ix; fy = y - iy
            row0 = h[iy]; row1 = h[iy + 1]
            old = ((row0[ix] * (1 - fx) + row0[ix + 1] * fx) * (1 - fy) +
                   (row1[ix] * (1 - fx) + row1[ix + 1] * fx) * fy)
            gx = (((row0[ix + 1] - row0[ix - 1]) * (1 - fx) +
                   (row0[ix + 2] - row0[ix]) * fx) * (1 - fy) +
                  ((row1[ix + 1] - row1[ix - 1]) * (1 - fx) +
                   (row1[ix + 2] - row1[ix]) * fx) * fy)
            rm1 = h[iy - 1]; rm2 = h[iy + 2]
            gy = (((row1[ix] - rm1[ix]) * (1 - fx) +
                   (row1[ix + 1] - rm1[ix + 1]) * fx) * (1 - fy) +
                  ((rm2[ix] - row0[ix]) * (1 - fx) +
                   (rm2[ix + 1] - row0[ix + 1]) * fx) * fy)
            ndx = dx * inertia - gx * (1 - inertia)
            ndy = dy * inertia - gy * (1 - inertia)
            dl = hyp(ndx, ndy)
            if dl < 1e-10:
                a = rng.uniform(0, 6.2831853)
                ndx, ndy, dl = math.cos(a), math.sin(a), 1.0
            dx = ndx / dl; dy = ndy / dl
            x += dx; y += dy
            if x < 1 or y < 1 or x > W - 3 or y > H - 3:
                break
            nix = int(x); niy = int(y); nfx = x - nix; nfy = y - niy
            nr0 = h[niy]; nr1 = h[niy + 1]
            new = ((nr0[nix] * (1 - nfx) + nr0[nix + 1] * nfx) * (1 - nfy) +
                   (nr1[nix] * (1 - nfx) + nr1[nix + 1] * nfx) * nfy)
            dh = new - old
            cap = max(-dh, min_slope) * speed * water * cap_f
            if sed > cap or dh > 0:
                amt = min(sed, dh) if dh > 0 else (sed - cap) * deposit_s
                w00 = amt * (1 - nfx) * (1 - nfy)
                w10 = amt * nfx * (1 - nfy)
                w01 = amt * (1 - nfx) * nfy
                w11 = amt * nfx * nfy
                h[niy, nix] += w00; h[niy, nix + 1] += w10
                h[niy + 1, nix] += w01; h[niy + 1, nix + 1] += w11
                sed -= amt
            else:
                amt = min((cap - sed) * erode_s, -dh)
                for by, bx, bw in brush:
                    yy = iy + by; xx = ix + bx
                    if 0 <= yy < H and 0 <= xx < W:
                        h[yy, xx] -= amt * bw
                sed += amt
            speed = sqrt(max(0.0, speed * speed - dh * gravity))
            water *= (1.0 - evap)
            if water < 0.005:
                break
    return h


if HAVE_NUMBA:
    @njit(cache=True)
    def _droplets_core(h, n_drops, lifetime, inertia, cap_f, erode_s,
                       deposit_s, evap, seed):
        np.random.seed(seed)
        H, W = h.shape
        for _i in range(n_drops):
            x = 1.0 + np.random.random() * (W - 4.0)
            y = 1.0 + np.random.random() * (H - 4.0)
            dx = 0.0; dy = 0.0; speed = 1.0; water = 1.0; sed = 0.0
            for _ in range(lifetime):
                ix = int(x); iy = int(y)
                if ix < 1 or iy < 1 or ix > W - 4 or iy > H - 4:
                    break
                fx = x - ix; fy = y - iy
                old = ((h[iy, ix] * (1 - fx) + h[iy, ix + 1] * fx) * (1 - fy) +
                       (h[iy + 1, ix] * (1 - fx) + h[iy + 1, ix + 1] * fx) * fy)
                gx = (((h[iy, ix + 1] - h[iy, ix - 1]) * (1 - fx) +
                       (h[iy, ix + 2] - h[iy, ix]) * fx) * (1 - fy) +
                      ((h[iy + 1, ix + 1] - h[iy + 1, ix - 1]) * (1 - fx) +
                       (h[iy + 1, ix + 2] - h[iy + 1, ix]) * fx) * fy)
                gy = (((h[iy + 1, ix] - h[iy - 1, ix]) * (1 - fx) +
                       (h[iy + 1, ix + 1] - h[iy - 1, ix + 1]) * fx) * (1 - fy) +
                      ((h[iy + 2, ix] - h[iy, ix]) * (1 - fx) +
                       (h[iy + 2, ix + 1] - h[iy, ix + 1]) * fx) * fy)
                ndx = dx * inertia - gx * (1 - inertia)
                ndy = dy * inertia - gy * (1 - inertia)
                dl = math.sqrt(ndx * ndx + ndy * ndy)
                if dl < 1e-10:
                    a = np.random.random() * 6.2831853
                    ndx = math.cos(a); ndy = math.sin(a); dl = 1.0
                dx = ndx / dl; dy = ndy / dl
                x += dx; y += dy
                if x < 1 or y < 1 or x > W - 3 or y > H - 3:
                    break
                nix = int(x); niy = int(y); nfx = x - nix; nfy = y - niy
                new = ((h[niy, nix] * (1 - nfx) + h[niy, nix + 1] * nfx) *
                       (1 - nfy) +
                       (h[niy + 1, nix] * (1 - nfx) + h[niy + 1, nix + 1] *
                        nfx) * nfy)
                dh = new - old
                cap = max(-dh, 0.001) * speed * water * cap_f
                if sed > cap or dh > 0:
                    if dh > 0:
                        amt = min(sed, dh)
                    else:
                        amt = (sed - cap) * deposit_s
                    h[niy, nix] += amt * (1 - nfx) * (1 - nfy)
                    h[niy, nix + 1] += amt * nfx * (1 - nfy)
                    h[niy + 1, nix] += amt * (1 - nfx) * nfy
                    h[niy + 1, nix + 1] += amt * nfx * nfy
                    sed -= amt
                else:
                    amt = min((cap - sed) * erode_s, -dh)
                    b = amt * 0.10; h[iy, ix] -= b
                    b = amt * 0.20
                    h[iy, ix + 1] -= b; h[iy, ix - 1] -= b
                    h[iy + 1, ix] -= b; h[iy - 1, ix] -= b
                    b = amt * 0.05
                    h[iy + 1, ix + 1] -= b; h[iy + 1, ix - 1] -= b
                    h[iy - 1, ix + 1] -= b; h[iy - 1, ix - 1] -= b
                    sed += amt
                sp2 = speed * speed - dh * 10.0
                speed = math.sqrt(sp2) if sp2 > 0.0 else 0.0
                water *= (1.0 - evap)
                if water < 0.005:
                    break
        return h


def hydraulic_numba(h, p, seed, progress, cancelled, snapshot=None):
    n = int(p['droplets'])
    lifetime = int(p['lifetime']); inertia = float(p['inertia'])
    cap_f = float(p['capacity']); erode_s = float(p['erode'])
    deposit_s = float(p['deposit']); evap = float(p['evaporate'])
    h = h.astype(np.float64, copy=True)
    chunk = max(2000, n // 40)
    done = 0
    while done < n:
        if cancelled():
            raise GenCancelled()
        c = min(chunk, n - done)
        h = _droplets_core(h, c, lifetime, inertia, cap_f, erode_s,
                           deposit_s, evap, (seed + done) & 0x7FFFFFFF)
        done += c
        progress(done / n)
        if snapshot is not None:
            snapshot(_down(h))
    return h


def _erode_chunk(args):
    h, p, seed = args
    return hydraulic_erode(h, p, seed, lambda f: None, lambda: False)


def hydraulic_parallel(h, p, seed, progress, cancelled, snapshot=None):
    total = int(p['droplets'])
    procs = min(8, max(2, total // 1500), os.cpu_count() or 1)
    if procs < 2 or total < 1500:
        return hydraulic_erode(h, p, seed, progress, cancelled, snapshot)
    per = max(200, total // procs)
    try:
        import multiprocessing as mp
        ctx = mp.get_context('fork' if hasattr(os, 'fork') else 'spawn')
        args = [(h, {**p, 'droplets': per},
                 (seed ^ (0x9E3779B9 * (i + 1))) & 0x7FFFFFFF)
                for i in range(procs)]
        results = []
        with ctx.Pool(procs) as pool:
            for res in pool.imap_unordered(_erode_chunk, args):
                if cancelled():
                    pool.terminate()
                    raise GenCancelled()
                results.append(res)
                progress(0.95 * len(results) / procs)
        return np.mean(np.stack(results), axis=0)
    except GenCancelled:
        raise
    except Exception:
        return hydraulic_erode(h, p, seed, progress, cancelled)


def bilinear_sample(a, X, Y):
    H, W = a.shape
    x0 = np.clip(np.floor(X).astype(np.int64), 0, W - 1)
    y0 = np.clip(np.floor(Y).astype(np.int64), 0, H - 1)
    x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
    fx = X - x0; fy = Y - y0
    top = a[y0, x0] * (1 - fx) + a[y0, x1] * fx
    bot = a[y1, x0] * (1 - fx) + a[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def resize2d(a, w2, h2):
    ys = np.linspace(0, a.shape[0] - 1, h2)
    xs = np.linspace(0, a.shape[1] - 1, w2)
    X, Y = np.meshgrid(xs, ys)
    return bilinear_sample(a, X, Y)


def pipe_erode(hgt, p, progress, cancelled, snapshot=None):
    f32 = np.float32
    H, W = hgt.shape
    h = hgt.astype(f32, copy=True)
    w = np.zeros((H, W), f32); s = np.zeros((H, W), f32)
    fL = np.zeros((H, W), f32); fR = np.zeros((H, W), f32)
    fT = np.zeros((H, W), f32); fB = np.zeros((H, W), f32)
    tA = np.empty((H, W), f32); tB = np.empty((H, W), f32)
    tC = np.empty((H, W), f32); tD = np.empty((H, W), f32)
    tE = np.empty((H, W), f32); tF = np.empty((H, W), f32)
    gx = np.empty((H, W), f32); gy = np.empty((H, W), f32)
    inflow = np.empty((H, W), f32)
    mB = np.empty((H, W), dtype=bool)
    hwP = np.empty((H + 2, W + 2), f32)
    hw = hwP[1:-1, 1:-1]
    colgrid = np.tile(np.arange(W, dtype=f32), (H, 1))
    rowgrid = np.tile(np.arange(H, dtype=f32)[:, None], (1, W))
    dt = f32(0.25)
    cKf = f32(dt * float(p['flow_flux']))
    rain = f32(p['flow_rain']); evap = f32(p['flow_evap'])
    Kc = f32(p['flow_capacity']); Ke = f32(p['flow_erode'])
    Kd = f32(p['flow_deposit']); iters = int(p['flow_iters'])
    halfdt = f32(0.125)
    c1 = f32(1.0); c4 = f32(4.0); c15 = f32(1.5); c001 = f32(0.01)
    c8 = f32(8.0); c1e_10 = f32(1e-10)
    for it in range(iters):
        if cancelled():
            raise GenCancelled()
        if it % 8 == 0:
            progress(it / iters)
        if snapshot is not None and it % 20 == 0:
            snapshot(_down(h))
        np.add(h, w, out=hw)
        hwP[0, 1:-1] = hw[0]; hwP[-1, 1:-1] = hw[-1]
        hwP[:, 0] = hwP[:, 1]; hwP[:, -1] = hwP[:, -2]
        np.subtract(hw, hwP[1:-1, :-2], out=tA); tA *= cKf; tA += fL
        np.maximum(tA, 0.0, out=fL)
        np.subtract(hw, hwP[1:-1, 2:], out=tA); tA *= cKf; tA += fR
        np.maximum(tA, 0.0, out=fR)
        np.subtract(hw, hwP[:-2, 1:-1], out=tA); tA *= cKf; tA += fT
        np.maximum(tA, 0.0, out=fT)
        np.subtract(hw, hwP[2:, 1:-1], out=tA); tA *= cKf; tA += fB
        np.maximum(tA, 0.0, out=fB)
        np.add(fL, fR, out=tA); tA += fT; tA += fB
        tA *= dt
        np.greater(tA, c1e_10, out=mB)
        np.divide(w, tA, out=tA, where=mB)
        np.logical_not(mB, out=mB)
        np.copyto(tA, c1, where=mB)
        np.minimum(tA, c1, out=tA)
        fL *= tA; fR *= tA; fT *= tA; fB *= tA
        inflow.fill(0.0)
        inflow[:, 1:] += fR[:, :-1]
        inflow[:, :-1] += fL[:, 1:]
        inflow[1:, :] += fT[:-1, :]
        inflow[:-1, :] += fB[1:, :]
        np.add(fL, fR, out=tB); tB += fT; tB += fB
        np.subtract(inflow, tB, out=tB); tB *= dt
        w += tB
        np.clip(w, 0.0, c8, out=w)
        np.subtract(fR, fL, out=tC); tC *= halfdt
        np.subtract(fB, fT, out=tD); tD *= halfdt
        np.subtract(h[:, 2:], h[:, :-2], out=gx[:, 1:-1]); gx[:, 1:-1] *= f32(0.5)
        gx[:, 0] = h[:, 1] - h[:, 0]; gx[:, -1] = h[:, -1] - h[:, -2]
        np.subtract(h[2:, :], h[:-2, :], out=gy[1:-1, :]); gy[1:-1, :] *= f32(0.5)
        gy[0, :] = h[1, :] - h[0, :]; gy[-1, :] = h[-1, :] - h[-2, :]
        np.multiply(gx, gx, out=tA); tA += gy * gy
        np.sqrt(tA, out=tA)
        np.multiply(tA, tA, out=tB); tB += c1
        np.sqrt(tB, out=tB)
        np.divide(tA, tB, out=tA)
        np.multiply(tC, tC, out=tB); tB += tD * tD
        np.sqrt(tB, out=tB)
        tA *= tB
        np.minimum(w, c15, out=tB); tB *= Kc; tA *= tB
        np.minimum(tA, c4, out=tA)
        np.subtract(tA, s, out=tB)
        np.maximum(tB, 0.0, out=tE); tE *= Ke
        np.minimum(tE, c001, out=tE)
        np.subtract(s, tA, out=tF)
        np.maximum(tF, 0.0, out=tF); tF *= Kd
        np.minimum(tF, s, out=tF)
        np.subtract(h, tE, out=h); np.add(h, tF, out=h)
        np.subtract(s, tF, out=s); np.add(s, tE, out=s)
        np.multiply(tC, -dt, out=tE); tE += colgrid
        np.multiply(tD, -dt, out=tF); tF += rowgrid
        np.clip(tE, 0.0, f32(W - 1.001), out=tE)
        np.clip(tF, 0.0, f32(H - 1.001), out=tF)
        s = bilinear_sample(s, tE, tF)
        w *= (f32(1.0) - evap)
    return h.astype(np.float64)


def flow_accumulation(h, seed=0, sea=None):
    H, W = h.shape
    rng = np.random.default_rng(seed if seed else 0x24635)
    hn = h + (rng.random((H, W)) - 0.5) * 1e-6
    pad = np.pad(hn, 1, mode='edge')
    nb = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
    dirs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    choice = np.argmin(nb, axis=0)
    lowv = np.take_along_axis(nb, choice[None, :, :], 0)[0]
    is_low = lowv < hn
    if sea is not None:
        is_low &= hn > float(sea)
    yy, xx = np.mgrid[0:H, 0:W]
    ty = np.clip(yy + dirs[choice][:, :, 0], 0, H - 1)
    tx = np.clip(xx + dirs[choice][:, :, 1], 0, W - 1)
    tgt = np.full(H * W, -1, dtype=np.int64)
    sel = is_low.ravel()
    tgt.ravel()[sel] = (ty * W + tx).ravel()[sel]
    order = np.argsort(hn.ravel())[::-1]
    order = order[tgt[order] >= 0]
    ol = order.tolist()
    tl = tgt[order].tolist()
    fl = np.ones(H * W, dtype=np.float64).tolist()
    for t, src in zip(tl, ol):
        fl[t] += fl[src]
    return np.asarray(fl, dtype=np.float64).reshape(H, W)

# ----------------------------------------------------------------------------
# Data fields (masks, lakes, contrast)
# ----------------------------------------------------------------------------

def wetness_mask(flow):
    m = np.log2(np.asarray(flow, np.float64) + 1.0)
    mn, mx = float(m.min()), float(m.max())
    return np.power((m - mn) / (mx - mn + 1e-9), 0.65)


def deposition_mask(h, sea, flow):
    gy, gx = np.gradient(np.maximum(h, sea))
    slope = np.sqrt(gx * gx + gy * gy)
    flat = np.clip(1.0 - slope / (float(slope.mean()) * 2.0 + 1e-9), 0.0, 1.0)
    f = np.log2(np.asarray(flow, np.float64) + 1.0)
    fn = (f - f.min()) / (f.max() - f.min() + 1e-9)
    return np.clip(fn * flat, 0.0, 1.0) ** 0.8


def curvature_field(h, sea):
    d = np.maximum(h, sea)
    p = np.pad(d, 1, mode='edge')
    lap = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * d)
    return np.tanh(lap * 150.0)


def walkable_mask(h, sea, thr):
    gy, gx = np.gradient(np.maximum(h, sea))
    return (np.sqrt(gx * gx + gy * gy) <= thr).astype(np.float64)


def autocontrast(h):
    lo, hi = np.percentile(h, (0.5, 99.5))
    if hi - lo < 1e-9:
        return h.copy()
    return np.clip((h - lo) / (hi - lo), 0.0, 1.0)


def detect_lakes(h, sea, max_iters=200):
    filled = h.copy()
    for _ in range(max_iters):
        p = np.pad(filled, 1, mode='edge')
        m = np.minimum(np.minimum(p[:-2, 1:-1], p[2:, 1:-1]),
                       np.minimum(p[1:-1, :-2], p[1:-1, 2:]))
        new = np.maximum(h, m)
        delta = float(np.max(np.abs(new - filled)))
        filled = new
        if delta < 1e-6:
            break
    return filled, (filled - h) > 1e-4


def _data_masks(h, flow, sea):
    if flow is not None:
        wet = wetness_mask(flow)
        dep = deposition_mask(h, sea, flow)
    else:
        wet = np.zeros_like(h); dep = np.zeros_like(h)
    return wet, dep, curvature_field(h, sea)


def _tint_lakes(rgb, lakes, alpha=0.55):
    lc = np.array([60, 130, 190], np.float32)
    m = lakes[:, :, None].astype(np.float32)
    out = rgb.astype(np.float32) * (1 - alpha * m) + lc * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def _autosave_path():
    try:
        d = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation)
    except Exception:
        d = ''
    if not d:
        d = os.path.expanduser('~')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'terrainforge_autosave.npz')

# ----------------------------------------------------------------------------
# Generation pipeline
# ----------------------------------------------------------------------------

def generate_terrain(p, progress, cancelled, snapshot=None):
    T = {}
    t0 = time.perf_counter()
    ps = {'25%': 0.25, '50%': 0.5, '100%': 1.0}[p['preview_scale']]
    w = max(64, int(round(int(p['width']) * ps)))
    hgt = max(64, int(round(int(p['height']) * ps)))
    seed = int(p['seed']); kind = p['noise_type']
    tiled = bool(p.get('tiled', False))
    scale_f = float(p['scale'])
    if tiled:
        period = max(1, min(256, int(round(scale_f))))
        eff_scale = float(period); lac = 2.0
    else:
        period = None; eff_scale = scale_f; lac = float(p['lacunarity'])
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(hgt, dtype=np.float64))
    cx = xs / max(w - 1, 1) + float(p['offset_x'])
    cy = ys / max(hgt - 1, 1) + float(p['offset_y'])
    progress(3)
    tw = time.perf_counter()
    X, Y = cx, cy
    if float(p['warp_strength']) > 0.001:
        if tiled:
            wf = float(max(1, min(256, round(float(p['warp_freq'])))))
        else:
            wf = float(p['warp_freq'])
        wx = fbm(cx * wf, cy * wf, seed + 911, 3, 0.5, 2.0, kind,
                 cancel=cancelled, period=(max(1, round(wf)) if tiled else None))
        wy = fbm(cx * wf, cy * wf, seed + 912, 3, 0.5, 2.0, kind,
                 cancel=cancelled, period=(max(1, round(wf)) if tiled else None))
        X = cx + float(p['warp_strength']) * wx * 2.0
        Y = cy + float(p['warp_strength']) * wy * 2.0
    T['warp'] = time.perf_counter() - tw
    progress(10)
    tb = time.perf_counter()
    base = fbm(X * eff_scale, Y * eff_scale, seed,
               int(p['octaves']), float(p['persistence']), lac, kind,
               ridged=(p['pattern'] == 'Ridged'),
               billow=(p['pattern'] == 'Billow'),
               cancel=cancelled, period=period,
               progress=lambda f: progress(10 + int(f * 32)))
    T['noise'] = time.perf_counter() - tb
    mn, mx = float(base.min()), float(base.max())
    hmap = (base - mn) / (mx - mn + 1e-12)
    del base
    hmap = hmap ** float(p['gamma'])
    progress(44)
    tm = time.perf_counter()
    hmap = apply_island_mask(hmap, p, seed, tiled)
    T['mask'] = time.perf_counter() - tm
    progress(46)
    if int(p['smoothing']) > 0:
        hmap = box_blur(hmap, int(p['smoothing']), wrap=tiled)
    progress(48)
    tt = time.perf_counter()
    if int(p['thermal_iters']) > 0:
        hmap = thermal_erode(hmap, int(p['thermal_iters']), float(p['talus']),
                             float(p['thermal_rate']),
                             lambda f: progress(48 + int(f * 12)), cancelled)
    T['thermal'] = time.perf_counter() - tt
    progress(60)
    if snapshot is not None:
        snapshot(_down(hmap))
    th = time.perf_counter()
    method = p['erosion_method']
    if method == 'Droplets (classic)' and int(p['droplets']) > 0:
        if HAVE_NUMBA and p.get('numba_droplets', True):
            hmap = hydraulic_numba(hmap, p, seed ^ 0x5EED,
                                   lambda f: progress(60 + int(f * 32)),
                                   cancelled, snapshot)
        elif p.get('erode_parallel', False):
            hmap = hydraulic_parallel(hmap, p, seed ^ 0x5EED,
                                      lambda f: progress(60 + int(f * 32)),
                                      cancelled)
        else:
            hmap = hydraulic_erode(hmap, p, seed ^ 0x5EED,
                                   lambda f: progress(60 + int(f * 32)),
                                   cancelled, snapshot)
    elif method == 'Flow (fast)':
        hmap = pipe_erode(hmap, p, lambda f: progress(60 + int(f * 32)),
                          cancelled, snapshot)
    T['hydraulic'] = time.perf_counter() - th
    progress(93)
    if int(p['terraces']) >= 2:
        hmap = terrace(hmap, int(p['terraces']))
    mn, mx = float(hmap.min()), float(hmap.max())
    hmap = (hmap - mn) / (mx - mn + 1e-12)
    progress(94)

    wA = None
    A = p.get('_layer_a')
    if p.get('blend_enable') and A is not None:
        Ar = resize2d(np.clip(np.asarray(A, np.float64), 0.0, 1.0), w, hgt)
        m = blend_mask_field(p, (hgt, w), seed, hmap, p.get('_blend_mask'))
        lo = float(p.get('blend_lo', 0.35))
        hi = max(float(p.get('blend_hi', 0.65)), lo + 0.02)
        wA = smoothstep(lo, hi, m)
        hmap = np.clip(Ar * wA + hmap * (1.0 - wA), 0.0, 1.0)
        if snapshot is not None:
            snapshot(_down(hmap))
    progress(96)

    flow_field = None
    tv = time.perf_counter()
    if bool(p['rivers']):
        flow_field = flow_accumulation(hmap, seed=seed ^ 24635,
                                       sea=float(p['sea_level']))
    T['flowacc'] = time.perf_counter() - tv
    progress(98)
    np.clip(hmap, 0.0, 1.0, out=hmap)
    T['total'] = time.perf_counter() - t0
    progress(100)
    return hmap, flow_field, T, wA

# ----------------------------------------------------------------------------
# Palettes & rendering
# ----------------------------------------------------------------------------

PALETTES = {
    'Terrain': {
        'water': [(0, (8, 28, 64)), (0.65, (20, 60, 120)), (0.92, (40, 110, 160)), (1, (170, 210, 200))],
        'land': [(0, (210, 190, 140)), (0.08, (95, 155, 75)), (0.35, (45, 110, 55)),
                 (0.62, (130, 120, 90)), (0.8, (150, 145, 140)), (0.92, (225, 228, 235)), (1, (255, 255, 255))]},
    'Alpine': {
        'water': [(0, (15, 35, 70)), (1, (80, 140, 170))],
        'land': [(0, (60, 90, 60)), (0.35, (110, 130, 90)), (0.6, (140, 130, 110)),
                 (0.8, (190, 185, 180)), (1, (255, 255, 255))]},
    'Desert': {
        'water': [(0, (20, 50, 90)), (1, (70, 150, 170))],
        'land': [(0, (240, 220, 170)), (0.3, (220, 180, 120)), (0.6, (190, 140, 90)),
                 (0.8, (140, 95, 70)), (1, (255, 250, 240))]},
    'Arctic': {
        'water': [(0, (10, 30, 60)), (1, (120, 170, 200))],
        'land': [(0, (200, 215, 225)), (0.5, (235, 240, 245)), (1, (255, 255, 255))]},
    'Volcanic': {
        'water': [(0, (10, 10, 25)), (1, (40, 30, 60))],
        'land': [(0, (30, 20, 20)), (0.4, (70, 45, 40)), (0.7, (120, 60, 40)),
                 (0.9, (230, 120, 30)), (0.97, (255, 220, 90)), (1, (255, 255, 220))]},
    'Alien': {
        'water': [(0, (30, 10, 60)), (1, (90, 40, 140))],
        'land': [(0, (40, 20, 60)), (0.3, (90, 30, 90)), (0.55, (160, 50, 110)),
                 (0.8, (240, 120, 60)), (1, (255, 240, 160))]},
    'Grayscale': {
        'water': [(0, (0, 0, 0)), (1, (120, 120, 120))],
        'land': [(0, (120, 120, 120)), (1, (255, 255, 255))]},
}

SPLAT_THEMES = {
    'Terrain': [('grass', (104, 168, 92)), ('rock', (132, 128, 124)),
                ('sand', (224, 198, 146)), ('snow', (245, 248, 252))],
    'Alpine': [('grass', (96, 138, 84)), ('rock', (122, 118, 114)),
               ('scree', (150, 146, 142)), ('snow', (245, 248, 252))],
    'Desert': [('dune', (226, 196, 138)), ('rock', (168, 124, 86)),
               ('cracked', (196, 150, 102)), ('salt', (240, 232, 214))],
    'Arctic': [('tundra', (150, 166, 150)), ('rock', (120, 120, 124)),
               ('ice', (190, 220, 235)), ('snow', (246, 250, 255))],
    'Volcanic': [('ash', (78, 72, 70)), ('basalt', (52, 48, 50)),
                 ('scoria', (122, 72, 58)), ('sulfur', (224, 196, 92))],
    'Alien': [('flora', (96, 180, 120)), ('rock', (110, 90, 120)),
              ('crystal', (160, 200, 220)), ('glow', (230, 240, 150))],
    'Grayscale': [('dark', (70, 70, 70)), ('mid', (120, 120, 120)),
                  ('light', (180, 180, 180)), ('white', (250, 250, 250))],
}

_BGRID = np.array([
    [138, 144, 150], [146, 158, 120], [188, 172, 120], [204, 178, 118],
    [120, 132, 112], [88, 128, 82], [140, 156, 92], [176, 158, 96],
    [132, 142, 138], [74, 116, 74], [52, 108, 58], [74, 134, 70],
    [148, 156, 160], [64, 108, 78], [38, 96, 50], [44, 124, 58],
], np.float32).reshape(4, 4, 3)

_LUT_CACHE = {}
_AO = {'key': None, 'cav': None}
_FLOG = {'id': None, 'm': None}
_MOIST = {'key': None, 'm': None}


def _lut(stops, n=256):
    pos = np.array([s[0] for s in stops], float)
    cols = np.array([s[1] for s in stops], float)
    t = np.linspace(0, 1, n)
    out = np.empty((n, 3))
    for c in range(3):
        out[:, c] = np.interp(t, pos, cols[:, c])
    return out


def _palette_luts(name):
    r = _LUT_CACHE.get(name)
    if r is None:
        pal = PALETTES[name]
        r = (_lut(pal['water']), _lut(pal['land']))
        _LUT_CACHE[name] = r
    return r


def _moisture(h, seed, freq):
    key = (id(h), seed, round(freq, 2))
    if _MOIST['key'] != key:
        H, W = h.shape
        xs = np.arange(W, dtype=np.float64) / max(W - 1, 1)
        ys = np.arange(H, dtype=np.float64) / max(H - 1, 1)
        cx, cy = np.meshgrid(xs, ys)
        m = value2d(cx * freq + 11.3, cy * freq + 5.9, seed ^ 0xB10B7)
        _MOIST.update(key=key, m=np.clip(m * 0.5 + 0.5, 0.0, 1.0))
    return _MOIST['m']


def _biome_rgb(h, p, sea):
    H, W = h.shape
    seed = int(p['seed']); freq = float(p.get('moist_scale', 3.5))
    m = _moisture(h, seed, freq)
    lat = np.abs(np.linspace(-1.0, 1.0, H))[:, None]
    temp = np.clip(1.0 - lat * 1.15 - np.maximum(h - sea, 0.0) * 1.3, 0.0, 1.0)
    fr = np.clip(m, 0, 1) * 3.0
    fc = temp * 3.0
    y0 = np.clip(np.floor(fr).astype(np.int64), 0, 2)
    x0 = np.clip(np.floor(fc).astype(np.int64), 0, 2)
    y1 = np.minimum(y0 + 1, 3); x1 = np.minimum(x0 + 1, 3)
    ty = (fr - y0)[:, :, None]; tx = (fc - x0)[:, :, None]
    B = (_BGRID[y0, x0] * (1 - ty) * (1 - tx) + _BGRID[y1, x0] * ty * (1 - tx) +
         _BGRID[y0, x1] * (1 - ty) * tx + _BGRID[y1, x1] * ty * tx)
    gy2, gx2 = np.gradient(np.maximum(h, sea))
    slope = np.sqrt(gx2 * gx2 + gy2 * gy2)
    rock = smoothstep(0.05, 0.13, slope)
    RC = np.array([126.0, 122.0, 118.0], np.float32)
    B = B * (1 - rock)[:, :, None] + RC[None, None, :] * rock[:, :, None]
    land = (h >= sea)
    snow = np.clip((0.14 - temp) * 7.0, 0, 1) * np.clip((h - sea) * 8.0, 0, 1)
    snow = np.maximum(snow, np.clip((h - 0.80) * 7.0, 0, 1))
    snow = np.clip(snow, 0, 1) * land * (1 - 0.5 * rock)
    SC = np.array([242.0, 246.0, 250.0], np.float32)
    B = B * (1 - snow)[:, :, None] + SC[None, None, :] * snow[:, :, None]
    beach = np.clip(1.0 - np.abs(h - sea) / 0.025, 0, 1) * land * (1 - rock) * (1 - snow)
    BC = np.array([224.0, 206.0, 160.0], np.float32)
    B = B * (1 - beach)[:, :, None] + BC[None, None, :] * beach[:, :, None]
    return B


def make_splat_rgba(h, sea, p=None):
    """R=base(grass), G=rock(slope), B=sand(coast), A=snow(altitude)."""
    p = p or {}
    gy, gx = np.gradient(np.maximum(h, sea))
    slope = np.sqrt(gx * gx + gy * gy)
    rm = float(p.get('spl_rock', 0.05))
    rock = smoothstep(rm, rm * 2.2, slope)
    bw = float(p.get('spl_beach', 0.045))
    sand = np.clip(1.0 - np.abs(h - sea) / max(bw, 1e-4), 0, 1) ** 1.5 * (1 - rock)
    sl = float(p.get('spl_snow', 0.79))
    snow = smoothstep(sl, min(sl + 0.18, 1.0), h) * (1 - 0.65 * rock)
    grass = np.clip(1.0 - rock - sand - snow, 0.0, 1.0)
    tot = grass + rock + sand + snow + 1e-6
    return np.dstack([grass, rock, sand, snow]) / tot[:, :, None]


def live_preview_img(a, p):
    a = np.clip(a, 0.0, 1.0)
    H, W = a.shape
    sea = max(float(p['sea_level']), 1e-6)
    pal = 'Terrain' if p.get('palette') == 'Biomes' else p.get('palette', 'Terrain')
    lut_w, lut_l = _palette_luts(pal)
    iw = (np.clip(a / sea, 0, 1) * 255).astype(np.uint8)
    il = (np.clip((a - sea) / max(1 - sea, 1e-6), 0, 1) * 255).astype(np.uint8)
    rgb = np.where((a < sea)[:, :, None], lut_w[iw], lut_l[il])
    disp = np.maximum(a, sea)
    gy, gx = np.gradient(disp)
    nx, ny = -gx * 6.0, -gy * 6.0
    nl = np.sqrt(nx * nx + ny * ny + 1.0)
    az = math.radians(float(p.get('sun_az', 315)))
    alt = math.radians(float(p.get('sun_alt', 45)))
    lx = math.cos(alt) * math.cos(az); ly = math.cos(alt) * math.sin(az)
    lz = math.sin(alt)
    sh = np.clip((nx * lx + ny * ly + lz) / nl, 0, 1)
    k = 0.55
    rgbf = rgb.astype(np.float32) * ((1 - k) + k * sh)[:, :, None]
    rgb = np.ascontiguousarray(np.clip(rgbf, 0, 255).astype(np.uint8))
    return QImage(rgb.tobytes(), W, H, 3 * W, QImage.Format.Format_RGB888).copy()


def render_terrain(h, p, flow=None, splat=None, blend_w=None, lakes=None):
    H, W = h.shape
    sea = float(p['sea_level'])
    mode = p['render_mode']
    gscale = float(p['relief']) * W / 16.0
    wrap = bool(p.get('tiled', False))

    def _img(rgb):
        rgb = np.ascontiguousarray(rgb, np.uint8)
        return QImage(rgb.tobytes(), W, H, 3 * W, QImage.Format.Format_RGB888).copy(), rgb

    if mode == 'Heightmap':
        g = (np.clip(h, 0, 1) * 255).astype(np.uint8)
        return _img(np.repeat(g[:, :, None], 3, axis=2))
    if mode == 'Normals':
        gy, gx = np.gradient(h)
        nx, ny = -gx * gscale, -gy * gscale
        nl = np.sqrt(nx * nx + ny * ny + 1.0)
        rgb = (np.clip(np.stack([nx / nl, ny / nl, 1.0 / nl], -1) * 0.5 + 0.5,
                       0, 1) * 255).astype(np.uint8)
        return _img(rgb)
    if mode == 'Slope':
        gy, gx = np.gradient(h)
        s = np.clip(np.sqrt(gx * gx + gy * gy) * gscale, 0, 1)
        g8 = (s * 255).astype(np.uint8)
        return _img(np.repeat(g8[:, :, None], 3, axis=2))
    if mode == 'Wetness':
        m = wetness_mask(flow) if flow is not None else np.zeros_like(h)
        m = np.where(h < sea, 0.0, m)
        c = np.array([20, 24, 28], np.float32) * (1 - m[:, :, None]) + \
            np.array([70, 160, 200], np.float32) * m[:, :, None]
        return _img(c)
    if mode == 'Deposition':
        m = deposition_mask(h, sea, flow) if flow is not None else np.zeros_like(h)
        m = np.where(h < sea, 0.0, m)
        c = np.array([30, 26, 22], np.float32) * (1 - m[:, :, None]) + \
            np.array([225, 180, 90], np.float32) * m[:, :, None]
        return _img(c)
    if mode == 'Curvature':
        c = curvature_field(h, sea)
        t = np.clip(c * 0.5 + 0.5, 0, 1)[:, :, None]
        cA = np.array([220, 90, 60], np.float32)
        cB = np.array([60, 120, 220], np.float32)
        return _img(cA * (1 - t) + cB * t)
    if mode == 'Blend mask':
        if blend_w is None:
            return _img(np.full((H, W, 3), 70, np.uint8))
        wv = np.clip(blend_w, 0, 1)[:, :, None].astype(np.float32)
        CA = np.array([235.0, 150.0, 70.0], np.float32)
        CB = np.array([70.0, 170.0, 190.0], np.float32)
        return _img(CA * wv + CB * (1.0 - wv))

    pal_name = p['palette']
    is_biome = (pal_name == 'Biomes')
    lut_w = _palette_luts('Terrain' if is_biome else pal_name)[0]
    iw = (np.clip(h / max(sea, 1e-6), 0, 1) * 255).astype(np.uint8)
    water_rgb = lut_w[iw]

    if mode == 'Walkable':
        gy2, gx2 = np.gradient(np.maximum(h, sea))
        sl = np.sqrt(gx2 * gx2 + gy2 * gy2)
        thr = float(p.get('walk_slope', 0.10))
        rgb = np.full((H, W, 3), 235.0, np.float32)
        rgb[(sl > thr) & (h >= sea)] = (205.0, 62.0, 50.0)
        rgb[h < sea] = (92.0, 118.0, 148.0)
        return _img(rgb)
    if mode == 'Strata':
        n = max(2, int(p.get('strata_n', 12)))
        XS, YS = np.meshgrid(np.linspace(0.0, 7.0, W), np.linspace(0.0, 5.0, H))
        jit = value2d(XS, YS, int(p['seed']) ^ 0x57A) * 0.02
        hb = np.clip(h + jit, 0.0, 0.99999)
        band = (np.floor(hb * n) + 0.5) / n
        pal2 = 'Terrain' if is_biome else pal_name
        rgb = _palette_luts(pal2)[1][np.clip(band * 255.0, 0, 255).astype(np.uint8)]
        gy2, gx2 = np.gradient(np.maximum(h, sea))
        sh = np.clip(1.0 - np.sqrt(gx2 * gx2 + gy2 * gy2) * gscale * 0.9,
                     0.5, 1.0)
        rgb = (rgb.astype(np.float32) * sh[:, :, None]).astype(np.uint8)
        rgb = np.where((h < sea)[:, :, None], water_rgb, rgb)
        return _img(rgb)

    if mode == 'Splat':
        wgt = splat if splat is not None else make_splat_rgba(h, sea, p)
        theme = SPLAT_THEMES.get('Terrain' if is_biome else pal_name,
                                 SPLAT_THEMES['Terrain'])
        rgbf = np.zeros((H, W, 3), np.float32)
        for i in range(4):
            rgbf += (wgt[:, :, i] * np.array(theme[i][1], np.float32))[:, :, None]
        rgbf = np.where((h < sea)[:, :, None], water_rgb.astype(np.float32), rgbf)
        if lakes is not None:
            rgbf = _tint_lakes(rgbf, lakes).astype(np.float32)
        return _img(np.clip(rgbf, 0, 255))

    if is_biome:
        base_rgb = _biome_rgb(h, p, sea)
        rgb = np.where((h < sea)[:, :, None], water_rgb, base_rgb)
    else:
        lut_l = _palette_luts(pal_name)[1]
        il = (np.clip((h - sea) / max(1 - sea, 1e-6), 0, 1) * 255).astype(np.uint8)
        rgb = np.where((h < sea)[:, :, None], water_rgb, lut_l[il])
    disp = np.maximum(h, sea)
    gy, gx = np.gradient(disp)
    nx, ny = -gx * gscale, -gy * gscale
    nl = np.sqrt(nx * nx + ny * ny + 1.0)
    az = math.radians(float(p['sun_az'])); alt = math.radians(float(p['sun_alt']))
    lx = math.cos(alt) * math.cos(az); ly = math.cos(alt) * math.sin(az)
    lz = math.sin(alt)
    shade = np.clip((nx * lx + ny * ly + lz) / nl, 0, 1)
    k = float(p['shade_strength'])
    S = (1 - k + k * shade)
    ao = float(p['ao'])
    if ao > 0.01:
        key = (id(h), round(float(p['relief']), 3), wrap)
        if _AO['key'] != key:
            bl = box_blur(h, 3, wrap=wrap)
            _AO.update(key=key, cav=np.clip((bl - h) * gscale * 0.8, 0.0, 1.5))
        S = S * np.clip(1.0 - ao * _AO['cav'], 0.3, 1.0)
    rgbf = rgb.astype(np.float32) * S[:, :, None]
    if bool(p['contours']):
        n = float(p['contour_n'])
        c = h * n
        line = np.abs(c - np.round(c)) < 0.05
        line &= h > sea
        line |= np.abs(h - sea) < 0.0015
        col = np.array(QColor(p['contour_color']).getRgb()[:3], float)
        rgbf = np.where(line[:, :, None], rgbf * 0.45 + col * 0.55, rgbf)
    if flow is not None and float(p['river_amount']) > 0.001:
        if _FLOG['id'] != id(flow):
            _FLOG.update(id=id(flow), m=np.log2(flow + 1.0))
        m = np.clip((_FLOG['m'] - float(p['river_threshold'])) / 1.6, 0, 1)
        m = np.where(h > sea, m, 0.0)
        if m.any():
            m = box_blur(m, 1, wrap=wrap) * float(p['river_amount'])
            riv = lut_w[40][None, None, :] * (0.45 + 0.55 * S)[:, :, None]
            rgbf = rgbf * (1 - m[:, :, None]) + riv * m[:, :, None]
    rgb = np.clip(rgbf, 0, 255).astype(np.uint8)
    if lakes is not None:
        rgb = _tint_lakes(rgb, lakes)
    return _img(rgb)

# ----------------------------------------------------------------------------
# Quick styles
# ----------------------------------------------------------------------------

STYLES = {
    'Continents': dict(pattern='fBm', scale=2.0, octaves=8, persistence=0.52,
        warp_strength=0.11, warp_freq=2.0, gamma=1.0, mask_type='None',
        sea_level=0.34, erosion_method='Flow (fast)', flow_iters=180,
        thermal_iters=20, droplets=4000, palette='Terrain', rivers=True,
        river_threshold=8, river_amount=0.8, smoothing=0, terraces=0),
    'Biome Realms': dict(pattern='fBm', scale=3.0, octaves=8, persistence=0.52,
        warp_strength=0.10, gamma=1.0, mask_type='None', sea_level=0.32,
        erosion_method='Flow (fast)', flow_iters=180, thermal_iters=20,
        palette='Biomes', rivers=True, river_threshold=8, moist_scale=3.5),
    'Archipelago': dict(mask_type='Archipelago (noise)', mask_size=0.8,
        mask_strength=0.9, sea_level=0.42, pattern='fBm', scale=3.2, octaves=7,
        warp_strength=0.12, erosion_method='Droplets (classic)', droplets=5000,
        thermal_iters=12, palette='Terrain', rivers=False, gamma=1.0),
    'Alpine Peaks': dict(pattern='Ridged', octaves=9, persistence=0.55, scale=3.4,
        sea_level=0.22, gamma=0.9, erosion_method='Flow (fast)', flow_iters=200,
        thermal_iters=35, palette='Alpine', rivers=True, river_threshold=9,
        warp_strength=0.06, mask_type='None'),
    'Canyonlands': dict(pattern='Billow', scale=4.5, octaves=7, persistence=0.5,
        warp_strength=0.16, gamma=1.15, sea_level=0.14, erosion_method='Flow (fast)',
        flow_iters=260, flow_capacity=4.5, thermal_iters=60, talus=0.004,
        palette='Desert', rivers=True, river_threshold=7, mask_type='None', terraces=0),
    'Volcano Isles': dict(mask_type='Volcano', mask_size=0.8, mask_strength=0.85,
        palette='Volcanic', sea_level=0.4, scale=3.0, octaves=6, pattern='fBm',
        warp_strength=0.08, erosion_method='Flow (fast)', flow_iters=120,
        thermal_iters=15, rivers=False, gamma=0.95),
    'Dune Sea': dict(noise_type='Value', pattern='Billow', scale=2.4, octaves=4,
        persistence=0.42, warp_strength=0.03, gamma=1.1, smoothing=2,
        sea_level=0.06, erosion_method='None', thermal_iters=0, palette='Desert',
        rivers=False, mask_type='None'),
    'Frozen North': dict(pattern='Ridged', octaves=8, scale=3.0, sea_level=0.26,
        palette='Arctic', erosion_method='Flow (fast)', flow_iters=160,
        thermal_iters=25, rivers=True, river_threshold=9, warp_strength=0.07,
        mask_type='Round island', mask_strength=0.6),
    'Badlands': dict(pattern='fBm', scale=4.0, octaves=7, terraces=10,
        thermal_iters=80, talus=0.005, erosion_method='None', sea_level=0.12,
        palette='Terrain', rivers=True, river_threshold=8, gamma=1.0,
        warp_strength=0.1, mask_type='None'),
    'Lakeland': dict(scale=5.0, octaves=8, sea_level=0.46,
        erosion_method='Droplets (classic)', droplets=8000, thermal_iters=20,
        palette='Terrain', rivers=True, river_threshold=7, warp_strength=0.1,
        gamma=1.0, pattern='fBm', mask_type='None'),
    'Stone Pavement': dict(noise_type='Worley', pattern='fBm', scale=6.0,
        octaves=3, persistence=0.5, gamma=1.25, smoothing=1, sea_level=0.05,
        erosion_method='None', thermal_iters=0, palette='Desert',
        rivers=False, mask_type='None'),
}

# ----------------------------------------------------------------------------
# Parameter widgets
# ----------------------------------------------------------------------------

class SliderRow(QWidget):
    changed = Signal()

    def __init__(self, label, mn, mx, val, dec=0, tip=None):
        super().__init__()
        self._dec = dec
        self._scale = 10 ** dec
        self._mn = mn; self._mx = mx
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 1, 0, 1); lay.setSpacing(6)
        self.lab = QLabel(label); self.lab.setFixedWidth(112)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(round(mn * self._scale), round(mx * self._scale))
        self.slider.setValue(round(val * self._scale))
        self.edit = QLineEdit()
        self.edit.setFixedWidth(58)
        self.edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.edit.setToolTip('Type an exact value')
        self.edit.editingFinished.connect(self._typed)
        self.slider.valueChanged.connect(self._on)
        lay.addWidget(self.lab); lay.addWidget(self.slider, 1); lay.addWidget(self.edit)
        if tip:
            self.setToolTip(tip)
        self._sync()

    def _on(self, _):
        self._sync(); self.changed.emit()

    def _typed(self):
        try:
            v = float(self.edit.text().replace(',', '.'))
        except ValueError:
            self._sync(); return
        v = round(min(self._mx, max(self._mn, v)), self._dec)
        self.slider.blockSignals(True)
        self.slider.setValue(round(v * self._scale))
        self.slider.blockSignals(False)
        self._sync()
        self.changed.emit()

    def _sync(self):
        self.edit.setText(f"{self.get():.{self._dec}f}")

    def get(self):
        return self.slider.value() / self._scale

    def set(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(round(float(v) * self._scale))
        self.slider.blockSignals(False)
        self._sync()


class CheckRow(QCheckBox):
    changed = Signal()

    def __init__(self, label, val=False, tip=None):
        super().__init__(label)
        self.setChecked(val)
        self.toggled.connect(lambda _: self.changed.emit())
        if tip:
            self.setToolTip(tip)

    def get(self): return self.isChecked()
    def set(self, v):
        self.blockSignals(True); self.setChecked(bool(v)); self.blockSignals(False)


class ComboRow(QWidget):
    changed = Signal()

    def __init__(self, label, items, val=None, tip=None):
        super().__init__()
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 1, 0, 1)
        lab = QLabel(label); lab.setFixedWidth(112)
        self.combo = QComboBox(); self.combo.addItems(items)
        if val:
            self.combo.setCurrentText(val)
        self.combo.currentTextChanged.connect(lambda _: self.changed.emit())
        lay.addWidget(lab); lay.addWidget(self.combo, 1)
        if tip:
            self.setToolTip(tip)

    def get(self): return self.combo.currentText()
    def set(self, v):
        self.combo.blockSignals(True)
        if v in [self.combo.itemText(i) for i in range(self.combo.count())]:
            self.combo.setCurrentText(v)
        self.combo.blockSignals(False)


class ColorRow(QWidget):
    changed = Signal()

    def __init__(self, label, color='#000000'):
        super().__init__()
        self._color = QColor(color)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 1, 0, 1)
        lab = QLabel(label); lab.setFixedWidth(112)
        self.btn = QPushButton(); self.btn.setFixedSize(64, 20)
        self.btn.clicked.connect(self._pick)
        lay.addWidget(lab); lay.addWidget(self.btn); lay.addStretch(1)
        self._apply()

    def _pick(self):
        c = QColorDialog.getColor(self._color, self, 'Pick colour')
        if c.isValid():
            self._color = c; self._apply(); self.changed.emit()

    def _apply(self):
        self.btn.setStyleSheet(
            f"background-color: {self._color.name()}; border: 1px solid #888;")

    def get(self): return self._color.name()
    def set(self, v):
        self._color = QColor(v); self._apply()


class SeedRow(QWidget):
    changed = Signal()

    def __init__(self, val=1337):
        super().__init__()
        self._hist = [int(val)]; self._idx = 0; self._nav = False
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 1, 0, 1); lay.setSpacing(4)
        lab = QLabel('Seed'); lab.setFixedWidth(112)
        self.spin = QSpinBox(); self.spin.setRange(0, 999999); self.spin.setValue(val)
        back = QPushButton('◀'); back.setFixedWidth(26)
        fwd = QPushButton('▶'); fwd.setFixedWidth(26)
        rnd = QPushButton('Random')
        self.spin.valueChanged.connect(self._on_val)
        back.clicked.connect(self._back); fwd.clicked.connect(self._fwd)
        rnd.clicked.connect(self.randomize)
        lay.addWidget(lab); lay.addWidget(self.spin, 1)
        lay.addWidget(back); lay.addWidget(fwd); lay.addWidget(rnd)

    def _on_val(self, v):
        if self._nav:
            self._nav = False
        else:
            del self._hist[self._idx + 1:]
            self._hist.append(int(v))
            self._idx = len(self._hist) - 1
        self.changed.emit()

    def _back(self):
        if self._idx > 0:
            self._idx -= 1; self._nav = True; self.spin.setValue(self._hist[self._idx])

    def _fwd(self):
        if self._idx < len(self._hist) - 1:
            self._idx += 1; self._nav = True; self.spin.setValue(self._hist[self._idx])

    def randomize(self):
        self.spin.setValue(random.randint(0, 999999))

    def get(self): return self.spin.value()
    def set(self, v):
        self.spin.blockSignals(True); self.spin.setValue(int(v)); self.spin.blockSignals(False)


class ToolPalette(QWidget):
    """Vertical tool strip with a ComboRow-compatible API."""
    changed = Signal()
    TOOLS = [
        ('✥', 'Off', 'Pan & inspect (default)'),
        ('▲', 'Raise', 'Raise terrain under the brush'),
        ('▼', 'Lower', 'Lower terrain under the brush'),
        ('≋', 'Smooth', 'Blur terrain under the brush'),
        ('▭', 'Flatten', 'Flatten towards the first height'),
        ('◉', 'Paint Splat', 'Paint R/G/B/A layer weights '
                             "(preview in 'Splat' render mode)"),
        ('◐', 'Paint Blend', 'Paint the Layer-A blend mask by hand'),
    ]

    def __init__(self, val='Off'):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._btns = {}
        for glyph, name, tip in self.TOOLS:
            b = QToolButton()
            b.setText(glyph)
            b.setCheckable(True)
            b.setFixedSize(40, 36)
            b.setToolTip(f'{name} — {tip}')
            b.clicked.connect(lambda _=False, n=name: self.changed.emit())
            self._group.addButton(b)
            self._btns[name] = b
            lay.addWidget(b)
        lay.addStretch(1)
        (self._btns.get(val) or self._btns['Off']).setChecked(True)

    def get(self):
        for n, b in self._btns.items():
            if b.isChecked():
                return n
        return 'Off'

    def set(self, v):
        b = self._btns.get(v)
        if b:
            b.setChecked(True)


class Group(QGroupBox):
    def __init__(self, title, parent_layout, collapsed=False):
        super().__init__(title)
        self.setCheckable(True)
        self.setChecked(not collapsed)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(8, 10, 8, 8)
        self.lay.setSpacing(3)
        self._rows = []
        self.toggled.connect(self._sync)
        parent_layout.addWidget(self)

    def addRow(self, w):
        self.lay.addWidget(w)
        self._rows.append(w)
        w.setVisible(self.isChecked())

    def _sync(self, on):
        for w in self._rows:
            w.setVisible(on)

# ----------------------------------------------------------------------------
# Analysis widgets
# ----------------------------------------------------------------------------

class ProfileWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._data = None; self._sea = 0.3
        self.setMinimumHeight(90)

    def set_data(self, data, sea):
        self._data = data; self._sea = sea
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 24))
        w, h = self.width(), self.height()
        if self._data is None or len(self._data) < 2:
            p.setPen(QColor(130, 130, 130))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Generate a terrain first')
            p.end(); return
        pad = 6.0
        n = len(self._data)
        xs = pad + np.arange(n) * (w - 2 * pad) / (n - 1)
        ys = (h - pad) - self._data * (h - 2 * pad)
        fill = QPolygonF([QPointF(float(xs[0]), h - pad)])
        fill += QPolygonF([QPointF(float(x), float(y)) for x, y in zip(xs, ys)])
        fill.append(QPointF(float(xs[-1]), h - pad))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(90, 150, 90, 70))
        p.drawPolygon(fill)
        p.setPen(QPen(QColor(140, 220, 140), 1))
        p.drawPolyline(QPolygonF([QPointF(float(x), float(y)) for x, y in zip(xs, ys)]))
        sy = (h - pad) - self._sea * (h - 2 * pad)
        p.setPen(QPen(QColor(90, 150, 210), 1, Qt.PenStyle.DashLine))
        p.drawLine(int(pad), int(sy), int(w - pad), int(sy))
        p.setPen(QColor(90, 150, 210))
        p.drawText(8, int(sy) - 3, 'sea')
        p.end()


class Histogram(QWidget):
    def __init__(self):
        super().__init__()
        self._counts = None; self._sea = 0.3
        self.setMinimumHeight(90); self.setMinimumWidth(220)

    def set_data(self, h, sea):
        self._sea = sea
        if h is None:
            self._counts = None
        else:
            counts, _ = np.histogram(h, bins=90, range=(0.0, 1.0))
            m = counts.max()
            self._counts = (counts / m if m > 0 else counts).astype(np.float64)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 24))
        w, h = self.width(), self.height()
        if self._counts is None:
            p.setPen(QColor(130, 130, 130))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Height distribution')
            p.end(); return
        n = len(self._counts)
        bw = w / n
        for i, c in enumerate(self._counts):
            centre = (i + 0.5) / n
            col = QColor(90, 140, 200) if centre < self._sea else QColor(110, 165, 110)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(col)
            bh = c * (h - 10)
            p.drawRect(QRectF(i * bw, h - 4 - bh, bw - 0.5, bh))
        p.setPen(QPen(QColor(255, 200, 80), 1, Qt.PenStyle.DashLine))
        p.drawLine(int(self._sea * w), 2, int(self._sea * w), h - 2)
        p.end()

# ----------------------------------------------------------------------------
# Background workers
# ----------------------------------------------------------------------------

class GenWorker(QObject):
    progress = Signal(int)
    snapshot = Signal(object)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, params):
        super().__init__()
        self.p = params
        self.cancel = False

    def run(self):
        try:
            snap = None
            if self.p.get('live_erosion', True):
                snap = lambda a: self.snapshot.emit(a)
            h, flow, T, bw = generate_terrain(self.p, self.progress.emit,
                                              lambda: self.cancel, snap)
            self.done.emit((h, flow, T, bw))
        except GenCancelled:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())


class ErodeWorker(QObject):
    """Applies thermal/hydraulic erosion to an EXISTING heightmap in place."""
    progress = Signal(int)
    snapshot = Signal(object)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, h, p, mode):
        super().__init__()
        self.h = h; self.p = p; self.mode = mode

    def run(self):
        try:
            snap = lambda a: self.snapshot.emit(a)
            seed = int(self.p['seed']) ^ 0x5EED
            h = self.h
            if self.mode.startswith('Hydraulic'):
                method = self.p.get('erosion_method', 'Flow (fast)')
                if (method == 'Droplets (classic)' and HAVE_NUMBA
                        and self.p.get('numba_droplets', True)):
                    h = hydraulic_numba(h, self.p, seed, self.progress.emit,
                                        lambda: False, snap)
                elif method == 'Droplets (classic)':
                    h = hydraulic_erode(h, self.p, seed, self.progress.emit,
                                        lambda: False, snap)
                else:
                    h = pipe_erode(h, self.p, self.progress.emit,
                                   lambda: False, snap)
            else:
                h = thermal_erode(h, int(self.p['thermal_iters']),
                                  float(self.p['talus']),
                                  float(self.p['thermal_rate']),
                                  self.progress.emit, lambda: False)
            self.done.emit(h)
        except Exception:
            self.failed.emit(traceback.format_exc())

# ----------------------------------------------------------------------------
# 2D canvas — zoom/pan + paint signals + brush cursor + tablet pressure
# ----------------------------------------------------------------------------

class Canvas(QGraphicsView):
    pixel = Signal(int, int)
    paint_at = Signal(float, float)
    paint_end = Signal()

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setBackgroundBrush(QColor(24, 24, 28))
        self.setMouseTracking(True)
        self._line = self._scene.addLine(
            0, 0, 0, 0, QPen(QColor(255, 150, 40), 0, Qt.PenStyle.DashLine))
        self._line.setZValue(10)
        self._line.hide()
        self._brush_item = self._scene.addEllipse(
            0, 0, 10, 10, QPen(QColor(255, 170, 40), 0, Qt.PenStyle.DashLine))
        self._brush_item.setZValue(11)
        self._brush_item.hide()
        self._paint_mode = False
        self._painting = False
        self.pressure = 1.0
        self._tablet_down = False

    def set_paint_mode(self, on):
        self._paint_mode = bool(on)
        self._painting = False
        self.setDragMode(QGraphicsView.DragMode.NoDrag if on
                         else QGraphicsView.DragMode.ScrollHandDrag)
        if not on:
            self._brush_item.hide()

    def set_brush_cursor(self, x, y, r):
        if self._paint_mode and r > 0:
            self._brush_item.setRect(x - r, y - r, 2 * r, 2 * r)
            self._brush_item.show()
        else:
            self._brush_item.hide()

    def set_image(self, img, fit=False):
        pm = QPixmap.fromImage(img)
        self._item.setPixmap(pm)
        self._scene.setSceneRect(QRectF(pm.rect()))
        if fit:
            self.fit()

    def set_profile_line(self, show, x0=0.0, y0=0.0, x1=0.0, y1=0.0):
        if show:
            self._line.setLine(x0, y0, x1, y1)
            self._line.show()
        else:
            self._line.hide()

    def fit(self):
        if not self._item.pixmap().isNull():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, e):
        if self._item.pixmap().isNull():
            return
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        m = self.transform().m11() * f
        if 0.03 < m < 80:
            self.scale(f, f)

    def tabletEvent(self, e):
        e.accept()
        sp = self.mapToScene(e.position().toPoint())
        try:
            pr = float(e.pressure())
        except Exception:
            pr = 1.0
        self.pressure = pr if pr > 0.01 else 1.0
        t = e.type()
        if t == QEvent.Type.TabletPress:
            self._tablet_down = True
            self._painting = True
            self.paint_at.emit(float(sp.x()), float(sp.y()))
        elif t == QEvent.Type.TabletMove and self._tablet_down:
            self.paint_at.emit(float(sp.x()), float(sp.y()))
        elif t == QEvent.Type.TabletRelease and self._tablet_down:
            self._tablet_down = False
            self._painting = False
            self.paint_end.emit()

    def mousePressEvent(self, e):
        if self._tablet_down:
            return
        if self._paint_mode and e.button() == Qt.MouseButton.LeftButton:
            self._painting = True
            sp = self.mapToScene(e.position().toPoint())
            self.paint_at.emit(float(sp.x()), float(sp.y()))
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._tablet_down:
            return
        sp = self.mapToScene(e.position().toPoint())
        pm = self._item.pixmap()
        inside = (not pm.isNull() and 0 <= sp.x() < pm.width()
                  and 0 <= sp.y() < pm.height())
        if inside:
            self.pixel.emit(int(sp.x()), int(sp.y()))
        else:
            self._brush_item.hide()
        if self._painting:
            self.paint_at.emit(float(sp.x()), float(sp.y()))
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._tablet_down:
            return
        if self._painting and e.button() == Qt.MouseButton.LeftButton:
            self._painting = False
            self.paint_end.emit()
            return
        super().mouseReleaseEvent(e)

# ----------------------------------------------------------------------------
# 3D preview — software fallback
# ----------------------------------------------------------------------------

class View3D(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 260)
        self.yaw = 35.0; self.elev = 55.0; self.zoom = 1.0
        self._h = None; self._rgb = None; self._p = None; self._lakes = None
        self._geo = None
        self._drag = None

    def set_terrain(self, h, rgb, p, lakes=None):
        self._h = h; self._rgb = rgb; self._p = p; self._lakes = lakes
        self._rebuild()

    def reset_view(self):
        self.yaw = 35.0; self.elev = 55.0; self.zoom = 1.0
        self.update()

    def _rebuild(self):
        if self._h is None or self._rgb is None or self._p is None:
            self._geo = None; self.update(); return
        p = self._p
        n = int(p['d3_density'])
        H, W = self._h.shape
        step = max(1, math.ceil(W / n)); stepy = max(1, math.ceil(H / n))
        hs = np.ascontiguousarray(self._h[::stepy, ::step])
        cs = self._rgb[::stepy, ::step].astype(np.float32)
        if self._lakes is not None:
            lk = self._lakes[::stepy, ::step]
            cs = np.where(lk[:, :, None],
                          np.array([60, 130, 190], np.float32), cs)
        sea = float(p['sea_level']); zs = float(p['d3_zscale'])
        zn = ((np.maximum(hs, sea) - sea) * zs).astype(np.float32)
        ny_, nx_ = zn.shape
        if ny_ < 2 or nx_ < 2:
            self._geo = None; self.update(); return
        cq = (cs[:-1, :-1] + cs[1:, :-1] + cs[:-1, 1:] + cs[1:, 1:]) * 0.25
        gx = (zn[:-1, 1:] + zn[1:, 1:] - zn[:-1, :-1] - zn[1:, :-1]) * (nx_ / 2.0)
        gy = (zn[1:, :-1] + zn[1:, 1:] - zn[:-1, :-1] - zn[:-1, 1:]) * (ny_ / 2.0)
        nl = np.sqrt(gx * gx + gy * gy + 1.0)
        az = math.radians(float(p['sun_az'])); alt = math.radians(float(p['sun_alt']))
        Lx = math.cos(alt) * math.cos(az); Ly = math.cos(alt) * math.sin(az)
        Lz = math.sin(alt)
        lam = np.clip((-gx * Lx - gy * Ly + Lz) / nl, 0.0, 1.0)
        shade = (0.35 + 0.65 * lam).astype(np.float32)
        self._geo = dict(zn=zn, cq=cq, shade=shade, nx=nx_, ny=ny_)
        self.update()

    def paintEvent(self, e):
        pnt = QPainter(self)
        w, h = self.width(), self.height()
        pnt.fillRect(self.rect(), QColor(18, 18, 22))
        if self._geo is None:
            pnt.setPen(QColor(160, 160, 160))
            pnt.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                         'Generate a terrain first')
            pnt.end(); return
        g = self._geo; p = self._p
        nx = g['nx']; ny = g['ny']
        yaw = math.radians(self.yaw); el = math.radians(self.elev)
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        ce, se = math.cos(el), math.sin(el)
        D = 2.4
        f = self.zoom * min(w, h) * 0.9
        Xn = np.linspace(-0.5, 0.5, nx, dtype=np.float32)[None, :]
        Yn = np.linspace(-0.5, 0.5, ny, dtype=np.float32)[:, None]
        zn = g['zn']
        xrn = Xn * cy_ - Yn * sy_
        yrn = Xn * sy_ + Yn * cy_
        upn = yrn * se + zn * ce
        depn = yrn * ce - zn * se
        perspn = (f / (D + depn)).astype(np.float32)
        sxn = (w / 2 + xrn * perspn)
        syn = (h / 2 - upn * perspn)
        qdep = 0.25 * (depn[:-1, :-1] + depn[1:, :-1] + depn[:-1, 1:] + depn[1:, 1:])
        col = g['cq'] * g['shade'][:, :, None]
        fog = float(p['d3_fog'])
        if fog > 0.001:
            dmin = qdep.min(); dr = qdep.max() - dmin + 1e-9
            fgn = (np.clip((qdep - dmin) / dr, 0, 1) * fog)[:, :, None]
            sky = np.array([150.0, 165.0, 185.0], np.float32)[None, None, :]
            col = col * (1 - fgn) + sky * fgn
        col8 = np.clip(col, 0, 255).astype(np.uint8)
        SX = sxn.tolist(); SY = syn.tolist(); CC = col8.tolist()
        QD = qdep.ravel().tolist()
        order = np.argsort(qdep, axis=None)[::-1].tolist()
        nq = nx - 1
        paths = {}
        for k in order:
            i = k // nq; j = k - i * nq
            c = CC[i][j]
            key = ((c[0] >> 3) << 10) | ((c[1] >> 3) << 5) | (c[2] >> 3)
            ent = paths.get(key)
            poly = QPolygonF([QPointF(SX[i][j], SY[i][j]),
                              QPointF(SX[i + 1][j], SY[i + 1][j]),
                              QPointF(SX[i + 1][j + 1], SY[i + 1][j + 1]),
                              QPointF(SX[i][j + 1], SY[i][j + 1])])
            if ent is None:
                path = QPainterPath()
                path.addPolygon(poly)
                paths[key] = [path, QD[k], c]
            else:
                ent[0].addPolygon(poly)
                if QD[k] > ent[1]:
                    ent[1] = QD[k]
        pnt.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        pnt.setPen(Qt.PenStyle.NoPen)
        for key in sorted(paths, key=lambda kk: paths[kk][1], reverse=True):
            path, _, c = paths[key]
            pnt.setBrush(QColor(c[0], c[1], c[2]))
            pnt.drawPath(path)
        sea = float(p.get('sea_level', 0.0))
        if sea > 0.015:
            corners = np.array([[-0.5, 0.0, -0.5], [-0.5, 0.0, 0.5],
                                [0.5, 0.0, 0.5], [0.5, 0.0, -0.5]],
                               np.float32)
            xr = corners[:, 0] * cy_ - corners[:, 2] * sy_
            yr = corners[:, 0] * sy_ + corners[:, 2] * cy_
            up = yr * se
            dep = yr * ce
            pp = f / (D + dep)
            poly = QPolygonF([QPointF(float(w / 2 + xx * pp[i]),
                                      float(h / 2 - up[i] * pp[i]))
                              for i, xx in enumerate(xr)])
            pnt.setBrush(QColor(40, 90, 140, 150))
            pnt.drawPolygon(poly)
        pnt.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = (e.position().x(), e.position().y(), self.yaw, self.elev)

    def mouseMoveEvent(self, e):
        if self._drag:
            x0, y0, yaw0, el0 = self._drag
            self.yaw = (yaw0 + (e.position().x() - x0) * 0.4) % 360.0
            self.elev = min(88.0, max(4.0, el0 - (e.position().y() - y0) * 0.3))
            self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = min(8.0, max(0.2, self.zoom * f))
        self.update()

# ----------------------------------------------------------------------------
# 3D preview — OpenGL (textured, sea plane)
# ----------------------------------------------------------------------------

if HAVE_GL:
    class GLView3D(QOpenGLWidget):
        VS = """
        attribute vec3 aPos;
        attribute vec3 aCol;
        attribute vec3 aNrm;
        attribute vec2 aUV;
        uniform mat4 uMVP;
        uniform vec3 uEye;
        uniform vec3 uLight;
        varying vec3 vCol;
        varying vec2 vUV;
        varying float vDist;
        void main() {
            gl_Position = uMVP * vec4(aPos, 1.0);
            float d = clamp(dot(normalize(aNrm), normalize(uLight)), 0.0, 1.0);
            vCol = aCol * (0.35 + 0.65 * d);
            vUV = aUV;
            vDist = distance(aPos, uEye);
        }"""
        FS = """
        varying vec3 vCol;
        varying vec2 vUV;
        varying float vDist;
        uniform float uFog;
        uniform vec3 uFogCol;
        uniform float uAlpha;
        uniform sampler2D uTex;
        uniform float uUseTex;
        void main() {
            vec3 c = mix(vCol, texture2D(uTex, vUV).rgb, uUseTex);
            float f = clamp(vDist / 4.0, 0.0, 1.0) * uFog;
            gl_FragColor = vec4(mix(c, uFogCol, f), uAlpha);
        }"""

        failed = Signal()

        def __init__(self):
            super().__init__()
            self.setMinimumSize(300, 260)
            self.yaw = 35.0; self.elev = 55.0; self.zoom = 1.0
            self._h = None; self._rgb = None; self._p = None; self._lakes = None
            self._data = None; self._index = None; self._count = 0
            self._zc = 0.2
            self._dirty = False; self._ok = False; self._drag = None
            self._prog = None; self._vbo = None; self._ibo = None
            self._wvbo = None; self._f = None
            self._tex = None; self._tex_img = None
            self._tex_dirty = False; self._use_tex = False

        def set_terrain(self, h, rgb, p, lakes=None):
            self._h = h; self._rgb = rgb; self._p = p; self._lakes = lakes
            self._build_arrays()
            self.update()

        def reset_view(self):
            self.yaw = 35.0; self.elev = 55.0; self.zoom = 1.0
            self.update()

        def _build_arrays(self):
            if self._h is None or self._rgb is None or self._p is None:
                return
            p = self._p
            n = int(p['d3_density'])
            H, W = self._h.shape
            step = max(1, math.ceil(W / n)); stepy = max(1, math.ceil(H / n))
            hs = self._h[::stepy, ::step]
            cs = self._rgb[::stepy, ::step].astype(np.float32)
            if self._lakes is not None:
                lk = self._lakes[::stepy, ::step]
                cs = np.where(lk[:, :, None],
                              np.array([60, 130, 190], np.float32), cs)
            sea = float(p['sea_level']); zs = float(p['d3_zscale'])
            zn = ((np.maximum(hs, sea) - sea) * zs).astype(np.float32)
            ny_, nx_ = zn.shape
            if ny_ < 2 or nx_ < 2:
                self._data = None; return
            xs = np.linspace(-0.5, 0.5, nx_, dtype=np.float32)
            zs_ = np.linspace(-0.5, 0.5, ny_, dtype=np.float32)
            Zg, Xg = np.meshgrid(zs_, xs)
            Ug, Vg = np.meshgrid(np.linspace(0, 1, nx_, dtype=np.float32),
                                 np.linspace(1, 0, ny_, dtype=np.float32))
            UV = np.dstack([Ug, Vg])
            dzn = zn.astype(np.float64)
            dzdy, dzdx = np.gradient(dzn)
            g1 = (-dzdx * (nx_ - 1)).astype(np.float32)
            g2 = (-dzdy * (ny_ - 1)).astype(np.float32)
            ln = np.sqrt(g1 * g1 + g2 * g2 + 1.0).astype(np.float32)
            N = np.dstack([g1 / ln, np.ones_like(ln), g2 / ln]).astype(np.float32)
            V = np.concatenate([np.dstack([Xg, Yg, Zg]).astype(np.float32),
                                cs.astype(np.float32), N, UV], axis=2)
            self._data = np.ascontiguousarray(V.reshape(-1, 11), dtype=np.float32)
            ii, jj = np.meshgrid(np.arange(ny_ - 1), np.arange(nx_ - 1),
                                 indexing='ij')
            v00 = (ii * nx_ + jj).astype(np.uint32).ravel()
            idx = np.concatenate([v00, v00 + nx_, v00 + 1,
                                  v00 + 1, v00 + nx_, v00 + nx_ + 1])
            self._index = np.ascontiguousarray(idx, dtype=np.uint32)
            self._count = int(len(idx))
            self._zc = float(zn.mean())
            self._tex_img = QImage(np.ascontiguousarray(self._rgb).tobytes(),
                                   W, H, 3 * W,
                                   QImage.Format.Format_RGB888).copy()
            self._tex_dirty = True
            self._dirty = True

        def initializeGL(self):
            try:
                self._f = self.context().functions()
                self._f.glEnable(GL_DEPTH_TEST)
                prog = QOpenGLShaderProgram(self)
                prog.bindAttributeLocation('aPos', 0)
                prog.bindAttributeLocation('aCol', 1)
                prog.bindAttributeLocation('aNrm', 2)
                prog.bindAttributeLocation('aUV', 3)
                if not prog.addShaderFromSourceCode(
                        QOpenGLShader.ShaderTypeBit.Vertex, self.VS):
                    raise RuntimeError(prog.log())
                if not prog.addShaderFromSourceCode(
                        QOpenGLShader.ShaderTypeBit.Fragment, self.FS):
                    raise RuntimeError(prog.log())
                if not prog.link():
                    raise RuntimeError(prog.log())
                self._prog = prog
                self._vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
                self._vbo.create()
                self._ibo = QOpenGLBuffer(QOpenGLBuffer.Type.IndexBuffer)
                self._ibo.create()
                s2 = 0.5
                wv = []
                for tri in ((0, 2, 1), (1, 2, 3)):
                    for ci in tri:
                        px = -s2 if ci in (0, 1) else s2
                        pz = -s2 if ci in (0, 3) else s2
                        wv += [px, 0.0, pz, 0.42, 0.62, 0.78, 0.0, 1.0, 0.0,
                               0.0, 0.0]
                self._wdata = np.asarray(wv, np.float32)
                self._wvbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
                self._wvbo.create()
                self._ok = True
            except Exception:
                self._ok = False
                QTimer.singleShot(0, self.failed.emit)

        def resizeGL(self, w, h):
            if self._f:
                dpr = self.devicePixelRatioF()
                self._f.glViewport(0, 0, int(w * dpr), int(h * dpr))

        def _bind_attribs(self):
            f = self._f
            f.glEnableVertexAttribArray(0)
            f.glVertexAttribPointer(0, 3, GL_FLOAT, False, 44, VoidPtr(0))
            f.glEnableVertexAttribArray(1)
            f.glVertexAttribPointer(1, 3, GL_FLOAT, False, 44, VoidPtr(12))
            f.glEnableVertexAttribArray(2)
            f.glVertexAttribPointer(2, 3, GL_FLOAT, False, 44, VoidPtr(24))
            f.glEnableVertexAttribArray(3)
            f.glVertexAttribPointer(3, 2, GL_FLOAT, False, 44, VoidPtr(36))

        def _unbind_attribs(self):
            f = self._f
            for i in (0, 1, 2, 3):
                f.glDisableVertexAttribArray(i)

        def paintGL(self):
            f = self._f
            if f is None:
                return
            f.glClearColor(0.07, 0.07, 0.09, 1.0)
            f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            if not self._ok or self._data is None:
                return
            if self._dirty:
                self._upload()
            p = self._p
            w = max(1, self.width()); h = max(1, self.height())
            yaw = math.radians(self.yaw); el = math.radians(self.elev)
            R = 2.2 / self.zoom
            eye = QVector3D(R * math.cos(el) * math.cos(yaw),
                            self._zc + R * math.sin(el),
                            R * math.cos(el) * math.sin(yaw))
            center = QVector3D(0.0, self._zc, 0.0)
            proj = QMatrix4x4()
            proj.perspective(45.0, w / h, 0.05, 50.0)
            view = QMatrix4x4()
            view.lookAt(eye, center, QVector3D(0.0, 1.0, 0.0))
            mvp = proj * view
            alt = math.radians(float(p['sun_alt']))
            az = math.radians(float(p['sun_az']))
            L = QVector3D(math.cos(alt) * math.cos(az), math.sin(alt),
                          math.cos(alt) * math.sin(az))
            self._prog.bind()
            self._prog.setUniformValue('uMVP', mvp)
            self._prog.setUniformValue('uEye', eye)
            self._prog.setUniformValue('uLight', L)
            self._prog.setUniformValue('uFog', float(p['d3_fog']))
            self._prog.setUniformValue('uFogCol', QVector3D(0.59, 0.65, 0.73))
            self._prog.setUniformValue('uUseTex', 1.0 if self._use_tex else 0.0)
            if self._use_tex:
                self._tex.bind()
            self._vbo.bind()
            self._bind_attribs()
            self._ibo.bind()
            self._prog.setUniformValue('uAlpha', 1.0)
            f.glDrawElements(GL_TRIANGLES, self._count, GL_UNSIGNED_INT,
                             VoidPtr(0))
            self._ibo.release()
            self._vbo.release()
            if self._use_tex:
                self._tex.release()
            if float(p.get('sea_level', 0.0)) > 0.015 and self._wvbo is not None:
                self._prog.setUniformValue('uUseTex', 0.0)
                self._wvbo.bind()
                self._bind_attribs()
                f.glEnable(GL_BLEND)
                f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                f.glDepthMask(False)
                self._prog.setUniformValue('uAlpha', 0.62)
                self._prog.setUniformValue('uFog', 0.0)
                f.glDrawArrays(GL_TRIANGLES, 0, 6)
                f.glDepthMask(True)
                f.glDisable(GL_BLEND)
                self._unbind_attribs()
                self._wvbo.release()
            self._unbind_attribs()
            self._prog.release()

        def _upload(self):
            self._vbo.bind()
            self._vbo.allocate(self._data.tobytes(), self._data.nbytes)
            self._vbo.release()
            self._ibo.bind()
            self._ibo.allocate(self._index.tobytes(), self._index.nbytes)
            self._ibo.release()
            if self._tex_dirty:
                if self._tex is not None:
                    self._tex.destroy()
                self._tex = QOpenGLTexture(self._tex_img)
                self._tex.setMinificationFilter(
                    QOpenGLTexture.Filter.LinearMipMapLinear)
                self._tex.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
                self._tex.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
                self._use_tex = self._tex.isCreated()
                self._tex_dirty = False
            self._wvbo.bind()
            self._wvbo.allocate(self._wdata.tobytes(), self._wdata.nbytes)
            self._wvbo.release()
            self._dirty = False

        def mousePressEvent(self, e):
            if e.button() == Qt.MouseButton.LeftButton:
                self._drag = (e.position().x(), e.position().y(),
                              self.yaw, self.elev)

        def mouseMoveEvent(self, e):
            if self._drag:
                x0, y0, yaw0, el0 = self._drag
                self.yaw = (yaw0 + (e.position().x() - x0) * 0.4) % 360.0
                self.elev = min(88.0, max(4.0, el0 -
                                          (e.position().y() - y0) * 0.3))
                self.update()

        def mouseReleaseEvent(self, e):
            self._drag = None

        def wheelEvent(self, e):
            f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
            self.zoom = min(8.0, max(0.2, self.zoom * f))
            self.update()


def create_3d_view():
    if HAVE_GL:
        try:
            return GLView3D()
        except Exception:
            pass
    return View3D()

# ----------------------------------------------------------------------------
# Export helpers
# ----------------------------------------------------------------------------

def _png_bytes(arr):
    a = np.ascontiguousarray(arr)
    fmt = (QImage.Format.Format_RGBA8888 if a.ndim == 3 and a.shape[2] == 4
           else QImage.Format.Format_RGB888)
    img = QImage(a.tobytes(), a.shape[1], a.shape[0],
                 a.shape[2] * a.shape[1], fmt).copy()
    buf = QBuffer()
    buf.open(QBuffer.OpenModeFlag.ReadWrite)
    img.save(buf, 'PNG')
    data = bytes(buf.data())
    buf.close()
    return data


def _raw16_bytes(a, big_endian):
    dt = '>u2' if big_endian else '<u2'
    return (np.clip(a, 0, 1) * 65535.0 + 0.5).astype(dt).tobytes()


def _f32_bytes(a):
    return np.ascontiguousarray(a, dtype='<f4').tobytes()


def _save_rgb_png(path, arr):
    with open(path, 'wb') as f:
        f.write(_png_bytes(arr))


def _save_gray_png(path, a):
    g = np.repeat((np.clip(a, 0, 1) * 255).astype(np.uint8)[:, :, None], 3, 2)
    _save_rgb_png(path, g)


def _save_raw16(path, a, big_endian):
    with open(path, 'wb') as f:
        f.write(_raw16_bytes(a, big_endian))


def prep_height(h, mode):
    if mode.startswith('Normal'):
        mn, mx = float(h.min()), float(h.max())
        if mx - mn > 1e-9:
            return (h - mn) / (mx - mn)
    return h


AUTO_NORM = {'unity': False, 'unreal': True, 'chunked': True, 'godot': False,
             'glb': False, 'obj': False, 'ply': False, 'stl': False}


def normal_map_rgb(h, strength):
    gy, gx = np.gradient(h)
    s = strength * h.shape[1] / 16.0
    nx, ny = -gx * s, -gy * s
    nl = np.sqrt(nx * nx + ny * ny + 1.0)
    return (np.clip(np.stack([nx / nl, ny / nl, 1.0 / nl], -1) * 0.5 + 0.5,
                    0, 1) * 255).astype(np.uint8)


def make_tile_texture(seed, tint, size=128):
    p1, p2 = 8, 24
    xs = np.arange(size, dtype=np.float64) / size
    X, Y = np.meshgrid(xs * p1, xs * p1)
    n = 0.65 * value2d_tiled(X, Y, seed, p1)
    X2, Y2 = np.meshgrid(xs * p2, xs * p2)
    n += 0.35 * value2d_tiled(X2, Y2, seed + 7, p2)
    n = np.clip(n * 0.5 + 0.5, 0.0, 1.0)
    col = np.array(tint, np.float32)[None, None, :]
    out = col * (0.72 + 0.56 * n)[:, :, None]
    g = np.abs(value2d_tiled(X2 * 1.7, Y2 * 1.7, seed + 13, p2))
    crack = np.clip(1.0 - g * 6.0, 0, 1) * 0.18
    out *= (1.0 - crack)[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def write_exr_r(path, data):
    H, W = data.shape

    def attr(name, typ, payload):
        return name.encode() + b'\0' + typ.encode() + b'\0' + \
               struct.pack('<i', len(payload)) + payload

    chlist = (b'R\0' + struct.pack('<i', 2) + b'\0\0\0\0' +
              struct.pack('<ii', 1, 1) + b'\0')
    hdr = struct.pack('<II', 0x01312F76, 2)
    hdr += attr('channels', 'chlist', chlist)
    hdr += attr('compression', 'compression', struct.pack('<B', 0))
    hdr += attr('dataWindow', 'box2i', struct.pack('<iiii', 0, 0, W - 1, H - 1))
    hdr += attr('displayWindow', 'box2i', struct.pack('<iiii', 0, 0, W - 1, H - 1))
    hdr += attr('lineOrder', 'lineOrder', struct.pack('<B', 0))
    hdr += attr('pixelAspectRatio', 'float', struct.pack('<f', 1.0))
    hdr += attr('screenWindowCenter', 'v2f', struct.pack('<ff', 0, 0))
    hdr += attr('screenWindowWidth', 'float', struct.pack('<f', 1.0))
    hdr += b'\0'
    d = np.ascontiguousarray(data, dtype='<f4')
    with open(path, 'wb') as f:
        f.write(hdr)
        offset_pos = f.tell()
        f.write(b'\0' * 8 * H)
        base = f.tell()
        offs = []
        for y in range(H):
            f.write(struct.pack('<ii', y, W * 4))
            f.write(d[y].tobytes())
            offs.append(f.tell() - base)
        f.seek(offset_pos)
        f.write(struct.pack('<' + 'Q' * H, *offs))


def _mesh_grid(h, rgb, world, height, step, ox=0, oy=0, gw=None, gh=None,
               uv=False):
    H, W = h.shape
    GWd = max((gw if gw is not None else W) - 1, 1)
    GHd = max((gh if gh is not None else H) - 1, 1)
    ys = np.arange(0, H, step); xs = np.arange(0, W, step)
    ny, nx = len(ys), len(xs)
    Z, X = np.meshgrid((ys + oy) / GHd * world, (xs + ox) / GWd * world,
                       indexing='ij')
    hs = h[np.ix_(ys, xs)]
    Y = hs * height
    pos = np.stack([X, Y, Z], -1).reshape(-1, 3)
    col = rgb[np.ix_(ys, xs)].astype(np.float32) / 255.0 if rgb is not None else None
    uvA = None
    if uv:
        U, Vv = np.meshgrid(xs / max(W - 1, 1), 1.0 - ys / max(H - 1, 1),
                            indexing='ij')
        uvA = np.stack([U, Vv], -1).reshape(-1, 2)
    dzdy, dzdx = np.gradient(hs.astype(np.float64))
    sx = world / GWd * step
    sz = world / GHd * step
    n1 = (-dzdx / max(sx, 1e-9)).astype(np.float32)
    n2 = (-dzdy / max(sz, 1e-9)).astype(np.float32)
    ln = np.sqrt(n1 * n1 + n2 * n2 + 1.0)
    nrm = np.dstack([n1 / ln, np.ones_like(ln), n2 / ln]).reshape(-1, 3)
    ii, jj = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing='ij')
    v00 = (ii * nx + jj).ravel()
    v01 = v00 + nx; v10 = v00 + 1; v11 = v00 + nx + 1
    tri = np.stack([v00, v01, v10, v10, v01, v11], 1)
    return pos, col, nrm.astype(np.float32), tri.astype(np.uint32), uvA, nx, ny


def write_glb(path, h, rgb, world, height, step, sea01=0.0, water=False,
              texture=True, vcol=True, albedo_png=None,
              ox=0, oy=0, gw=None, gh=None):
    pos, col, nrm, tri, uvA, _, _ = _mesh_grid(h, rgb, world, height, step,
                                               ox, oy, gw, gh, uv=True)
    pos = pos.astype('<f4')
    nrm = np.ascontiguousarray(nrm, '<f4')
    uvA = np.ascontiguousarray(uvA, '<f4')
    use_tex = bool(texture and albedo_png)
    use_col = bool(vcol and col is not None)
    col = np.ascontiguousarray(col, '<f4') if use_col else None
    idx = tri.astype('<u4')

    sections = [pos.tobytes(), nrm.tobytes(), uvA.tobytes()]
    kinds = ['pos', 'nrm', 'uv']
    if use_col:
        sections.append(col.tobytes()); kinds.append('col')
    sections.append(idx.tobytes()); kinds.append('tri')
    if water:
        wy = np.float32(sea01 * height)
        wpos = np.array([[0, wy, 0], [0, wy, world], [world, wy, world],
                         [world, wy, 0]], '<f4')
        wnrm = np.tile(np.array([0, 1, 0], '<f4'), (4, 1))
        wuv = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], '<f4')
        widx = np.array([0, 1, 2, 0, 2, 3], '<u4')
        sections += [wpos.tobytes(), wnrm.tobytes(), wuv.tobytes(),
                     widx.tobytes()]
        kinds += ['wpos', 'wnrm', 'wuv', 'wtri']
    if use_tex:
        sections.append(albedo_png); kinds.append('png')

    def pad(b):
        return b + b'\0' * (-len(b) % 4)

    padded = [pad(s) for s in sections]
    offsets = {}
    cur = 0
    for k, s in zip(kinds, padded):
        offsets[k] = cur
        cur += len(s)
    total = cur
    secmap = dict(zip(kinds, sections))

    def view(k, target=None):
        v = {"buffer": 0, "byteOffset": offsets[k],
             "byteLength": len(secmap[k])}
        if target:
            v["target"] = target
        return v

    pos_max = [float(pos[:, i].max()) for i in range(3)]
    pos_min = [float(pos[:, i].min()) for i in range(3)]
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": len(pos),
         "type": "VEC3", "min": pos_min, "max": pos_max},
        {"bufferView": 1, "componentType": 5126, "count": len(nrm), "type": "VEC3"},
        {"bufferView": 2, "componentType": 5126, "count": len(uvA), "type": "VEC2"},
    ]
    bufferViews = [view('pos', 34962), view('nrm', 34962), view('uv', 34962)]
    tri_acc = 3
    if use_col:
        accessors.append({"bufferView": 3, "componentType": 5126,
                          "count": len(col), "type": "VEC3"})
        bufferViews.append(view('col', 34962))
        tri_acc = 4
    accessors.append({"bufferView": tri_acc, "componentType": 5125,
                      "count": len(idx), "type": "SCALAR"})
    bufferViews.append(view('tri', 34963))
    water_acc = None
    if water:
        base_a = len(accessors)
        water_acc = dict(wpos=base_a, wnrm=base_a + 1,
                         wuv=base_a + 2, wtri=base_a + 3)
        wy = float(sea01 * height)
        accessors += [
            {"bufferView": len(bufferViews), "componentType": 5126, "count": 4,
             "type": "VEC3", "min": [0.0, wy, 0.0],
             "max": [float(world), wy, float(world)]},
            {"bufferView": len(bufferViews) + 1, "componentType": 5126,
             "count": 4, "type": "VEC3"},
            {"bufferView": len(bufferViews) + 2, "componentType": 5126,
             "count": 4, "type": "VEC2"},
            {"bufferView": len(bufferViews) + 3, "componentType": 5125,
             "count": 6, "type": "SCALAR"},
        ]
        for k in ('wpos', 'wnrm', 'wuv'):
            bufferViews.append(view(k, 34962))
        bufferViews.append(view('wtri', 34963))

    mat_terrain = {"name": "Terrain", "doubleSided": True,
                   "pbrMetallicRoughness": {"metallicFactor": 0.0,
                                            "roughnessFactor": 1.0,
                                            "baseColorFactor": [1, 1, 1, 1]}}
    imgs = textures = samplers = None
    if use_tex:
        bufferViews.append(view('png'))
        imgs = [{"bufferView": len(bufferViews) - 1, "mimeType": "image/png"}]
        samplers = [{"magFilter": 9729, "minFilter": 9987,
                     "wrapS": 10497, "wrapT": 10497}]
        textures = [{"sampler": 0, "source": 0}]
        mat_terrain["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}

    attrs = {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}
    if use_col:
        attrs["COLOR_0"] = 3
    prims = [{"attributes": attrs, "indices": tri_acc, "material": 0}]
    mats = [mat_terrain]
    if water:
        prims.append({"attributes": {"POSITION": water_acc['wpos'],
                                     "NORMAL": water_acc['wnrm'],
                                     "TEXCOORD_0": water_acc['wuv']},
                      "indices": water_acc['wtri'], "material": 1})
        mats.append({"name": "Water", "doubleSided": True,
                     "alphaMode": "BLEND",
                     "pbrMetallicRoughness": {
                         "baseColorFactor": [0.10, 0.33, 0.46, 0.72],
                         "metallicFactor": 0.1, "roughnessFactor": 0.12}})

    js = {
        "asset": {"version": "2.0", "generator": "Terrain Forge v12"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Terrain"}],
        "meshes": [{"primitives": prims, "name": "TerrainMesh"}],
        "materials": mats,
        "buffers": [{"byteLength": total}],
        "bufferViews": bufferViews,
        "accessors": accessors,
    }
    if imgs:
        js["images"] = imgs; js["samplers"] = samplers; js["textures"] = textures
    jbin = json.dumps(js, separators=(',', ':')).encode()
    jbin += b' ' * (-len(jbin) % 4)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(jbin) + 8 + total))
        f.write(struct.pack('<I', len(jbin))); f.write(b'JSON'); f.write(jbin)
        f.write(struct.pack('<I', total)); f.write(b'BIN\x00')
        for s in padded:
            f.write(s)


def write_ply(path, h, rgb, world, height, step):
    pos, col, nrm, tri, _, _, _ = _mesh_grid(h, rgb, world, height, step)
    n = len(pos)
    P = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                           ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
                           ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    P['x'], P['y'], P['z'] = pos[:, 0], pos[:, 1], pos[:, 2]
    P['nx'], P['ny'], P['nz'] = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    c8 = (np.clip(col, 0, 1) * 255).astype(np.uint8)
    P['r'], P['g'], P['b'] = c8[:, 0], c8[:, 1], c8[:, 2]
    F = np.empty(len(tri), dtype=[('n', 'u1'), ('a', '<u4'), ('b', '<u4'), ('c', '<u4')])
    F['n'] = 3
    F['a'], F['b'], F['c'] = tri[:, 0], tri[:, 1], tri[:, 2]
    hdr = ("ply\nformat binary_little_endian 1.0\ncomment Terrain Forge v12\n"
           f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
           "property float nx\nproperty float ny\nproperty float nz\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           f"element face {len(tri)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, 'wb') as f:
        f.write(hdr.encode()); P.tofile(f); F.tofile(f)


def write_stl(path, h, world, height, step):
    """Watertight solid: top surface + skirt walls + flat base."""
    H, W = h.shape
    ys = np.arange(0, H, step); xs = np.arange(0, W, step)
    ny, nx = len(ys), len(xs)
    Xg, Yg = np.meshgrid(xs / max(W - 1, 1) * world,
                         ys / max(H - 1, 1) * world, indexing='ij')
    Zg = h[np.ix_(ys, xs)] * height
    tris = []
    ap = tris.append
    for i in range(ny - 1):
        for j in range(nx - 1):
            a = (Xg[i, j], Yg[i, j], Zg[i, j])
            b = (Xg[i, j + 1], Yg[i, j + 1], Zg[i, j + 1])
            c = (Xg[i + 1, j], Yg[i + 1, j], Zg[i + 1, j])
            d = (Xg[i + 1, j + 1], Yg[i + 1, j + 1], Zg[i + 1, j + 1])
            ap((a, b, c)); ap((b, d, c))
            a0 = (a[0], a[1], 0.0); b0 = (b[0], b[1], 0.0)
            c0 = (c[0], c[1], 0.0); d0 = (d[0], d[1], 0.0)
            ap((a0, c0, b0)); ap((b0, c0, d0))
    for i in range(ny - 1):
        t0 = (Xg[i, 0], Yg[i, 0], Zg[i, 0])
        t1 = (Xg[i + 1, 0], Yg[i + 1, 0], Zg[i + 1, 0])
        b0 = (t0[0], t0[1], 0.0); b1 = (t1[0], t1[1], 0.0)
        ap((t0, t1, b0)); ap((t1, b1, b0))
    for i in range(ny - 1):
        t0 = (Xg[i, nx - 1], Yg[i, nx - 1], Zg[i, nx - 1])
        t1 = (Xg[i + 1, nx - 1], Yg[i + 1, nx - 1], Zg[i + 1, nx - 1])
        b0 = (t0[0], t0[1], 0.0); b1 = (t1[0], t1[1], 0.0)
        ap((t0, b0, t1)); ap((t1, b0, b1))
    for j in range(nx - 1):
        t0 = (Xg[0, j], Yg[0, j], Zg[0, j])
        t1 = (Xg[0, j + 1], Yg[0, j + 1], Zg[0, j + 1])
        b0 = (t0[0], t0[1], 0.0); b1 = (t1[0], t1[1], 0.0)
        ap((t0, b0, t1)); ap((t1, b0, b1))
    for j in range(nx - 1):
        t0 = (Xg[ny - 1, j], Yg[ny - 1, j], Zg[ny - 1, j])
        t1 = (Xg[ny - 1, j + 1], Yg[ny - 1, j + 1], Zg[ny - 1, j + 1])
        b0 = (t0[0], t0[1], 0.0); b1 = (t1[0], t1[1], 0.0)
        ap((t0, t1, b0)); ap((t1, b1, b0))
    tv = np.asarray(tris, dtype=np.float32)
    nrm = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(ln, 1e-12)
    rec = np.zeros(len(tv), dtype=[('n', '<f4', (3,)),
                                   ('v', '<f4', (3, 3)), ('attr', '<u2')])
    rec['n'] = nrm; rec['v'] = tv
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(tv)))
        rec.tofile(f)


def write_obj_scaled(path, h, rgb, world, height, step, textured=False):
    H, W = h.shape
    base = path[:-4] if path.lower().endswith('.obj') else path
    mtl_path = None
    if textured:
        png_path = base + '_albedo.png'
        _save_rgb_png(png_path, rgb)
        mtl_path = base + '.mtl'
        with open(mtl_path, 'w') as f:
            f.write("newmtl terrain\nKa 1 1 1\nKd 1 1 1\n"
                    f"map_Kd {os.path.basename(png_path)}\n")
    pos, col, nrm, tri, uvA, nx, ny = _mesh_grid(h, rgb, world, height, step,
                                                 uv=textured)
    lines = [f"# Terrain Forge v12 OBJ — Y-up, meters, {nx}x{ny} grid"]
    if textured:
        lines.append(f"mtllib {os.path.basename(mtl_path)}")
        lines.append("usemtl terrain")
    ap = lines.append
    for v in pos:
        ap(f"v {v[0]:.3f} {v[1]:.4f} {v[2]:.3f}")
    for v in nrm:
        ap(f"vn {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
    if textured:
        for t in uvA:
            ap(f"vt {t[0]:.4f} {t[1]:.4f}")
    idx = tri + 1
    if textured:
        for a, b, c in idx:
            ap(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
    else:
        for a, b, c in idx:
            ap(f"f {a}//{a} {b}//{b} {c}//{c}")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return mtl_path

# ----------------------------------------------------------------------------
# Unity .unitypackage writer
# ----------------------------------------------------------------------------

def _uguid(rel):
    return hashlib.md5(('terrain-forge-v12:' + rel).encode()).hexdigest()[:32]

_META_FOLDER = """fileFormatVersion: 2
guid: {g}
folderAsset: yes
DefaultImporter:
  externalObjects: {{}}
  userData: 
  assetBundleName: 
  assetBundleVariant: 
"""

_META_DEFAULT = """fileFormatVersion: 2
guid: {g}
DefaultImporter:
  externalObjects: {{}}
  userData: 
  assetBundleName: 
  assetBundleVariant: 
"""

_META_CS = """fileFormatVersion: 2
guid: {g}
MonoImporter:
  externalObjects: {{}}
  serializedVersion: 2
  defaultReferences: []
  executionOrder: 0
  icon: {{instanceID: 0}}
"""

_META_TEX = """fileFormatVersion: 2
guid: {g}
TextureImporter:
  internalIDToNameTable: []
  externalObjects: {{}}
  serializedVersion: 12
  mipmaps:
    mipMapMode: 0
    enableMipMap: 1
  textureSettings:
    serializedVersion: 2
    filterMode: 1
    aniso: 1
    wrapU: 1
    wrapV: 1
  nPOTScale: 0
  textureType: 0
  textureShape: 1
"""


def write_unitypackage(path, files):
    with tarfile.open(path, 'w:gz') as tar:
        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data); info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
        dirs = {'Assets/TerrainForge'}
        for rel in files:
            parts = rel.split('/')[:-1]
            for i in range(1, len(parts) + 1):
                dirs.add('/'.join(parts[:i]))
        for d in sorted(dirs):
            g = _uguid(d)
            add(f'{g}/pathname', (d + '\n').encode())
            add(f'{g}/asset.meta', _META_FOLDER.format(g=g).encode())
        for rel, data in files.items():
            g = _uguid(rel)
            add(f'{g}/pathname', (rel + '\n').encode())
            add(f'{g}/asset', data)
            ext = rel.rsplit('.', 1)[-1].lower()
            if ext == 'cs':
                meta = _META_CS.format(g=g)
            elif ext in ('png', 'jpg', 'tga'):
                meta = _META_TEX.format(g=g)
            else:
                meta = _META_DEFAULT.format(g=g)
            add(f'{g}/asset.meta', meta.encode())

# ----------------------------------------------------------------------------
# Generated import scripts
# ----------------------------------------------------------------------------

UNITY_CS = """// Terrain Forge v12 -> Unity
// 1. Double-click TerrainForge.unitypackage (or Assets > Import Package).
// 2. Menu: Tools > Terrain Forge > Import Terrain.
#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;

public static class TerrainForgeImport
{
    const int RES = %%RES%%;
    const bool BIG_ENDIAN = %%BE%%;
    const bool NORMALIZE = %%NORM%%;
    const float WORLD = %%WORLD%%f;
    const float HEIGHT = %%HEIGHT%%f;
    const float SEA = %%SEA%%f;
    const string F = "TerrainForge";

    [MenuItem("Tools/Terrain Forge/Import Terrain")]
    public static void Import()
    {
        string dir = Path.Combine(Application.dataPath, F);
        string raw = Path.Combine(dir, "heightmap.raw");
        if (!File.Exists(raw)) { Debug.LogError("[TF] heightmap.raw missing in Assets/" + F); return; }

        byte[] b = File.ReadAllBytes(raw);
        var h = new float[RES, RES];
        float mn = float.MaxValue, mx = float.MinValue;
        for (int i = 0; i < RES * RES; i++)
        {
            float v = (BIG_ENDIAN ? ((b[2*i] << 8) | b[2*i+1]) : (b[2*i] | (b[2*i+1] << 8))) / 65535f;
            h[i / RES, i % RES] = v;
            if (v < mn) mn = v; if (v > mx) mx = v;
        }
        if (NORMALIZE)
        {
            float r = Mathf.Max(mx - mn, 1e-6f);
            for (int z = 0; z < RES; z++) for (int x = 0; x < RES; x++) h[z, x] = (h[z, x] - mn) / r;
        }

        var td = new TerrainData { heightmapResolution = RES, size = new Vector3(WORLD, HEIGHT, WORLD) };

        string[] names = { "%%L0%%", "%%L1%%", "%%L2%%", "%%L3%%" };
        string[] texs  = { "layer_0_%%L0%%.png", "layer_1_%%L1%%.png",
                           "layer_2_%%L2%%.png", "layer_3_%%L3%%.png" };
        var layers = new TerrainLayer[4];
        for (int i = 0; i < 4; i++)
        {
            var tl = new TerrainLayer { name = names[i], tileSize = new Vector2(WORLD / 24f, WORLD / 24f) };
            var t = LoadTex(dir, texs[i]);
            if (t != null) tl.diffuseTexture = t;
            layers[i] = tl;
        }
        td.terrainLayers = layers;

        string sp = Path.Combine(dir, "splatmap.png");
        if (File.Exists(sp))
        {
            var st = LoadTex(sp);
            if (st != null)
            {
                int sw = st.width;
                td.alphamapResolution = sw;
                var px = st.GetPixels32();
                var am = new float[sw, sw, 4];
                for (int z = 0; z < sw; z++)
                    for (int x = 0; x < sw; x++)
                    {
                        var c = px[z * sw + x];
                        am[z,x,0]=c.r/255f; am[z,x,1]=c.g/255f;
                        am[z,x,2]=c.b/255f; am[z,x,3]=c.a/255f;
                    }
                td.SetAlphamaps(0, 0, am);
            }
        }
        td.SetHeights(0, 0, h);
        AssetDatabase.CreateAsset(td, "Assets/" + F + "/TerrainData.asset");
        var go = Terrain.CreateTerrainGameObject(td);
        go.name = "TerrainForge";
        CreateWater();
        Selection.activeGameObject = go;
        Debug.Log("[TF] Terrain " + RES + "x" + RES + " @ " + WORLD + "x" + WORLD + " m created.");
    }

    static void CreateWater()
    {
        var w = GameObject.CreatePrimitive(PrimitiveType.Plane);
        w.name = "TF_Water";
        w.transform.position = new Vector3(WORLD / 2f, SEA * HEIGHT, WORLD / 2f);
        w.transform.localScale = new Vector3(WORLD / 10f, 1f, WORLD / 10f);
        var m = new Material(Shader.Find("Standard"));
        if (m.shader == null) m = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        if (m.HasProperty("_Mode"))
        {
            m.SetFloat("_Mode", 2f);
            m.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            m.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            m.SetInt("_ZWrite", 0);
            m.DisableKeyword("_ALPHATEST_ON");
            m.EnableKeyword("_ALPHABLEND_ON");
            m.renderQueue = 3000;
        }
        Color wc = new Color(0.12f, 0.35f, 0.50f, 0.70f);
        if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", wc);
        if (m.HasProperty("_Color")) m.SetColor("_Color", wc);
        w.GetComponent<MeshRenderer>().sharedMaterial = m;
    }

    static Texture2D LoadTex(string dir, string name)
    {
        string p = Path.Combine(dir, name);
        if (!File.Exists(p)) return null;
        var t = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
        t.LoadImage(File.ReadAllBytes(p));
        t.wrapMode = TextureWrapMode.Repeat;
        return t;
    }
}
#endif
"""


UE_PY = """# Terrain Forge v12 -> Unreal Engine 5 (editor Python)
import os
import unreal

SRC = r"%%FOLDER%%"
DEST = "/Game/TerrainForge"
RES = %%RES%%
WORLD = %%WORLD%%
HEIGHT = %%HEIGHT%%

MASKS = [("%%L0%%", "mask_0_%%L0%%.png"), ("%%L1%%", "mask_1_%%L1%%.png"),
         ("%%L2%%", "mask_2_%%L2%%.png"), ("%%L3%%", "mask_3_%%L3%%.png")]


def log(msg):
    unreal.log_warning("[TerrainForge] " + str(msg))


def import_texture(full_path, name):
    at = unreal.AssetToolsHelpers.get_asset_tools()
    task = unreal.AssetImportTask()
    task.filename = full_path
    task.destination_path = DEST
    task.destination_name = name
    task.automated = True
    task.save = False
    at.import_asset_tasks([task])
    return unreal.load_asset(DEST + "/" + name)


def ensure_layer_infos():
    eal = unreal.EditorAssetLibrary
    for lname, _ in MASKS:
        p = DEST + "/LI_" + lname
        if eal.does_asset_exist(p):
            continue
        try:
            unreal.AssetToolsHelpers.get_asset_tools().create_asset(
                "LI_" + lname, DEST, unreal.LandscapeLayerInfoObject, None)
            log("Created layer info LI_" + lname)
        except Exception as e:
            log("Layer info " + lname + ": " + str(e))


def connect_layer(mel, blend, layer_name, expr):
    for pin in ("Layer " + layer_name, layer_name, "Layer" + layer_name):
        try:
            if mel.connect_material_expressions(expr, "RGB", blend, pin):
                return True
        except Exception:
            pass
    log("Could not connect layer pin: " + layer_name)
    return False


def build_material():
    mel = unreal.MaterialEditingLibrary
    mat = mel.create_material_asset("M_TerrainForge", DEST, unreal.Material)
    if mat is None:
        log("Material creation failed")
        return None
    coords = mel.create_material_expression(
        mat, unreal.MaterialExpressionLandscapeLayerCoords, -700, 0)
    blend = mel.create_material_expression(
        mat, unreal.MaterialExpressionLandscapeLayerBlend, 0, 0)
    layers = []
    for lname, _ in MASKS:
        layers.append(unreal.LandscapeLayerBlendLayer(
            layer_name=lname,
            blend_type=unreal.LandscapeLayerBlendMode.LB_WEIGHT_MAP_BLEND,
            preview_weight=1.0))
    blend.set_editor_property("layers", layers)
    for i, (lname, fname) in enumerate(MASKS):
        p = os.path.join(SRC, fname)
        if not os.path.exists(p):
            log("Missing mask: " + p)
            continue
        ts = mel.create_material_expression(
            mat, unreal.MaterialExpressionTextureSample, -380, i * 200 - 300)
        tex = import_texture(p, "T_Mask_" + lname)
        if tex:
            ts.set_editor_property("texture", tex)
        mel.connect_material_expressions(coords, "", ts, "UVs")
        connect_layer(mel, blend, lname, ts)
    mel.recompile_material(mat)
    mel.save_material_asset(mat)
    log("Material M_TerrainForge created with layers: " +
        ", ".join(n for n, _ in MASKS))
    return mat


xy_pct = WORLD / (RES - 1) * 100.0
z_pct = HEIGHT / 5.12
log("Landscape import settings for this export:")
log("  XY scale = %.1f %%   Z scale = %.1f %%   (target %gx%g m, height %g m)"
    % (xy_pct, z_pct, WORLD, WORLD, HEIGHT))
ensure_layer_infos()
build_material()
log("Done. Landscape > Import from File > heightmap.raw / heightmap.png, "
    "then assign M_TerrainForge.")
"""


GODOT_GD = """# Terrain Forge v12 -> Godot 4
extends Node3D

const F32_PATH := "res://terrain.f32"
const ALBEDO_PATH := "res://albedo.png"
const WIDTH := %%W%%
const DEPTH := %%H%%
const WORLD_SIZE := %%WORLD%%
const HEIGHT_METERS := %%HEIGHT%%


func _ready() -> void:
    var f := FileAccess.open(F32_PATH, FileAccess.READ)
    if f == null:
        push_error("terrain.f32 not found in res://")
        return
    var data := f.get_buffer(WIDTH * DEPTH * 4).to_float32_array()
    _build_mesh(data)
    var col := get_node_or_null("TerrainCollision")
    if col:
        col.scale = Vector3(WORLD_SIZE / float(WIDTH - 1), HEIGHT_METERS,
                            WORLD_SIZE / float(DEPTH - 1))
    else:
        _build_collision(data)


func _build_mesh(d: PackedFloat32Array) -> void:
    var img: Image = null
    if FileAccess.file_exists(ProjectSettings.globalize_path(ALBEDO_PATH)):
        img = Image.load_from_file(ProjectSettings.globalize_path(ALBEDO_PATH))
    var st := SurfaceTool.new()
    st.begin(Mesh.PRIMITIVE_TRIANGLES)
    var sx := WORLD_SIZE / float(WIDTH - 1)
    var sy := WORLD_SIZE / float(DEPTH - 1)
    for y in DEPTH - 1:
        for x in WIDTH - 1:
            _vtx(st, d, img, x, y, sx, sy)
            _vtx(st, d, img, x + 1, y, sx, sy)
            _vtx(st, d, img, x, y + 1, sx, sy)
            _vtx(st, d, img, x + 1, y, sx, sy)
            _vtx(st, d, img, x + 1, y + 1, sx, sy)
            _vtx(st, d, img, x, y + 1, sx, sy)
    st.generate_tangents()
    var mi := MeshInstance3D.new()
    mi.mesh = st.commit()
    mi.name = "TerrainMesh"
    var mat := StandardMaterial3D.new()
    mat.vertex_color_use_as_albedo = true
    mat.roughness = 1.0
    mi.material_override = mat
    add_child(mi)


func _build_collision(d: PackedFloat32Array) -> void:
    var body := StaticBody3D.new()
    body.name = "TerrainCollision"
    var cs := CollisionShape3D.new()
    var shape := HeightMapShape3D.new()
    shape.map_width = WIDTH
    shape.map_depth = DEPTH
    shape.map_data = d
    cs.shape = shape
    body.add_child(cs)
    body.scale = Vector3(WORLD_SIZE / float(WIDTH - 1), HEIGHT_METERS,
                         WORLD_SIZE / float(DEPTH - 1))
    add_child(body)


func _vtx(st: SurfaceTool, d: PackedFloat32Array, img: Image,
        x: int, y: int, sx: float, sy: float) -> void:
    var hgt: float = d[y * WIDTH + x]
    st.set_normal(_normal(d, x, y, sx))
    st.set_uv(Vector2(float(x) / float(WIDTH - 1), float(y) / float(DEPTH - 1)))
    if img:
        var px := mini(int(float(x) / (WIDTH - 1) * (img.get_width() - 1)),
                       img.get_width() - 1)
        var py := mini(int(float(y) / (DEPTH - 1) * (img.get_height() - 1)),
                       img.get_height() - 1)
        st.set_color(img.get_pixel(px, py))
    else:
        st.set_color(Color(0.55, 0.55, 0.55))
    st.add_vertex(Vector3(x * sx, hgt * HEIGHT_METERS, y * sy))


func _normal(d: PackedFloat32Array, x: int, y: int, sx: float) -> Vector3:
    var hl: float = d[y * WIDTH + maxi(x - 1, 0)]
    var hr: float = d[y * WIDTH + mini(x + 1, WIDTH - 1)]
    var hu: float = d[maxi(y - 1, 0) * WIDTH + x]
    var hd: float = d[mini(y + 1, DEPTH - 1) * WIDTH + x]
    return Vector3((hl - hr) * HEIGHT_METERS, 2.0 * sx,
                   (hu - hd) * HEIGHT_METERS).normalized()
"""


BLENDER_PY = """# Terrain Forge -> Blender (self-contained)
import bpy, zlib, base64
import numpy as np

WIDTH  = %%NX%%
DEPTH  = %%NY%%
WORLD  = %%WORLD%%
HEIGHT = %%HEIGHT%%
CW, CH = %%CXW%%, %%CYX%%

_h = np.frombuffer(zlib.decompress(base64.b64decode("%%HB64%%")), dtype='<u2')
_h = _h.reshape(DEPTH, WIDTH).astype(np.float32) / 65535.0
_c = np.frombuffer(zlib.decompress(base64.b64decode("%%CB64%%")), dtype=np.uint8)
_c = _c.reshape(CH, CW, 3).astype(np.float32) / 255.0

verts, faces = [], []
sx = WORLD / max(WIDTH - 1, 1)
sy = WORLD / max(DEPTH - 1, 1)
for y in range(DEPTH):
    for x in range(WIDTH):
        verts.append((x * sx, y * sy, float(_h[y, x]) * HEIGHT))
for y in range(DEPTH - 1):
    for x in range(WIDTH - 1):
        i = y * WIDTH + x
        faces.append((i, i + WIDTH, i + 1))
        faces.append((i + 1, i + WIDTH, i + WIDTH + 1))

mesh = bpy.data.meshes.new("TerrainForge")
mesh.from_pydata(verts, [], faces)
mesh.validate()
mesh.update()
for p in mesh.polygons:
    p.use_smooth = True

yy = np.clip((np.arange(DEPTH) / max(DEPTH - 1, 1) * (CH - 1)).astype(int), 0, CH - 1)
xx = np.clip((np.arange(WIDTH) / max(WIDTH - 1, 1) * (CW - 1)).astype(int), 0, CW - 1)
colors = _c[yy][:, xx]
ca = mesh.color_attributes.new(name="Col", type='FLOAT_COLOR', domain='POINT')
rgba = np.ones((DEPTH, WIDTH, 4), np.float32)
rgba[:, :, :3] = colors
try:
    ca.data.foreach_set("color", rgba.ravel())
except Exception:
    for i in range(WIDTH * DEPTH):
        ca.data[i].color = tuple(rgba.flat[i * 4:i * 4 + 4])

obj = bpy.data.objects.new("TerrainForge", mesh)
bpy.context.collection.objects.link(obj)
mat = bpy.data.materials.new("TerrainForge_Mat")
mat.use_nodes = True
bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
attr = mat.node_tree.nodes.new("ShaderNodeAttribute")
attr.attribute_name = "Col"
mat.node_tree.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
obj.data.materials.append(mat)
print("Terrain Forge mesh created:", len(verts), "verts,", len(faces), "faces")
"""

# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------

def _meta_json(p, res, world, hmet, splat_names=None, height_mode=None,
               extra=None):
    d = {
        "generator": "Terrain Forge", "version": "12.0",
        "created": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "seed": int(p['seed']),
        "resolution": res,
        "world_size_m": world,
        "max_height_m": hmet,
        "sea_level_01": float(p['sea_level']),
        "units": "meters", "up_axis": "Y",
        "height_mode": height_mode or "sea-anchored",
        "splat_layers": splat_names or ["grass", "rock", "sand", "snow"],
        "splat_painted": bool(p.get('_splat_painted', False)),
        "blend_enabled": bool(p.get('blend_enable', False)),
        "erosion": p.get('erosion_method'),
        "palette": p.get('palette'),
        "params": {k: v for k, v in p.items()
                   if not isinstance(v, np.ndarray)},
    }
    if extra:
        d.update(extra)
    return json.dumps(d, indent=2)

# ----------------------------------------------------------------------------
# Engine export dialog
# ----------------------------------------------------------------------------

class EngineExportDialog(QDialog):
    TARGETS = [
        ('Unity (.unitypackage)', 'unity'),
        ('Unreal Engine 5 (raw + material script)', 'unreal'),
        ('Godot 4 (tscn + glb + f32/EXR)', 'godot'),
        ('Blender (self-contained script)', 'blender'),
        ('Chunked tiles (streaming + LODs)', 'chunked'),
        ('glTF 2.0 binary (universal)', 'glb'),
        ('Wavefront OBJ (scaled, Y-up)', 'obj'),
        ('Binary PLY (vertex colors + normals)', 'ply'),
        ('Binary STL (watertight, 3D print)', 'stl'),
    ]

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self._albedo = None
        self.setWindowTitle('Export for Engine')
        self.setMinimumWidth(680)
        self._rows = {}

        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.listw = QListWidget()
        for name, _ in self.TARGETS:
            self.listw.addItem(name)
        self.listw.setFixedWidth(250)
        self.desc = QLabel()
        self.desc.setWordWrap(True)
        self.desc.setMinimumHeight(120)
        self.desc.setStyleSheet('color: #99a;')
        left.addWidget(self.listw, 1)
        left.addWidget(self.desc)
        lay.addLayout(left, 1)

        formw = QWidget()
        form = QFormLayout(formw)
        form.setSpacing(8)

        self.res_combo = QComboBox()
        self._rows['res'] = self._row('Heightmap resolution', self.res_combo)
        form.addRow(self._rows['res'])

        self.endian_combo = QComboBox()
        self.endian_combo.addItems(['Little (Windows)', 'Big (Mac)'])
        self._rows['endian'] = self._row('Raw byte order', self.endian_combo)
        form.addRow(self._rows['endian'])

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Auto per engine', 'Sea-anchored (absolute)',
                                  'Normalized (min→max)'])
        self._rows['mode'] = self._row('Height encoding', self.mode_combo)
        form.addRow(self._rows['mode'])

        self.world_spin = QDoubleSpinBox(); self.world_spin.setRange(10, 20000)
        self.world_spin.setValue(1000); self.world_spin.setSuffix(' m')
        form.addRow(self._row('World size (X/Z)', self.world_spin))

        self.height_spin = QDoubleSpinBox(); self.height_spin.setRange(1, 5000)
        self.height_spin.setValue(120); self.height_spin.setSuffix(' m')
        form.addRow(self._row('Max height', self.height_spin))

        self.step_spin = QSpinBox(); self.step_spin.setRange(1, 8); self.step_spin.setValue(2)
        self._rows['step'] = self._row('Mesh decimation step', self.step_spin)
        form.addRow(self._rows['step'])

        self.grid_combo = QComboBox()
        self.grid_combo.addItems(['2×2', '3×3', '4×4', '6×6', '8×8'])
        self._rows['grid'] = self._row('Chunk grid', self.grid_combo)
        form.addRow(self._rows['grid'])

        self.overlap_spin = QSpinBox(); self.overlap_spin.setRange(0, 8)
        self.overlap_spin.setValue(1)
        self._rows['overlap'] = self._row('Chunk overlap (cells)', self.overlap_spin)
        form.addRow(self._rows['overlap'])

        self.chk_lods = QCheckBox('LOD meshes (½, ¼, ⅛ grid) per chunk')
        self.chk_lods.setChecked(True)
        self._rows['lods'] = self._row('Chunk LODs', self.chk_lods)
        form.addRow(self._rows['lods'])

        self.chk_zip = QCheckBox('Also create a .zip of the bundle')
        self._rows['zip'] = self._row('ZIP archive', self.chk_zip)
        form.addRow(self._rows['zip'])

        self.chk_water = QCheckBox('Include water plane (alpha blend)')
        self.chk_water.setChecked(True)
        self._rows['water'] = self._row('Water plane', self.chk_water)
        form.addRow(self._rows['water'])

        self.chk_embed = QCheckBox('Embed albedo texture (OBJ: +.mtl +.png)')
        self.chk_embed.setChecked(True)
        self._rows['embed'] = self._row('Texture', self.chk_embed)
        form.addRow(self._rows['embed'])

        self.chk_vcol = QCheckBox('Keep vertex colors as well')
        self.chk_vcol.setChecked(True)
        self._rows['vcol'] = self._row('Vertex colors', self.chk_vcol)
        form.addRow(self._rows['vcol'])

        self.chk_splat = QCheckBox('Write splatmap / layer masks')
        self.chk_splat.setChecked(True)
        self._rows['splat'] = self._row('Splatmap / layer masks', self.chk_splat)
        form.addRow(self._rows['splat'])

        self.chk_masks = QCheckBox('wetness / deposition / curvature / '
                                   'lakes / walkable')
        self._rows['masks'] = self._row('Data masks (PNG)', self.chk_masks)
        form.addRow(self._rows['masks'])

        p0 = main._collect()
        theme = SPLAT_THEMES.get(p0.get('palette', 'Terrain'),
                                 SPLAT_THEMES['Terrain'])
        names_w = QWidget(); nl = QHBoxLayout(names_w)
        nl.setContentsMargins(0, 0, 0, 0); nl.setSpacing(4)
        self.layer_edits = []
        for i in range(4):
            e = QLineEdit(theme[i][0]); e.setFixedWidth(70)
            self.layer_edits.append(e); nl.addWidget(e)
        self._rows['names'] = self._row('Layer names (R/G/B/A)', names_w)
        form.addRow(self._rows['names'])

        self.chk_normal = QCheckBox('Save normal map PNG')
        self._rows['normal'] = self._row('Normal map', self.chk_normal)
        form.addRow(self._rows['normal'])

        self.chk_script = QCheckBox('Include import helper script')
        self.chk_script.setChecked(True)
        self._rows['script'] = self._row('Helper script', self.chk_script)
        form.addRow(self._rows['script'])

        if main._splat is not None:
            note = QLabel('● Painted splat layers will be exported.')
            note.setStyleSheet('color: #7ec97e;')
            form.addRow(note)

        if p0.get('preview_scale') != '100%':
            H, W = main._hmap.shape
            note = QLabel(f"⚠ Preview quality is {p0['preview_scale']} — exporting "
                          f"{W}×{H}. Set 100% and regenerate for full resolution.")
            note.setWordWrap(True)
            note.setStyleSheet('color: #e0a040;')
            form.addRow(note)

        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel = QPushButton('Close')
        cancel.clicked.connect(self.reject)
        self.export_btn = QPushButton('Export…')
        self.export_btn.setDefault(True)
        self.export_btn.clicked.connect(self._do_export)
        btns.addWidget(cancel); btns.addWidget(self.export_btn)
        form.addRow(btns)
        lay.addWidget(formw, 1)

        self.listw.currentRowChanged.connect(lambda _: self._apply_vis())
        self.listw.setCurrentRow(0)

    def _row(self, label, w):
        c = QWidget(); l = QHBoxLayout(c)
        l.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(label); lab.setMinimumWidth(150)
        l.addWidget(lab); l.addWidget(w, 1)
        return c

    def _code(self):
        return self.TARGETS[self.listw.currentRow()][1]

    def _layer_names(self):
        out = []
        for i, e in enumerate(self.layer_edits):
            nm = re.sub(r'[^A-Za-z0-9_]', '', e.text().strip()) or f'layer{i}'
            out.append(nm)
        return out

    def _height_mode(self, target):
        t = self.mode_combo.currentText()
        if t.startswith('Auto'):
            return 'normalized' if AUTO_NORM.get(target, False) else 'sea-anchored'
        return 'normalized' if t.startswith('Normal') else 'sea-anchored'

    def _apply_vis(self):
        code = self._code()
        vis = {'res': code in ('unity', 'unreal', 'godot'),
               'endian': code in ('unity', 'unreal'),
               'mode': code in ('unity', 'unreal', 'godot', 'chunked'),
               'splat': code in ('unity', 'unreal'),
               'names': code in ('unity', 'unreal'),
               'masks': code in ('unity', 'unreal', 'godot', 'chunked'),
               'normal': code in ('unity', 'unreal', 'godot'),
               'script': code in ('unity', 'godot', 'blender'),
               'step': code in ('glb', 'obj', 'ply', 'stl', 'chunked', 'godot'),
               'grid': code == 'chunked',
               'overlap': code == 'chunked',
               'lods': code == 'chunked',
               'zip': code == 'chunked',
               'water': code in ('glb', 'godot'),
               'embed': code in ('glb', 'godot', 'obj'),
               'vcol': code == 'glb'}
        for key, w in self._rows.items():
            w.setVisible(vis.get(key, False))
        self.desc.setText({
            'unity': "Real .unitypackage. Splat comes from your painted layers "
                     "if you used Paint Splat, else the procedural thresholds.",
            'unreal': "heightmap.raw/.png + layer masks + TF_UE5_Import.py "
                      "(imports masks, creates layer infos, builds the layered "
                      "material, logs exact scales).",
            'godot': "terrain_scene.tscn (embedded collision), GDScript with "
                     "UVs/tangents, terrain.glb (textured + water), f32 + EXR.",
            'blender': "One self-contained Python script. Scripting tab > Run.",
            'chunked': "N×N world-space chunks: raw + textured GLB + LODs + "
                       "manifest.json, optional ZIP.",
            'glb': "Y-up meters, normals, UVs, embedded albedo, optional "
                   "vertex colors + alpha water plane.",
            'obj': "Scaled OBJ; with texture enabled also .mtl + albedo.png "
                   "with UVs and normals.",
            'ply': "Binary PLY with vertex colors AND per-vertex normals.",
            'stl': "Watertight solid: surface + walls + flat base "
                   "(slicer-ready).",
        }[code])
        if code in ('unity', 'unreal', 'godot'):
            cur = self.res_combo.currentText()
            RES_OPTS = {
                'unity': ['Native (max side)', '129', '257', '513', '1025', '2049', '4097'],
                'unreal': ['127', '253', '505', '1009', '2017', '4033'],
                'godot': ['Native (max side)', '129', '257', '513', '1025'],
            }[code]
            self.res_combo.blockSignals(True)
            self.res_combo.clear()
            self.res_combo.addItems(RES_OPTS)
            if code == 'unreal':
                W = self.main._hmap.shape[1]
                near = min([127, 253, 505, 1009, 2017, 4033],
                           key=lambda r: abs(r - W))
                self.res_combo.setCurrentText(str(near))
            elif cur in RES_OPTS:
                self.res_combo.setCurrentText(cur)
            self.res_combo.blockSignals(False)

    def _parse_res(self):
        t = self.res_combo.currentText()
        if t.startswith('Native'):
            return max(self.main._hmap.shape)
        return int(t)

    def _clean_rgb(self, p):
        if self._albedo is None:
            p2 = dict(p)
            p2['render_mode'] = 'Shaded'
            p2['shade_strength'] = 0.0
            p2['ao'] = 0.0
            p2['contours'] = False
            _, rgb = render_terrain(self.main._hmap, p2, self.main._flow)
            self._albedo = rgb
        return self._albedo

    def _info_open(self, title, text, path):
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        ob = box.addButton('Open Folder', QMessageBox.ButtonRole.ActionRole)
        box.addButton('OK', QMessageBox.ButtonRole.AcceptRole)
        box.exec()
        if box.clickedButton() is ob:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _write_data_masks(self, sub, h, p):
        self.main._ensure_flow()
        wet, dep, cur = _data_masks(h, self.main._flow, float(p['sea_level']))
        _save_gray_png(os.path.join(sub, 'mask_wetness.png'), wet)
        _save_gray_png(os.path.join(sub, 'mask_deposition.png'), dep)
        _save_gray_png(os.path.join(sub, 'mask_curvature.png'),
                       (cur + 1.0) * 0.5)
        _f, lk = detect_lakes(h, float(p['sea_level']))
        _save_gray_png(os.path.join(sub, 'mask_lakes.png'),
                       lk.astype(np.float64))
        _save_gray_png(os.path.join(sub, 'mask_walkable.png'),
                       walkable_mask(h, float(p['sea_level']),
                                     float(p.get('walk_slope', 0.10))))

    def _do_export(self):
        main = self.main
        if main._hmap is None:
            return
        code = self._code()
        p = main._collect()
        p['_splat_painted'] = main._splat is not None
        h_raw = main._hmap
        H, W = h_raw.shape
        world = self.world_spin.value()
        hmet = self.height_spin.value()
        hmode = self._height_mode(code)
        layers = self._layer_names()
        h = prep_height(h_raw, hmode)

        if code in ('unity', 'unreal', 'godot', 'blender', 'chunked'):
            if code == 'unity':
                ok, msg, path = self._export_unity(h, h_raw, p, world, hmet,
                                                   layers, hmode)
                if not ok:
                    return
                self._info_open('Export complete', msg, path)
            else:
                base = QFileDialog.getExistingDirectory(self, 'Choose output folder')
                if not base:
                    return
                sub = os.path.join(base, f'TerrainForge_{code}')
                os.makedirs(sub, exist_ok=True)
                try:
                    if code == 'unreal':
                        msg = self._export_unreal(sub, h, h_raw, p, world, hmet,
                                                  layers, hmode)
                    elif code == 'godot':
                        msg = self._export_godot(sub, h, p, world, hmet, hmode)
                    elif code == 'chunked':
                        msg = self._export_chunked(sub, h, p, world, hmet, hmode)
                    else:
                        msg = self._export_blender(sub, h, p, world, hmet, hmode)
                except Exception:
                    QMessageBox.critical(self, 'Export failed',
                                         traceback.format_exc())
                    return
                self._info_open('Export complete', f'{msg}\n\nFolder:\n{sub}', sub)
        else:
            ext = {'glb': '.glb', 'obj': '.obj', 'ply': '.ply', 'stl': '.stl'}
            filt = {'glb': 'glTF binary (*.glb)', 'obj': 'OBJ (*.obj)',
                    'ply': 'PLY (*.ply)', 'stl': 'STL (*.stl)'}
            path, _ = QFileDialog.getSaveFileName(
                self, 'Export mesh', 'terrain' + ext[code], filt[code])
            if not path:
                return
            step = self.step_spin.value()
            nverts = math.ceil(H / step) * math.ceil(W / step)
            if nverts > 2_000_000:
                QMessageBox.warning(self, 'Mesh too large',
                                    f'{nverts:,} vertices — increase the decimation step.')
                return
            rgb = self._clean_rgb(p)
            sea01 = float(p['sea_level'])
            has_sea = bool((h < sea01).any()) and sea01 > 0.015
            water = self.chk_water.isChecked() and hmode == 'sea-anchored' and has_sea
            try:
                if code == 'glb':
                    write_glb(path, h, rgb, world, hmet, step,
                              sea01=sea01, water=water,
                              texture=self.chk_embed.isChecked(),
                              vcol=self.chk_vcol.isChecked(),
                              albedo_png=_png_bytes(rgb))
                elif code == 'obj':
                    write_obj_scaled(path, h, rgb, world, hmet, step,
                                     textured=self.chk_embed.isChecked())
                elif code == 'ply':
                    write_ply(path, h, rgb, world, hmet, step)
                else:
                    write_stl(path, h, world, hmet, step)
            except Exception:
                QMessageBox.critical(self, 'Export failed', traceback.format_exc())
                return
            ntris = 2 * (math.ceil(H / step) - 1) * (math.ceil(W / step) - 1)
            self._info_open('Export complete',
                            f'{nverts:,} vertices / {ntris:,} triangles\n{path}',
                            os.path.dirname(path) or '.')

    def _export_unity(self, h, h_raw, p, world, hmet, layers, hmode):
        res = self._parse_res()
        big = self.endian_combo.currentText().startswith('Big')
        norm = hmode.startswith('norm')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Unity package', 'TerrainForge.unitypackage',
            'Unity package (*.unitypackage)')
        if not path:
            return False, None, None
        hr = resize2d(h, res, res)
        splat = (self.main._active_splat(p) * 255 + 0.5).astype(np.uint8)
        L = layers
        cs = (UNITY_CS.replace('%%RES%%', str(res))
                      .replace('%%BE%%', 'true' if big else 'false')
                      .replace('%%NORM%%', 'true' if norm else 'false')
                      .replace('%%WORLD%%', f'{world:g}')
                      .replace('%%HEIGHT%%', f'{hmet:g}')
                      .replace('%%SEA%%', f"{float(p['sea_level']):.3f}"))
        for i in range(4):
            cs = cs.replace(f'%%L{i}%%', L[i])
        files = {
            'Assets/TerrainForge/heightmap.raw': _raw16_bytes(hr, big),
            'Assets/TerrainForge/albedo.png': _png_bytes(self._clean_rgb(p)),
            'Assets/TerrainForge/splatmap.png': _png_bytes(splat),
            'Assets/TerrainForge/TerrainForgeImport.cs': cs.encode(),
            'Assets/TerrainForge/metadata.json': _meta_json(
                p, res, world, hmet, layers, hmode,
                {'target': 'unity'}).encode(),
            'Assets/TerrainForge/README.md': (
                "# Terrain Forge → Unity\n\n"
                "1. This package installs into Assets/TerrainForge.\n"
                "2. Menu: Tools → Terrain Forge → Import Terrain.\n"
                f"3. Size {world:g}×{world:g} m, height {hmet:g} m, "
                f"sea {float(p['sea_level']):.2f} ({hmode} heights).\n"
                f"Layers: {', '.join(layers)}.\n").encode(),
        }
        seed = int(p['seed'])
        theme = SPLAT_THEMES.get(p.get('palette', 'Terrain'),
                                 SPLAT_THEMES['Terrain'])
        for i, nm in enumerate(L):
            files[f'Assets/TerrainForge/layer_{i}_{nm}.png'] = _png_bytes(
                make_tile_texture(seed + i * 101, theme[i][1]))
        if self.chk_normal.isChecked():
            files['Assets/TerrainForge/normalmap.png'] = _png_bytes(
                normal_map_rgb(h_raw, float(p['relief'])))
        if self.chk_masks.isChecked():
            if self.main._flow is None:
                self.main._ensure_flow()
            wet, dep, cur = _data_masks(h, self.main._flow,
                                        float(p['sea_level']))
            files['Assets/TerrainForge/mask_wetness.png'] = _png_bytes(
                np.repeat((wet * 255).astype(np.uint8)[:, :, None], 3, 2))
            files['Assets/TerrainForge/mask_deposition.png'] = _png_bytes(
                np.repeat((dep * 255).astype(np.uint8)[:, :, None], 3, 2))
            files['Assets/TerrainForge/mask_curvature.png'] = _png_bytes(
                np.repeat((((cur + 1) * 0.5) * 255).astype(np.uint8)[:, :, None], 3, 2))
            files['Assets/TerrainForge/mask_walkable.png'] = _png_bytes(
                np.repeat((walkable_mask(h, float(p['sea_level']),
                                         float(p.get('walk_slope', 0.10))) * 255
                           ).astype(np.uint8)[:, :, None], 3, 2))
        write_unitypackage(path, files)
        return True, (f'Unity package written: {len(files)} assets, heightmap '
                      f'{res}×{res}, layers {", ".join(L)}, {hmode} heights.'), \
               (os.path.dirname(path) or '.')

    def _export_unreal(self, sub, h, h_raw, p, world, hmet, layers, hmode):
        res = int(self.res_combo.currentText())
        big = self.endian_combo.currentText().startswith('Big')
        hr = resize2d(h, res, res)
        _save_raw16(os.path.join(sub, 'heightmap.raw'), hr, big)
        a16 = (np.clip(hr, 0, 1) * 65535.0 + 0.5).astype('<u2')
        img = QImage(a16.tobytes(), res, res, res * 2,
                     QImage.Format.Format_Grayscale16).copy()
        img.save(os.path.join(sub, 'heightmap.png'), 'PNG')
        if self.chk_splat.isChecked():
            splat = self.main._active_splat(p)
            for i, nm in enumerate(layers):
                _save_gray_png(os.path.join(sub, f'mask_{i}_{nm}.png'),
                               splat[:, :, i])
        if self.chk_masks.isChecked():
            self._write_data_masks(sub, h, p)
        if self.chk_normal.isChecked():
            _save_rgb_png(os.path.join(sub, 'normalmap.png'),
                          normal_map_rgb(h_raw, float(p['relief'])))
        if self.chk_script.isChecked():
            py = (UE_PY.replace('%%FOLDER%%', os.path.abspath(sub))
                       .replace('%%RES%%', str(res))
                       .replace('%%WORLD%%', f'{world:g}')
                       .replace('%%HEIGHT%%', f'{hmet:g}'))
            for i in range(4):
                py = py.replace(f'%%L{i}%%', layers[i])
            with open(os.path.join(sub, 'TF_UE5_Import.py'), 'w') as f:
                f.write(py)
        with open(os.path.join(sub, 'metadata.json'), 'w') as f:
            f.write(_meta_json(p, res, world, hmet, layers, hmode, extra={
                'target': 'unreal',
                'unreal_xy_scale_pct': round(world / (res - 1) * 100.0, 1),
                'unreal_z_scale_pct': round(hmet / 5.12, 1)}))
        xy = world / (res - 1) * 100.0
        with open(os.path.join(sub, 'README.md'), 'w') as f:
            f.write(f"""# Terrain Forge → Unreal Engine 5

1. Run TF_UE5_Import.py (Python Editor Script Plugin) — builds the material
   and prints the scales.
2. Landscape > Import from File > heightmap.raw ({res}×{res}).
3. XY ≈ {xy:.1f} % · Z ≈ {hmet / 5.12:.1f} % · heights: {hmode}.
4. Layers: {', '.join(layers)} (R/G/B/A order).
""")
        return (f'Unreal bundle written: raw+png {res}×{res}, '
                f'{len(layers)} masks, import script, README.')

    def _export_godot(self, sub, h, p, world, hmet, hmode):
        res = self._parse_res()
        hg = resize2d(h, res, res)
        with open(os.path.join(sub, 'terrain.f32'), 'wb') as f:
            f.write(_f32_bytes(hg))
        _save_rgb_png(os.path.join(sub, 'albedo.png'), self._clean_rgb(p))
        write_exr_r(os.path.join(sub, 'heightmap.exr'), hg)
        rgb = self._clean_rgb(p)
        step = self.step_spin.value()
        sea01 = float(p['sea_level'])
        water = (self.chk_water.isChecked() and hmode == 'sea-anchored'
                 and sea01 > 0.015 and bool((h < sea01).any()))
        write_glb(os.path.join(sub, 'terrain.glb'), h, rgb, world, hmet, step,
                  sea01=sea01, water=water,
                  texture=self.chk_embed.isChecked(),
                  vcol=self.chk_vcol.isChecked(),
                  albedo_png=_png_bytes(rgb))
        if self.chk_masks.isChecked():
            self._write_data_masks(sub, h, p)
        if self.chk_normal.isChecked():
            _save_rgb_png(os.path.join(sub, 'normalmap.png'),
                          normal_map_rgb(h, float(p['relief'])))
        if self.chk_script.isChecked():
            gd = (GODOT_GD.replace('%%W%%', str(res))
                          .replace('%%H%%', str(res))
                          .replace('%%WORLD%%', f'{world:.1f}')
                          .replace('%%HEIGHT%%', f'{hmet:.1f}'))
            with open(os.path.join(sub, 'terrain_godot.gd'), 'w') as f:
                f.write(gd)
            flats = np.clip(hg, 0, 1).reshape(-1)
            txt = ','.join(f'{v:.6f}' for v in flats.tolist())
            tscn = f"""[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://terrain_godot.gd" id="1"]

[sub_resource type="HeightMapShape3D" id="hms"]
map_width = {res}
map_depth = {res}
map_data = PackedFloat32Array({txt})

[node name="TerrainForge" type="Node3D"]
script = ExtResource("1")

[node name="TerrainCollision" type="StaticBody3D" parent="."]

[node name="CollisionShape3D" type="CollisionShape3D" parent="TerrainCollision"]
shape = SubResource("hms")
"""
            with open(os.path.join(sub, 'terrain_scene.tscn'), 'w') as f:
                f.write(tscn)
        with open(os.path.join(sub, 'metadata.json'), 'w') as f:
            f.write(_meta_json(p, res, world, hmet,
                               ['grass', 'rock', 'sand', 'snow'], hmode,
                               {'target': 'godot'}))
        with open(os.path.join(sub, 'README.md'), 'w') as f:
            f.write(f"""# Terrain Forge → Godot 4

A) Open terrain_scene.tscn (collision pre-built). B) Instance terrain.glb.
World {world:g} m, height {hmet:g} m, heights {hmode}.
Terrain3D: import heightmap.exr at scale 1.0.
""")
        return (f'Godot bundle written: tscn + gd + glb + f32/EXR {res}×{res}.')

    def _export_chunked(self, sub, h, p, world, hmet, hmode):
        H, W = h.shape
        grid = int(self.grid_combo.currentText().split('×')[0])
        ov = self.overlap_spin.value()
        step = self.step_spin.value()
        rgb = self._clean_rgb(p)
        sea01 = float(p['sea_level'])
        chH = math.ceil(H / grid); chW = math.ceil(W / grid)
        cellx = world / max(W - 1, 1); cellz = world / max(H - 1, 1)
        tiles = []
        for r in range(grid):
            for c in range(grid):
                y0 = max(0, r * chH - ov); y1 = min(H, (r + 1) * chH + ov)
                x0 = max(0, c * chW - ov); x1 = min(W, (c + 1) * chW + ov)
                name = f'chunk_r{r}_c{c}'
                d = os.path.join(sub, name)
                os.makedirs(d, exist_ok=True)
                sh = h[y0:y1, x0:x1]
                sc = rgb[y0:y1, x0:x1]
                _save_raw16(os.path.join(d, 'heightmap.raw'), sh, False)
                write_glb(os.path.join(d, 'mesh.glb'), sh, sc, world, hmet,
                          step, sea01=0.0, water=False, texture=True,
                          vcol=True, albedo_png=_png_bytes(sc),
                          ox=x0, oy=y0, gw=W, gh=H)
                lods = []
                if self.chk_lods.isChecked():
                    for li in (1, 2, 3):
                        ls = step * (2 ** li)
                        if ls < min(sh.shape):
                            write_glb(os.path.join(d, f'mesh_lod{li}.glb'),
                                      sh, sc, world, hmet, ls, sea01=0.0,
                                      water=False, texture=True, vcol=False,
                                      albedo_png=_png_bytes(sc),
                                      ox=x0, oy=y0, gw=W, gh=H)
                            lods.append(f'mesh_lod{li}.glb')
                tiles.append({
                    "dir": name,
                    "cells": [int(x1 - x0), int(y1 - y0)],
                    "cell_origin": [int(x0), int(y0)],
                    "world_origin_m": [round(x0 * cellx, 3), round(y0 * cellz, 3)],
                    "heightmap": "heightmap.raw", "mesh": "mesh.glb",
                    "lod_meshes": lods})
        if self.chk_masks.isChecked():
            self._write_data_masks(sub, h, p)
        manifest = {
            "chunk_grid": [grid, grid],
            "overlap_cells": ov,
            "cell_size_m": round(cellx, 4),
            "world_size_m": world,
            "height_m": hmet,
            "height_mode": hmode,
            "sea_level_01": sea01,
            "tiles": tiles,
            "metadata": json.loads(_meta_json(p, W, world, hmet,
                                              ['grass', 'rock', 'sand', 'snow'],
                                              hmode)),
        }
        with open(os.path.join(sub, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
        with open(os.path.join(sub, 'README.md'), 'w') as f:
            f.write(f"# Terrain Forge → Chunked tiles ({grid}×{grid}, overlap {ov})\n"
                    "World-space chunks; manifest.json has origins + LODs.\n")
        if self.chk_zip.isChecked():
            zpath = sub + '.zip'
            with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
                for root, _, fs in os.walk(sub):
                    for fn in fs:
                        fp = os.path.join(root, fn)
                        z.write(fp, os.path.relpath(fp, sub))
            return (f'Chunked export: {len(tiles)} chunks + {zpath}')
        return (f'Chunked export written: {len(tiles)} chunks.')

    def _export_blender(self, sub, h, p, world, hmet, hmode):
        H, W = h.shape
        step = max(1, math.ceil(max(H, W) / 513))
        hs = h[::step, ::step]
        ny, nx = hs.shape
        h64 = base64.b64encode(zlib.compress(
            (np.clip(hs, 0, 1) * 65535).astype('<u2').tobytes(), 9)).decode()
        cs0 = self._clean_rgb(p)[::step, ::step]
        cstep = max(1, math.ceil(nx / 256))
        cs = cs0[::cstep, ::cstep]
        cy, cx = cs.shape[:2]
        c64 = base64.b64encode(zlib.compress(cs.astype(np.uint8).tobytes(), 9)).decode()
        script = (BLENDER_PY.replace('%%NX%%', str(nx)).replace('%%NY%%', str(ny))
                            .replace('%%WORLD%%', f'{world:.1f}')
                            .replace('%%HEIGHT%%', f'{hmet:.1f}')
                            .replace('%%HB64%%', h64).replace('%%CB64%%', c64)
                            .replace('%%CXW%%', str(cx)).replace('%%CYX%%', str(cy)))
        with open(os.path.join(sub, 'terrain_blender.py'), 'w') as f:
            f.write(script)
        with open(os.path.join(sub, 'metadata.json'), 'w') as f:
            f.write(_meta_json(p, max(H, W), world, hmet,
                               ['grass', 'rock', 'sand', 'snow'], hmode,
                               {'target': 'blender'}))
        with open(os.path.join(sub, 'README.md'), 'w') as f:
            f.write(f"# Terrain Forge → Blender\n\nScripting tab → Open → "
                    f"Run terrain_blender.py ({nx}×{ny}).\n")
        return (f'Blender script written: {nx}×{ny} grid.')

# ----------------------------------------------------------------------------
# CLI — headless mode with batch export
# ----------------------------------------------------------------------------

CLI_DEFAULTS = dict(
    seed=1337, width=512, height=512, preview_scale='100%',
    noise_type='Perlin', pattern='fBm', tiled=False,
    scale=3.0, octaves=6, persistence=0.5, lacunarity=2.0,
    offset_x=0.0, offset_y=0.0, warp_strength=0.06, warp_freq=2.0,
    gamma=1.0, terraces=0, smoothing=0,
    mask_type='None', mask_size=0.75, mask_strength=0.8,
    erosion_method='Flow (fast)', droplets=4000, lifetime=30,
    erode=0.35, deposit=0.30, capacity=3.0, evaporate=0.03, inertia=0.06,
    numba_droplets=HAVE_NUMBA, erode_parallel=not HAVE_NUMBA,
    live_erosion=False,
    flow_iters=160, flow_flux=18, flow_rain=0.04, flow_evap=0.02,
    flow_capacity=3.0, flow_erode=0.35, flow_deposit=0.30,
    thermal_iters=15, talus=0.003, thermal_rate=0.5,
    sea_level=0.30, palette='Terrain', moist_scale=3.5,
    rivers=True, river_threshold=8, river_amount=0.8,
    render_mode='Shaded', relief=1.0, ao=0.35, shade_strength=0.6,
    sun_az=315, sun_alt=45, contours=False, contour_n=16,
    contour_color='#101010',
    spl_rock=0.05, spl_beach=0.045, spl_snow=0.79,
    walk_slope=0.10, strata_n=12,
    blend_enable=False, blend_mask='Noise', blend_freq=3.0,
    blend_lo=0.35, blend_hi=0.65, blend_invert=False,
)


def _cli_load_hmap(path, W, H):
    ext = path.rsplit('.', 1)[-1].lower()
    if ext == 'f32':
        a = np.fromfile(path, '<f4')
        if a.size < W * H:
            raise ValueError(f'need {W * H} floats, file has {a.size}')
        h = a[:W * H].reshape(H, W).astype(np.float64)
        if h.max() > 1.01 or h.min() < -0.01:
            mn, mx = h.min(), h.max()
            h = (h - mn) / max(mx - mn, 1e-9)
        return resize2d(np.clip(h, 0, 1), W, H)
    if ext == 'raw':
        a = np.fromfile(path, '<u2')
        side = int(math.isqrt(a.size))
        if side * side != a.size:
            raise ValueError(f'raw is not square ({a.size} samples)')
        return resize2d(a.reshape(side, side).astype(np.float64) / 65535.0, W, H)
    img = QImage(path)
    if img.isNull():
        raise ValueError('could not decode image')
    img = img.convertToFormat(QImage.Format.Format_Grayscale16)
    iw, ih = img.width(), img.height()
    arr = np.frombuffer(img.constBits(), np.uint16).reshape(
        ih, img.bytesPerLine() // 2)[:, :iw]
    return resize2d(arr.astype(np.float64) / 65535.0, W, H)


def _parse_batch(spec):
    key, _, expr = spec.partition('=')
    key = key.strip()
    if not key or not expr:
        raise ValueError(f"batch spec needs key=value, got '{spec}'")
    vals = []
    for part in expr.split(','):
        part = part.strip()
        if '..' in part:
            a_s, rest = part.split('..', 1)
            b_s, _, st_s = rest.partition(':')
            a = float(a_s); b = float(b_s)
            step = float(st_s) if st_s else 1.0
            if step <= 0:
                raise ValueError('batch step must be > 0')
            v = a
            while v <= b + 1e-9 and len(vals) < 512:
                vals.append(v)
                v += step
        else:
            vals.append(float(part))
    if not vals:
        raise ValueError('batch produced no values')
    if all(abs(v - round(v)) < 1e-9 for v in vals):
        vals = [int(round(v)) for v in vals]
    return key, vals


def _cli_export(p, h, flow, img, rgb, wants, out_dir, args, quiet):
    wants = set(wants)
    if 'bundle' in wants:
        wants |= {'render', 'albedo', 'heightmap', 'normal', 'masks',
                  'metadata', 'preset'}
        wants.discard('bundle')
    written = []

    def done(name):
        written.append(name)
        if not quiet:
            print(f'  + {name}')

    H, W = h.shape
    sea = float(p['sea_level'])
    if 'render' in wants:
        img.save(os.path.join(out_dir, 'render.png'), 'PNG'); done('render.png')
    if 'albedo' in wants:
        p2 = dict(p); p2.update(render_mode='Shaded', shade_strength=0.0,
                                ao=0.0, contours=False)
        _, alb = render_terrain(h, p2, flow)
        _save_rgb_png(os.path.join(out_dir, 'albedo.png'), alb); done('albedo.png')
        rgb = alb
    if 'heightmap' in wants:
        a16 = (np.clip(h, 0, 1) * 65535.0 + 0.5).astype('<u2')
        im16 = QImage(a16.tobytes(), W, H, W * 2,
                      QImage.Format.Format_Grayscale16).copy()
        im16.save(os.path.join(out_dir, 'heightmap_16bit.png'), 'PNG')
        with open(os.path.join(out_dir, 'heightmap.f32'), 'wb') as f:
            f.write(_f32_bytes(h))
        done('heightmap_16bit.png'); done('heightmap.f32')
    if 'normal' in wants:
        _save_rgb_png(os.path.join(out_dir, 'normalmap.png'),
                      normal_map_rgb(h, float(p['relief']))); done('normalmap.png')
    if 'masks' in wants:
        fm = flow if flow is not None else flow_accumulation(
            h, seed=int(p['seed']) ^ 24635, sea=sea)
        wet, dep, cur = _data_masks(h, fm, sea)
        _save_gray_png(os.path.join(out_dir, 'mask_wetness.png'), wet)
        _save_gray_png(os.path.join(out_dir, 'mask_deposition.png'), dep)
        _save_gray_png(os.path.join(out_dir, 'mask_curvature.png'),
                       (cur + 1.0) * 0.5)
        _f, lk = detect_lakes(h, sea)
        _save_gray_png(os.path.join(out_dir, 'mask_lakes.png'),
                       lk.astype(np.float64))
        _save_gray_png(os.path.join(out_dir, 'mask_walkable.png'),
                       walkable_mask(h, sea, float(p.get('walk_slope', 0.10))))
        done('masks (5 PNG)')
    if 'glb' in wants:
        write_glb(os.path.join(out_dir, 'terrain.glb'), h, rgb,
                  args.world, args.max_height, 2, sea01=sea,
                  water=sea > 0.015 and bool((h < sea).any()),
                  texture=True, vcol=True, albedo_png=_png_bytes(rgb))
        done('terrain.glb')
    if 'obj' in wants:
        write_obj_scaled(os.path.join(out_dir, 'terrain.obj'), h, rgb,
                         args.world, args.max_height, 2, textured=True)
        done('terrain.obj (+mtl/albedo)')
    if 'ply' in wants:
        write_ply(os.path.join(out_dir, 'terrain.ply'), h, rgb,
                  args.world, args.max_height, 2)
        done('terrain.ply')
    if 'stl' in wants:
        write_stl(os.path.join(out_dir, 'terrain.stl'), h,
                  args.world, args.max_height, 2)
        done('terrain.stl')
    if 'unity' in wants:
        res = min(max(args.resolution, 65), 4097)
        hr = resize2d(h, res, res)
        theme = SPLAT_THEMES.get(p['palette'], SPLAT_THEMES['Terrain'])
        L = [t[0] for t in theme]
        cs = (UNITY_CS.replace('%%RES%%', str(res))
                      .replace('%%BE%%', 'false')
                      .replace('%%NORM%%', 'false')
                      .replace('%%WORLD%%', f'{args.world:g}')
                      .replace('%%HEIGHT%%', f'{args.max_height:g}')
                      .replace('%%SEA%%', f'{sea:.3f}'))
        for i in range(4):
            cs = cs.replace(f'%%L{i}%%', L[i])
        splat = (make_splat_rgba(h, sea, p) * 255 + 0.5).astype(np.uint8)
        files = {
            'Assets/TerrainForge/heightmap.raw': _raw16_bytes(hr, False),
            'Assets/TerrainForge/albedo.png': _png_bytes(rgb),
            'Assets/TerrainForge/splatmap.png': _png_bytes(splat),
            'Assets/TerrainForge/TerrainForgeImport.cs': cs.encode(),
            'Assets/TerrainForge/metadata.json': _meta_json(
                p, res, args.world, args.max_height, L, 'sea-anchored',
                {'target': 'unity'}).encode(),
        }
        for i, nm in enumerate(L):
            files[f'Assets/TerrainForge/layer_{i}_{nm}.png'] = _png_bytes(
                make_tile_texture(int(p['seed']) + i * 101, theme[i][1]))
        write_unitypackage(os.path.join(out_dir, 'TerrainForge.unitypackage'), files)
        done('TerrainForge.unitypackage')
    if 'metadata' in wants:
        with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
            f.write(_meta_json(p, max(H, W), args.world, args.max_height,
                               None, 'sea-anchored', {'target': 'cli'}))
        done('metadata.json')
    if 'preset' in wants:
        with open(os.path.join(out_dir, 'preset.json'), 'w') as f:
            json.dump({k: v for k, v in p.items()
                       if not isinstance(v, np.ndarray)}, f, indent=2)
        done('preset.json')
    return len(written)


def run_cli(argv):
    import argparse
    ap = argparse.ArgumentParser(
        prog='TerrainForge', description='Headless terrain generation/export.')
    ap.add_argument('--preset', help='JSON preset to start from')
    ap.add_argument('--style', choices=list(STYLES.keys()))
    ap.add_argument('--seed', type=int)
    ap.add_argument('--width', type=int)
    ap.add_argument('--height', type=int)
    ap.add_argument('--palette')
    ap.add_argument('--sea', type=float, help='sea level 0..1')
    ap.add_argument('--out', default='forge_out')
    ap.add_argument('--world', type=float, default=1000.0)
    ap.add_argument('--max-height', type=float, default=120.0)
    ap.add_argument('--resolution', type=int, default=513)
    ap.add_argument('--export', default='bundle',
                    help='comma list: bundle,render,albedo,heightmap,normal,'
                         'masks,glb,obj,ply,stl,unity,metadata,preset')
    ap.add_argument('--export-many', '--batch', dest='batch', metavar='SPEC',
                    help="batch over a numeric param, e.g. \"seed=100..110\" "
                         "\"seed=1,5,9\" \"sea=0.2..0.6:0.1\" — one output "
                         "subfolder per value")
    ap.add_argument('--out-pattern', default='{key}_{value}',
                    help='batch subfolder name pattern ({key}/{value})')
    ap.add_argument('--blend-a', metavar='FILE',
                    help='heightmap file to blend as Layer A')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    p = dict(CLI_DEFAULTS)
    if args.preset:
        with open(args.preset) as f:
            data = json.load(f)
        for k, v in data.items():
            if k in p:
                p[k] = v
    if args.style:
        p.update(STYLES[args.style])
    if args.seed is not None: p['seed'] = args.seed
    if args.width is not None: p['width'] = args.width
    if args.height is not None: p['height'] = args.height
    if args.palette: p['palette'] = args.palette
    if args.sea is not None: p['sea_level'] = args.sea
    if args.blend_a:
        p['blend_enable'] = True
        p['_layer_a'] = _cli_load_hmap(args.blend_a,
                                       int(p['width']), int(p['height']))

    from PySide6.QtCore import QCoreApplication
    app = QCoreApplication.instance() or QCoreApplication([sys.argv[0]])

    quiet = args.quiet
    last = [0]

    def prog(v):
        if not quiet and v - last[0] >= 5:
            last[0] = v
            sys.stderr.write(f'\r  {v:3d}%')
            sys.stderr.flush()

    wants = {w.strip() for w in args.export.split(',') if w.strip()}

    def run_one(params, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.monotonic()
        h, flow, T, _bw = generate_terrain(params, prog, lambda: False)
        if not quiet:
            sys.stderr.write('\n')
        img, rgb = render_terrain(h, params, flow)
        n = _cli_export(params, h, flow, img, rgb, wants, out_dir, args, quiet)
        print(f'  -> {n} files in {time.monotonic() - t0:.1f}s '
              f'-> {os.path.abspath(out_dir)}')
        return n

    total = 0
    t_all = time.monotonic()
    if args.batch:
        try:
            key, vals = _parse_batch(args.batch)
        except ValueError as e:
            print(f'error: {e}', file=sys.stderr)
            return 2
        if key not in CLI_DEFAULTS:
            print(f"error: unknown batch key '{key}'. Valid: "
                  + ', '.join(sorted(k for k in CLI_DEFAULTS
                                     if not k.startswith('_'))),
                  file=sys.stderr)
            return 2
        print(f'Batch: {key} over {len(vals)} values -> {args.out}/'
              f'{args.out_pattern}')
        for v in vals:
            pv = dict(p)
            pv[key] = v
            fv = int(v) if float(v).is_integer() else round(float(v), 4)
            sub = (args.out_pattern.replace('{key}', key)
                   .replace('{value}', str(fv)))
            print(f'=== {key}={fv} ===')
            total += run_one(pv, os.path.join(args.out, sub))
    else:
        total = run_one(p, args.out)
    print(f'Terrain Forge CLI done: {total} files in '
          f'{time.monotonic() - t_all:.1f}s')
    return 0

# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------

class MainWindow(QMainWindow):
    RENDER_KEYS = {'sea_level', 'palette', 'render_mode', 'relief', 'ao',
                   'shade_strength', 'sun_az', 'sun_alt', 'contours', 'contour_n',
                   'contour_color', 'river_threshold', 'river_amount',
                   'moist_scale', 'spl_rock', 'spl_beach', 'spl_snow',
                   'walk_slope', 'strata_n', 'show_lakes'}
    VIEW3D_KEYS = {'d3_density', 'd3_zscale', 'd3_fog'}
    IGNORE_KEYS = {'auto_update', 'obj_step', 'obj_vscale', 'fast_preview',
                   'tool', 'brush_size', 'brush_str', 'live_erosion',
                   'splat_layer'}

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Terrain Forge v12')
        self.resize(1400, 900)
        self.settings = QSettings('TerrainForge', 'TerrainForge')
        self.controls = {}
        self._hmap = None; self._flow = None; self._rgb = None; self._image = None
        self._worker = None; self._thread = None; self._pending = False
        self._last_size = None; self._booted = False; self._gen_time = 0.0
        self._timings = None
        self._auto_pending = False
        self._full_refresh_pending = False
        self._dirty3d = False
        self._undo_stack = []
        self._redo_stack = []
        self._flat_ref = None
        self._layer_a = None
        self._splat = None
        self._blend_w = None
        self._blend_mask_painted = None
        self._lakes = None; self._lakes_key = None
        self._snap_pending = None
        self._er_worker = None; self._er_thread = None; self._er_t0 = 0.0
        self._recovered = False
        self._autosa_path = _autosave_path()
        self._recent = self.settings.value('recent', []) or []
        if isinstance(self._recent, str):
            self._recent = [self._recent]
        self._build_ui()
        self._queue_timer = QTimer(self)
        self._queue_timer.setSingleShot(True)
        self._queue_timer.setInterval(300)
        self._queue_timer.timeout.connect(self._start_generation)
        self._full_timer = QTimer(self)
        self._full_timer.setSingleShot(True)
        self._full_timer.setInterval(700)
        self._full_timer.timeout.connect(self._start_generation)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(40)
        self._render_timer.timeout.connect(self._rerender)
        self._v3d_timer = QTimer(self)
        self._v3d_timer.setSingleShot(True)
        self._v3d_timer.setInterval(120)
        self._v3d_timer.timeout.connect(self._rebuild_3d)
        self._snap_timer = QTimer(self)
        self._snap_timer.setSingleShot(True)
        self._snap_timer.setInterval(80)
        self._snap_timer.timeout.connect(self._flush_snapshot)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(120000)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start()
        self._make_shortcuts()
        self._restore_session()
        self._prompt_recovery()

    # ---------- registration helpers ----------

    def _register(self, key, row):
        self.controls[key] = row
        if key not in self.IGNORE_KEYS:
            row.changed.connect(lambda k=key: self._on_control(k))
        return row

    def _add(self, group, key, row):
        self._register(key, row)
        group.addRow(row)

    def _sub(self, text):
        lab = QLabel(text)
        lab.setStyleSheet('color: #8ab; font-weight: 600; margin-top: 4px;')
        lab.setWordWrap(True)
        return lab

    # ---------- UI construction ----------

    def _build_ui(self):
        self._build_toolbar()
        self._build_docks()
        self._build_central()
        self._build_statusbar()
        self._build_menus()
        self.controls['tool'].changed.connect(self._on_tool)
        self._on_tool()

    def _build_toolbar(self):
        tb = QToolBar('Main')
        tb.setObjectName('mainToolbar')
        tb.setMovable(False)
        self.addToolBar(tb)
        self.toolbar = tb

        self.gen_btn = QPushButton('▶  Generate')
        self.gen_btn.setObjectName('genBtn')
        self.gen_btn.setToolTip('Generate terrain from noise (Ctrl+G)')
        self.gen_btn.clicked.connect(self._generate_now)
        tb.addWidget(self.gen_btn)

        self.stop_btn = QPushButton('■  Stop')
        self.stop_btn.setObjectName('stopBtn')
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        tb.addWidget(self.stop_btn)
        tb.addSeparator()

        b_undo = QToolButton(); b_undo.setText('↶ Undo')
        b_undo.setToolTip('Undo stroke (Ctrl+Z)')
        b_undo.clicked.connect(self._undo_sculpt)
        b_redo = QToolButton(); b_redo.setText('↷ Redo')
        b_redo.setToolTip('Redo (Ctrl+Y)')
        b_redo.clicked.connect(self._redo_sculpt)
        tb.addWidget(b_undo); tb.addWidget(b_redo)
        tb.addSeparator()

        b_imp = QToolButton(); b_imp.setText('⭱ Import')
        b_imp.setToolTip('Import heightmap (Ctrl+I)')
        b_imp.clicked.connect(self._import_heightmap)
        b_exp = QToolButton(); b_exp.setText('⭳ Engine…')
        b_exp.setToolTip('Export for Unity / Unreal / Godot / … (Ctrl+Shift+E)')
        b_exp.clicked.connect(self._engine_export)
        tb.addWidget(b_imp); tb.addWidget(b_exp)
        tb.addSeparator()

        lab = QLabel(' Style ')
        lab.setStyleSheet('color:#8a93a0;')
        tb.addWidget(lab)
        self.style_combo = QComboBox()
        self.style_combo.addItems(list(STYLES.keys()))
        self.style_combo.setFixedWidth(150)
        self.style_combo.setToolTip('Quick style preset')
        tb.addWidget(self.style_combo)
        b_apply = QToolButton(); b_apply.setText('Apply')
        b_apply.clicked.connect(self._apply_style)
        b_rand = QToolButton(); b_rand.setText('🎲')
        b_rand.setToolTip('Randomize noise settings + seed')
        b_rand.clicked.connect(self._randomize)
        b_mut = QToolButton(); b_mut.setText('🧬')
        b_mut.setToolTip('Mutate (small tweak + new seed)')
        b_mut.clicked.connect(self._mutate)
        tb.addWidget(b_apply); tb.addWidget(b_rand); tb.addWidget(b_mut)
        tb.addSeparator()

        b_fit = QToolButton(); b_fit.setText('⤢ Fit')
        b_fit.setToolTip('Fit view (Ctrl+F)')
        b_fit.clicked.connect(lambda: self.canvas.fit())
        tb.addWidget(b_fit)

    def _make_tab(self, title):
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        sa.setWidget(page)
        self.set_tabs.addTab(sa, title)
        return v

    def _build_docks(self):
        self.setDockNestingEnabled(True)

        # ---- Tools dock (far left) ----
        self.tool_dock = QDockWidget('Tools', self)
        self.tool_dock.setObjectName('toolsDock')
        tools_w = QWidget()
        tv = QVBoxLayout(tools_w)
        tv.setContentsMargins(0, 4, 0, 4)
        tv.setSpacing(6)
        self.controls['tool'] = ToolPalette('Off')
        tv.addWidget(self.controls['tool'])
        tv.addWidget(self._register('brush_size', SliderRow(
            'Size', 1, 40, 8, 0, 'Brush size, % of the smaller map side')))
        tv.addWidget(self._register('brush_str', SliderRow(
            'Strength', 0.02, 1, 0.3, 2, 'Brush strength')))
        tv.addWidget(self._register('splat_layer', ComboRow(
            'Layer', ['R', 'G', 'B', 'A'], 'R',
            'Target splat layer for Paint Splat')))
        rowb = QWidget(); hb = QHBoxLayout(rowb)
        hb.setContentsMargins(6, 0, 6, 0); hb.setSpacing(4)
        rst = QPushButton('Reset'); rst.setToolTip('Reset splat to procedural')
        rst.clicked.connect(self._reset_splat)
        hb.addWidget(rst)
        tv.addWidget(rowb)
        tv.addStretch(1)
        self.tool_dock.setWidget(tools_w)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.tool_dock)

        # ---- Settings dock (tabbed) ----
        self.set_dock = QDockWidget('Settings', self)
        self.set_dock.setObjectName('settingsDock')
        self.set_tabs = QTabWidget()
        self.set_dock.setWidget(self.set_tabs)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.set_dock)

        # ===== Tab: Terrain =====
        v = self._make_tab('Terrain')
        g = Group('Map', v)
        self._add(g, 'seed', SeedRow(1337))
        self._add(g, 'width', SliderRow('Width', 64, 1024, 512, 0))
        self._add(g, 'height', SliderRow('Height', 64, 1024, 512, 0))
        self._add(g, 'preview_scale', ComboRow('Preview quality',
                  ['25%', '50%', '100%'], '100%',
                  'Lower = faster iteration. Set 100% before final export.'))
        self._add(g, 'fast_preview', CheckRow('Fast preview while adjusting', True))
        self._add(g, 'auto_update', CheckRow('Auto-update on change', True))
        v.addStretch(1)

        v = self._make_tab('Noise')
        g = Group('Base Noise', v)
        self._add(g, 'noise_type', ComboRow('Type',
                  ['Perlin', 'Value', 'Worley'], 'Perlin',
                  'Worley = cellular (badlands, cracked mud). Heavier — '
                  'keep octaves ≤ 5.'))
        self._add(g, 'pattern', ComboRow('Pattern',
                  ['fBm', 'Ridged', 'Billow'], 'fBm'))
        self._add(g, 'tiled', CheckRow('Seamless tiling', False))
        self._add(g, 'scale', SliderRow('Scale', 0.5, 12, 3.0, 1))
        self._add(g, 'octaves', SliderRow('Octaves', 1, 12, 6, 0))
        self._add(g, 'persistence', SliderRow('Persistence', 0.05, 0.9, 0.5, 2))
        self._add(g, 'lacunarity', SliderRow('Lacunarity', 1.5, 4.0, 2.0, 2))
        self._add(g, 'offset_x', SliderRow('Offset X', -4, 4, 0.0, 2))
        self._add(g, 'offset_y', SliderRow('Offset Y', -4, 4, 0.0, 2))
        g = Group('Domain Warp', v)
        self._add(g, 'warp_strength', SliderRow('Strength', 0, 0.3, 0.06, 2))
        self._add(g, 'warp_freq', SliderRow('Frequency', 0.5, 8, 2.0, 1))
        v.addStretch(1)

        v = self._make_tab('Shaping')
        g = Group('Height Shaping', v)
        self._add(g, 'gamma', SliderRow('Gamma / exponent', 0.2, 3.0, 1.0, 2))
        self._add(g, 'terraces', SliderRow('Terraces (0=off)', 0, 24, 0, 0))
        self._add(g, 'smoothing', SliderRow('Smoothing passes', 0, 10, 0, 0))
        ac = QPushButton('Auto-contrast (0.5–99.5%)')
        ac.clicked.connect(self._autocontrast)
        g.addRow(ac)
        g = Group('Island Mask', v)
        self._add(g, 'mask_type', ComboRow('Shape',
                  ['None', 'Round island', 'Square island',
                   'Archipelago (noise)', 'Volcano'], 'None'))
        self._add(g, 'mask_size', SliderRow('Size', 0.4, 1.2, 0.75, 2))
        self._add(g, 'mask_strength', SliderRow('Strength', 0, 1, 0.8, 2))
        g = Group('Layer Blend (A × mask + B)', v)
        g.addRow(self._sub('Capture the map as A, change settings, Generate.'))
        cap = QPushButton('Capture current as Layer A')
        cap.clicked.connect(self._capture_a)
        g.addRow(cap)
        clr = QPushButton('Clear Layer A')
        clr.clicked.connect(self._clear_a)
        g.addRow(clr)
        self._add(g, 'blend_enable', CheckRow('Enable blend on generate', False))
        self._add(g, 'blend_mask', ComboRow('Mask',
                  ['Noise', 'Radial', 'Gradient X', 'Gradient Y',
                   'Height ramp', 'Painted'], 'Noise'))
        self._add(g, 'blend_freq', SliderRow('Noise freq', 0.5, 8, 3.0, 1))
        self._add(g, 'blend_lo', SliderRow('Ramp low', 0, 1, 0.35, 2))
        self._add(g, 'blend_hi', SliderRow('Ramp high', 0, 1, 0.65, 2))
        self._add(g, 'blend_invert', CheckRow('Invert mask', False))
        g.addRow(self._sub('Preview the mask: render mode "Blend mask".'))
        v.addStretch(1)

        # ===== Tab: Erosion =====
        v = self._make_tab('Erosion')
        g = Group('Hydraulic & Thermal', v)
        self._add(g, 'live_erosion', CheckRow('Live erosion preview', True,
                  'Stream the heightfield to the 2D view while erosion runs.'))
        self._add(g, 'erosion_method', ComboRow('Method',
                  ['None', 'Droplets (classic)', 'Flow (fast)'], 'Flow (fast)'))
        g.addRow(self._sub('Droplets'))
        chk = CheckRow('Numba JIT droplets', True)
        if not HAVE_NUMBA:
            chk.setEnabled(False)
            chk.setToolTip('numba not installed — run: pip install numba')
        self._add(g, 'numba_droplets', chk)
        self._add(g, 'erode_parallel', CheckRow(
            'Parallel droplets (multi-core)', not HAVE_NUMBA))
        self._add(g, 'droplets', SliderRow('Droplets', 500, 30000, 4000, 0))
        self._add(g, 'lifetime', SliderRow('Droplet lifetime', 10, 80, 30, 0))
        self._add(g, 'erode', SliderRow('Erode speed', 0.05, 1, 0.35, 2))
        self._add(g, 'deposit', SliderRow('Deposit speed', 0.05, 1, 0.30, 2))
        self._add(g, 'capacity', SliderRow('Sediment capacity', 0.5, 10, 3.0, 1))
        self._add(g, 'evaporate', SliderRow('Evaporation', 0.005, 0.2, 0.03, 3))
        self._add(g, 'inertia', SliderRow('Inertia', 0, 0.9, 0.06, 2))
        g.addRow(self._sub('Flow (pipe model)'))
        self._add(g, 'flow_iters', SliderRow('Iterations', 20, 400, 160, 0))
        self._add(g, 'flow_flux', SliderRow('Flux rate', 4, 40, 18, 0))
        self._add(g, 'flow_rain', SliderRow('Rainfall', 0.005, 0.1, 0.04, 3))
        self._add(g, 'flow_evap', SliderRow('Evaporation', 0.005, 0.08, 0.02, 3))
        self._add(g, 'flow_capacity', SliderRow('Capacity', 0.5, 8, 3.0, 1))
        self._add(g, 'flow_erode', SliderRow('Erode speed', 0.05, 0.8, 0.35, 2))
        self._add(g, 'flow_deposit', SliderRow('Deposit speed', 0.05, 0.8, 0.30, 2))
        g.addRow(self._sub('Thermal (slippage)'))
        self._add(g, 'thermal_iters', SliderRow('Iterations', 0, 200, 15, 0))
        self._add(g, 'talus', SliderRow('Talus angle', 0.0002, 0.01, 0.003, 4))
        self._add(g, 'thermal_rate', SliderRow('Rate', 0.05, 1, 0.5, 2))
        rowe = QWidget(); he = QHBoxLayout(rowe)
        he.setContentsMargins(0, 2, 0, 0); he.setSpacing(4)
        self.apply_hyd_btn = QPushButton('Apply Erosion to Map')
        self.apply_hyd_btn.setToolTip('Runs the current hydraulic settings on '
            'the map IN PLACE — works on imported, sculpted and blended maps.')
        self.apply_th_btn = QPushButton('Apply Thermal')
        self.apply_hyd_btn.clicked.connect(lambda: self._apply_erosion('Hydraulic'))
        self.apply_th_btn.clicked.connect(lambda: self._apply_erosion('Thermal'))
        he.addWidget(self.apply_hyd_btn, 1); he.addWidget(self.apply_th_btn, 1)
        g.addRow(rowe)
        v.addStretch(1)

        # ===== Tab: Surface =====
        v = self._make_tab('Surface')
        g = Group('Water & Colour', v)
        self._add(g, 'sea_level', SliderRow('Sea level', 0, 0.9, 0.30, 2))
        self._add(g, 'palette', ComboRow('Palette',
                  list(PALETTES.keys()) + ['Biomes'], 'Terrain'))
        self._add(g, 'moist_scale', SliderRow('Biome moisture', 1, 8, 3.5, 1))
        self._add(g, 'rivers', CheckRow('Rivers (flow accumulation)', True))
        self._add(g, 'river_threshold', SliderRow('River cutoff', 4, 14, 8, 0))
        self._add(g, 'river_amount', SliderRow('River strength', 0, 1, 0.8, 2))
        self._add(g, 'show_lakes', CheckRow('Detect & show lakes', False,
                  'Fills depressions to find lakes: overlay in 2D, lake tint + '
                  'translucent sea plane in 3D.'))
        g = Group('Splat Layers', v)
        g.addRow(self._sub('Procedural thresholds; painted strokes override. '
                           'Preview with render mode "Splat".'))
        self._add(g, 'spl_rock', SliderRow('Rock on slopes', 0.02, 0.15, 0.05, 3))
        self._add(g, 'spl_beach', SliderRow('Beach width', 0.01, 0.12, 0.045, 3))
        self._add(g, 'spl_snow', SliderRow('Snowline', 0.4, 1.0, 0.79, 2))
        v.addStretch(1)

        # ===== Tab: View & Export =====
        v = self._make_tab('View & Export')
        g = Group('Rendering', v)
        self._add(g, 'render_mode', ComboRow('Mode',
                  ['Shaded', 'Heightmap', 'Normals', 'Slope', 'Splat',
                   'Blend mask', 'Wetness', 'Deposition', 'Curvature',
                   'Walkable', 'Strata'], 'Shaded'))
        self._add(g, 'relief', SliderRow('Relief exaggeration', 0.1, 5, 1.0, 2))
        self._add(g, 'ao', SliderRow('Ambient occlusion', 0, 1, 0.35, 2))
        self._add(g, 'shade_strength', SliderRow('Hillshade strength', 0, 1, 0.6, 2))
        self._add(g, 'sun_az', SliderRow('Sun azimuth', 0, 360, 315, 0))
        self._add(g, 'sun_alt', SliderRow('Sun altitude', 5, 90, 45, 0))
        self._add(g, 'contours', CheckRow('Contour lines', False))
        self._add(g, 'contour_n', SliderRow('Contour count', 2, 50, 16, 0))
        self._add(g, 'contour_color', ColorRow('Contour colour', '#101010'))
        self._add(g, 'walk_slope', SliderRow('Walkable slope ≤', 0.02, 0.3,
                  0.10, 3, 'Cells steeper than this are flagged.'))
        self._add(g, 'strata_n', SliderRow('Strata bands', 2, 40, 12, 0))
        g = Group('3D Preview', v)
        self._add(g, 'd3_density', ComboRow('Mesh density',
                  ['48', '64', '96', '128', '160'], '96'))
        self._add(g, 'd3_zscale', SliderRow('Vertical scale', 0.3, 4, 1.6, 2))
        self._add(g, 'd3_fog', SliderRow('Fog', 0, 1, 0.35, 2))
        gl_lab = QLabel('Renderer: ' + ('OpenGL (textured)' if HAVE_GL else 'software'))
        gl_lab.setStyleSheet('color:#8a93a0;')
        g.addRow(gl_lab)
        reset = QPushButton('Reset 3D camera')
        reset.clicked.connect(lambda: self.view3d.reset_view())
        g.addRow(reset)
        g = Group('Export', v)
        self._add(g, 'obj_step', SliderRow('OBJ grid step', 1, 8, 2, 0))
        self._add(g, 'obj_vscale', SliderRow('OBJ vertical scale', 0.1, 5, 1.0, 2))
        eng = QPushButton('Export for Engine…  (Ctrl+Shift+E)')
        eng.setStyleSheet('font-weight: 600;')
        eng.clicked.connect(self._engine_export)
        g.addRow(eng)
        bundle_btn = QPushButton('Export Everything (bundle)…')
        bundle_btn.clicked.connect(self._export_bundle)
        g.addRow(bundle_btn)
        wrap = QWidget(); grid = QGridLayout(wrap)
        grid.setContentsMargins(0, 0, 0, 0); grid.setSpacing(4)
        exports = (('Render PNG', self._export_png),
                   ('Heightmap 16-bit', self._export_heightmap),
                   ('Normal map', self._export_normal),
                   ('Raw float32', self._export_raw),
                   ('Save heightmap…', self._save_height),
                   ('Preset JSON', self._save_preset))
        for i, (text, slot) in enumerate(exports):
            b = QPushButton(text)
            b.clicked.connect(slot)
            grid.addWidget(b, i // 2, i % 2)
        g.addRow(wrap)
        v.addStretch(1)

        self._build_docks_analysis()
        self.resizeDocks([self.tool_dock, self.set_dock], [60, 384],
                         Qt.Orientation.Horizontal)

    def _build_analysis_widget(self):
        w = QWidget()
        bl = QHBoxLayout(w)
        bl.setContentsMargins(6, 4, 6, 4)
        leftcol = QVBoxLayout()
        prow = QHBoxLayout()
        prow.addWidget(QLabel('Cross-section'))
        self.prof_axis = QComboBox(); self.prof_axis.addItems(['Row', 'Column'])
        prow.addWidget(self.prof_axis)
        prow.addWidget(QLabel('Pos'))
        self.prof_pos = QSlider(Qt.Orientation.Horizontal)
        self.prof_pos.setRange(0, 1000); self.prof_pos.setValue(500)
        prow.addWidget(self.prof_pos, 1)
        self.prof_show = QCheckBox('Show line'); self.prof_show.setChecked(True)
        prow.addWidget(self.prof_show)
        leftcol.addLayout(prow)
        self.profile = ProfileWidget()
        leftcol.addWidget(self.profile, 1)
        bl.addLayout(leftcol, 1)
        rightcol = QVBoxLayout()
        self.hist = Histogram()
        rightcol.addWidget(self.hist, 1)
        self.stats = QLabel('—')
        rightcol.addWidget(self.stats)
        bl.addLayout(rightcol)
        self.prof_pos.valueChanged.connect(lambda _: self._update_analysis())
        self.prof_axis.currentTextChanged.connect(lambda _: self._update_analysis())
        self.prof_show.toggled.connect(lambda _: self._update_analysis())
        return w

    def _build_docks_analysis(self):
        self.anal_dock = QDockWidget('Analysis', self)
        self.anal_dock.setObjectName('analysisDock')
        self.anal_dock.setWidget(self._build_analysis_widget())
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.anal_dock)
        self.resizeDocks([self.anal_dock], [190], Qt.Orientation.Vertical)

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        rv = QVBoxLayout(central)
        rv.setContentsMargins(4, 4, 4, 4)
        self.tabs = QTabWidget()
        page2 = QWidget(); p2 = QVBoxLayout(page2)
        p2.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        fit_btn = QPushButton('Fit view')
        fit_btn.clicked.connect(lambda: self.canvas.fit())
        self.hint = QLabel('Hover: height · drag: pan · wheel: zoom')
        self.hint.setStyleSheet('color:#8a93a0;')
        bar.addWidget(fit_btn); bar.addStretch(1); bar.addWidget(self.hint)
        p2.addLayout(bar)
        self.canvas = Canvas()
        self.canvas.pixel.connect(self._on_pixel)
        self.canvas.paint_at.connect(self._on_paint_at)
        self.canvas.paint_end.connect(self._on_paint_end)
        p2.addWidget(self.canvas, 1)
        self.view3d = create_3d_view()
        if isinstance(self.view3d, GLView3D):
            self.view3d.failed.connect(self._on_gl_failed)
        self.tabs.addTab(page2, '2D View')
        self.tabs.addTab(self.view3d, '3D Preview')
        self.tabs.currentChanged.connect(self._on_tab)
        rv.addWidget(self.tabs, 1)

    def _build_statusbar(self):
        self.progress = QProgressBar()
        self.progress.setFixedWidth(200); self.progress.hide()
        self.cursor_label = QLabel(''); self.cursor_label.setFixedWidth(200)
        self.size_label = QLabel('')
        self.statusBar().addPermanentWidget(self.cursor_label)
        self.statusBar().addPermanentWidget(self.size_label)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage('Ready')

    def _build_menus(self):
        mb = self.menuBar()
        m_file = mb.addMenu('&File')
        m_file.addAction('Import Heightmap…', self._import_heightmap)
        m_file.addAction('Save Heightmap…', self._save_height)
        m_file.addSeparator()
        m_file.addAction('Open Preset…', self._load_preset)
        m_file.addAction('Save Preset As…', self._save_preset)
        self.menu_recent = m_file.addMenu('Recent presets')
        m_file.addSeparator()
        m_file.addAction('Quit', self.close)
        self._rebuild_recent_menu()

        m_exp = mb.addMenu('&Export')
        m_exp.addAction('For Engine…', self._engine_export)
        m_exp.addAction('Everything (bundle)…', self._export_bundle)
        m_exp.addSeparator()
        m_exp.addAction('Render PNG', self._export_png)
        m_exp.addAction('Heightmap 16-bit', self._export_heightmap)
        m_exp.addAction('Normal map', self._export_normal)
        m_exp.addAction('Raw float32', self._export_raw)
        m_exp.addAction('OBJ Mesh', self._export_obj)

        m_view = mb.addMenu('&View')
        m_view.addAction(self.toolbar.toggleViewAction())
        for d in (self.tool_dock, self.set_dock, self.anal_dock):
            m_view.addAction(d.toggleViewAction())
        m_view.addSeparator()
        m_view.addAction('Reset Layout', self._reset_layout)

        m_help = mb.addMenu('&Help')
        m_help.addAction('Shortcuts & Features (F1)', self._show_help)

    def _reset_layout(self):
        self.settings.remove('winlayout')
        for d in (self.tool_dock, self.set_dock, self.anal_dock):
            d.setFloating(False)
            d.show()
        self.toolbar.show()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.tool_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.set_dock)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.anal_dock)
        self.resizeDocks([self.tool_dock, self.set_dock], [60, 384],
                         Qt.Orientation.Horizontal)
        self.resizeDocks([self.anal_dock], [190], Qt.Orientation.Vertical)

    def _rebuild_recent_menu(self):
        self.menu_recent.clear()
        if not self._recent:
            self.menu_recent.addAction('(empty)').setEnabled(False)
            return
        for path in self._recent[:8]:
            self.menu_recent.addAction(
                path, lambda pp=path: self._load_preset_path(pp))
        self.menu_recent.addSeparator()
        self.menu_recent.addAction('Clear', self._clear_recent)

    def _clear_recent(self):
        self._recent = []
        self.settings.setValue('recent', self._recent)
        self._rebuild_recent_menu()

    def _add_recent(self, path):
        if path in self._recent:
            self._recent.remove(path)
        self._recent.insert(0, path)
        self._recent = self._recent[:8]
        self.settings.setValue('recent', self._recent)
        self._rebuild_recent_menu()

    def _make_shortcuts(self):
        QShortcut(QKeySequence('Ctrl+G'), self, self._generate_now)
        QShortcut(QKeySequence('Ctrl+N'), self, self._new_seed)
        QShortcut(QKeySequence('Ctrl+S'), self, self._export_png)
        QShortcut(QKeySequence('Ctrl+Shift+S'), self, self._export_heightmap)
        QShortcut(QKeySequence('Ctrl+E'), self, self._export_obj)
        QShortcut(QKeySequence('Ctrl+Shift+E'), self, self._engine_export)
        QShortcut(QKeySequence('Ctrl+F'), self, lambda: self.canvas.fit())
        QShortcut(QKeySequence('Ctrl+T'), self,
                  lambda: self.tabs.setCurrentIndex((self.tabs.currentIndex() + 1) % 2))
        QShortcut(QKeySequence('Ctrl+Z'), self, self._undo_sculpt)
        QShortcut(QKeySequence('Ctrl+Y'), self, self._redo_sculpt)
        QShortcut(QKeySequence('Ctrl+Shift+Z'), self, self._redo_sculpt)
        QShortcut(QKeySequence('Ctrl+I'), self, self._import_heightmap)
        QShortcut(QKeySequence('F1'), self, self._show_help)

    # ---------- session ----------

    def _restore_session(self):
        data = self.settings.value('session')
        if data:
            try:
                saved = json.loads(data)
                for k, v in saved.items():
                    if k in self.controls:
                        try:
                            self.controls[k].set(v)
                        except Exception:
                            pass
                self.statusBar().showMessage('Restored last session', 4000)
            except Exception:
                pass
        st = self.settings.value('winlayout')
        if st is not None:
            self.restoreState(st)

    def _save_session(self):
        try:
            self.settings.setValue('winlayout', self.saveState())
            self.settings.setValue('session', json.dumps(self._collect()))
        except Exception:
            pass

    # ---------- live erosion snapshot ----------

    def _on_snapshot(self, arr):
        if self.tabs.currentIndex() != 0 or self.canvas._paint_mode:
            return
        self._snap_pending = arr
        if not self._snap_timer.isActive():
            self._snap_timer.start()

    def _flush_snapshot(self):
        a = self._snap_pending
        self._snap_pending = None
        if a is None:
            return
        p = self._collect()
        img = live_preview_img(a, p)
        self.canvas.set_image(img, fit=False)
        self.size_label.setText(f'{img.width()} × {img.height()} (live)')

    # ---------- layer blend ----------

    def _capture_a(self):
        if self._hmap is None:
            QMessageBox.information(self, 'Layer Blend',
                                    'Generate a terrain first.')
            return
        self._layer_a = self._hmap.copy()
        self.statusBar().showMessage(
            'Layer A captured — change settings and Generate to blend', 6000)
        if self.controls['auto_update'].get():
            self._auto_pending = True
            self._queue_timer.start()

    def _clear_a(self):
        self._layer_a = None
        self._blend_mask_painted = None
        self.statusBar().showMessage('Layer A + painted mask cleared', 3000)

    # ---------- sculpt / splat paint ----------

    def _on_tool(self):
        tool = self.controls['tool'].get()
        self.canvas.set_paint_mode(tool != 'Off')
        if tool == 'Paint Splat':
            self.hint.setText('Left-drag paints the R/G/B/A splat layer · '
                              'preview in "Splat" mode')
        elif tool == 'Paint Blend':
            self.controls['blend_enable'].set(True)
            self.controls['blend_mask'].set('Painted')
            self.hint.setText('Left-drag paints the blend mask (A side) · '
                              'preview in "Blend mask" mode')
        elif tool != 'Off':
            self.hint.setText('Left-drag paints · set Brush=Off to pan')
        else:
            self.hint.setText('Hover: height · drag: pan · wheel: zoom')

    def _reset_splat(self):
        self._splat = None
        self._rerender()
        self.statusBar().showMessage('Splat reset to procedural', 3000)

    def _active_splat(self, p):
        if self._splat is not None:
            return self._splat
        return make_splat_rgba(self._hmap, float(p['sea_level']), p)

    def _brush_radius(self):
        if self._hmap is None:
            return 0
        H, W = self._hmap.shape
        size = float(self.controls['brush_size'].get())
        return max(3.0, size / 100.0 * min(H, W) * 0.5)

    def _on_pixel(self, x, y):
        if self._hmap is None:
            return
        H, W = self._hmap.shape
        if 0 <= y < H and 0 <= x < W:
            hv = float(self._hmap[y, x])
            sea = float(self._collect()['sea_level'])
            kind = 'water' if hv < sea else 'land'
            self.cursor_label.setText(f'x{x} y{y} · h {hv:.3f} ({kind})')
            if self.controls['tool'].get() != 'Off':
                self.canvas.set_brush_cursor(x, y, self._brush_radius())
        else:
            self.canvas.set_brush_cursor(0, 0, 0)

    def _push_undo(self):
        if self._hmap is None:
            return
        self._undo_stack.append((
            self._hmap.copy(),
            None if self._splat is None else self._splat.copy(),
            None if self._blend_mask_painted is None
            else self._blend_mask_painted.copy()))
        if len(self._undo_stack) > 16:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _restore_state(self, h, sp, bm):
        self._hmap = h; self._splat = sp; self._blend_mask_painted = bm
        self._bust_caches(); self._lakes_key = None
        self._recompute_flow()
        self._refresh_blend_preview()
        self._rerender()

    def _undo_sculpt(self):
        if not self._undo_stack or self._hmap is None:
            return
        self._redo_stack.append((
            self._hmap.copy(),
            None if self._splat is None else self._splat.copy(),
            None if self._blend_mask_painted is None
            else self._blend_mask_painted.copy()))
        self._restore_state(*self._undo_stack.pop())
        self.statusBar().showMessage('Undo', 2000)

    def _redo_sculpt(self):
        if not self._redo_stack or self._hmap is None:
            return
        self._undo_stack.append((
            self._hmap.copy(),
            None if self._splat is None else self._splat.copy(),
            None if self._blend_mask_painted is None
            else self._blend_mask_painted.copy()))
        self._restore_state(*self._redo_stack.pop())
        self.statusBar().showMessage('Redo', 2000)

    def _refresh_blend_preview(self):
        p = self._collect()
        if (p.get('blend_enable') and p.get('blend_mask') == 'Painted'
                and self._blend_mask_painted is not None):
            lo = float(p['blend_lo']); hi = max(float(p['blend_hi']), lo + 0.02)
            self._blend_w = smoothstep(lo, hi, self._blend_mask_painted)

    def _ensure_flow(self):
        if self._flow is None and self._hmap is not None:
            p = self._collect()
            self._flow = flow_accumulation(self._hmap,
                                           seed=int(p['seed']) ^ 24635,
                                           sea=float(p['sea_level']))
            _FLOG['id'] = None

    def _get_lakes(self):
        if self._hmap is None:
            return None
        key = id(self._hmap)
        if self._lakes_key != key:
            _f, lake = detect_lakes(self._hmap,
                                    float(self._collect()['sea_level']))
            self._lakes = lake
            self._lakes_key = key
        return self._lakes

    def _bust_caches(self):
        _AO['key'] = None
        _FLOG['id'] = None
        _MOIST['key'] = None

    def _recompute_flow(self):
        self._flow = None
        if self._hmap is None:
            return
        p = self._collect()
        if p.get('rivers'):
            self._flow = flow_accumulation(self._hmap,
                                           seed=int(p['seed']) ^ 24635,
                                           sea=float(p['sea_level']))
            _FLOG['id'] = None

    def _on_paint_at(self, x, y):
        if self._hmap is None:
            return
        H, W = self._hmap.shape
        tool = self.controls['tool'].get()
        if tool == 'Off' or not (0 <= x < W and 0 <= y < H):
            return
        if not getattr(self, '_stroke_active', False):
            self._stroke_active = True
            self._flat_ref = None
            self._push_undo()
        r = self._brush_radius()
        s = float(self.controls['brush_str'].get())
        pres = getattr(self.canvas, 'pressure', 1.0)
        s *= (0.3 + 0.7 * max(0.0, min(1.0, pres)))
        ix0 = max(0, int(x - r)); ix1 = min(W, int(x + r) + 1)
        iy0 = max(0, int(y - r)); iy1 = min(H, int(y + r) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        cy, cx = y - iy0, x - ix0
        dy, dx = np.mgrid[0:iy1 - iy0, 0:ix1 - ix0]
        d2 = (dy - cy) ** 2 + (dx - cx) ** 2
        w = np.clip(1.0 - d2 / (r * r), 0.0, 1.0)
        w = w * w

        if tool == 'Paint Splat':
            p = self._collect()
            if self._splat is None:
                self._splat = make_splat_rgba(self._hmap,
                                              float(p['sea_level']), p)
            li = 'RGBA'.index(self.controls['splat_layer'].get())
            reg = self._splat[iy0:iy1, ix0:ix1, :]
            reg[:, :, li] += w * (0.25 + 1.1 * s)
            tot = self._splat.sum(-1, keepdims=True)
            self._splat /= np.maximum(tot, 1e-6)
            self._render_timer.start()
            return

        if tool == 'Paint Blend':
            if self._blend_mask_painted is None:
                self._blend_mask_painted = np.zeros((H, W), np.float64)
            reg = self._blend_mask_painted[iy0:iy1, ix0:ix1]
            reg += w * (0.3 + 1.2 * s)
            np.clip(reg, 0.0, 1.0, out=reg)
            self._refresh_blend_preview()
            self._render_timer.start()
            return

        h = self._hmap
        region = h[iy0:iy1, ix0:ix1]
        if tool == 'Raise':
            region += (s * 0.02) * w
        elif tool == 'Lower':
            region -= (s * 0.02) * w
        elif tool == 'Smooth':
            pr = np.pad(region, 1, mode='edge')
            bl = (pr[:-2, :-2] + pr[:-2, 1:-1] + pr[:-2, 2:] +
                  pr[1:-1, :-2] + pr[1:-1, 1:-1] + pr[1:-1, 2:] +
                  pr[2:, :-2] + pr[2:, 1:-1] + pr[2:, 2:]) / 9.0
            region += (bl - region) * (s * 0.8 * w)
        elif tool == 'Flatten':
            if self._flat_ref is None:
                self._flat_ref = float(h[int(np.clip(y, 0, H - 1)),
                                         int(np.clip(x, 0, W - 1))])
            region += (self._flat_ref - region) * (s * 0.35 * w)
        np.clip(h, 0.0, 1.0, out=h)
        self._bust_caches()
        self._render_timer.start()

    def _on_paint_end(self):
        self._stroke_active = False
        self._flat_ref = None
        if self._hmap is None:
            return
        self._recompute_flow()
        self._refresh_blend_preview()
        self._rerender()
        self._autosave()
        self.statusBar().showMessage('Stroke applied', 2500)

    # ---------- import / save heightmap ----------

    def _import_heightmap(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Import heightmap', '',
            'Heightmaps (*.png *.jpg *.bmp *.pgm *.raw *.f32)')
        if not path:
            return
        try:
            W = int(self.controls['width'].get())
            Hd = int(self.controls['height'].get())
            h = _cli_load_hmap(path, W, Hd)
        except Exception as e:
            QMessageBox.critical(self, 'Import failed', str(e))
            return
        self._push_undo()
        self._hmap = np.clip(h, 0.0, 1.0)
        self.controls['auto_update'].set(False)
        self._bust_caches(); self._lakes_key = None
        self._recompute_flow()
        self._rerender()
        self._autosave()
        self.statusBar().showMessage(
            f'Imported {path} — auto-update paused (Generate restarts from noise)',
            8000)

    def _save_height(self):
        if self._hmap is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save heightmap', 'terrain_height',
            'PNG 16-bit (*.png);;Raw float32 (*.f32);;Raw 16-bit (*.raw)')
        if not path:
            return
        ext = path.rsplit('.', 1)[-1].lower()
        if ext == 'f32':
            with open(path, 'wb') as f:
                f.write(_f32_bytes(self._hmap))
        elif ext == 'raw':
            _save_raw16(path, self._hmap, False)
        else:
            a16 = (np.clip(self._hmap, 0, 1) * 65535.0 + 0.5).astype('<u2')
            img = QImage(a16.tobytes(), self._hmap.shape[1], self._hmap.shape[0],
                         self._hmap.shape[1] * 2,
                         QImage.Format.Format_Grayscale16).copy()
            img.save(path if ext == 'png' else path + '.png', 'PNG')
        self.statusBar().showMessage(f'Saved {path}', 5000)

    # ---------- params / render ----------

    def _collect(self):
        return {k: w.get() for k, w in self.controls.items()}

    def _on_control(self, key):
        if key in self.RENDER_KEYS:
            self._render_timer.start()
        elif key in self.VIEW3D_KEYS:
            self._v3d_timer.start()
        elif key in self.IGNORE_KEYS:
            pass
        elif self.controls['auto_update'].get():
            self._auto_pending = True
            self._queue_timer.start()
        else:
            self.statusBar().showMessage('Settings changed — press Generate', 2500)

    def _rerender(self):
        if self._hmap is None:
            return
        p = self._collect()
        if p['render_mode'] in ('Wetness', 'Deposition') and self._flow is None:
            self._ensure_flow()
        lakes = self._get_lakes() if p.get('show_lakes') else None
        img, rgb = render_terrain(self._hmap, p, self._flow,
                                  splat=self._splat, blend_w=self._blend_w,
                                  lakes=lakes)
        self._image, self._rgb = img, rgb
        size = (img.width(), img.height())
        fit = (self._last_size != size)
        self.canvas.set_image(img, fit=fit)
        self._last_size = size
        self.size_label.setText(f'{size[0]} × {size[1]}')
        self._update_analysis()
        self._rebuild_3d()

    def _rebuild_3d(self):
        if self._hmap is None or self._rgb is None:
            return
        if self.tabs.currentWidget() is not self.view3d:
            self._dirty3d = True
            return
        self._dirty3d = False
        lakes = self._get_lakes() if self._collect().get('show_lakes') else None
        self.view3d.set_terrain(self._hmap, self._rgb, self._collect(), lakes)

    def _on_tab(self, _):
        if self.tabs.currentWidget() is self.view3d and self._dirty3d:
            self._rebuild_3d()

    def _on_gl_failed(self):
        idx = self.tabs.indexOf(self.view3d)
        if idx < 0:
            return
        self.view3d.deleteLater()
        self.view3d = View3D()
        self.tabs.removeTab(idx)
        self.tabs.insertTab(idx, self.view3d, '3D Preview')
        self.tabs.setCurrentIndex(idx)
        if self._hmap is not None and self._rgb is not None:
            self.view3d.set_terrain(self._hmap, self._rgb, self._collect())
        self.statusBar().showMessage('OpenGL unavailable — software 3D fallback', 5000)

    def _update_analysis(self):
        if self._hmap is None:
            self.profile.set_data(None, 0.3)
            self.hist.set_data(None, 0.3)
            self.stats.setText('—')
            self.canvas.set_profile_line(False)
            return
        h = self._hmap
        p = self._collect()
        sea = float(p['sea_level'])
        H, W = h.shape
        self.hist.set_data(h, sea)
        frac = self.prof_pos.value() / 1000.0
        show = self.prof_show.isChecked() and self._image is not None
        if self.prof_axis.currentText() == 'Row':
            idx = int(round(frac * (H - 1)))
            data = h[idx]
            self.canvas.set_profile_line(show, 0, idx, W - 1, idx)
        else:
            idx = int(round(frac * (W - 1)))
            data = h[:, idx]
            self.canvas.set_profile_line(show, idx, 0, idx, H - 1)
        self.profile.set_data(data, sea)
        water = float((h < sea).mean()) * 100.0
        t = self._timings or {}
        ero = t.get('hydraulic', 0.0) + t.get('thermal', 0.0)
        perf = ''
        if t:
            perf = (f" · noise {t.get('noise', 0):.2f}s"
                    f" · erode {ero:.2f}s"
                    f" · rivers {t.get('flowacc', 0):.2f}s")
        blend = ' · blended' if self._blend_w is not None else ''
        splat = ' · splat painted' if self._splat is not None else ''
        self.stats.setText(
            f"{W}×{H}  min {h.min():.2f}  max {h.max():.2f}\n"
            f"water {water:.0f}%  ·  gen {self._gen_time:.1f}s{perf}"
            f"{blend}{splat}")

    # ---------- generation ----------

    def _generate_now(self):
        self._auto_pending = False
        self._full_timer.stop()
        self._start_generation()

    def _start_generation(self):
        if self._worker is not None or self._er_worker is not None:
            self._pending = True
            self.statusBar().showMessage('Busy — job queued…')
            return
        self._queue_timer.stop()
        self._full_timer.stop()
        params = self._collect()
        if self.controls['blend_enable'].get() and self._layer_a is not None:
            params['_layer_a'] = self._layer_a
        if (params.get('blend_mask') == 'Painted'
                and self._blend_mask_painted is not None):
            params['_blend_mask'] = self._blend_mask_painted
        quick = (self._auto_pending and self.controls['fast_preview'].get()
                 and params['preview_scale'] != '25%'
                 and int(params['width']) * int(params['height']) >= 200000)
        self._auto_pending = False
        self._full_refresh_pending = quick
        if quick:
            params['preview_scale'] = '25%'
        self._thread = QThread(self)
        self._worker = GenWorker(params)
        self._worker.moveToThread(self._thread)
        self._worker.progress.connect(self._on_progress)
        self._worker.snapshot.connect(self._on_snapshot)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._thread.started.connect(self._worker.run)
        self._thread.start()
        self._t0 = time.monotonic()
        self.progress.setValue(0); self.progress.show()
        self.gen_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.statusBar().showMessage(
            'Quick preview…' if quick else 'Generating terrain…')

    def _on_progress(self, v):
        self.progress.setValue(v)

    def _on_done(self, result):
        self._hmap, self._flow, self._timings, self._blend_w = result
        self._splat = None
        self._bust_caches(); self._lakes_key = None
        self._rerender()
        self._cleanup()
        self._gen_time = time.monotonic() - self._t0
        was_quick = self._full_refresh_pending
        self._full_refresh_pending = False
        self._update_analysis()
        self._autosave()
        if self._pending:
            self._pending = False
            self._start_generation()
            return
        if was_quick and self.controls['auto_update'].get():
            self.statusBar().showMessage('Quick preview done — refining…')
            self._full_timer.start()
        else:
            bl = ' (blended)' if self._blend_w is not None else ''
            self.statusBar().showMessage(
                f'Done in {self._gen_time:.2f} s{bl}', 5000)

    def _on_failed(self, tb):
        self._cleanup()
        QMessageBox.critical(self, 'Generation failed', tb)

    def _on_cancelled(self):
        self._cleanup()
        self.statusBar().showMessage('Cancelled', 3000)

    def _cleanup(self):
        if self._thread:
            self._thread.quit(); self._thread.wait()
            self._thread.deleteLater(); self._thread = None
        if self._worker:
            self._worker.deleteLater(); self._worker = None
        self.progress.hide()
        self.gen_btn.setEnabled(True); self.stop_btn.setEnabled(False)

    def _stop(self):
        self._queue_timer.stop()
        self._full_timer.stop()
        if self._worker is not None:
            self._worker.cancel = True
            self.statusBar().showMessage('Cancelling…')

    # ---------- styles / seeds ----------

    def _apply_style(self):
        st = STYLES.get(self.style_combo.currentText())
        if not st:
            return
        for k, v in st.items():
            if k in self.controls:
                try:
                    self.controls[k].set(v)
                except Exception:
                    pass
        self._generate_now()

    def _randomize(self):
        c = self.controls
        c['pattern'].set(random.choice(['fBm', 'fBm', 'Ridged', 'Billow']))
        c['scale'].set(round(random.uniform(1.6, 6.0), 2))
        c['octaves'].set(random.randint(4, 9))
        c['persistence'].set(round(random.uniform(0.4, 0.62), 2))
        c['warp_strength'].set(round(random.uniform(0.0, 0.16), 2))
        c['warp_freq'].set(round(random.uniform(1.0, 4.0), 1))
        c['gamma'].set(round(random.uniform(0.75, 1.35), 2))
        c['mask_type'].set(random.choice(
            ['None', 'None', 'None', 'Round island', 'Archipelago (noise)']))
        c['sea_level'].set(round(random.uniform(0.18, 0.42), 2))
        c['seed'].randomize()

    def _mutate(self):
        c = self.controls
        c['scale'].set(min(12, max(0.5, c['scale'].get() * random.uniform(0.85, 1.18))))
        c['warp_strength'].set(min(0.3, max(0.0, c['warp_strength'].get() +
                                            random.uniform(-0.03, 0.03))))
        c['gamma'].set(min(3, max(0.2, c['gamma'].get() + random.uniform(-0.1, 0.1))))
        c['seed'].randomize()

    def _new_seed(self):
        self.controls['seed'].randomize()
        if not self.controls['auto_update'].get():
            self._generate_now()

    # ---------- apply erosion / auto-contrast ----------

    def _apply_erosion(self, mode):
        if self._hmap is None:
            QMessageBox.information(self, 'Nothing to erode',
                                    'Generate or import a map first.')
            return
        if self._worker is not None or self._er_worker is not None:
            self.statusBar().showMessage('Busy — wait for the current job', 3000)
            return
        p = self._collect()
        if mode.startswith('Hydraulic') and p.get('erosion_method') in (None, 'None'):
            p['erosion_method'] = 'Flow (fast)'
        self._push_undo()
        self._er_thread = QThread(self)
        self._er_worker = ErodeWorker(self._hmap.copy(), p, mode)
        self._er_worker.moveToThread(self._er_thread)
        self._er_worker.progress.connect(self._on_progress)
        self._er_worker.snapshot.connect(self._on_snapshot)
        self._er_worker.done.connect(self._on_erode_done)
        self._er_worker.failed.connect(self._on_erode_failed)
        self._er_thread.started.connect(self._er_worker.run)
        self._er_thread.start()
        self._er_t0 = time.monotonic()
        self.progress.setValue(0); self.progress.show()
        self.apply_hyd_btn.setEnabled(False); self.apply_th_btn.setEnabled(False)
        self.statusBar().showMessage(f'Applying {mode.lower()} erosion…')

    def _on_erode_done(self, h):
        self._hmap = np.clip(h, 0.0, 1.0)
        self._cleanup_erode()
        self._bust_caches(); self._lakes_key = None
        self._recompute_flow()
        self._rerender()
        self._autosave()
        self.statusBar().showMessage(
            f'Erosion applied in {time.monotonic() - self._er_t0:.1f}s '
            '(Ctrl+Z to undo · painted splat kept)', 6000)

    def _on_erode_failed(self, tb):
        self._cleanup_erode()
        QMessageBox.critical(self, 'Erosion failed', tb)

    def _cleanup_erode(self):
        if self._er_thread:
            self._er_thread.quit(); self._er_thread.wait()
            self._er_thread.deleteLater(); self._er_thread = None
        if self._er_worker:
            self._er_worker.deleteLater(); self._er_worker = None
        self.progress.hide()
        if hasattr(self, 'apply_hyd_btn'):
            self.apply_hyd_btn.setEnabled(True)
            self.apply_th_btn.setEnabled(True)

    def _autocontrast(self):
        if self._hmap is None:
            return
        self._push_undo()
        self._hmap = autocontrast(self._hmap)
        self._bust_caches(); self._lakes_key = None
        self._recompute_flow()
        self._rerender()
        self._autosave()
        self.statusBar().showMessage('Auto-contrast applied', 3000)

    # ---------- autosave / recovery ----------

    def _autosave(self):
        if self._hmap is None:
            return
        try:
            data = {'hmap': self._hmap.astype('<f4'),
                    'params': json.dumps(self._collect())}
            if self._splat is not None:
                data['splat'] = self._splat.astype('<f4')
            if self._blend_mask_painted is not None:
                data['blendmask'] = self._blend_mask_painted.astype('<f4')
            if self._layer_a is not None:
                data['layera'] = self._layer_a.astype('<f4')
            with open(self._autosa_path, 'wb') as f:
                np.savez_compressed(f, **data)
        except Exception:
            pass

    def _prompt_recovery(self):
        p = self._autosa_path
        try:
            if not os.path.exists(p):
                return
            if (time.time() - os.path.getmtime(p)) / 3600.0 > 48:
                return
            z = np.load(p)
            if 'hmap' not in z.files:
                return
        except Exception:
            return
        mb = QMessageBox(self)
        mb.setWindowTitle('Terrain Forge')
        mb.setText('Found an autosaved terrain from a previous session.\n'
                   'Recover it now?')
        yes = mb.addButton('Recover', QMessageBox.ButtonRole.AcceptRole)
        mb.addButton('Discard', QMessageBox.ButtonRole.DestructiveRole)
        mb.exec()
        if mb.clickedButton() is not yes:
            try:
                os.remove(p)
            except Exception:
                pass
            return
        try:
            z = np.load(p)
            self._hmap = np.clip(z['hmap'].astype(np.float64), 0.0, 1.0)
            self._splat = (np.clip(z['splat'].astype(np.float64), 0.0, 1.0)
                           if 'splat' in z.files else None)
            self._blend_mask_painted = (z['blendmask'].astype(np.float64)
                                        if 'blendmask' in z.files else None)
            self._layer_a = (z['layera'].astype(np.float64)
                             if 'layera' in z.files else None)
            self._recovered = True
            self.controls['auto_update'].set(False)
            self._bust_caches(); self._lakes_key = None
            self._refresh_blend_preview()
            self._recompute_flow()
            self._rerender()
            self.statusBar().showMessage(
                'Recovered autosaved terrain — auto-update paused '
                '(Generate restarts from noise)', 9000)
        except Exception:
            pass

    # ---------- help ----------

    def _show_help(self):
        QMessageBox.information(self, 'Terrain Forge — Help', HELP_TEXT)

    # ---------- engine export ----------

    def _engine_export(self):
        if self._hmap is None:
            QMessageBox.information(self, 'Nothing to export',
                                    'Generate a terrain first.')
            return
        EngineExportDialog(self).exec()

    # ---------- everything bundle ----------

    def _export_bundle(self):
        if self._hmap is None:
            QMessageBox.information(self, 'Nothing to export',
                                    'Generate a terrain first.')
            return
        base = QFileDialog.getExistingDirectory(self, 'Choose bundle folder')
        if not base:
            return
        sub = os.path.join(base, f'TerrainForge_bundle_'
                                 f"{int(self._collect()['seed'])}")
        os.makedirs(sub, exist_ok=True)
        p = self._collect()
        h = self._hmap
        H, W = h.shape
        world, hmet = 1000.0, 120.0
        sea = float(p['sea_level'])
        try:
            self._image.save(os.path.join(sub, 'render.png'), 'PNG')
            p2 = dict(p); p2.update(render_mode='Shaded', shade_strength=0.0,
                                    ao=0.0, contours=False)
            _, alb = render_terrain(h, p2, self._flow)
            _save_rgb_png(os.path.join(sub, 'albedo.png'), alb)
            a16 = (np.clip(h, 0, 1) * 65535.0 + 0.5).astype('<u2')
            img = QImage(a16.tobytes(), W, H, W * 2,
                         QImage.Format.Format_Grayscale16).copy()
            img.save(os.path.join(sub, 'heightmap_16bit.png'), 'PNG')
            with open(os.path.join(sub, 'heightmap.f32'), 'wb') as f:
                f.write(_f32_bytes(h))
            _save_rgb_png(os.path.join(sub, 'normalmap.png'),
                          normal_map_rgb(h, float(p['relief'])))
            write_glb(os.path.join(sub, 'terrain.glb'), h, alb, world, hmet, 2,
                      sea01=sea, water=sea > 0.015 and bool((h < sea).any()),
                      texture=True, vcol=True, albedo_png=_png_bytes(alb))
            self._ensure_flow()
            wet, dep, cur = _data_masks(h, self._flow, sea)
            _save_gray_png(os.path.join(sub, 'mask_wetness.png'), wet)
            _save_gray_png(os.path.join(sub, 'mask_deposition.png'), dep)
            _save_gray_png(os.path.join(sub, 'mask_curvature.png'),
                           (cur + 1.0) * 0.5)
            _f, lk = detect_lakes(h, sea)
            _save_gray_png(os.path.join(sub, 'mask_lakes.png'),
                           lk.astype(np.float64))
            _save_gray_png(os.path.join(sub, 'mask_walkable.png'),
                           walkable_mask(h, sea, float(p.get('walk_slope', 0.10))))
            with open(os.path.join(sub, 'metadata.json'), 'w') as f:
                f.write(_meta_json(p, max(H, W), world, hmet,
                                   ['grass', 'rock', 'sand', 'snow'],
                                   'sea-anchored', {'target': 'bundle'}))
            with open(os.path.join(sub, 'preset.json'), 'w') as f:
                json.dump(p, f, indent=2)
        except Exception:
            QMessageBox.critical(self, 'Bundle failed', traceback.format_exc())
            return
        box = QMessageBox(self)
        box.setWindowTitle('Bundle complete')
        box.setText(f'Everything exported to:\n{sub}')
        ob = box.addButton('Open Folder', QMessageBox.ButtonRole.ActionRole)
        box.addButton('OK', QMessageBox.ButtonRole.AcceptRole)
        box.exec()
        if box.clickedButton() is ob:
            QDesktopServices.openUrl(QUrl.fromLocalFile(sub))

    # ---------- quick exports ----------

    def _warn_no_map(self):
        if self._hmap is None:
            QMessageBox.information(self, 'Nothing to export', 'Generate a terrain first.')
            return False
        return True

    def _export_png(self):
        if not self._warn_no_map():
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save render', 'terrain.png', 'PNG (*.png)')
        if path:
            self._image.save(path, 'PNG')
            self.statusBar().showMessage(f'Saved {path}', 5000)

    def _export_heightmap(self):
        if not self._warn_no_map():
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save 16-bit heightmap',
                                              'terrain_height.png', 'PNG (*.png)')
        if path:
            a16 = (np.clip(self._hmap, 0, 1) * 65535.0 + 0.5).astype('<u2')
            img = QImage(a16.tobytes(), a16.shape[1], a16.shape[0],
                         a16.shape[1] * 2, QImage.Format.Format_Grayscale16).copy()
            img.save(path, 'PNG')
            self.statusBar().showMessage(f'Saved {path}', 5000)

    def _export_normal(self):
        if not self._warn_no_map():
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save normal map',
                                              'terrain_normal.png', 'PNG (*.png)')
        if path:
            _save_rgb_png(path, normal_map_rgb(self._hmap,
                                               float(self._collect()['relief'])))
            self.statusBar().showMessage(f'Saved {path}', 5000)

    def _export_raw(self):
        if not self._warn_no_map():
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save raw heightmap',
                                              'terrain.f32', 'Raw float32 (*.f32 *.raw)')
        if path:
            with open(path, 'wb') as f:
                f.write(_f32_bytes(self._hmap))
            H, W = self._hmap.shape
            self.statusBar().showMessage(
                f'Saved {path} ({W}×{H} float32, row-major)', 5000)

    def _export_obj(self):
        if not self._warn_no_map():
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export OBJ', 'terrain.obj', 'OBJ (*.obj)')
        if not path:
            return
        p = self._collect()
        step = max(1, int(p['obj_step'])); vs = float(p['obj_vscale'])
        H, W = self._hmap.shape
        if math.ceil(H / step) * math.ceil(W / step) > 600000:
            QMessageBox.warning(self, 'Mesh too large', 'Increase the OBJ grid step.')
            return
        write_obj_scaled(path, self._hmap, self._rgb, 100.0 * vs, 30.0 * vs, step)
        self.statusBar().showMessage(f'Exported OBJ (+normals): {path}', 5000)

    # ---------- presets ----------

    def _save_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save preset', 'preset.json', 'JSON (*.json)')
        if path:
            with open(path, 'w') as f:
                json.dump(self._collect(), f, indent=2)
            self._add_recent(path)
            self.statusBar().showMessage(f'Saved preset {path}', 5000)

    def _load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load preset', '', 'JSON (*.json)')
        if path:
            self._load_preset_path(path)

    def _load_preset_path(self, path):
        try:
            with open(path) as f:
                data = json.load(f)
            for k, v in data.items():
                if k in self.controls:
                    try:
                        self.controls[k].set(v)
                    except Exception:
                        pass
            self._add_recent(path)
            self.statusBar().showMessage(f'Loaded preset {path}', 5000)
            if self.controls['auto_update'].get():
                self._generate_now()
        except Exception as e:
            QMessageBox.critical(self, 'Load failed', str(e))

    # ---------- lifecycle ----------

    def showEvent(self, e):
        super().showEvent(e)
        if not self._booted:
            self._booted = True
            if not self._recovered:
                self._auto_pending = True
                self._start_generation()

    def closeEvent(self, e):
        self._save_session()
        if self._er_worker is not None:
            self._er_thread.quit()
            self._er_thread.wait(3000)
        if self._worker is not None:
            self._worker.cancel = True
            if self._thread:
                self._thread.quit()
                self._thread.wait(3000)
        e.accept()


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def _warm_numba():
    if not HAVE_NUMBA:
        return
    import threading

    def go():
        try:
            a = np.zeros((16, 16), np.float64)
            _droplets_core(a, 12, 8, 0.06, 3.0, 0.35, 0.3, 0.03, 12345)
        except Exception:
            pass
    threading.Thread(target=go, daemon=True).start()


def main():
    if '--cli' in sys.argv:
        cli_args = [a for a in sys.argv[1:] if a != '--cli']
        sys.exit(run_cli(cli_args))
    if HAVE_GL:
        from PySide6.QtGui import QSurfaceFormat
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        QSurfaceFormat.setDefaultFormat(fmt)
    _warm_numba()
    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()