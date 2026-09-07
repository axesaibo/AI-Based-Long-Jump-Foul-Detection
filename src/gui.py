"""
Long Jump Foul Detection System — Enhanced Modern tkinter GUI.

Dark theme with gradient accents, hover animations, color-coded status,
real-time FPS, grid overlay, and animated UI elements.

Run:
    python src/gui.py
"""

import cv2
import os
import sys
import time
import math
import threading
import queue
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from datetime import datetime

# Add src directory to path
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

from detector import ShoeDetector
from foul_checker import FoulChecker
from camera import Camera
from utils import (
    load_config,
    get_foul_line_from_config,
    FoulLine,
    FoulEvent,
    FrameBuffer,
    create_video_writer,
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
DISPLAY_W = 1100

# Black-frame detection thresholds
_BLACK_MEAN_THRESH = 20.0
_BLACK_ZERO_RATIO = 0.95
_MAX_FRAMES_TO_SCAN = 60


def _first_visible_frame(cap, max_scan=_MAX_FRAMES_TO_SCAN):
    """Skip initial black frames common in H.264/H.265 encoded videos."""
    fallback = None
    for i in range(max_scan):
        ok, frame = cap.read()
        if not ok:
            break
        if fallback is None:
            fallback = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = float(gray.mean())
        if mean_val >= _BLACK_MEAN_THRESH:
            return frame, i, ""
        zero_ratio = float(np.sum(gray < 10)) / gray.size
        if zero_ratio < _BLACK_ZERO_RATIO:
            return frame, i, ""
    if fallback is not None:
        return fallback, 0, (
            f"Could not find a non-black frame within the first "
            f"{max_scan} frames.  Displaying frame 0.")
    return None, 0, "Cannot read any frames from the video."


# ===================================================================
#  GEOMETRY HELPERS
# ===================================================================

def _order_corners(box):
    """TL, TR, BR, BL ordering via centroid-angle sort."""
    pts = np.float32(box).reshape(-1, 2)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    order = np.argsort(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    return np.float32([pts[order[1]], pts[order[2]],
                       pts[order[3]], pts[order[0]]])


def _warp_point(pt, M):
    p = np.float32([[pt[0], pt[1]]]).reshape(-1, 1, 2)
    r = cv2.perspectiveTransform(p, M)[0][0]
    return float(r[0]), float(r[1])


# ===================================================================
#  THEME & COLOUR PALETTE
# ===================================================================

class C:
    """Centralised colour palette in BGR order (OpenCV-native)."""
    BG           = (23, 17, 13)
    SURFACE      = (34, 27, 22)
    SURFACE_ALT  = (46, 37, 30)
    BORDER       = (61, 54, 48)
    ACCENT       = (255, 166, 88)
    ACCENT2      = (255, 140, 188)
    SUCCESS      = (80, 185, 63)
    WARNING      = (34, 153, 210)
    DANGER       = (73, 81, 248)
    TEXT          = (243, 237, 230)
    TEXT_DIM      = (158, 148, 139)
    ROI_GREEN    = (136, 255, 0)
    FOUL_RED     = (87, 71, 255)


def _mix(c1, c2, t):
    """Linearly interpolate between two BGR colours."""
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _hex(bgr):
    """Convert a BGR tuple to an RGB hex string for tkinter."""
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"


# ===================================================================
#  CUSTOM HOVER BUTTON
# ===================================================================

class _HoverButton(tk.Canvas):
    """Rounded button with smooth hover colour-shift animation."""

    def __init__(self, parent, text, command, *,
                 bg=None, hover=None, fg=(255, 255, 255),
                 disabled_fg=(70, 75, 82),
                 font=("Segoe UI", 10, "bold"),
                 w=150, h=38, r=8, **kw):
        self._bw = w
        self._bh = h
        super().__init__(parent, width=w, height=h,
                         highlightthickness=0, bg=_hex(C.BG))
        self._cmd = command
        self._base = bg or C.ACCENT
        self._hover_c = hover or _mix(self._base, (255, 255, 255), 0.22)
        self._fg = fg
        self._dis_fg = disabled_fg
        self._font = font
        self._r = r
        self._text = text
        self._enabled = True
        self._t = 0.0           # animation progress 0..1
        self._aid = None
        self._draw()
        self.bind("<Enter>", self._on_in)
        self.bind("<Leave>", self._on_out)
        self.bind("<Button-1>", self._on_click)

    # -- drawing ---------------------------------------------------
    def _draw(self):
        self.delete("all")
        x1, y1 = 1, 1
        x2, y2 = self._bw - 1, self._bh - 1
        r = self._r
        # Rounded polygon points for shadow + button body
        pts = []
        for a in range(180, 271, 10):
            rad = math.radians(a)
            pts.append([x1 + r + r * math.cos(rad), y1 + r + r * math.sin(rad)])
        for a in range(270, 361, 10):
            rad = math.radians(a)
            pts.append([x2 - r + r * math.cos(rad), y1 + r + r * math.sin(rad)])
        for a in range(0, 91, 10):
            rad = math.radians(a)
            pts.append([x2 - r + r * math.cos(rad), y2 - r + r * math.sin(rad)])
        for a in range(90, 181, 10):
            rad = math.radians(a)
            pts.append([x1 + r + r * math.cos(rad), y2 - r + r * math.sin(rad)])
        flat = [v for p in pts for v in p]
        # Drop shadow (offset darker polygon)
        shadow = [v + (0 if i % 2 else 0) + (3 if i % 2 else 1)
                  for i, v in enumerate(flat)]
        sc = _mix(C.BG, (0, 0, 0), 0.6)
        self.create_polygon(shadow, fill=_hex(sc), outline="", smooth=True)
        # Button body
        col = _mix(self._base, self._hover_c, self._t)
        self.create_polygon(flat, fill=_hex(col), outline="", smooth=True)
        tc = self._fg if self._enabled else self._dis_fg
        self.create_text(self._bw // 2, self._bh // 2,
                         text=self._text, fill=_hex(tc), font=self._font)

    # -- animation -------------------------------------------------
    def _on_in(self, _e=None):
        if not self._enabled:
            return
        self._cancel()
        self._run(1.0)

    def _on_out(self, _e=None):
        self._cancel()
        self._run(0.0)

    def _cancel(self):
        if self._aid:
            self.after_cancel(self._aid)
            self._aid = None

    def _run(self, target):
        d = target - self._t
        if abs(d) < 0.03:
            self._t = target
            self._draw()
            return
        self._t += 0.18 * d
        self._draw()
        self._aid = self.after(14, lambda: self._run(target))

    def _on_click(self, _e=None):
        if not self._enabled:
            return
        self._t = 1.0
        self._draw()
        self.after(90, lambda: self._run(0.0))
        if self._cmd:
            self._cmd()

    # -- public API ------------------------------------------------
    def configure(self, **kw):
        if "text" in kw:
            self._text = kw["text"]
        if "state" in kw:
            self._enabled = (kw["state"] != tk.DISABLED)
        if "command" in kw:
            self._cmd = kw["command"]
        self._draw()


# ===================================================================
#  TOOLTIP
# ===================================================================

class _Tip:
    """Minimal tooltip that appears after a short delay."""

    def __init__(self, widget, text, delay=600):
        self._w = widget
        self._txt = text
        self._delay = delay
        self._id = None
        self._tip = None
        widget.bind("<Enter>", self._sched, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _sched(self, _e=None):
        self._id = self._w.after(self._delay, self._show)

    def _show(self):
        x = self._w.winfo_rootx() + 10
        y = self._w.winfo_rooty() + self._w.winfo_height() + 4
        self._tip = tk.Toplevel(self._w)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(self._tip, text=self._txt, bg="#1e2430", fg="#c9d1d9",
                       font=("Segoe UI", 9), padx=8, pady=3,
                       highlightbackground="#30363d", highlightthickness=1)
        lbl.pack()

    def _hide(self, _e=None):
        if self._id:
            self._w.after_cancel(self._id)
            self._id = None
        if self._tip:
            self._tip.destroy()
            self._tip = None


# ===================================================================
#  MAIN APPLICATION
# ===================================================================

class FoulDetectionApp:
    """Single-window enhanced tkinter application for foul detection."""

    def __init__(self, root):
        self.root = root
        self.root.title("Long Jump Foul Detection System")
        self.root.geometry("1300x900")
        self.root.minsize(800, 550)
        self.root.configure(bg=_hex(C.BG))

        # ---------- project config ----------
        cfg_path = os.path.join(_PROJECT_ROOT, "config.yaml")
        self.config = load_config(cfg_path)

        # ---------- state ----------
        self.mode = "idle"
        self.source_path = None
        self.frame = None
        self.display_frame = None
        self._photo = None
        self.roi_corners = []
        self.foul_pts = []
        self._scale = 1.0
        self.foul_events = []
        self._after_id = None
        self._fps_val = 0.0
        self._det_count = 0
        self._foul_count = 0
        self._placeholder_rot = 0.0
        self._placeholder_aid = None

        self._hdr_last_w = 0

        # ---------- UI (build first so user sees loading screen) ----------
        self._build_ui()
        self.root.update()
        self._show_loading_screen()
        # Defer heavy model loading so the loading screen renders first
        self.root.after(120, self._load_model)

    def _show_loading_screen(self):
        """Display an animated loading splash before model init."""
        self.canvas.delete("all")
        cw = self.canvas.winfo_width() or 1100
        ch = self.canvas.winfo_height() or 650
        self._draw_grid(cw, ch)
        cx, cy = cw // 2, ch // 2
        self.canvas.create_text(cx, cy - 40, text="\u26a1",
                                fill=_hex(C.ACCENT),
                                font=("Segoe UI Emoji", 28))
        self.canvas.create_text(
            cx, cy + 5,
            text="Initializing AI Model\u2026",
            fill=_hex(C.TEXT), font=("Segoe UI", 16, "bold"))
        self.canvas.create_text(
            cx, cy + 30,
            text="Loading shoe segmentation network",
            fill=_hex(C.TEXT_DIM), font=("Segoe UI", 10))
        bw, bh = 260, 6
        self._ld_bar_bg = self.canvas.create_rectangle(
            cx - bw // 2, cy + 50, cx + bw // 2, cy + 50 + bh,
            fill=_hex(C.SURFACE_ALT), outline="")
        self._ld_bar = self.canvas.create_rectangle(
            cx - bw // 2, cy + 50, cx - bw // 2 + 2, cy + 50 + bh,
            fill=_hex(C.ACCENT), outline="")
        self._ld_pulse = 0.0
        self._animate_loading()

    def _animate_loading(self):
        """Indeterminate progress bar animation."""
        self._ld_pulse += 0.06
        t = (math.sin(self._ld_pulse * 2.0) + 1.0) / 2.0
        col = _mix(C.ACCENT, C.ACCENT2, t)
        if hasattr(self, '_ld_bar') and self.canvas.winfo_exists():
            self.canvas.itemconfigure(self._ld_bar, fill=_hex(col))
            cw = self.canvas.winfo_width() or 1100
            cx = cw // 2
            bw = 260
            w = int(40 + 180 * ((math.sin(self._ld_pulse * 1.5) + 1) / 2))
            self.canvas.coords(self._ld_bar,
                               cx - bw // 2, 0, cx - bw // 2 + w, 0)
            # Restore y coords (coords resets all 4)
            cy = self.canvas.winfo_height() // 2 if self.canvas.winfo_height() > 10 else 325
            self.canvas.coords(self._ld_bar,
                               cx - bw // 2, cy + 50,
                               cx - bw // 2 + w, cy + 50 + 6)
            self._ld_aid = self.root.after(30, self._animate_loading)

    def _load_model(self):
        """Load the ML model after the UI has rendered."""
        if hasattr(self, '_ld_aid'):
            self.root.after_cancel(self._ld_aid)

        model_cfg = self.config.get("model", {})
        det_cfg = self.config.get("detection", {})

        weights = model_cfg.get("weights", "models/shoe_seg.pt")
        if not os.path.isabs(weights):
            weights = os.path.join(_PROJECT_ROOT, weights)

        self.detector = ShoeDetector(
            weights_path=weights,
            confidence=model_cfg.get("confidence_threshold", 0.5),
            iou=model_cfg.get("iou_threshold", 0.45),
            device=model_cfg.get("device", "auto"),
            imgsz=model_cfg.get("imgsz", 640),
            class_name=det_cfg.get("class_name", "shoe"),
        )
        if not self.detector.load():
            messagebox.showerror("Error", "Could not load the model.\n"
                                 "Make sure models/shoe_seg.pt exists.")
            self.root.destroy()
            return

        self._start_header_anim()
        self._start_placeholder_anim()
        self._show_placeholder()

    # ---------------------------------------------------------------
    #  UI CONSTRUCTION
    # ---------------------------------------------------------------
    def _build_ui(self):
        # ---- Header ----
        hf = tk.Frame(self.root, height=62, bg=_hex(C.BG))
        hf.pack(fill=tk.X)
        hf.pack_propagate(False)

        self._hdr_canvas = tk.Canvas(hf, height=62, highlightthickness=0,
                                     bg=_hex(C.BG))
        self._hdr_canvas.pack(fill=tk.X)
        self._hdr_pulse = 0.0
        self._hdr_bars = []

        # Draw header when canvas is actually sized (and on every resize)
        self._hdr_canvas.bind("<Configure>", self._on_header_resize)

        # ---- Toolbar ----
        tbf = tk.Frame(self.root, bg=_hex(C.BG))
        tbf.pack(fill=tk.X, padx=14, pady=(6, 2))
        # accent border on top
        tk.Frame(tbf, height=2, bg=_hex(C.ACCENT)).pack(fill=tk.X, pady=(0, 8))

        tb = tk.Frame(tbf, bg=_hex(C.BG))
        tb.pack(fill=tk.X)

        self.btn_upload = _HoverButton(
            tb, text="\u2b06  Upload Video", command=self._on_upload,
            bg=C.SUCCESS, w=165, h=38)
        self.btn_upload.pack(side=tk.LEFT, padx=(0, 6))
        _Tip(self.btn_upload, "Load a video file for foul analysis")

        self.btn_cam = _HoverButton(
            tb, text="\U0001f4f7  Live Camera", command=self._on_camera,
            bg=C.ACCENT, w=165, h=38)
        self.btn_cam.pack(side=tk.LEFT, padx=(0, 6))
        _Tip(self.btn_cam, "Start live webcam feed")

        tk.Frame(tb, width=16, bg=_hex(C.BG)).pack(side=tk.LEFT)

        self.btn_reset = _HoverButton(
            tb, text="\u21ba  Reset", command=self._on_reset,
            bg=C.SURFACE_ALT, w=110, h=38,
            font=("Segoe UI", 10))
        self.btn_reset.pack(side=tk.LEFT, padx=(0, 6))
        _Tip(self.btn_reset, "Clear all selections and reset")

        self.btn_start = _HoverButton(
            tb, text="\u25b6  Start Detection", command=self._on_start,
            bg=C.DANGER, w=195, h=38)
        self.btn_start.pack(side=tk.RIGHT)
        self.btn_start.configure(state=tk.DISABLED)
        _Tip(self.btn_start, "Begin AI-powered foul detection")

        # ---- Canvas area ----
        cf = tk.Frame(self.root, width=1100, height=650, bg=_hex(C.BG))
        cf.pack(fill=tk.BOTH, expand=True, padx=14, pady=(6, 4))
        cf.pack_propagate(False)

        self.canvas = tk.Canvas(cf, bg=_hex(C.BG), highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_click)

        # ---- Status bar ----
        sb = tk.Frame(self.root, bg=_hex(C.SURFACE), height=38)
        sb.pack(fill=tk.X, padx=0, pady=0)
        sb.pack_propagate(False)

        inner = tk.Frame(sb, bg=_hex(C.SURFACE))
        inner.pack(fill=tk.BOTH, expand=True, padx=14)

        self._dot = tk.Label(inner, text="\u25cf", fg=_hex(C.SUCCESS),
                             bg=_hex(C.SURFACE), font=("Segoe UI Symbol", 13))
        self._dot.pack(side=tk.LEFT, padx=(0, 6))

        self.status_var = tk.StringVar(
            value="Ready \u2014 upload a video or start the camera")
        tk.Label(inner, textvariable=self.status_var, fg=_hex(C.TEXT),
                 bg=_hex(C.SURFACE),
                 font=("Segoe UI", 10)).pack(side=tk.LEFT)

        self._foul_lbl = tk.Label(inner, text="", fg=_hex(C.DANGER),
                                  bg=_hex(C.SURFACE),
                                  font=("Consolas", 10, "bold"))
        self._foul_lbl.pack(side=tk.RIGHT, padx=(10, 0))

        self._fps_lbl = tk.Label(inner, text="FPS: \u2014", fg=_hex(C.TEXT_DIM),
                                 bg=_hex(C.SURFACE),
                                 font=("Consolas", 10, "bold"))
        self._fps_lbl.pack(side=tk.RIGHT, padx=(10, 0))

        self.mode_var = tk.StringVar(value="\u25c9 Idle")
        self._mode_lbl = tk.Label(inner, textvariable=self.mode_var,
                                  fg=_hex(C.ACCENT), bg=_hex(C.SURFACE),
                                  font=("Segoe UI", 10, "bold"))
        self._mode_lbl.pack(side=tk.RIGHT)

    def _on_header_resize(self, event):
        """Redraw header whenever the canvas gets its real (or new) width."""
        cw = event.width
        if cw < 10 or cw == self._hdr_last_w:
            return
        self._hdr_last_w = cw
        self._draw_header_title()

    def _draw_header_title(self):
        """Draw title text and gradient bars on the header canvas."""
        cw = self._hdr_canvas.winfo_width() or self.root.winfo_width()
        if cw < 10:
            return
        self._hdr_canvas.delete("all")
        ch = 62
        mid = ch // 2
        # Pre-create gradient bar segments spanning the full width
        n_bars = 40
        seg_w = max(cw // n_bars + 1, 2)
        self._hdr_bars = []
        for i in range(n_bars):
            x = i * (cw / n_bars)
            bar = self._hdr_canvas.create_rectangle(
                x, ch - 3, x + seg_w, ch,
                fill=_hex(C.ACCENT), outline="")
            self._hdr_bars.append(bar)
        self._hdr_canvas.create_text(
            cw // 2, mid - 10,
            text="\u26a1  LONG JUMP FOUL DETECTION SYSTEM",
            fill=_hex(C.ACCENT), font=("Segoe UI", 16, "bold"))
        self._hdr_canvas.create_text(
            cw // 2, mid + 14,
            text="AI-Powered Analysis",
            fill=_hex(C.TEXT_DIM), font=("Segoe UI", 9))

    # ---------------------------------------------------------------
    #  HEADER PULSE ANIMATION
    # ---------------------------------------------------------------
    def _start_header_anim(self):
        self._hdr_pulse = 0.0
        self._pulse_tick()

    def _pulse_tick(self):
        self._hdr_pulse += 0.04
        n = len(self._hdr_bars)
        for i, bar in enumerate(self._hdr_bars):
            phase = self._hdr_pulse + i * 0.18
            t = (math.sin(phase) + 1.0) / 2.0
            c = _mix(_mix(C.BG, C.ACCENT, 0.3),
                     _mix(C.ACCENT, C.ACCENT2, t * 0.3), t)
            self._hdr_canvas.itemconfigure(bar, fill=_hex(c))
        self.root.after(45, self._pulse_tick)

    # ---------------------------------------------------------------
    #  CANVAS HELPERS
    # ---------------------------------------------------------------
    def _set_mode(self, mode, text=""):
        self.mode = mode
        label = mode.replace("_", " ").title()
        colours = {
            "idle": C.TEXT_DIM,
            "select_roi": C.WARNING,
            "select_foul": C.WARNING,
            "running": C.SUCCESS,
        }
        c = colours.get(mode, C.ACCENT)
        self._mode_lbl.config(fg=_hex(c))
        self.mode_var.set(f"\u25c9 {label}")
        self.status_var.set(text)
        # Status dot
        dots = {"idle": C.SUCCESS, "select_roi": C.WARNING,
                "select_foul": C.WARNING, "running": C.ACCENT}
        self._dot.config(fg=_hex(dots.get(mode, C.SUCCESS)))

    def _draw_grid(self, cw, ch):
        """Subtle reference grid on empty canvas."""
        sp = 50
        gc = _hex(C.SURFACE)
        for x in range(sp, cw, sp):
            self.canvas.create_line(x, 0, x, ch, fill=gc, width=1)
        for y in range(sp, ch, sp):
            self.canvas.create_line(0, y, cw, y, fill=gc, width=1)
        cc = _hex(C.SURFACE_ALT)
        self.canvas.create_line(cw // 2, 0, cw // 2, ch, fill=cc, width=1)
        self.canvas.create_line(0, ch // 2, cw, ch // 2, fill=cc, width=1)

    def _show_placeholder(self):
        """Instruction screen on the dark canvas."""
        self.canvas.delete("all")
        cw = self.canvas.winfo_width() or 1100
        ch = self.canvas.winfo_height() or 650
        self._draw_grid(cw, ch)

        # rotating decorative ring
        cx, cy, rad = cw // 2, ch // 2 - 30, 52
        rot = self._placeholder_rot
        for a in range(0, 360, 8):
            t = (math.sin(math.radians(a) + rot * 2) + 1) / 2
            alpha = 0.3 + 0.7 * t
            col = _mix(C.BG, C.ACCENT, alpha)
            r1 = math.radians(a + rot)
            r2 = math.radians(a + 4 + rot)
            self.canvas.create_line(
                cx + rad * math.cos(r1), cy + rad * math.sin(r1),
                cx + rad * math.cos(r2), cy + rad * math.sin(r2),
                fill=_hex(col), width=3)

        self.canvas.create_text(
            cx, cy - 4,
            text="\u26a1", fill=_hex(C.ACCENT),
            font=("Segoe UI Emoji", 22))
        self.canvas.create_text(
            cx, cy + 50,
            text="Upload a video or start the camera",
            fill=_hex(C.TEXT_DIM), font=("Segoe UI", 17))
        self.canvas.create_text(
            cx, cy + 82,
            text="to begin AI-powered foul detection",
            fill=_hex(C.TEXT_DIM), font=("Segoe UI", 11))

    def _start_placeholder_anim(self):
        """Start the rotating ring animation on the placeholder."""
        self._placeholder_rot = 0.0
        self._rotate_placeholder()

    def _rotate_placeholder(self):
        if self.mode != "idle" or self.frame is not None:
            return
        self._placeholder_rot += 1.5
        self._show_placeholder()
        self._placeholder_aid = self.root.after(50, self._rotate_placeholder)

    def _show_frame(self, frame, overlay_fn=None):
        """Resize BGR frame to canvas, apply overlay, blit."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 1100, 650

        fh, fw = frame.shape[:2]
        scale = min(cw / fw, ch / fh, 1.0)
        dw, dh = int(fw * scale), int(fh * scale)

        disp = cv2.resize(frame, (dw, dh))
        if overlay_fn:
            overlay_fn(disp, scale)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)

        # Subtle border around displayed frame
        ox = (cw - dw) // 2
        oy = (ch - dh) // 2
        self.canvas.create_rectangle(
            ox - 1, oy - 1, ox + dw, oy + dh,
            outline=_hex(C.BORDER), width=1)
        # Draw selection progress badge if in selection mode
        if self.mode in ("select_roi", "select_foul"):
            self._draw_selection_badge()
        return scale

    def _draw_selection_badge(self):
        """Small progress badge in the top-left corner during selection."""
        if self.mode == "select_roi":
            n = len(self.roi_corners)
            total = 4
            label = "ROI"
            col = C.ROI_GREEN
        elif self.mode == "select_foul":
            n = len(self.foul_pts)
            total = 2
            label = "FOUL LINE"
            col = C.FOUL_RED
        else:
            return
        cx, cy = 38, 38
        self.canvas.create_oval(cx - 20, cy - 20, cx + 20, cy + 20,
                                fill=_hex(col), outline=_hex(C.BG), width=2)
        self.canvas.create_text(cx, cy - 3, text=str(n),
                                fill=_hex(C.BG),
                                font=("Segoe UI", 14, "bold"))
        self.canvas.create_text(cx, cy + 10, text=f"/{total}",
                                fill=_hex(C.BG),
                                font=("Segoe UI", 7, "bold"))
        self.canvas.create_text(cx + 30, cy, text=label,
                                fill=_hex(col), anchor="w",
                                font=("Segoe UI", 9, "bold"))

    def _draw_info_panel(self, disp, scale):
        """Semi-transparent overlay with mode, frame, FPS, detections."""
        dh, dw = disp.shape[:2]
        pw, ph, m = 215, 105, 10
        x0, y0 = dw - pw - m, m

        panel = disp.copy()
        cv2.rectangle(panel, (x0, y0), (x0 + pw, y0 + ph), C.SURFACE, -1)
        cv2.addWeighted(panel, 0.82, disp, 0.18, 0, disp)
        cv2.rectangle(disp, (x0, y0), (x0 + pw, y0 + ph), C.BORDER, 1)

        mode_c = {"idle": C.TEXT_DIM, "select_roi": C.WARNING,
                  "select_foul": C.WARNING, "running": C.SUCCESS}
        mc = mode_c.get(self.mode, C.ACCENT)
        ml = self.mode.replace("_", " ").title()
        cv2.circle(disp, (x0 + 14, y0 + 20), 5, mc, -1)
        cv2.putText(disp, ml, (x0 + 26, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, mc, 1, cv2.LINE_AA)

        fn = getattr(self, "_frame_num", 0)
        cv2.putText(disp, f"Frame: {fn}", (x0 + 12, y0 + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C.TEXT, 1,
                    cv2.LINE_AA)

        fc = C.SUCCESS if self._fps_val >= 25 else (
            C.WARNING if self._fps_val >= 15 else C.DANGER)
        cv2.putText(disp, f"FPS: {self._fps_val:.1f}", (x0 + 12, y0 + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, fc, 1, cv2.LINE_AA)

        cv2.putText(disp, f"Detections: {self._det_count}", (x0 + 12, y0 + 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C.TEXT, 1,
                    cv2.LINE_AA)

    # ---------------------------------------------------------------
    #  SOURCE SELECTION
    # ---------------------------------------------------------------
    def _on_upload(self):
        if self.mode == "running":
            return
        path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._cancel_after()
        self.source_path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open:\n{path}")
            return
        frame, skipped, warning = _first_visible_frame(cap)
        cap.release()
        if frame is None:
            messagebox.showerror("Error", warning)
            return
        if skipped > 0:
            print(f"[Upload] Skipped {skipped} black frame(s)")
        if warning:
            messagebox.showwarning("Warning", warning)
        self._prepare_frame(frame)
        self._set_mode("select_roi",
                       "Click 4 corners of the take-off board area (green)")

    def _on_camera(self):
        if self.mode == "running":
            return
        self._cancel_after()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame, skipped, warning = _first_visible_frame(cap)
        cap.release()
        if frame is None:
            messagebox.showerror("Error", "Cannot read webcam frame.")
            return
        if skipped > 0:
            print(f"[Camera] Skipped {skipped} black frame(s)")
        if warning:
            messagebox.showwarning("Warning", warning)
        self.source_path = 0
        self._prepare_frame(frame)
        self._set_mode("select_roi",
                       "Click 4 corners of the take-off board area (green)")

    def _prepare_frame(self, frame):
        """Store frame and reset ROI/foul state, then show for ROI."""
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.roi_corners = []
        self.foul_pts = []
        self.foul_events = []
        self.root.after(50, self._show_frame, self.frame, self._draw_roi_overlay)

    # ---------------------------------------------------------------
    #  MOUSE CLICK HANDLER
    # ---------------------------------------------------------------
    def _on_click(self, event):
        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(), 400)
        ch = max(self.canvas.winfo_height(), 300)

        fh, fw = self.frame.shape[:2]
        scale = min(cw / fw, ch / fh, 1.0)
        dw, dh = int(fw * scale), int(fh * scale)
        ox = (cw - dw) / 2
        oy = (ch - dh) / 2

        rx, ry = event.x - ox, event.y - oy
        if rx < 0 or ry < 0 or rx >= dw or ry >= dh:
            return

        self._scale = scale

        if self.mode == "select_roi":
            self._handle_roi_click(rx, ry, scale)
        elif self.mode == "select_foul":
            self._handle_foul_click(rx, ry, scale)

    # ---------------------------------------------------------------
    #  ROI SELECTION  (4 points -> green quadrilateral)
    # ---------------------------------------------------------------
    def _handle_roi_click(self, rx, ry, scale):
        if len(self.roi_corners) >= 4:
            return
        self.roi_corners.append((rx, ry))
        n = len(self.roi_corners)
        self._set_mode("select_roi",
                       f"ROI corners: {n}/4 \u2014 " +
                       ("press Reset to redo" if n < 4 else
                        "click 'Start Detection' when ready"))
        if n == 4:
            self._show_frame(self.frame, self._draw_roi_overlay)
            self.foul_pts = []
            self._set_mode("select_foul",
                           "ROI set. Now click 2 points for the foul line (red).")
        else:
            self._show_frame(self.frame, self._draw_roi_overlay)

    def _draw_roi_overlay(self, disp, scale):
        """Green ROI corners + connecting edges on *disp* (in-place)."""
        for i, (x, y) in enumerate(self.roi_corners):
            ix, iy = int(x), int(y)
            glow = _mix(C.BG, C.ROI_GREEN, 0.4)
            cv2.circle(disp, (ix, iy), 10, glow, -1)
            cv2.circle(disp, (ix, iy), 6, C.ROI_GREEN, -1)
            cv2.putText(disp, f"C{i+1}", (ix + 12, iy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C.ROI_GREEN, 1,
                        cv2.LINE_AA)
        if len(self.roi_corners) >= 2:
            pts = [(int(x), int(y)) for x, y in self.roi_corners]
            for i in range(len(pts)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(pts)]
                cv2.line(disp, p1, p2, C.ROI_GREEN, 2, cv2.LINE_AA)

    # ---------------------------------------------------------------
    #  FOUL LINE SELECTION  (2 points -> red line)
    # ---------------------------------------------------------------
    def _handle_foul_click(self, rx, ry, scale):
        if len(self.foul_pts) >= 2:
            return
        self.foul_pts.append((rx, ry))
        n = len(self.foul_pts)
        self._set_mode("select_foul",
                       f"Foul line points: {n}/2" +
                       (" \u2014 click one more" if n < 2 else
                        " \u2014 press 'Start Detection'"))
        if n == 2:
            self._set_mode("select_foul",
                           "Foul line set (red). Click 'Start Detection'.")
            self.btn_start.configure(state=tk.NORMAL)
        self._show_frame(self.frame, self._draw_all_overlay)

    def _draw_all_overlay(self, disp, scale):
        """ROI (green) + foul line (red) on *disp* (in-place)."""
        if len(self.roi_corners) >= 2:
            pts = [(int(x), int(y)) for x, y in self.roi_corners]
            for i in range(len(pts)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(pts)]
                cv2.line(disp, p1, p2, C.ROI_GREEN, 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(self.foul_pts):
            ix, iy = int(x), int(y)
            glow = _mix(C.BG, C.FOUL_RED, 0.4)
            cv2.circle(disp, (ix, iy), 10, glow, -1)
            cv2.circle(disp, (ix, iy), 6, C.FOUL_RED, -1)
            cv2.putText(disp, f"P{i+1}", (ix + 12, iy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C.FOUL_RED, 1,
                        cv2.LINE_AA)
        if len(self.foul_pts) == 2:
            p1 = (int(self.foul_pts[0][0]), int(self.foul_pts[0][1]))
            p2 = (int(self.foul_pts[1][0]), int(self.foul_pts[1][1]))
            cv2.line(disp, p1, p2, C.FOUL_RED, 3, cv2.LINE_AA)

    # ---------------------------------------------------------------
    #  START DETECTION
    # ---------------------------------------------------------------
    def _on_start(self):
        if self.mode == "running":
            return
        if self.frame is None or len(self.roi_corners) != 4 or len(self.foul_pts) != 2:
            messagebox.showwarning(
                "Not ready",
                "Upload a video, select 4 ROI corners, and 2 foul-line points first.")
            return

        scale = self._scale

        # Warp from ORIGINAL frame coordinates
        orig_roi = np.float32([
            (cx / scale, cy / scale) for cx, cy in self.roi_corners
        ])
        rect = cv2.minAreaRect(orig_roi)
        box = cv2.boxPoints(rect)
        ordered = _order_corners(box)
        roi_w = int(cv2.norm(ordered[0] - ordered[1]))
        roi_h = int(cv2.norm(ordered[0] - ordered[3]))

        if roi_w < 20 or roi_h < 20:
            messagebox.showwarning("ROI too small", "Select a larger region.")
            return

        dst = np.float32([[0, 0], [roi_w, 0], [roi_w, roi_h], [0, roi_h]])
        M_warp = cv2.getPerspectiveTransform(ordered, dst)
        M_inv = cv2.getPerspectiveTransform(dst, ordered)

        fl1 = (self.foul_pts[0][0] / scale, self.foul_pts[0][1] / scale)
        fl2 = (self.foul_pts[1][0] / scale, self.foul_pts[1][1] / scale)

        r1 = _warp_point(fl1, M_warp)
        r2 = _warp_point(fl2, M_warp)
        fl_roi = FoulLine(
            x1=int(r1[0]), y1=int(r1[1]),
            x2=int(r2[0]), y2=int(r2[1]),
            tolerance_px=self.config.get("foul_line", {}).get("tolerance_px", 5))

        self.foul_checker = FoulChecker(
            foul_line=fl_roi,
            min_consecutive_frames=self.config.get(
                "detection", {}).get("min_consecutive_frames", 2),
            tolerance_px=fl_roi.tolerance_px,
        )

        self._M_warp = M_warp
        self._M_inv = M_inv
        self._roi_w = roi_w
        self._roi_h = roi_h
        self._fl_orig = (fl1, fl2)
        # Preserve original user-clicked corners (not minAreaRect)
        self._roi_quad = np.array(
            [(cx / scale, cy / scale) for cx, cy in self.roi_corners],
            dtype=np.float32)
        self.foul_events = []

        self.camera = Camera(
            source=self.source_path,
            width=self.config.get("camera", {}).get("width", 1280),
            height=self.config.get("camera", {}).get("height", 720),
            fps=self.config.get("camera", {}).get("fps", 60),
        )
        if not self.camera.open():
            messagebox.showerror("Error", "Cannot open video source.")
            return

        out_cfg = self.config.get("output", {})
        self._save = out_cfg.get("save_video", True)
        self._output_dir = out_cfg.get("output_dir", "output")
        if not os.path.isabs(self._output_dir):
            self._output_dir = os.path.join(_PROJECT_ROOT, self._output_dir)
        self._writer = None
        if self._save:
            os.makedirs(self._output_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(self._output_dir, f"foul_check_{ts}.mp4")
            self._writer = create_video_writer(
                out_path, self.camera.fps,
                self.camera.width, self.camera.height)

        self._set_mode("running", "Running detection\u2026")
        self._dot.config(fg=_hex(C.SUCCESS))
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_upload.configure(state=tk.DISABLED)
        self.btn_cam.configure(state=tk.DISABLED)

        self._frame_num = 0
        self._fps_t = time.time()
        self._fps_cnt = 0
        self._fps_val = 0.0

        # Background processing: producer/consumer via queue
        self._result_queue = queue.Queue(maxsize=8)
        self._stop_event = threading.Event()
        self._worker_error = None
        self._worker = threading.Thread(
            target=self._detection_worker, daemon=True)
        self._worker.start()
        self._poll_results()

    # ---------------------------------------------------------------
    #  BACKGROUND DETECTION WORKER  (runs on a separate thread)
    # ---------------------------------------------------------------
    def _detection_worker(self):
        """Read frames + run YOLO inference off the main thread."""
        from utils import Detection as Det

        try:
            while not self._stop_event.is_set():
                ok, frame = self.camera.read()
                if not ok:
                    self._result_queue.put(None)   # sentinel
                    return

                fh, fw = frame.shape[:2]
                roi_frame = cv2.warpPerspective(frame, self._M_warp,
                                                (self._roi_w, self._roi_h))
                dets_roi = self.detector.detect_with_fallback(roi_frame)

                # Map detections back to original frame coordinates
                dets_orig = []
                for d in dets_roi:
                    x1, y1, x2, y2 = d.bbox
                    tl = _warp_point((x1, y1), self._M_inv)
                    br = _warp_point((x2, y2), self._M_inv)
                    mask = None
                    if d.mask is not None:
                        mask = cv2.warpPerspective(d.mask, self._M_inv, (fw, fh))
                    dets_orig.append(Det(bbox=(tl[0], tl[1], br[0], br[1]),
                                         confidence=d.confidence, mask=mask,
                                         class_name=d.class_name))

                # Put result — foul check done on main thread (needs frame_num)
                result = (frame, dets_roi, dets_orig, fh, fw)
                self._result_queue.put(result)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._worker_error = str(exc)
            # Always send sentinel so the polling loop can shut down cleanly
            try:
                self._result_queue.put(None)
            except Exception:
                pass

    # ---------------------------------------------------------------
    #  POLL RESULTS  (runs on tkinter main thread via after())
    # ---------------------------------------------------------------
    def _poll_results(self):
        """Consume results from the worker thread and update the UI."""
        try:
            while True:
                result = self._result_queue.get_nowait()
                if result is None:
                    self._finish_detection()
                    return
                try:
                    self._render_frame(result)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    print(f"[Poll] Render error (skipping frame): {exc}")
        except queue.Empty:
            pass
        if self.mode == "running":
            self._after_id = self.root.after(5, self._poll_results)

    def _render_frame(self, result):
        """Draw annotations and display the processed frame."""
        frame, dets_roi, dets_orig, fh, fw = result

        self._det_count = len(dets_orig)

        # Foul check runs on main thread (safe access to _frame_num)
        frame_is_foul, confirmed_foul = self.foul_checker.check_frame(
            dets_roi, self._frame_num)

        # --- Draw annotations ---
        ann = frame.copy()
        overlay = ann.copy()
        for d in dets_orig:
            bx1, by1, bx2, by2 = [int(v) for v in d.bbox]
            col = C.FOUL_RED if frame_is_foul else C.ROI_GREEN
            if d.mask is not None:
                mr = cv2.resize(d.mask, (fw, fh))
                overlay[mr.astype(bool)] = (
                    overlay[mr.astype(bool)] * 0.5 +
                    np.array(col) * 0.5
                ).astype(np.uint8)
            cv2.rectangle(ann, (bx1, by1), (bx2, by2), col, 2)
            lbl = f"{d.class_name} {d.confidence:.2f}"
            tw, th_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(ann, (bx1, by1 - th_ - 8),
                          (bx1 + tw + 4, by1), col, -1)
            cv2.putText(ann, lbl, (bx1 + 2, by1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        ann = cv2.addWeighted(overlay, 0.6, ann, 0.4, 0)

        # ROI outline
        cv2.polylines(ann, [np.int32(self._roi_quad)], True, C.ROI_GREEN, 2)

        # Foul line
        fl1, fl2 = self._fl_orig
        cv2.line(ann,
                 (int(fl1[0]), int(fl1[1])),
                 (int(fl2[0]), int(fl2[1])),
                 C.FOUL_RED, 3)

        # Foul / No foul banner
        if confirmed_foul:
            cv2.rectangle(ann, (0, 0), (fw, 70), (0, 0, 180), -1)
            cv2.putText(ann, "FOUL", (fw // 2 - 80, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
        else:
            cv2.putText(ann, "NO FOUL", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        # FPS calculation
        self._fps_cnt += 1
        elapsed = time.time() - self._fps_t
        if elapsed >= 1.0:
            self._fps_val = self._fps_cnt / elapsed
            self._fps_cnt = 0
            self._fps_t = time.time()

        # Update FPS label with colour coding
        if self._fps_val >= 25:
            self._fps_lbl.config(fg=_hex(C.SUCCESS))
        elif self._fps_val >= 15:
            self._fps_lbl.config(fg=_hex(C.WARNING))
        else:
            self._fps_lbl.config(fg=_hex(C.DANGER))
        self._fps_lbl.config(text=f"FPS: {self._fps_val:.1f}")

        # Foul event
        if confirmed_foul and self.foul_checker.foul_frame_number == self._frame_num:
            self.foul_events.append(self._frame_num)
            self._foul_count = len(self.foul_events)
            self._foul_lbl.config(
                text=f"\u26a0 {self._foul_count} FOUL"
                     f"{'S' if self._foul_count > 1 else ''}")
            self._set_mode("running",
                           f"FOUL detected at frame {self._frame_num}!  "
                           f"Running\u2026 ({self._frame_num} frames)")
            self._dot.config(fg=_hex(C.DANGER))
            if self._save:
                cv2.imwrite(os.path.join(self._output_dir,
                                         f"foul_frame_{self._frame_num}.jpg"),
                            ann)

        if self._writer:
            self._writer.write(ann)

        # Display with info panel
        self._show_frame(ann, self._draw_info_panel)
        self._frame_num += 1

    def _finish_detection(self):
        """Called when the video ends or camera disconnects."""
        total = self._frame_num
        nf = len(self.foul_events)

        if self._writer:
            self._writer.release()
            self._writer = None
        self.camera.release()

        self.btn_upload.configure(state=tk.NORMAL)
        self.btn_cam.configure(state=tk.NORMAL)

        # Surface worker errors so the user isn't left guessing
        if getattr(self, '_worker_error', None):
            messagebox.showerror(
                "Detection Error",
                f"The detection worker crashed:\n{self._worker_error}\n\n"
                f"Processed {total} frame{'s' if total != 1 else ''} before failure.")
            self._worker_error = None

        if nf > 0:
            self._set_mode("idle",
                           f"Done \u2014 {total} frames | "
                           f"FOUL at frame {self.foul_events[0]}")
        else:
            self._set_mode("idle",
                           f"Done \u2014 {total} frames | NO FOUL detected")

        # Enhanced result banner with summary statistics
        if self.frame is not None:
            verdict = self.frame.copy()
            fh, fw = verdict.shape[:2]
            if nf > 0:
                cv2.rectangle(verdict, (0, 0), (fw, 110), (0, 0, 180), -1)
                cv2.putText(verdict, "RESULT: FOUL DETECTED",
                            (fw // 2 - 250, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                            (255, 255, 255), 3)
                cv2.putText(verdict,
                            f"{nf} foul{'s' if nf > 1 else ''}  |  "
                            f"{total} frames  |  "
                            f"peak {self._fps_val:.0f} FPS",
                            (fw // 2 - 200, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(verdict, (0, 0), (fw, 110), (0, 100, 0), -1)
                cv2.putText(verdict, "RESULT: NO FOUL",
                            (fw // 2 - 180, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                            (255, 255, 255), 3)
                cv2.putText(verdict,
                            f"{total} frames processed  |  "
                            f"peak {self._fps_val:.0f} FPS",
                            (fw // 2 - 180, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 255, 200), 1, cv2.LINE_AA)
            self._show_frame(verdict)

    # ---------------------------------------------------------------
    #  RESET / CLEANUP
    # ---------------------------------------------------------------
    def _stop_worker(self):
        """Signal the background worker to stop and wait for it."""
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
        if hasattr(self, '_worker') and self._worker.is_alive():
            self._worker.join(timeout=2.0)

    def _cancel_after(self):
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if hasattr(self, '_placeholder_aid') and self._placeholder_aid:
            self.root.after_cancel(self._placeholder_aid)
            self._placeholder_aid = None
        self._stop_worker()

    def _on_reset(self):
        self._cancel_after()
        if hasattr(self, "camera") and self.camera.is_open():
            self.camera.release()
        if hasattr(self, "_writer") and self._writer:
            self._writer.release()
            self._writer = None

        self.frame = None
        self.display_frame = None
        self.roi_corners = []
        self.foul_pts = []
        self.foul_events = []
        self._photo = None
        self._fps_val = 0.0
        self._det_count = 0
        self._foul_count = 0

        self.btn_upload.configure(state=tk.NORMAL)
        self.btn_cam.configure(state=tk.NORMAL)
        self.btn_start.configure(state=tk.DISABLED)
        self._fps_lbl.config(text="FPS: \u2014", fg=_hex(C.TEXT_DIM))
        self._foul_lbl.config(text="")
        self._set_mode("idle", "Ready \u2014 upload a video or start the camera")
        self._start_placeholder_anim()
        self._show_placeholder()

    def on_close(self):
        self._cancel_after()
        if hasattr(self, "camera") and self.camera.is_open():
            self.camera.release()
        if hasattr(self, "_writer") and self._writer:
            self._writer.release()
        self.root.destroy()


# ===================================================================
#  ENTRY POINT
# ===================================================================

def main():
    root = tk.Tk()
    app = FoulDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
