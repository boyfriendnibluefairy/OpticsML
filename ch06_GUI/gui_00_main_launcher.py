"""
    filename :  gui_00_main_launcher.py

    Settings:
    1) This is the first module to be called by main_gui.py to construct OpticsML Desktop Application.
    2) The central area of QMainWindow is divided into 4 quadrants with different relative sizes.
       A QSplitter places a draggable handle between child widgets.
    3) See ch06_GUI/gui_05_color_palette.py for color/style templates.
    4) Imports quadrant modules from gui_01_quadrant.py to gui_04_quadrant.py.
       If quandrant is empty, it falls back to placeholder widgets.
    5) QScreen.availableGeometry() (via self.screen() or QGuiApplication.primaryScreen()) access the
       available geometry of the desktop screen. It then shrinks the available QMainWindow rectangle by
       1 cm on all sides (converted to pixels using the screen DPI).

    TO DO:
    Expected (optional) API in each quadrant module (recommended):
        def build(parent: QWidget | None = None) -> QWidget

    Expected (optional) API in gui_05_color_palette.py (recommended):
        def get_main_stylesheet() -> str
    or  def apply(app: QApplication) -> None

"""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QIcon, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ch06_GUI.gui_05_color_palette import *

def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _icon_path() -> Path:
    return _repo_root() / "ch06_GUI" / "media" / "opticsml_icon.png"


def _cm_to_px(cm: float, dpi: float) -> int:
    # 1 inch = 2.54 cm
    inches = cm / 2.54
    return int(round(dpi * inches))


def _make_placeholder(title: str) -> QWidget:
    frame = QFrame()
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    frame.setObjectName("QuadrantFrame")

    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(8)

    lbl = QLabel(title)
    lbl.setObjectName("QuadrantTitle")
    lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    hint = QLabel("Placeholder (module not implemented yet).")
    hint.setObjectName("QuadrantHint")
    hint.setWordWrap(True)

    layout.addWidget(lbl)
    layout.addWidget(hint)
    layout.addStretch(1)

    return frame


def _load_quadrant_widget(import_path: str, fallback_title: str, builder_name: str = "build") -> QWidget:
    try:
        module = __import__(import_path, fromlist=["*"])
        builder = getattr(module, builder_name, None)
        if callable(builder):
            w = builder(None)
            if isinstance(w, QWidget):
                return w
        return _make_placeholder(fallback_title)
    except Exception:
        return _make_placeholder(fallback_title)


def _default_stylesheet() -> str:
    return f"""
    QMainWindow {{
        background: {MAIN_BG};
    }}

    QWidget {{
        color: #E8EEF9;
        font-size: 11pt;
    }}

    QFrame#QuadrantFrame {{
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 10px;
    }}

    QLabel#QuadrantTitle {{
        font-size: 12pt;
        font-weight: 600;
        color: #FFFFFF;
    }}

    QLabel#QuadrantHint {{
        font-size: 10pt;
        color: rgba(232, 238, 249, 0.75);
    }}

    QSplitter::handle {{
        background: rgba(255, 255, 255, 0.12);
    }}
    QSplitter::handle:hover {{
        background: rgba(255, 255, 255, 0.20);
    }}
    QSplitter::handle:pressed {{
        background: rgba(255, 255, 255, 0.28);
    }}
    """


def _apply_palette(app: QApplication, window: QMainWindow) -> None:
    try:
        from ch06_GUI import gui_05_color_palette as palette  # type: ignore

        apply_fn = getattr(palette, "apply", None)
        if callable(apply_fn):
            apply_fn(app)
            window.setStyleSheet(f"QMainWindow {{ background: {MAIN_BG}; }}")
            return

        get_css = getattr(palette, "get_main_stylesheet", None)
        if callable(get_css):
            css = str(get_css())
            css = f"QMainWindow {{ background: {MAIN_BG}; }}\n" + css
            window.setStyleSheet(css)
            return

    except Exception:
        pass

    window.setStyleSheet(_default_stylesheet())


class OpticsMLMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(APP_TITLE)

        icon_file = _icon_path()
        if icon_file.exists():
            self.setWindowIcon(QIcon(str(icon_file)))

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(0)

        # ---- Build quadrant widgets (or placeholders) ----
        q1 = _load_quadrant_widget(
            "ch06_GUI.gui_01_quadrant",
            "Quadrant 1: Optical System 2D Layout",
        )
        q3 = _load_quadrant_widget(
            "ch06_GUI.gui_03_quadrant",
            "Quadrant 3: Lens System Data",
        )
        q4 = _load_quadrant_widget(
            "ch06_GUI.gui_04_quadrant",
            "Quadrant 4: System Settings",
        )

        q2 = self._build_quadrant_2_container()

        # ---- Splitters for draggable resizing ----
        top_split = QSplitter(Qt.Orientation.Horizontal)
        top_split.addWidget(q1)
        top_split.addWidget(q2)

        bottom_split = QSplitter(Qt.Orientation.Horizontal)
        bottom_split.addWidget(q3)
        bottom_split.addWidget(q4)

        main_split = QSplitter(Qt.Orientation.Vertical)
        main_split.addWidget(top_split)
        main_split.addWidget(bottom_split)

        # Initial weights (user can drag afterwards)
        top_split.setSizes([2, 3])
        bottom_split.setSizes([2, 3])
        main_split.setSizes([3, 2])

        top_split.setHandleWidth(8)
        bottom_split.setHandleWidth(8)
        main_split.setHandleWidth(8)

        outer.addWidget(main_split)

        _apply_palette(QApplication.instance(), self)  # type: ignore[arg-type]

        # ---- NEW: Fit window to desktop with 1 cm margin, respecting taskbar/menu/dock ----
        self._fit_to_available_screen(margin_cm=1.0)

    def _fit_to_available_screen(self, margin_cm: float = 1.0) -> None:
        """
        Resizes/moves the main window to fill the screen's *availableGeometry*,
        inset by margin_cm on all sides.

        availableGeometry() excludes taskbar (Windows), menu bar + Dock (macOS),
        and similar desktop UI areas (Linux), when reported by the OS/WM.
        """
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            # Fallback: do nothing if screen info isn't available
            return

        avail = screen.availableGeometry()

        # Convert cm margin to pixels using logical DPI (per-screen).
        dpi_x = float(screen.logicalDotsPerInchX())
        dpi_y = float(screen.logicalDotsPerInchY())
        mx = _cm_to_px(margin_cm, dpi_x)
        my = _cm_to_px(margin_cm, dpi_y)

        target = QRect(
            avail.x() + mx,
            avail.y() + my,
            max(200, avail.width() - 2 * mx),
            max(200, avail.height() - 2 * my),
        )

        # Apply geometry (position + size)
        self.setGeometry(target)

        # Optional: ensure we're not maximized (so margins are visible)
        if self.isMaximized():
            self.showNormal()

    def _build_quadrant_2_container(self) -> QWidget:
        q2_widget = _load_quadrant_widget(
            "ch06_GUI.gui_02_quadrant",
            "Quadrant 2: Reconstruction Display",
        )

        # If placeholder, provide 3 uneven columns with a splitter
        if isinstance(q2_widget, QFrame) and q2_widget.objectName() == "QuadrantFrame":
            col1 = _make_placeholder("Q2 - Column 1")
            col2 = _make_placeholder("Q2 - Column 2")
            col3 = _make_placeholder("Q2 - Column 3")

            split = QSplitter(Qt.Orientation.Horizontal)
            split.addWidget(col1)
            split.addWidget(col2)
            split.addWidget(col3)

            split.setSizes([2, 6, 2]) # previous default: split.setSizes([5, 3, 2])
            split.setHandleWidth(6)
            return split

        return q2_widget