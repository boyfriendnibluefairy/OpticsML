import sys
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QLineEdit, QGroupBox
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# for wave propagation methods
from wave_prop.wave_propagation import *

# =========================
#  GUI Application
# =========================

class DiffractionDemo(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fresnel / Fraunhofer / Rayleigh-Sommerfeld Demo")

        # ---- Default parameters (similar to your MATLAB example) ----
        self.wavelength = 0.6e-6        # m
        self.M = 512                    # pixels (square grid)
        self.N = self.M
        self.dx = 10 * self.wavelength  # m, pixel pitch
        self.dy = self.dx
        self.L = self.M * self.dx       # m, physical side length

        self.aperture_radius_mm = 0.5   # mm, initial radius
        self.z_m = 0.07                 # m, initial propagation distance

        # Preallocate object and fields
        self.U_object = None
        self.fields = {}

        # Build UI
        self._init_ui()
        self.recompute_all()

    # ---------- UI setup ----------

    def _init_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)

        # Top panel: controls
        controls_layout = QHBoxLayout()

        # Text boxes group
        params_group = QGroupBox("Parameters")
        params_layout = QGridLayout(params_group)

        # Wavelength
        self.lambda_edit = QLineEdit(f"{self.wavelength:.3e}")
        params_layout.addWidget(QLabel("Wavelength (m):"), 0, 0)
        params_layout.addWidget(self.lambda_edit, 0, 1)

        # Object size in pixels (M)
        self.M_edit = QLineEdit(str(self.M))
        params_layout.addWidget(QLabel("Object size (pixels):"), 1, 0)
        params_layout.addWidget(self.M_edit, 1, 1)

        # Pixel pitch
        self.dx_edit = QLineEdit(f"{self.dx:.3e}")
        params_layout.addWidget(QLabel("Pixel pitch (m):"), 2, 0)
        params_layout.addWidget(self.dx_edit, 2, 1)

        # Physical side length
        self.L_edit = QLineEdit(f"{self.L:.3e}")
        params_layout.addWidget(QLabel("Side length (m):"), 3, 0)
        params_layout.addWidget(self.L_edit, 3, 1)

        # Connect text edits
        self.lambda_edit.editingFinished.connect(self.on_lambda_changed)
        self.M_edit.editingFinished.connect(self.on_M_changed)
        self.dx_edit.editingFinished.connect(self.on_dx_changed)
        self.L_edit.editingFinished.connect(self.on_L_changed)

        controls_layout.addWidget(params_group)

        # Sliders group
        sliders_group = QGroupBox("Controls")
        sliders_layout = QGridLayout(sliders_group)

        # Aperture radius slider (0.0–10.0 mm, step 0.1 mm)
        self.ap_slider = QSlider(Qt.Orientation.Horizontal)
        self.ap_slider.setMinimum(0)
        self.ap_slider.setMaximum(100)  # 0..10 mm in 0.1 mm steps
        self.ap_slider.setSingleStep(1)
        self.ap_slider.setValue(int(self.aperture_radius_mm * 10))

        self.ap_edit = QLineEdit(f"{self.aperture_radius_mm:.1f}")

        sliders_layout.addWidget(QLabel("Aperture radius (mm):"), 0, 0)
        sliders_layout.addWidget(self.ap_slider, 0, 1)
        sliders_layout.addWidget(self.ap_edit, 0, 2)

        # z slider (0–2.0 m in 0.01 m steps = 0–200)
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(200)
        self.z_slider.setSingleStep(1)
        self.z_slider.setValue(int(self.z_m * 100))

        self.z_edit = QLineEdit(f"{self.z_m:.2f}")

        sliders_layout.addWidget(QLabel("Propagation distance z (m):"), 1, 0)
        sliders_layout.addWidget(self.z_slider, 1, 1)
        sliders_layout.addWidget(self.z_edit, 1, 2)

        # Connect sliders and edits
        self.ap_slider.valueChanged.connect(self.on_ap_slider_changed)
        self.z_slider.valueChanged.connect(self.on_z_slider_changed)

        self.ap_edit.editingFinished.connect(self.on_ap_edit_changed)
        self.z_edit.editingFinished.connect(self.on_z_edit_changed)

        controls_layout.addWidget(sliders_group)

        main_layout.addLayout(controls_layout)

        # Bottom panel: Matplotlib figure
        self.fig = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.fig)

        # 2x4 grid of axes
        self.axes = self.fig.subplots(2, 4)

        # Object at [0,0]
        self.ax_object = self.axes[0, 0]
        self.ax_object.set_title("Object (Circular aperture)")

        # We won't use [0,1] and [0,2] for images
        self.axes[0, 1].axis('off')
        self.axes[0, 2].axis('off')

        # Fraunhofer at [0,3]
        self.ax_fraunhofer = self.axes[0, 3]
        self.ax_fraunhofer.set_title("Fraunhofer")

        # RS kernel at [1,0]
        self.ax_rs_kernel = self.axes[1, 0]
        self.ax_rs_kernel.set_title("RS (kernel)")

        # RS freq at [1,1]
        self.ax_rs_freq = self.axes[1, 1]
        self.ax_rs_freq.set_title("RS (freq)")

        # Fresnel kernel at [1,2]
        self.ax_fresnel_kernel = self.axes[1, 2]
        self.ax_fresnel_kernel.set_title("Fresnel (kernel)")

        # Fresnel freq at [1,3]
        self.ax_fresnel_freq = self.axes[1, 3]
        self.ax_fresnel_freq.set_title("Fresnel (freq)")

        # Turn off axes ticks for all
        for ax_row in self.axes:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])

        # Placeholders for imshow objects
        self.im_object = None
        self.im_fraunhofer = None
        self.im_rs_kernel = None
        self.im_rs_freq = None
        self.im_fresnel_kernel = None
        self.im_fresnel_freq = None

        main_layout.addWidget(self.canvas)

        self.setCentralWidget(central)

    # ---------- Parameter handlers ----------

    def on_lambda_changed(self):
        try:
            val = float(self.lambda_edit.text())
            if val <= 0:
                return
            self.wavelength = val
            # adjust dx default maybe proportional? keep as is
            self.recompute_all()
        except ValueError:
            self.lambda_edit.setText(f"{self.wavelength:.3e}")

    def on_M_changed(self):
        try:
            val = int(float(self.M_edit.text()))
            if val <= 0:
                return
            self.M = self.N = val
            self.L = self.M * self.dx
            self.L_edit.setText(f"{self.L:.3e}")
            self.recompute_all()
        except ValueError:
            self.M_edit.setText(str(self.M))

    def on_dx_changed(self):
        try:
            val = float(self.dx_edit.text())
            if val <= 0:
                return
            self.dx = self.dy = val
            self.L = self.M * self.dx
            self.L_edit.setText(f"{self.L:.3e}")
            self.recompute_all()
        except ValueError:
            self.dx_edit.setText(f"{self.dx:.3e}")

    def on_L_changed(self):
        try:
            val = float(self.L_edit.text())
            if val <= 0:
                return
            self.L = val
            # adjust dx based on new L and current M
            self.dx = self.dy = self.L / self.M
            self.dx_edit.setText(f"{self.dx:.3e}")
            self.recompute_all()
        except ValueError:
            self.L_edit.setText(f"{self.L:.3e}")

    def on_ap_slider_changed(self, value):
        # value is 0..100 -> radius in mm
        self.aperture_radius_mm = value / 10.0
        self.ap_edit.setText(f"{self.aperture_radius_mm:.1f}")
        self.recompute_all()

    def on_z_slider_changed(self, value):
        # value is 0..200 -> z in meters (0..2.0)
        self.z_m = value / 100.0
        self.z_edit.setText(f"{self.z_m:.2f}")
        self.recompute_all()

    def on_ap_edit_changed(self):
        try:
            val = float(self.ap_edit.text())
            if val < 0:
                val = 0.0
            if val > 10.0:
                val = 10.0
            self.aperture_radius_mm = val
            self.ap_slider.setValue(int(val * 10.0))
            self.recompute_all()
        except ValueError:
            self.ap_edit.setText(f"{self.aperture_radius_mm:.1f}")

    def on_z_edit_changed(self):
        try:
            val = float(self.z_edit.text())
            if val < 0:
                val = 0.0
            if val > 2.0:
                val = 2.0
            self.z_m = val
            self.z_slider.setValue(int(val * 100.0))
            self.recompute_all()
        except ValueError:
            self.z_edit.setText(f"{self.z_m:.2f}")

    # ---------- Core computations ----------

    def build_object(self):
        """Build circular aperture object field based on current parameters."""
        M = self.M
        dx = self.dx
        radius_m = self.aperture_radius_mm * 1e-3  # mm -> m

        # Coordinate grid centered at zero
        coords = (np.arange(M) - M / 2) * dx
        X, Y = np.meshgrid(coords, coords)
        R = np.sqrt(X*X + Y*Y)

        aperture = (R <= radius_m).astype(np.complex128)
        self.U_object = aperture

    def propagate_all(self):
        """Compute propagated fields using all methods."""
        self.fields = {}
        z = self.z_m
        lam = self.wavelength
        dx = self.dx
        dy = self.dy

        # If z == 0, just copy the object as "no propagation"
        if z <= 0:
            U = self.U_object
            self.fields['fraunhofer'] = U
            self.fields['rs_kernel'] = U
            self.fields['rs_freq'] = U
            self.fields['fresnel_kernel'] = U
            self.fields['fresnel_freq'] = U
            return

        Uin = self.U_object

        # Fraunhofer
        U_fraunhofer, _, _ = Fraunhofer(Uin, dx, dy, lam, z)
        self.fields['fraunhofer'] = U_fraunhofer

        # Rayleigh-Sommerfeld kernel
        U_rs_kernel = RayleighSommerfeld(Uin, dx, dy, lam, z, method='kernel')
        self.fields['rs_kernel'] = U_rs_kernel

        # Rayleigh-Sommerfeld freq
        U_rs_freq = RayleighSommerfeld(Uin, dx, dy, lam, z, method='freq')
        self.fields['rs_freq'] = U_rs_freq

        # Fresnel kernel
        U_fresnel_kernel = Fresnel(Uin, dx, dy, lam, z, method='kernel')
        self.fields['fresnel_kernel'] = U_fresnel_kernel

        # Fresnel freq
        U_fresnel_freq = Fresnel(Uin, dx, dy, lam, z, method='freq')
        self.fields['fresnel_freq'] = U_fresnel_freq

    def recompute_all(self):
        """Rebuild object, propagate, and refresh plots."""
        self.build_object()
        self.propagate_all()
        self.update_plots()

    # ---------- Plotting ----------

    @staticmethod
    def _intensity_normalized(U):
        I = np.abs(U)**2
        max_val = I.max()
        if max_val > 0:
            I = I / max_val
        return I

    def update_plots(self):
        # Object
        obj_int = self._intensity_normalized(self.U_object)

        if self.im_object is None:
            self.im_object = self.ax_object.imshow(
                obj_int, cmap='gray', vmin=0.0, vmax=1.0
            )
        else:
            self.im_object.set_data(obj_int)

        # Fraunhofer
        fra_int = self._intensity_normalized(self.fields['fraunhofer'])
        if self.im_fraunhofer is None:
            self.im_fraunhofer = self.ax_fraunhofer.imshow(
                fra_int, cmap='gray', vmin=0.0, vmax=1.0
            )
        else:
            self.im_fraunhofer.set_data(fra_int)

        # RS kernel
        rs_k_int = self._intensity_normalized(self.fields['rs_kernel'])
        if self.im_rs_kernel is None:
            self.im_rs_kernel = self.ax_rs_kernel.imshow(
                rs_k_int, cmap='gray', vmin=0.0, vmax=1.0
            )
        else:
            self.im_rs_kernel.set_data(rs_k_int)

        # RS freq
        rs_f_int = self._intensity_normalized(self.fields['rs_freq'])
        if self.im_rs_freq is None:
            self.im_rs_freq = self.ax_rs_freq.imshow(
                rs_f_int, cmap='gray', vmin=0.0, vmax=1.0
            )
        else:
            self.im_rs_freq.set_data(rs_f_int)

        # Fresnel kernel
        fr_k_int = self._intensity_normalized(self.fields['fresnel_kernel'])
        if self.im_fresnel_kernel is None:
            self.im_fresnel_kernel = self.ax_fresnel_kernel.imshow(
                fr_k_int, cmap='gray', vmin=0.0, vmax=1.0
            )
        else:
            self.im_fresnel_kernel.set_data(fr_k_int)

        # Fresnel freq
        fr_f_int = self._intensity_normalized(self.fields['fresnel_freq'])
        if self.im_fresnel_freq is None:
            self.im_fresnel_freq = self.ax_fresnel_freq.imshow(
                fr_f_int, cmap='gray', vmin=0.0, vmax=1.0
            )
        else:
            self.im_fresnel_freq.set_data(fr_f_int)

        self.canvas.draw_idle()


# =========================
#  Main entry point
# =========================


