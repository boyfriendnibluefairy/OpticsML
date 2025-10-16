import sys
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

class Lens_FFTPSF_Window(QMainWindow):
    def __init__(self):
        super().__init__()

        ### Main Window Specs
        self.setWindowTitle(" OpticsML - FFT PSF")
        self.setWindowIcon(QIcon("media/opticsml_icon.png"))
        self.setStyleSheet("QMainWindow { background-color: #0e121a; }")
        self.setGeometry(300,200,800,400)
        self.setContentsMargins(10, 10, 10, 10)
        self.main = QWidget()
        self.setCentralWidget(self.main)
        self.main_layout = QVBoxLayout()
        self.main.setLayout(self.main_layout)

        ### Grid Layout
        self.grid = QGridLayout()
        self.main_layout.addLayout(self.grid)  # grid is a layout, not a widget