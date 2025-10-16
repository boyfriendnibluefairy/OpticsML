import sys
from PyQt6.QtWidgets import QApplication
from pyqt.pyqt_fftpsf import Lens_FFTPSF_Window

def main():
    app = QApplication(sys.argv)
    win = Lens_FFTPSF_Window()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()