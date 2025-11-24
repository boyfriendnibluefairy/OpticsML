from gui.gui_prop_methods import *

"""
    DiffractionDemo() GUI compares the performance of Fresnel, Fraunhofer, RS wave propagation methods
    at various object aperture and propagation distances.
"""
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DiffractionDemo()
    win.show()
    sys.exit(app.exec())