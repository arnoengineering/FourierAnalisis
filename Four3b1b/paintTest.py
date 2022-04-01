import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QMainWindow
# QToolBar, QStatusBar
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen
from TSP.NN import  load_f
from numpy import sin, cos, pi
from scipy import fft


class AppDemo(QWidget):
    def __init__(self):
        super().__init__()

        self._set_screen()
        self._set_vars()

        mainLayout = QVBoxLayout()
        self.photoViewer = QLabel()
        mainLayout.addWidget(self.photoViewer)

        self.setLayout(mainLayout)
        # self.but = QPushButton()
        self.file_path = None
        self.timer = QTimer()

    def _set_screen(self):
        self.min_max_size = ((200, 200), (5000, 5000))
        self.resize(400, 400)

        self.setMinimumSize(*self.min_max_size[0])
        self.setMaximumSize(*self.min_max_size[1])
        self.setAcceptDrops(True)

    def _set_vars(self):
        self.fps = 60
        self.t0 = 0
        self.inter = int(1000 / 60)
        self.for_items = []
        self.n_four_items = []
        self.full_path = []
        self.paint = False

    def set_image(self, file_path):
        self.photoViewer.set_pixmap(QPixmap(file_path))

    def paintEvent(self, event):
        print('PAINT STARTING')
        painter = QPainter(self)
        painter.setPen(QPen(Qt.green, 8, Qt.SolidLine))
        print('pen set')
        if self.paint:
            painter.begin(self)
            cur_p = np.array([100, 60])  # start at origen
            print('starting print')

            for freq, xy in zip(*self.n_four_items):  # mag is complex point at t=0
                mag = np.linalg.norm(xy)
                print(f'T: {self.t0}, mag: {mag}, ')
                painter.drawEllipse(int(cur_p[0]), int(cur_p[1]), int(mag), int(mag))
                theta = -2 * pi * freq * self.t0
                print(f'Theta: {theta},\n')
                n_y = mag * np.array(
                    [cos(theta), -sin(theta)]) + cur_p  # updates vector head position then adds to start

                painter.drawLine(int(cur_p[0]), int(cur_p[1]), int(n_y[0]), int(n_y[1]))
                cur_p = n_y  # next vect starts at old
                self.full_path.append(cur_p)
            painter.end()
    # def draw_full_path(points):
    #     # todo at efry t and fade out
    #     return np.sum(points)

    def time_u(self):
        self.t0 += 1000
        print(self.t0 / 1000)
        self.update()

    def draw_all(self):
        # path raced by all vectors should resemble image
        self.t0 = 0
        print('time starting')
        # timer.setInterval(1000)
        self.timer.timeout.connect(self.time_u)
        self.timer.start(1000)
        self.paint = True
        # self.paintEven()

    def four(self, samples):
        print('STARTING FOURIER ANALISIS')
        print('samples: ', len(samples))
        four_samples = samples[:, 0] + 1j * samples[:, 1]
        for_mag = fft.fft(four_samples) / four_samples.size
        four_size = for_mag.size

        four_freq = fft.fftfreq(four_size, 1 / four_size)
        print('calculated Fourier')
        print(for_mag[:10])
        self.for_items = [four_freq, for_mag]

    def up_four(self, n_vect):
        self.n_four_items = [self.for_items[0][:n_vect], self.for_items[1][:n_vect]]

    def run_edge_nn_data(self):
        print('attempting pre-edge')
        points = load_f()
        print('got points')
        self.four(points)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()  # todo set max size
        self.setWindowTitle('QMainWindow')
        self.cen = QWidget(self)
        self.setCentralWidget(self.cen)

        self.layout = QVBoxLayout()
        self.dem = AppDemo()
        self.layout.addWidget(self.dem)

        self.cen.setLayout(self.layout)

        self.vec_vals = {'vect': 5}
        self.has_updated = {}
        self.but = QPushButton()
        self.but.clicked.connect(self.on_start)
        self.layout.addWidget(self.but)

    def on_start(self):
        self.dem.run_edge_nn_data()
        print('size ready to update')
        print('VECTOR UPDATE')

        self.up_vect()
        self.dem.draw_all()
        # todo after debug switch

    def up_vect(self):
        print('vect n= ', 5)
        self.dem.up_four(5)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
