import sys

import numpy as np
from PyQt5.QtWidgets import *
# QToolBar, QStatusBar
import os
from PyQt5.QtCore import Qt, QTimer, QPointF, QPoint
from PyQt5.QtGui import  QPainter, QPen, QImage, QFont
from Four3b1b.edge import simple_edge_detection
from TSP.NN import im_2_points, load_f
from numpy import sin, cos, pi, fft
from functools import partial

# def set_size(size):
#     print(size)
#     return size
## adding short cut for save action
#         saveAction.setShortcut("Ctrl + S")
#




# noinspection PyArgumentList
class SaLd(QFileDialog):
    def __init__(self, save=False, par=None):
        super().__init__(par)
        self.setModal(True)
        self.window_width, self.window_height = 800, 200
        # self.setMinimumSize(self.window_width, self.window_height)
        self.actions = ['Load', 'Save']
        self.save_options = {'Jpeg Bitmap', 'png Bitmap', 'Fourier csv', 'Nearest neighbor csv'}
        self.file_op = {'img files': ['.png', '.jpg'], 'dataset': ['.csv']}
        self.filenames = None
        self.save = save
        self._setfil()
        # self.accept.connect(self.get_file_name)
        # todo on cancle

    def _setfil(self):
        def img(z, y):
            img_fil = ', '.join(['*' + x for x in y])
            return f'{z}: ({img_fil})'

        # if self.mode[0] == 0:
        self.filter = ';; '.join(
            [img(k, f) for k, f in self.file_op.items()])  # can propbably ignore for if save seperate
        # else:
        #     self.filter = self.img(self.file_op['img files'])

    def get_file_name(self):
        fn = self.getOpenFileName(
            parent=self,
            caption='Select a data file',
            directory=os.getcwd(),
            filter=self.filter)
        # todo add endswith
        return self.end_ret(fn)

    def save_file(self):  # todo creat dir
        # self.filenames = 'Data File (*.xlsx *.csv *.dat);; Excel File (*.xlsx *.xls)'
        fn = self.getSaveFileName(
            parent=self,
            caption='Select a data file',
            directory=os.getcwd(),
            filter=self.filter,
            )

        path = os.path.dirname(fn[0])
        if not os.path.exists(path):
            os.mkdir(path)
        return self.end_ret(fn)

    def end_ret(self, fn):
        fn = fn[0]
        if fn.endswith(','):
            fn = fn[:-1]
        if any(fn.endswith(x) for x in self.file_op['img files']):  # todo add clear, start, end
            t = 'img'
        else:
            t = 'sng'
        print('loading fn: ', fn)
        return fn, t


class CannyEdge(QWidget):
    def __init__(self):
        super().__init__()
        print('\n___________\nCanny Edge Widget\n___________')

        self.setMinimumSize(200, 200)
        self.sizes = (200, 200)
        self.initial_image = np.zeros(self.sizes)
        self.normal_image = QImage(self.size(), QImage.Format_RGB32)
        self.lastPoint = QPoint()
        print(self.size())
        self.setMouseTracking(True)
        self._set_image_clear()
        self._set_brush()

    def _set_brush(self):
        print('setting Brush')
        self.drawing = (False, False)
        self.active = False
        self.brush_size = 10
        self.r = 2  # todo scale to iomg size,
        print('sett Brush')

    def _set_image_clear(self):
        print('setting img clear')
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        # self.image = QPixmap(img)
        # self.setPixmap(self.image)
        print('sett img clear')

    def set_image(self, arr):
        print('setting img')
        self.initial_image = arr
        self.image = QImage(arr)
        # self.image = QPixmap(img)
        # self.setPixmap(self.image)
        print('sett img')

    def set_img_from_file(self, file):
        pass

    def scale_image(self):
        pass
        pass

    def img_to_array(self):
        '''  Converts a QImage into an opencv MAT format  '''

        img = self.image.convertToFormat(QImage.Format.Format_RGB32)
        width = img.width()
        height = img.height()

        ptr = img.constBits()
        arr = np.array(ptr).reshape((height, width, 4))  # Copies the data

    # method for checking mouse cicks
    # paint event
    def mousePressEvent(self, event):
        # if left mouse button is pressed
        if event.button() == Qt.LeftButton:
            self.active = True
            self.lastPoint = event.pos()

    # method for tracking mouse activity
    def mouseMoveEvent(self, event):
        # print('size: ', self.size())
        self.normal_image.fill(Qt.black)
        vect_painter = QPainter(self.normal_image)
        painter = QPainter(self.image)
        rad = self.brush_size // 2
        if any(self.drawing):
            if event.buttons() & Qt.LeftButton:
                if self.drawing[0]:
                    col = Qt.white

                else:
                    col = Qt.black

                # painter.setBrush(col)
                print('set pen')
                painter.setPen(QPen(col, self.brush_size))  # Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin

                # draw line from the last point of cursor to the current point
                # this will draw only one step
                print('drawp')
                painter.drawLine(self.lastPoint, event.pos())

                # change the last point
                print('set last_p')
                self.lastPoint = event.pos()
                # painter.drawEllipse(event.pos(), rad, rad)
                vect_painter.drawImage(self.normal_image.rect(), self.image, self.image.rect())
            else:
                vect_painter.drawImage(self.normal_image.rect(), self.image, self.image.rect())
                vect_painter.setPen(QPen(Qt.green, 2,  # todo dash for temp
                                    Qt.DashLine))  # , Qt.RoundCap, Qt.RoundJoin

                vect_painter.drawEllipse(event.pos(), rad, rad)
            # print(event.pos())
            # painter.drawLine(event.pos())  # todo erase
        else:
            vect_painter.setPen(Qt.blue)
            vect_painter.setFont(QFont("Arial", 30))
            # vect_painter.drawLine(0, 0, 100,100)
            vect_painter.drawText(self.normal_image.rect(), Qt.AlignCenter, 'Fourier')
            # self.setText()  # todo will reset?

            # method for mouse left button release
            # update
        self.update()
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            # make drawing flag false
            self.active = False


    def paintEvent(self, event):
        # create a canvas
        # if self.active:
        canvasPainter = QPainter(self)

        # draw rectangle  on the canvas
        canvasPainter.drawImage(self.rect(), self.normal_image, self.normal_image.rect())

    def resizeEvent(self, event):
        self.normal_image = self.normal_image.scaled(self.size())
        self.image = self.image.scaled(self.size())


class FourierDisp(QWidget):  # todo optional img back
    def __init__(self):
        super().__init__()
        print('\n________________\nfourier disp\n___________')  # todo add format for this
        # self.setText('Fourier')

        self._set_screen()
        self.path_rect = QImage(self.size(), QImage.Format_RGB32)
        self.normal_image = QImage(self.size(), QImage.Format_RGB32)
        # todo draw onto rect then draw rect, other to other
        self._set_vars()
        self.timer = QTimer()
        self.draw = False
        # self._set_image_clear()

    def _set_image_clear(self):
        print('setting img clear')

        self.normal_image.fill(Qt.white)

        # self.setPixmap(QPixmap(self.normal_image))
        print('sett img clear')

    def _set_screen(self):
        self.min_max_size = ((200, 200), (5000, 5000))
        self.resize(400, 400)

        self.setMinimumSize(*self.min_max_size[0])
        self.setMaximumSize(*self.min_max_size[1])
        self.setAcceptDrops(True)

    def _set_vars(self):
        self.c_map = [Qt.green, Qt.blue, Qt.yellow, Qt.red]
        self.fps = 60
        self.t0 = 0
        self.inter = int(1000 / 60)
        self.time_scale = 1
        self.scale = 1
        self.for_items = []
        self.n_four_items = []
        self.full_path = []

    def paintEvent(self, event):
        print('hi')
        self.normal_image.fill(Qt.white)
        vect_painter = QPainter(self.normal_image)

        fulp = QPainter(self)
        if self.draw:
            cl = len(self.c_map)
            print('PAINT STARTING')

            path_painter = QPainter(self.path_rect)

            print('t0=', self.t0)
            cur_p = QPointF(*[int(x/2) for x in self.photoViewer.n_size])  # start at origen
            print('st p')
            n = 0
            for freq, xy in zip(*self.n_four_items):  # mag is complex point at t=0

                mag = np.linalg.norm(xy)
                vect_painter.setPen(QPen(self.c_map[n], 1, Qt.SolidLine))
                n = (n + 1) % cl
                vect_painter.drawEllipse(cur_p, int(mag), int(mag))
                theta = -2 * pi * np.real(freq) * self.t0/100000
                print("   theta=", theta)
                n_y = cur_p + QPointF(cos(theta), -sin(theta)) * mag  # updates vector head position then adds to start
                vect_painter.drawLine(cur_p, n_y)
                cur_p = n_y  # next vect starts at old
            self.full_path.append(cur_p)
            path_painter.drawPoint(cur_p)  # todo on full loop switch to path and no draw
            vect_painter.drawImage(self.normal_image.rect(), self.path_rect, self.path_rect.rect())

            fulp.drawImage(self.rect(), self.normal_image, self.normal_image.rect())
        else:
            vect_painter.setPen(Qt.blue)
            vect_painter.setFont(QFont("Arial", 30))
            # vect_painter.drawLine(0, 0, 100,100)
            vect_painter.drawText(self.normal_image.rect(), Qt.AlignCenter, "Qt")
            print('text')
        fulp.drawImage(self.rect(), self.normal_image, self.normal_image.rect())
        print('pm')

        # def draw_full_path(points):
        #     # todo at efry t and fade out
        #     return np.sum(points)

    def time_u(self):
        self.t0 += 1000/self.fps
        print(self.t0/1000)
        self.update()

    def draw_all(self):
        # path raced by all vectors should resemble image
        self.setText('')

        print('time starting')
        # timer.setInterval(1000)
        self.draw = True
        self.timer.timeout.connect(self.time_u)
        self.t0 = 0
        self.timer.start(1000/self.fps)

    def calculate_fourier(self, samples):
        print('STARTING FOURIER ANALISIS')
        print('samples: ', len(samples))
        four_samples = samples[:, 0] + 1j * samples[:, 1]
        for_mag = fft.fft(four_samples) / four_samples.size
        four_size = for_mag.size

        four_freq = fft.fftfreq(four_size, 1 / four_size)

        print('calculated Fourier')

        print(for_mag[:10])
        for_items = np.array((four_freq[four_size//2:], for_mag[four_size//2:]))
        # self.for_items = [four_freq, for_mag]
        for_items_index = np.abs(for_items[1])
        self.for_items = for_items[:, np.argsort(for_items_index)]
        print('sorted')
        # self.update_fourier_vector_cnt(5)
        
    def update_fourier_vector_cnt(self, n_vect):
        self.n_four_items = self.for_items[:, -n_vect:][:,::-1]


class MainImgDisp(QLabel):
    def __init__(self):
        super().__init__()
        self.pix = None
        self.size_0 = (200, 200)
        self.n_size = (200, 200)
        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa
            }
        ''')

    def set_pixmap(self, image):
        self.pix = image
        si = self.pix.size()
        self.size_0 = (si.width(), si.height())
        # self.resize(self.size)
        self.setPixmap(image)

    def update_pix(self, percent):
        self.n_size = (int(self.size_0[0] * percent), int(self.size_0[1] * percent))
        self.pix.scaled(*self.n_size)

        self.resize(*self.n_size)
        print('scaled')


class ToolWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QGridLayout()
        self._set_dial()
        self._set_play()
        self._set_num()

    def _set_play(self):
        self.but = {}
        buts = ['play', 'pause']
        for n, i in enumerate(buts,1):
            self.but[i] = QPushButton(i)
            self.layout.addWidget(self.but[i], 1, n)

    def _set_dial(self):
        self.cnts = {}
        self.line = {}
        buts = ['vect', 'im']
        self.dial = QDial()

        self.layout.addWidget(self.dial, 0,0)
        v = 5
        for n, i in enumerate(buts, 1):
            j = QSlider()
            j.setOrientation(Qt.Vertical)
            k = QLineEdit()

            j.setMinimum(0)
            j.setMaximum(10)
            j.setValue(v)

            k.setReadOnly(False)
            k.setText(str(v))

            kn = QLabel(i)

            self.cnts[i] = j
            self.line[i] = k
            self.layout.addWidget(j, 2, n, rowSpan=3)
            self.layout.addWidget(k, 1, n)
            self.layout.addWidget(kn, 5, n)
            # todo img dialog
            # todo init vals
            self.has_updated[i] = True

    def _set_num(self):
        self.lcd = []
        for i in range(2):
            j = QLCDNumber()  # to edit
            self.layout.addWidget(j, 2 + i, 1, columnSpan=2)
            self.lcd.append(j)

    def _set_img_cont(self):
        # control_lay = QGridLayout()
        self.buts = {}
        self.vect_min_max = (1, 1000)
        self.image_min_max = (1, 100)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()  # todo set max size
        self.setWindowTitle('QMainWindow')

        self.load_menu = {'file': ['load bitmap', 'load init img', 'load nn data', 'load fourier data'],
                          'save': ['save'],
                          'edit': ['start', 'stop'],
                          'view': ['img', 'bitmap', 'img,bitmap on output']}
        # self.tool_menu = {  # 'start': 'but',
        #                   # 'stop': 'but',
        #                   'img rescale': 'slider',
        #                   'brush': 'erase, draw',
        #                   'brush size': 'slider',
        #                   # 'recalc': '[all, bit, calculate_fourier, nn]: but',
        #                   'img_rescale': 'line',
        #                 'framerate': 'slider'
        #                  }

        self.dem = FourierDisp()

        self.setCentralWidget(self.dem)
        self._create_docks()

        # self.vec_vals = {'vect': 5}
        self.has_updated = {}

        self._create_tools()

    def _create_docks(self):
        # self.img_dock = QDockWidget('Image to analize')
        # self.img = MainImgDisp()
        # self.addDockWidget(Qt.TopDockWidgetArea, self.img_dock)
        # self.img_dock.setWidget(self.img)

        self.canny_dock = QDockWidget('Canny_results')
        self.canny = CannyEdge()
        # todo canny after,
        # todo data vis
        print('setting canny')
        self.addDockWidget(Qt.BottomDockWidgetArea, self.canny_dock)
        self.canny_dock.setWidget(self.canny)
        print('sett canny')

    def _create_tools(self):

        self.tool_dock = QDockWidget('ToolBar')
        self.tools = QWidget(self)
        self.tool_dock.setWidget(self.tools)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.tool_dock)

        self.tool_layout = QVBoxLayout()
        self.recalc_layout = QGridLayout()
        self.start_layout = QHBoxLayout()
        self.bush_layout = QGridLayout()

        self.brush_act = ['erase', 'draw']
        self.bs = [False, False]

        self.action_lst = {}
        for i in ['start', 'stop']:
            self.action_lst[i] = QPushButton(i)
            self.action_lst[i].clicked.connect(partial(self.play_pause, i))
            # todo dialog foor update all
            self.start_layout.addWidget(self.action_lst[i])

        lab = QLabel('Recalculate')

        bj = ['all', 'bit', 'calculate_fourier', 'nn']
        for n in range(2):
            for m in range(2):
                i = bj[(n+1) * m]
                j = QPushButton(i)
                #
                j.clicked.connect(partial(self.on_recalc, i))
                self.action_lst[i] = j
                self.recalc_layout.addWidget(j, n, m)
        self.recalc_layout.addWidget(lab, 0, 3, 2, 1)

        for n, i in enumerate(self.brush_act):
            j = QPushButton(i)
            j.clicked.connect(partial(self.brush_set, n))
            self.action_lst[i] = j
            self.bush_layout.addWidget(j, n, 1)

        bs = 'Brush size'
        b_lab = QLabel(bs)
        self.bush_layout.addWidget(b_lab)
        brush_slide = QSlider(Qt.Vertical)
        self.action_lst[bs] = brush_slide
        brush_slide.setMinimum(1)
        brush_slide.setMaximum(50)
        brush_slide.valueChanged.connect(self.brush_size)

        vls = {'Img': (0, 1000), 'Vect': (0, 100)}  # todo resize
        col = [2, 4]
        for n, i, in enumerate(vls.keys()):
            j = QLineEdit()
            a = int(np.average(vls[i]))
            j.setText(str(a))
            sl = QSlider(Qt.Vertical)
            sl.setMinimum(vls[i][0])
            sl.setMaximum(vls[i][1])
            sl.setValue(a)
            self.has_updated[i] = a
            la = QLabel(i)
            j.editingFinished.connect(partial(self.slide_2_other, i))
            # sl.valueChanged.connect(partial(self.up_vect, i))  # todo since can be user or other, the inputs can just connect to slide
            self.action_lst[i + ' Slide'] = sl
            self.action_lst[i + ' Vect'] = j

            self.bush_layout.addWidget(sl, 0, col[n], 2, 1)
            self.bush_layout.addWidget(j, 0, col[n] + 1)
            self.bush_layout.addWidget(la, 1, col[n] + 1)

        self.tool_layout.addLayout(self.start_layout)
        self.tool_layout.addLayout(self.recalc_layout)
        self.tool_layout.addLayout(self.bush_layout)
        self.tools.setLayout(self.tool_layout)

    def brush_size(self):
        v = self.action_lst['Brush size'].value()
        self.canny.brush_size = v  # todo lambda or other

    def set_anim_spd(self):
        pass

    def draw(self):
        if self.clicked:  # todo on correct win and erase
            if self.bs[0]:
                pass
            elif self.bs[1]:
                pass
            # set impage pix map there to 1 or zero, in circle rad brush

    def brush_set(self, n):
        print('but set', n)
        if not self.bs[n]:  # tus it is off, becomming on

            self.bs[(n+1)%2] = False
            #todo toggle brush
            print('toggle othger')
        self.bs[n] = not self.bs[n]
        self.canny.drawing = self.bs
        print('toggle self')
        for n in range(2):
            i = self.brush_act[n]
            print('set ss: ', i)
            if self.bs[n]:

                self.action_lst[i].setStyleSheet('background-color:green;')
                print('set ss true')
            else:
                self.action_lst[i].setStyleSheet('background-color:grey;')
                print('set ss false')

    def slide_2_other(self, na):
        if na == 'img':
            val = self.buts[na].value()
            self.buts[na+' Vect'].setText(str(val))

        else:
            val = self.buts[na].text()
            val = int(val)
            print(val)
            # self.buts[i_sp[0] + '_slide'].setValue(val)
        self.vec_vals[na] = val
        self.has_updated[na] = True

    def on_recalc(self, i):  # todo add output scale
        if i == 'all':  # todo bimap is same, qdia are you sure
            self.dem.run_edge()
        elif i == 'nn':
            self.dem.run_nn()
        elif i == 'for':
            self.dem.calculate_fourier()

        if self.has_updated['img']:
            print('size ready to update')
            if 'im' in self.vec_vals:
                self.up_size()
            self.dem.run_edge_nn_data()
            self.has_updated['im'] = False
        if self.has_updated['vect']:
            print('VECTOR UPDATE')
            self.up_vect()
            self.dem.draw_all()
            self.has_updated['vect'] = False
        # todo after debug switch

    def run_edge(self):
        # size = (400, 600)
        print('attempting edge')
        if self.file_path:
            self.edge_mat = simple_edge_detection(self.file_path, self.photoViewer.n_size)
        else:
            print('no file Loaded')

    def run_nn(self):
        points = im_2_points(self.edge_mat, True)
        self.calculate_fourier(points)  # todo first chech changes befor updating

    def run_edge_nn_data(self):
        print('attempting pre-edge')
        points = load_f()
        print('got points')
        self.calculate_fourier(points)

    def play_pause(self, i):
        if i == 'start':
            self.dem.draw_all()  # todo add check if data, if canny not all black
        else:
            self.dem.stop_all()

    # def resizeEvent(self, event):
    #     print("Window has been resized")
    #     if not self.image:
    #         print('scale')
    #         pass
    #
    #     QMainWindow.resizeEvent(self, event)
    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            self.file_path = event.mimeData().urls()[0].toLocalFile()
            print('received image')
            self.set_image(self.file_path)
            print('set image')
            # self.run_edge_nn(file_path)

            event.accept()

        else:
            event.ignore()

    def load_image(self):
        file_d = SaLd(True, self)
        file, t = file_d.get_file_name()
        hist = Dia(self)  # todo on drag

        if hist.exec_():
            print('same img')
            self.hist.save_img(file)
        else:
            print('same hist')
            self.hist.save_hist(file)

    def save_f(self):
        file_d = SaLd(True, self)
        file, t = file_d.save_file()
        if t == 'Forier':
            self.save_vects(file)  # todo check
        elif t == 'NN':
            self.save_nn(file)
        elif t == 'bitmap':
            self.save_im(file)

class Dia(QDialog):
    # noinspection PyArgumentList
    def __init__(self, ty, par=None):
        super().__init__(par)
        self.setModal(True)
        self.buttonBox = QDialogButtonBox(self)
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.button_box.addButton('Full Img', self.button_box.AcceptRole)
        self.button_box.addButton('Histogram', self.button_box.RejectRole)
        self.buttonBox.setCenterButtons(True)

        self.layout = QVBoxLayout()
        self.label = QLabel('Save What?')
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())

"""todo add load img, load img sorted, load svg, load excel calculate_fourier, add scale, 
img loc = self.img.getrect
mouse = mouse.onclick.getpos
rel_mouse = mouse-img pos  # pos on imge,  todo check scale of pix,
rad = mouserad_slide
for all pix in rad from relpix, 
    if brush: pix = 1 else 0"""




