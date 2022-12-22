import sys

import numpy as np
import scipy.interpolate
from PyQt5.QtWidgets import *
# QToolBar, QStatusBar
import os
from PyQt5.QtCore import Qt, QTimer, QPointF, QPoint, QSettings, QRect
from PyQt5.QtGui import QPainter, QPen, QImage, QFont, QPixmap
from Four3b1b.edge import simple_edge_detection
from TSP.NN import im_2_points, load_f
from numpy import sin, cos, pi, fft
from functools import partial

from super_bc import SuperCombo, SuperButton, SuperSlider

# def set_size(size):
#     print(size)
#     return size
# adding short cut for save action
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
        self.active = False
        print(self.size())
        self.setMouseTracking(True)
        self._set_image_clear()
        self._set_brush()

    def _set_brush(self):
        print('setting Brush')
        self.drawing = [False, False]

        self.brush_size = 10
        self.r = 2
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

                painter.setPen(QPen(col, self.brush_size))  # Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin

                # draw line from the last point of cursor to the current point
                # this will draw only one step

                painter.drawLine(self.lastPoint, event.pos())

                # change the last point

                self.lastPoint = event.pos()
                # painter.drawEllipse(event.pos(), rad, rad)
                vect_painter.drawImage(self.normal_image.rect(), self.image, self.image.rect())
            else:
                vect_painter.drawImage(self.normal_image.rect(), self.image, self.image.rect())
                vect_painter.setPen(QPen(Qt.green, 2,
                                    Qt.DashLine))  # , Qt.RoundCap, Qt.RoundJoin

                vect_painter.drawEllipse(event.pos(), rad, rad)
            # print(event.pos())
            # painter.drawLine(event.pos())
        else:
            vect_painter.setPen(Qt.blue)
            vect_painter.setFont(QFont("Arial", 30))
            # vect_painter.drawLine(0, 0, 100,100)
            vect_painter.drawText(self.normal_image.rect(), Qt.AlignCenter, 'Fourier')

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

    def draw_spline(self):  # on any mouse, and also for all splines
        def bezier(p,t_space=50):
            t = np.linspace(0,1,t_space)
            b = np.zeros(t_space)
            n = p.size
            i = np.arange(n)
            bi = scipy.special.binom(n,i)
            for tt in range(t_space):
                b[tt] = bi*(1-t[tt])**(n-i)*t[tt]**i*p
            return b
        tol = 3  # pixel tol
        mouse = np.array([0,2])
        painter = QPainter()
        spline = bezier(self.spline_points+mouse)
        for pp in range(len(self.spline_points)):
            painter.drawLine(pp)  # todo dashed, last point = mouse dif color

        for sp in range(len(spline)-1):
            painter.drawLine(*spline[sp],*spline[sp+1])  # dif color

        if mouse:  # click
            if any(np.abs(mouse-x)<tol for x in self.spline_points):
                # point x is active wil click num no new mouse but move point # todo for any point type
                pass
            self.spline_points.append(mouse)

        if 'escape': # key
            self.path_points.append(self.spline_points)


class FourierDisp(QWidget):
    def __init__(self,par):
        super().__init__()
        print('\n________________\nfourier disp\n___________')
        # self.setText('Fourier')
        self.par = par
        self.name = 'Fourier Display'
        # self.dock = QDockWidget(self.name)

        self._set_screen()
        self.path_rect = QPixmap(self.rect().height(), self.rect().width())
        self.path_rect.fill(Qt.transparent)
        self._set_vars()
        self.timer = QTimer()
        self.draw = False
        self.pause = False
        # self._set_image_clear()

    def v_scale(self):
        # todo shift control scale
        # is complex or real, connect in to out, reverse one
        # opacity scale

        pass

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
        self.scale = 2
        self.for_items = []
        self.n_four_items = []
        self.full_path = []
        self.t_scale = 0

    def paint_sin_0(self,t0):
        si_range = (self.t_l[1] - self.t_l[0])*1000
        t_range = 2000  # sec, posible x range, add scale
        start_x = self.size().width() * 2 // 3
        end_pos = self.size().width()
        pos_range = end_pos - start_x
        pos_scale = pos_range/t_range

        if t0 > t_range + self.t_scale:
            self.t_scale += self.ts_scale   # offsets by one frame, put in later

        def point_f(p):
            pos_x = []
            y_fun = []
            for ti in range(2):
                t = self.full_path[p+ti][1]
                tx = t-self.t_scale
                pos_x.append(tx*pos_scale+start_x)
                t_fun =(t+self.t_l[0]) % si_range
                y_res = t_fun/1000*self.scale**self.par.four_slide['disp scale']
                y_fun.append(-y_res+self.rect().center().y())
                # print(f'tx ({tx}) t({t}) yfunc({y_res}), tfun({t_fun})')

            return pos_x,y_fun

        return point_f

    # paint fuction with repeat
    def paintEvent(self, event):

        # vect_painter = QPainter(self.normal_image)

        vect_painter = QPainter(self)
        vect_painter.eraseRect(self.rect())
        if self.draw:
            cl = len(self.c_map)
            print('PAINT STARTING')

            print('t0=', self.t0)
            cur_p = self.rect().center()  # start at origen
            print('st p')
            paint_sin = self.paint_sin_0(self.t0)
            n = 0
            for freq, xy in zip(*self.n_four_items):  # mag is complex point at t=0

                mag = np.linalg.norm(xy)*self.scale**self.par.four_slide['disp scale']
                vect_painter.setPen(QPen(self.c_map[n], 1, Qt.SolidLine))
                n = (n + 1) % cl
                vect_painter.drawEllipse(cur_p, int(mag), int(mag))
                theta = -2 * pi * np.real(freq) * self.t0/1000

                n_y = cur_p + QPointF(cos(theta), -sin(theta)) * mag  # updates vector head position then adds to start
                vect_painter.drawLine(cur_p, n_y)
                cur_p = n_y  # next vect starts at old

            self.full_path.append((cur_p, self.t0))
            if len(self.full_path)>0:

                xiz = 0
                if len(self.full_path) > 50:
                    xiz = len(self.full_path)-50
                for p in range(xiz, len(self.full_path)-1):
                    xt, ft= paint_sin(p)

                    # painting result 1 d
                    vect_painter.setPen(QPen(Qt.gray, 2, Qt.DashLine))
                    vect_painter.drawLine(xt[0], self.full_path[p][0].y(), xt[1], self.full_path[p + 1][0].y())

                    # painting fx 1d
                    vect_painter.setPen(QPen(Qt.blue, 2, Qt.DashLine))
                    vect_painter.drawLine(xt[0], ft[0], xt[1], ft[1])

                    # painting result 2d
                    vect_painter.setPen(QPen(Qt.red, 2, Qt.DashLine))
                    vect_painter.drawLine(self.full_path[p][0],self.full_path[p+1][0])

        else:
            vect_painter.setPen(Qt.blue)
            vect_painter.setFont(QFont("Arial", 30))
            # vect_painter.drawLine(0, 0, 100,100)
            vect_painter.drawText(self.rect(), Qt.AlignCenter, "Qt")

    def time_u(self):
        self.ts_scale = 1000/(self.par.four_slide['fps']*self.par.four_slide['speed'])
        self.t0 += self.ts_scale
        print(1/self.ts_scale)
        self.update()

    def draw_all(self):
        # path raced by all vectors should resemble image
        # self.setText('')

        print('time starting')
        # timer.setInterval(1000)
        self.draw = True
        self.timer.timeout.connect(self.time_u)
        self.t0 = 0
        self.timer.start(int(1000/self.par.four_slide['fps']))

    def play_set(self,x):
        if x == 'play':
            print('play')
            if not self.draw:
                self.draw_all()
            elif self.pause:
                self.pause = False
                self.timer.start()

        elif x == 'reset':
            print('reset')
            self.t0 = 0
            self.timer.stop()
            self.draw = False
            self.pause=False
            vect_painter = QPainter(self)
            vect_painter.eraseRect(self.rect())
        else:
            print('stop')
            self.timer.stop()
            self.pause = True

    def calculate_fourier(self, samples):
        print('STARTING FOURIER ANALISIS')
        print('samples: ', len(samples))
        four_samples = samples[:, 0] + 1j * samples[:, 1]
        self.calc_fourier_main(four_samples)

    def calc_fourier_main(self, four_samples):
        for_mag = fft.fft(four_samples) / four_samples.size
        four_size = for_mag.size
        if len(self.t_l)>0:
            t_final = self.t_l[-1]
        else:
            t_final = 1

        t_space = t_final/four_size

        four_freq = fft.fftfreq(four_size, t_space)

        print('calculated Fourier')

        print(for_mag[:10])
        for_items = np.array((four_freq[four_size // 2:], for_mag[four_size // 2:]))
        # self.for_items = [four_freq, for_mag]
        for_items_index = np.abs(for_items[1])
        self.for_items = for_items[:, np.argsort(for_items_index)]
        print('sorted')
        self.update_fourier_vector_cnt(self.par.four_slide['vect'])
        
    def update_fourier_vector_cnt(self, n_vect):
        self.n_four_items = self.for_items[:, -n_vect:][:,::-1]

    def fourier_samples(self):
        self.t_l = (0,3)
        x = np.linspace(*self.t_l)
        self.main_f = np.linspace(*self.t_l)
        self.calc_fourier_main(self.main_f)
        self.par.update_tools('fun')


class MainImgDisp(QLabel):
    def __init__(self, par):
        super().__init__()
        self.par = par
        self.pix = None
        self.size_0 = (200, 200)
        self.n_size = (200, 200)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa
            }
        ''')

    def set_pixmap(self, image):
        self.pix = QPixmap(image)
        si = self.pix.size()
        self.size_0 = (si.width(), si.height())
        # self.resize(self.size)
        self.setPixmap(self.pix)

    def update_pix(self, percent):
        self.n_size = (int(self.size_0[0] * percent), int(self.size_0[1] * percent))
        self.pix.scaled(*self.n_size)

        self.resize(*self.n_size)
        print('scaled')

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
            self.par.new_img(event.mimeData().urls()[0].toLocalFile())
            self.set_pixmap(self.par.file_path)
            event.accept()

        else:
            event.ignore()


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings('Claassens Fourier', 'Fourier')
        self.setWindowTitle('Fourier')

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

        self.dem = FourierDisp(self)

        self.setCentralWidget(self.dem)
        self._create_docks()

        # self.vec_vals = {'vect': 5}
        self.has_updated = {}

        self._create_tools()
        self._update_set()

    def run_cmd(self,b):
        if b in self.const_tools.but:
            self.on_recalc(b)
        elif b in self.pre_process_tools.but:
            self.brush_set(b)
        elif b in self.fourier_tools.but:
            self.play_pause(b)

    def update_tools(self, fun):
        if fun == 'fun':
            self.addToolBar(self.fourier_tools.tool_bar)
            self.addToolBar(self.four_slide.tool_bar)

    def _create_docks(self):
        # img-on load->canney-edit-->main
        self.img_dock = QDockWidget('Image to analize')
        self.img = MainImgDisp(self)
        self.addDockWidget(Qt.TopDockWidgetArea, self.img_dock)
        self.img_dock.setWidget(self.img)

        self.canny_dock = QDockWidget('Canny_results')
        self.canny = CannyEdge()

        print('setting canny')
        self.addDockWidget(Qt.BottomDockWidgetArea, self.canny_dock)
        self.canny_dock.setWidget(self.canny)
        print('sett canny')

    def _create_tools(self):
        self.vec_vals = {'img':[0,False], 'vect':[0,False]}
        self.const_tools = SuperButton('reculc',self,vals=['all', 'bit', 'calculate fourier', 'nn', 'fun'])
        self.pre_process_tools = SuperButton('Draw',self,vals=['erase', 'draw', 'line', 'spline'])
        self.fourier_tools = SuperButton('Fourier',self,vals=['play', 'pause', 'reset'])
        fourier_slide = {'img':[20,1000], 'vect':[1,500], 'fps':[20,60], 'speed':[1,100], 'disp scale': [0,100]}
        # self.tool_dock = QDockWidget('ToolBar')
        self.tools = QToolBar()
        self.addToolBar(self.const_tools.tool_bar)
        self.four_slide = SuperSlider('FourierTools', self,fourier_slide)
        self.brush_slider=SuperSlider('BrushTools', self, {'brush size': [1,20]})
        self.addToolBar(self.pre_process_tools.tool_bar)  # todo remove and add widits

        self.tool_layout = QVBoxLayout()
        self.recalc_layout = QGridLayout()
        self.start_layout = QHBoxLayout()
        self.bush_layout = QGridLayout()

        self.bs = [False, False]

    def brush_size(self):
        v = self.action_lst['Brush size'].value()
        self.canny.brush_size = v

    def set_anim_spd(self):
        pass

    def new_img(self, path):
        self.file_path = path
        print('received image')
        self.canny.set_image(self.file_path)
        print('set image')
        # self.run_edge_nn(file_path)

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

    def slide_2_other(self, na, val):

        if na in self.vec_vals and val >self.vec_vals[na][0]:
            self.vec_vals[na] = [val,True]
            # self.has_updated[na] = True

    def on_recalc(self, i):  # todo add output scale,,  todo if has updated
        if i == 'all':  # todo bimap is same, qdia are you sure
            self.run_edge()
        elif i == 'nn':
            self.run_nn()
        elif i == 'for':
            self.dem.calculate_fourier(self.vec_vals['vect'])
        elif i == 'fun':
            self.dem.fourier_samples()
            print('loaded predefined func')

        # if self.has_updated['img']:
        #     # print('size ready to update')
        #     # if 'im' in self.vec_vals:
       #  self.up_size()
        # self.dem.run_edge_nn_data()
        #    self.has_updated['im'] = False
        #if self.has_updated['vect']:
        #    print('VECTOR UPDATE')
        # self.up_vect()
        # self.dem.draw_all()
        #    self.has_updated['vect'] = False

    def run_edge(self):
        # size = (400, 600)
        print('attempting edge')
        if self.file_path:
            self.edge_mat = simple_edge_detection(self.file_path, self.img.n_size)
            self.canny.set_image(self.edge_mat)
        else:
            print('no file Loaded')

    def run_nn(self):
        points = im_2_points(self.edge_mat, True)
        self.dem.calculate_fourier(points)  # todo first chech changes befor updating

    def run_edge_nn_data(self):
        print('attempting pre-edge')
        points = load_f()
        print('got points')
        self.dem.calculate_fourier(points)

    def play_pause(self, i):
        self.dem.play_set(i)

    # def resizeEvent(self, event):
    #     print("Window has been resized")
    #     if not self.image:
    #         print('scale')
    #         pass
    #
    #     QMainWindow.resizeEvent(self, event)

    def load_image(self):
        file_d = SaLd(True, self)
        file, t = file_d.get_file_name()
        hist = Dia(self)

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
            self.save_vects(file)
        elif t == 'NN':
            self.save_nn(file)
        elif t == 'bitmap':
            self.save_im(file)

    def _update_set(self):
        self.settings.beginGroup('nums')
        for i, j in self.four_slide.but.items():
            jk = self.settings.value(i, j['slide'].value())
            j['slide'].setValue(int(jk))

        self.settings.endGroup()
        k = self.settings.allKeys()

        for i, j in [(self.restoreGeometry, "Geometry"), (self.restoreState, "windowState")]:
            if j in k:
                va = self.settings.value(j)
                i(va)

    def user_settings(self):

        self.settings.setValue("Geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.beginGroup('nums')
        for i,j in self.four_slide.but.items():
            self.settings.setValue(i, j['slide'].value())

        self.settings.endGroup()

    def closeEvent(self, event):
        self.user_settings()
        super().closeEvent(event)


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
    if brush: pix = 1 else 0
    todo scale image save data """




