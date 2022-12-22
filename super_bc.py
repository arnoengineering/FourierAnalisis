from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon  # QPainter, QPen,QBrush,

from PyQt5.QtWidgets import *

from functools import partial


class SuperCombo(QComboBox):
    def __init__(self, name, par, orient_v=True, vals=None, show_lab=True, run=True):
        super().__init__()
        self.tool_bar = QToolBar()
        self.par = par
        self.orient_v = orient_v
        self.show_lab = show_lab
        self.name = name
        self.item_ls = []
        self.wig = QWidget()

        self.lab = QLabel(self.name)

        self._layout_set()

        if vals is not None:
            for v in vals:
                if '_' in v:
                    v, ic = v.split('_', 1)
                    self.addItem(QIcon(f'icons/{ic}.png'), v)
                elif self.name == 'Active Doc':
                    self.addItem(self.load_doc(v), v)
                else:
                    self.addItem(v)
                self.item_ls.append(v)

        if run:
            self.currentTextChanged.connect(lambda x: self.par.run_cmd(self.name, x))

    def load_doc(self, cell):
        id2 = self.par.doc_data.loc[self.par.doc_data['Doc'] == cell, 'Gender']
        if id2.values[0] == 'Male':
            ids = 'icons/user-medical.png'
        else:
            ids = 'icons/user-medical-female.png'
        return QIcon(ids)

    # noinspection PyArgumentList
    def _layout_set(self):
        if self.orient_v:
            self.layout = QVBoxLayout()
        else:
            self.layout = QHBoxLayout()
        self.layout.addWidget(self)
        self.layout.addWidget(self.lab)
        self.wig.setLayout(self.layout)

    def reset_show(self, show_lab=False, flip=False):
        if flip:
            self.orient_v = not self.orient_v
            self._layout_set()
        if show_lab:
            self.show_lab = not self.show_lab
            if self.show_lab:
                self.layout.addWidget(self.lab)
            else:
                self.layout.removeWidget(self.lab)


class SuperButton(QWidget):
    def __init__(self, name, par, orient_v=True, vals=None, show_lab=True):
        super().__init__()
        self.tool_bar = QToolBar()

        self.par = par
        self.orient_v = orient_v
        self.show_lab = show_lab
        self.name = name
        self.but = {}
        self.lab = QLabel(self.name)

        if vals:
            for i in vals:
                if '_' in i:
                    i, ic = i.split('_', 1)
                    j = QPushButton(QIcon(f'icons/{ic}.png'), "")
                else:
                    j = QPushButton(i)
                j.clicked.connect(partial(self.par.run_cmd, i))

                self.but[i] = j

        self._layout_set()

    def _layout_set(self):
        self.layout = QGridLayout()
        n = 0
        if self.orient_v:

            for i in self.but.keys():
                self.layout.addWidget(self.but[i], 0, n)
                n += 1
            self.layout.addWidget(self.lab, 1, 0, 1, n)
        else:
            for i in self.but.keys():
                self.layout.addWidget(self.but[i], n, 0)
                n += 1
            self.layout.addWidget(self.lab, 0, 1, n, 1)
        self.setLayout(self.layout)
        self.tool_bar.addWidget(self)

    def reset_show(self, show_lab=False, flip=False):
        if flip:
            self.orient_v = not self.orient_v
            self._layout_set()
        if show_lab:
            self.show_lab = not self.show_lab
            if not self.show_lab:
                self.layout.removeWidget(self.lab)


class SuperSlider(QWidget):
    def __init__(self, name, par, vals=None, show_lab=True):
        super().__init__()
        self.tool_bar = QToolBar()

        self.par = par
        self.show_lab = show_lab
        self.name = name
        self.but = {x:{} for x in vals.keys()}
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self.lab = QLabel(self.name)
        # sl.valueChanged.connect(partial(self.up_vect, i))  # todo since can be user or other, the inputs can just connect to slide

        if vals:
            for i,k in vals.items():
                layout = QGridLayout(self)
                j = QLineEdit()
                kk = (k[1]-k[0])//2
                j.setText(str(k[0]))
                j.editingFinished.connect(partial(self.add_file,i))

                sl = QSlider(Qt.Horizontal)
                sl.setMinimum(k[0])
                sl.setMaximum(k[1])
                sl.setValue(kk)
                sl.valueChanged.connect(partial(self.add_file, i, line=False))

                la = QLabel(i)
                layout.addWidget(sl,0,0,1,2)
                layout.addWidget(la, 1, 0)
                layout.addWidget(j, 1, 1)
                self.layout.addLayout(layout)
                self.but[i]['line'] = j
                self.but[i]['slide'] = sl

        self.tool_bar.addWidget(self)

    def add_file(self, na, line=True):
        if line:
            val = self.but[na]['line'].text()
            self.but[na]['slide'].setValue(int(val))
        else:
            v = self.but[na]['slide'].value()
            self.but[na]['line'].setText(str(v))
            self.par.slide_2_other(na,v)
            # self.par.self.par.run_cmd, i)

    def __getitem__(self, item):  # will give int form
        return self.but[item]['slide'].value()

    def __setitem__(self, key, value): # sets correct and will auto update
        self.but[key]['slide'].setValue(value)
