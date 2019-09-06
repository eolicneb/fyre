# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 14:10:28 2019

@author: nicoB
"""

import sys
import time
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg

class GenImage():
    def __init__(self, size=100):

        self.boost = 8
        self.period = .015
        self.batch = 2
        self.void = 2
        self.clip = (.1, 1.)
        self.size = size+2*self.void

        self.kiln = np.zeros((self.size, self.size+self.void))
        self.a, self.b = 1, self.size-1
        self.kiln[self.a:self.b,0] = np.random.random(self.b-self.a)*self.boost
        self.lastupdate = time.time()

    def __call__(self):
        for _ in range(self.batch):
            self.kiln[self.a:self.b,0] = self._coal()
            for h in np.arange(self.size-1, 0, -1):
                self.kiln[:,h] *= .01
                self.kiln[:,h] += .94 * self.kiln[:,h-1]
                self.kiln[1:-1,h] = .96*self.kiln[1:-1,h] + .02*(self.kiln[:-2,h]+self.kiln[2:,h])
        while time.time() < self.lastupdate + self.period:
            pass
        self.lastupdate = time.time()
        return self.kiln[self.void:-self.void,self.void:-self.void].clip(*self.clip)
    
    def _coal(self):
        return np.random.random(self.b-self.a)**4.2*self.boost + .6*self.kiln[self.a:self.b,0]



class App(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)

        self.size = 100

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(0,0, self.size, self.size))

        #  image plot
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)

        #### Set Data  #####################
        self.data_provider = GenImage(size=self.size)

        self.fps = 0.
        self.lastupdate = time.time()

        #### Start  #####################
        self._update()

    def _update(self):

        now = time.time()
        dt = (now-self.lastupdate)
        if dt <= 0:
            dt = 0.000000000001
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.9 + fps2 * 0.1
        tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps )
        self.label.setText(tx)


        self.img.setImage(self.data_provider())
        QtCore.QTimer.singleShot(1, self._update)


if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())