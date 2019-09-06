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

        self.boost = 1.
        self.period = .005
        self.batch = 7
        self.void = 3
        self.offset = -1
        self.clip = (.6, 1.)

        self.size = size+2*self.void
        self.h_frame = slice(self.void+self.offset, -self.void+self.offset, 1)

        self.kiln = np.zeros((self.size, self.size+self.void))
        self.a, self.b = 1, self.size-1
        self.kiln[self.a:self.b,0] = np.random.random(self.b-self.a)*self.boost
        self.lastupdate = time.time()
        

    def __call__(self):
        self.kiln[self.a:self.b,0] = self._coal()
        for _ in range(self.batch):
            for h in np.arange(self.size-1, 0, -1):
                self.kiln[1:-1,h] = .6*self.kiln[1:-1,h] + .25*(self.kiln[:-2,h]+self.kiln[2:,h])
                self.kiln[:,h] *= .085
                self.kiln[1:-1,h] += .8 * self.kiln[1:-1,h-1] + .05*(self.kiln[:-2,h-1]+self.kiln[2:,h-1])
        
        while time.time() < self.lastupdate + self.period:
            pass
        self.lastupdate = time.time()

        return (self.kiln[self.void:-self.void,self.h_frame].clip(0,1)**.5*self.boost).clip(*self.clip)
    
    def _coal(self):
        coal, log_len, n = np.zeros(self.b-self.a)+.5, self.b-self.a, 1
        while log_len >= 1:
            degrees = np.random.randn(n+1)
            log_slices = [slice(i*log_len,(i+1)*log_len) for i in range(n)]
            for log, deg in zip(log_slices, degrees[:-1]):
                coal[log] += deg
            coal[n*log_len:] += degrees[-1]
            n *= 2
            log_len //= 2
        coal += .4
        coal = coal.clip(.2,5.)
        coal += np.random.randn(self.b-self.a)/10
        # coal += np.random.random(self.b-self.a)
        return coal*.2 + .8*self.kiln[self.a:self.b,0]



class App(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)

        self.size = 200

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