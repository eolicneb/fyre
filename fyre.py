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
        self.batch = 5
        self.coal_batch = 40
        self.void = 10
        self.offset = -2
        self.clip = (.6, 1.)

        self.size = size+2*self.void
        self.h_frame = slice(self.void+self.offset, -self.void+self.offset, 1)

        self.kiln = np.zeros((self.size, self.size+self.void))
        self.a, self.b = 1, self.size-1
        self.kiln[self.a:self.b,0] = np.random.random(self.b-self.a)*self.boost
        self.lastupdate = time.time()

        self.coal_count = 0
        # self.kiln[self.a:self.b,0] = 10
        self.coal = self.kiln[self.a:self.b,0]
        

    def __call__(self):
        for _ in range(self.batch):
            self.kiln[self.a:self.b,0] = self._coal()
            for h in np.arange(self.size-1, 0, -1):
                self.kiln[1:-1,h] = .6*self.kiln[1:-1,h] \
                                + .25*(self.kiln[:-2,h]+self.kiln[2:,h])
                self.kiln[:,h] *= .073
                self.kiln[1:-1,h] += .851 * self.kiln[1:-1,h-1] \
                                + .03*(self.kiln[:-2,h-1]+self.kiln[2:,h-1])
        
        while time.time() < self.lastupdate + self.period:
            pass
        self.lastupdate = time.time()

        data = self.kiln[self.void:-self.void,self.h_frame]**1.1
        data *= self.boost
        # data = data
        return np.stack((data**.5,data**.7,data**2), axis=2).clip(*self.clip)
    
    def _coal(self):
        
        # temperature profile changes only every coal_batch iters
        if self.coal_count == self.coal_batch:
            log_len, n = self.b-self.a, 1
            while log_len >= 1:
                degrees = np.random.random(n+1)
                log_slices = [slice(i*log_len,(i+1)*log_len) for i in range(n)]
                for log, deg in zip(log_slices, degrees[:-1]):
                    self.coal[log] *= deg
                self.coal[n*log_len:] *= degrees[-1]
                n *= 2
                log_len //= 2
            self.coal_count = 0
        else:
            self.coal_count += 1
            
        # coal += np.random.random(self.b-self.a)
        # return coal*.4 + .6*self.kiln[self.a:self.b,0]
        return self.coal + np.random.randn(self.b-self.a).clip(0)



class App(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)

        self.size = 160

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