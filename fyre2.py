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

        self.period = .02

        self.heat_cons = .24
        self.base_heat = 1.8
        self.fire_rnd = .0
        self.coal_smooth = .84
        self.coal_batch = 5

        self.diffusion = .17
        self.convection = .987
        self.convection_batch = 3
        self.batch = 8

        self.clip_min = .862
        self.power = 1.648
        self.shift = 0.308
        
        self.clip_max = 1.

        self.void = 3
        self.offset = -1


        self._init_move()
        self._init_kernel()

        self.size = size+2*self.void
        self.h_frame = slice(self.void+self.offset, -self.void+self.offset, 1)
        self.a, self.b = 1, self.size-1

        self.reset()

    def reset(self):
        self.lastupdate = time.time()
        self.coal_count = 0
        self.coal = np.ones(self.b-self.a)
        self.kiln = np.zeros((self.size, self.size+self.void))
       
    def __call__(self):

        # Safety measure for when coal_batch is 
        # reduced while coal_count is still high.
        if self.coal_count > self.coal_batch:
            self.coal_count = 0

        self._move()        
        while time.time() < self.lastupdate + self.period:
            pass
        self.lastupdate = time.time()

        data = self.kiln[self.void:-self.void,self.h_frame].clip(0)
        data **= self.power
        data += self.shift
        return self.dataRGB(data.clip(0,1))

    def dataRGB(self, data):
        pR, pG, pB = .5, .7, 2.
        dataRGB = np.stack((data**pR,data**pG,data**pB), axis=2)
        # dataRGB = np.where(dataRGB < self.clip_min, 0, dataRGB)
        # dataRGB = dataRGB.clip(self.clip_min,self.clip_max)
        # np.where(dataRGB > self.clip_max, 1, dataRGB)
        dataRGB[0,0,:], dataRGB[-1,-1,:] = 1, 0
        clips = np.array((pR, pG, pB))
        clips = self.clip_min**clips
        dataRGB /= self.clip_min # clips.reshape((1,1,3))
        # print(clips.reshape((1,1,3)))
        dataRGB -= 1.
        return dataRGB.clip(0,1)
    
    def _coal(self):
        # temperature profile changes only every coal_batch iters
        if self.coal_count == self.coal_batch:
            log_len, n = self.b-self.a, 1
            while log_len > 1:
                
                degrees = np.random.randn(n+1)
                log_slices = [slice(i*log_len,(i+1)*log_len) for i in range(n)]
                for log, deg in zip(log_slices, degrees[:-1]):
                    self.coal[log] += deg
                self.coal[n*log_len:] += degrees[-1]
                
                n *= 2
                log_len = (self.b-self.a)//n

            self.coal_count = 0
            self.coal -= self.coal.min()
            self.coal *= (self.b-self.a) / self.coal.sum()*2
            # self.coal *= self.base_heat
        else:
            self.coal_count += 1
            
        new_coal = np.random.randn(self.b-self.a)*self.fire_rnd
        new_coal += self.coal.copy()
        return new_coal.clip(0)

    def _move(self):

        for _ in range(self.batch):
            #  Heat source
            bonfire = self.base_heat
            bonfire *= self._coal() # /(self.convection_batch+1)
            self.kiln[self.a:self.b,0] *= self.coal_smooth
            self.kiln[self.a:self.b,0] += bonfire*(1-self.coal_smooth)

            #  Accelerated convection
            for _ in range(self.convection_batch):
                new_kiln = self.kiln.clip(0,100)
                new_kiln[:,1:] *= (1-self.convection*self.kiln[:,:-1])
                new_kiln[:,1:] += self.convection*(self.kiln[:,:-1].clip(0,1))**2
                self.kiln[:,1:] = new_kiln[:,1:]*self.heat_cons_()
                
            #  Difussion process
            new_kiln = np.zeros(self.kiln.shape)
            for slice_, k in zip(self.slices, self.kernel_()):
                new_kiln[1:-1,1:-1] += self.kiln[slice_]*k*self.heat_cons_()

            #  Fix boundaries
            new_kiln[0,1:-1] = new_kiln[1,1:-1]
            new_kiln[-1,1:-1] =  new_kiln[-2,1:-1]
            new_kiln[:,-1] = new_kiln[:,-2]

            #  Renew kiln values
            self.kiln[:,1:] = new_kiln[:,1:]
        
    def _init_move(self):
        # Slice objects for pseudo-convolution
        slide = np.linspace(-1,1,num=3, dtype=np.dtype(int))
        I, J = np.meshgrid(slide, slide)
        slides = tuple(zip(I.reshape(-1), J.reshape(-1)))
        self.slices = tuple([ np.s_[1+i:(-1+i if i<1 else None), 
                                    1+j:(-1+j if j<1 else None)] \
                                        for i, j in slides])
            
    def _init_kernel(self):
        sq2 = np.sqrt(.5)
        self.ring = np.array(((sq2, 1, sq2), (1, 0, 1), (sq2, 1, sq2)))
        self.cent = np.zeros((3,3))
        self.cent[1, 1] = 1.
        self.bott = np.zeros((3,3))
        self.bott[0, 1] = 1.
    
    def kernel_(self):
        kernel = self.ring * self.diffusion \
                        + self.cent * (1-self.diffusion)
        # kernel *= (1-self.convection) / kernel.sum()
        # kernel += self.bott * self.convection
        kernel = kernel.reshape(9) / kernel.sum()
        return kernel

    def heat_cons_(self):
        return np.float(0.95**(1 / (self.heat_cons * (self.b-self.a))))


class AttrSlider(QtGui.QSlider):

    def __init__(self, *args, **kwargs):
        self.object = kwargs.pop('object', None)
        self.attr = kwargs.pop('attr', '')
        self.callback = kwargs.pop('callback', None)
        self.min = kwargs.pop('min', 0)
        self.max = kwargs.pop('max', 1)
        self.span = (self.max-self.min)/1000 # Check for range setting

        self.owner = kwargs.pop('owner', None)

        if 'orientation' not in kwargs:
            kwargs['orientation'] = QtCore.Qt.Horizontal
        super().__init__(*args, **kwargs)
        self.setMaximum(1000)

        self.adjustValue(self.getValue())

        self.valueChanged[int].connect(self.changeValue)

    def getValue(self):
        return getattr(self.object, self.attr)

    def adjustValue(self, value):
        if self.callback:
            value = self.callback(value, True)
        value = (value-self.min)/self.span
        self.setValue(value)
            
    def changeValue(self, value):
        val = np.float(self.min + value*self.span)
        if self.callback:
            val = self.callback(val, False)
        setattr(self.object, self.attr, val)
        if self.owner:
            self.owner.display()


class SliderBox(QtGui.QWidget):
    def __init__(self, **kwargs):
        box_kwargs = self._clean(kwargs.copy())
        super().__init__(**box_kwargs)
        self.setLayout(QtGui.QVBoxLayout())

        #  Slider
        kwargs['owner'] = self
        self.sld = AttrSlider(**kwargs)

        #  Title
        self.lb_attr = QtGui.QLabel(self)
        self.lb_attr.setText(self.sld.attr.replace('_', ' '))

        #  Display
        self.lb_display = QtGui.QLabel(self)
        self.display()
        
        self.layout().addWidget(self.lb_attr)
        self.layout().addWidget(self.sld)
        self.layout().addWidget(self.lb_display)
    
    def display(self):
        self.lb_display.setNum(self.sld.getValue())        
    
    def _clean(self, kwargs):
        kwargs.pop('object', None)
        kwargs.pop('attr', None)
        kwargs.pop('callback', None)
        kwargs.pop('min', None)
        kwargs.pop('max', None)
        return kwargs


class App(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)

        self.size = 140

        self.setGeometry(100, 100, 1000, 300)

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QGridLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas, 0, 1)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label, 1, 1)

        #  Plots box
        self.plots_canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.plots_canvas, 0, 0, 2, 1)

        #  Controls box
        self.controls_box = QtGui.QWidget(self.mainbox)
        self.controls_box.setMinimumWidth(400)
        self.mainbox.layout().addWidget(self.controls_box, 0, 2, 2, 1)
        self.controls_box.setLayout(QtGui.QGridLayout())
        self.controls_label = QtGui.QLabel()
        self.controls_box.layout().addWidget(self.controls_label, 0, 0)
        self.controls_label.setText('CONTROLS BOX')

        #  display canvas
        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(0,0, self.size, self.size))

        #  image plot
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)
        
        #  line plot 1
        self.plot1 = self.plots_canvas.addPlot(0,1)
        self.plot1.setWindowTitle('Bonfire temperature')
        self.plot1.setRange(yRange=(0,10))
        self.h1 = self.plot1.plot(pen='y')
        
        # self.plots_canvas.nextRow()
        #  line plot 2
        self.plot2 = self.plots_canvas.addPlot(1,1)
        self.plots_canvas.setWindowTitle('Coal temperature')
        self.plot2.setRange(yRange=(0,10))
        self.h2 = self.plot2.plot(pen='r')

        #### Set Data  #####################
        self.data_provider = GenImage(size=self.size)

        #  Sliders
        #  . heat conservation
        sld_heat_cons = SliderBox( object=self.data_provider,
                                  attr='heat_cons',
                                  min=.001)
        self.controls_box.layout().addWidget(sld_heat_cons, 1, 0)
        #  . base heat
        sld_base_heat = SliderBox( object=self.data_provider,
                                  attr='base_heat',
                                  min=.01, max=3.01)
        self.controls_box.layout().addWidget(sld_base_heat, 1, 1)
        #  . fire ramdomness
        sld_fire_rnd = SliderBox( object=self.data_provider,
                                  attr='fire_rnd',
                                  min=0, max=1)
        self.controls_box.layout().addWidget(sld_fire_rnd)
        #  . coal_smooth
        sld_coal_smooth = SliderBox( object=self.data_provider,
                                  attr='coal_smooth')
        self.controls_box.layout().addWidget(sld_coal_smooth)
        #  . coal_batch
        sld_coal_batch = SliderBox( object=self.data_provider,
                                  attr='coal_batch',
                                  callback=lambda x,y: int(x),
                                  min=1,
                                  max=30)
        self.controls_box.layout().addWidget(sld_coal_batch)

        #  . diffusion
        sld_diffusion = SliderBox( object=self.data_provider,
                                  attr='diffusion')
        self.controls_box.layout().addWidget(sld_diffusion)
        #  . convection
        sld_convection = SliderBox( object=self.data_provider,
                                  attr='convection')
        self.controls_box.layout().addWidget(sld_convection)        
        #  . convection_batch
        sld_convection_batch = SliderBox( object=self.data_provider,
                                  attr='convection_batch',
                                  callback=lambda x,y: int(x),
                                  max=10)
        self.controls_box.layout().addWidget(sld_convection_batch)
        #  . batch
        sld_batch = SliderBox( object=self.data_provider,
                                  attr='batch',
                                  callback=lambda x,y: int(x),
                                  min=1,
                                  max=30)
        self.controls_box.layout().addWidget(sld_batch)
        
        #  . clip min
        sld_clip_min = SliderBox( object=self.data_provider,
                                  attr='clip_min')
        self.controls_box.layout().addWidget(sld_clip_min)
        #  . power
        sld_power = SliderBox( object=self.data_provider,
                                  attr='power',
                                  min=0.2,
                                  max=5.0)
        self.controls_box.layout().addWidget(sld_power)
        #  . shift
        sld_shift = SliderBox( object=self.data_provider,
                                  attr='shift',
                                  min=-0.2,
                                  max=0.6)
        self.controls_box.layout().addWidget(sld_shift)

        # Reset
        btn_reset = QtGui.QPushButton('RESET', self.controls_box)
        btn_reset.clicked.connect(self.data_provider.reset)
        self.controls_box.layout().addWidget(btn_reset)


        #### Start  #####################
        self.fps = 0.
        self.lastupdate = time.time()
        self._update()

    def _update(self):

        now = time.time()
        dt = (now-self.lastupdate)
        if dt <= 0:
            dt = 0.000000000001
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.5 + fps2 * 0.5
        tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps )
        self.label.setText(tx)

        dp = self.data_provider
        self.h1.setData(dp.kiln[dp.a:dp.b,0])
        self.h2.setData(dp.coal)
        self.img.setImage(dp())
        QtCore.QTimer.singleShot(1, self._update)


if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())