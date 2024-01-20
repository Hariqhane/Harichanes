import pylab
import numpy as np
from gwpy.timeseries import TimeSeries
ts = TimeSeries(np.zeros(8*4096))
ts.dt=1/4096
tt = ts.spectrogram2(fftlength=0.1,  window='hann') ** (1/2.)
tt=tt.crop_frequencies(30,1625)
qq = np.load('C:/Users/Harichane/Desktop/y_pred_10_z.npy').reshape(-1,160,160)
for i in range(86):
        z=qq[i]
        tt.value[:][:]=z
        plot = tt.plot()
        ax = plot.gca()
        ax.set_yscale('log')
        ax.set_ylim(30,1600)
        ax.colorbar(
        clim=(0, 1),
        #norm="log",
        label=r"normalized strain noise [$1/\sqrt{\mathrm{Hz}}$]",)
        plot.savefig('C:/Users/Harichane/Desktop/tu_1/image_p%s.png'%i)
        plot.close()