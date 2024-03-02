```python
#总体思路：
#高斯噪声对比时域or时频域
#真实噪声对比更合适的数据方式，进行参数推断（H，Ωm定死）(透镜模型以及波形的泛化性)
#真实噪声利用folw尝试信号类型判断
#真实噪声尝试宇宙学参数推断

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import quad
#函数定义
def lcdm_distance_redshift(z, H0, Omega_m, Omega_Lambda):
    c = 299792.458  # 光速，单位：km/s
    # 定义被积函数
    integrand = lambda z: 1 / np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)
    # 积分计算
    result, _ = quad(integrand, 0, z)
    # 计算距离
    distance = c / H0 * result*(1+z)
    return distance


dl_all=[]
for zz in np.linspace(0,10,5000):
    dl_all.append(lcdm_distance_redshift(zz,70,0.3,0.7))
    
f_dl_z = scipy.interpolate.interp1d(dl_all, np.linspace(0,10,5000), kind='linear')

dl_all=[]
for zz in np.linspace(0,10,5000):
    dl_all.append(lcdm_distance_redshift(zz,70,0.3,0.7))

plt.plot(np.linspace(0,10,5000),dl_all)
plt.yscale('log')

#1024——glitch生成
import matplotlib.pyplot as pp
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform,get_fd_waveform
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import sigma
from pycbc import types
from pycbc.detector import Detector
from pycbc.filter import matched_filter
import pycbc
from pycbc.psd import welch
import pycbc.noise
import pycbc.psd
from pycbc import waveform
import scipy
import h5py
import os
import random
import numpy as np

from multiprocessing import Pool

from pycbc import distributions

from scipy.interpolate import interp1d
from gwpy.timeseries import TimeSeries

import numpy as np
from scipy.integrate import quad
#函数定义
def lcdm_distance_redshift(z, H0, Omega_m, Omega_Lambda):
    c = 299792.458  # 光速，单位：km/s
    # 定义被积函数
    integrand = lambda z: 1 / np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)
    # 积分计算
    result, _ = quad(integrand, 0, z)
    # 计算距离
    distance = c / H0 * result*(1+z)
    return distance


dl_all=[]
for zz in np.linspace(0,10,5000):
    dl_all.append(lcdm_distance_redshift(zz,70,0.3,0.7))
    
f_dl_z = scipy.interpolate.interp1d(dl_all, np.linspace(0,10,5000), kind='linear')

import numpy as np
#点质量透镜模型
class Point_mass_lens_model:
    def __init__(self):
        pass
    
    def F_f(self, f, M_LZ, epsilon, D_L, xi_0, D_s):
        return np.abs(self.miu_p(self.y(epsilon, D_L, xi_0, D_s), self.beta(self.y(epsilon, D_L, xi_0, D_s))))**0.5 - \
            1j * np.abs(self.miu_c(self.y(epsilon, D_L, xi_0, D_s), self.beta(self.y(epsilon, D_L, xi_0, D_s))))**0.5 * \
            np.exp(2j * np.pi * f * self.dt(M_LZ, self.y(epsilon, D_L, xi_0, D_s), self.beta(self.y(epsilon, D_L, xi_0, D_s))))
    
    def miu_p(self, y, beta):
        return 0.5 + (y**2 + 2) / (2 * y * beta)
    
    def miu_c(self, y, beta):
        return 0.5 - (y**2 + 2) / (2 * y * beta)
    #G:m³/kg·s²
    def dt(self, M_LZ, y, beta):#单位s
        return 4 * M_LZ*(y * beta / 2 + np.log((beta + y) / (beta - y))) * scipy.constants.G / scipy.constants.c**3 *(1.989e30)
    
    def beta(self, y):
        return (y**2 + 4)**0.5
    
    def y(self, epsilon, D_L, xi_0, D_s):
        return epsilon * D_L / xi_0 / D_s


def M_L_Z(M_L,z):
    return M_L*(1+z)
    
def x_i_0(M_L,D_LS,D_L,D_S): #G:m³/kg·s² #m^1 kg^-1 Mc^1  Mpc^1 单位转化成MPc,,,,1.989×1030 千克,,,3.0857E+22
    return ((4*scipy.constants.G*M_L/scipy.constants.c**2)*(D_LS*D_L/D_S) /(3.0857e22)*(1.989e30))**0.5

def mc_q_to_m1_m2(mc,q):
    m1=(mc**5*(1+q)/q**3)**(1/5)
    m2=m1*q
    return m1,m2


def get_snr(data,T_obs,fs,psd):
    #波形、1、频率、psd
    N = T_obs*fs
    delta_f = 1.0/T_obs
    delta_t = 1.0/fs

#     win = tukey(N,alpha=1.0/8.0)
    idx = np.argwhere(psd==0.0)
    psd[idx] = 1e300    

    xf = np.fft.rfft(data)*delta_t
    #fig = plt.figure()
    #plt.plot(np.real(xf))
    #plt.plot(np.imag(xf))
    SNRsq = 4.0*np.sum((np.abs(xf)**2)/psd)*delta_f
    return np.sqrt(SNRsq)

#固定的参数
calculator = Point_mass_lens_model()
T_obs=10
N_fs=4096#采样率为4096Hz
N_s=T_obs*N_fs#对应时域为10s

flen = round(N_s/2)+1
delta_f = 1.0 / T_obs
delta_t=1/N_fs
f_low=30
flow=30

tsamples = int(T_obs / delta_t)


# 打开txt文件以读取内容
with open('/home/suntianyang/work5/ligo/aligo_O4high.txt', 'r') as file:
    lines = file.readlines()


# 解析数据
data_lines = [line.strip().split() for line in lines]

data = [[float(value) for value in line] for line in data_lines]

list_los = []  # 用于存储第一列数据
list_losvail = []  # 用于存储第二列数据
# 遍历数据并将其添加到相应的列表中
for row in data:
    list_los.append(row[0])  # 将第一列数据添加到list_los
    list_losvail.append(row[1]**2)  # 将第二列数据添加到list_losvail
    
    
psdh = pycbc.psd.aLIGO140MpcT1800545(flen, delta_f, flow)
interp_function = interp1d(list_los, list_losvail, kind='linear')
y_new = interp_function(psdh.sample_frequencies)  # 生成的 y 值
y_new[psdh.sample_frequencies<f_low]=0
psdh.data=y_new
psdh_snr = psdh.copy()

pp.plot(psdh.sample_frequencies, psdh**0.5, label='O4')
pp.yscale('log')
pp.xscale('log') 
pp.xlim(10,2048)
pp.legend()
pp.show()

# 打开txt文件以读取内容
with open('/home/suntianyang/work5/ligo/aligo_O4high.txt', 'r') as file:
    lines = file.readlines()


# 解析数据
data_lines = [line.strip().split() for line in lines]

data = [[float(value) for value in line] for line in data_lines]

list_los = []  # 用于存储第一列数据
list_losvail = []  # 用于存储第二列数据
# 遍历数据并将其添加到相应的列表中
for row in data:
    list_los.append(row[0])  # 将第一列数据添加到list_los
    list_losvail.append(row[1]**2)  # 将第二列数据添加到list_losvail
    
    
psdl = pycbc.psd.aLIGO140MpcT1800545(flen, delta_f, flow)
interp_function = interp1d(list_los, list_losvail, kind='linear')
y_new = interp_function(psdl.sample_frequencies)  # 生成的 y 值
y_new[psdl.sample_frequencies<f_low]=0
psdl.data=y_new
psdl_snr = psdl.copy()


pp.plot(psdl.sample_frequencies, psdl**0.5, label='O4')
pp.yscale('log')
pp.xscale('log') 
pp.xlim(10,2048)
pp.legend()
pp.show()

# 打开txt文件以读取内容
with open('/home/suntianyang/work5/ligo/avirgo_O4high.txt', 'r') as file:
    lines = file.readlines()


# 解析数据
data_lines = [line.strip().split() for line in lines]

data = [[float(value) for value in line] for line in data_lines]

list_los = []  # 用于存储第一列数据
list_losvail = []  # 用于存储第二列数据
# 遍历数据并将其添加到相应的列表中
for row in data:
    list_los.append(row[0])  # 将第一列数据添加到list_los
    list_losvail.append(row[1]**2)  # 将第二列数据添加到list_losvail
    
    
psdv = pycbc.psd.aLIGO140MpcT1800545(flen, delta_f, flow)
interp_function = interp1d(list_los, list_losvail, kind='linear')
y_new = interp_function(psdv.sample_frequencies)  # 生成的 y 值
y_new[psdv.sample_frequencies<f_low]=0
psdv.data=y_new
psdv_snr = psdv.copy()


pp.plot(psdv.sample_frequencies, psdv**0.5, label='O4')
pp.yscale('log')
pp.xscale('log') 
pp.xlim(10,2048)
pp.legend()
pp.show()

# 打开txt文件以读取内容
with open('/home/suntianyang/work5/ligo/k1_o4_high.txt', 'r') as file:
    lines = file.readlines()


# 解析数据
data_lines = [line.strip().split() for line in lines]

data = [[float(value) for value in line] for line in data_lines]

list_los = []  # 用于存储第一列数据
list_losvail = []  # 用于存储第二列数据
# 遍历数据并将其添加到相应的列表中
for row in data:
    list_los.append(row[0])  # 将第一列数据添加到list_los
    list_losvail.append(row[1]**2)  # 将第二列数据添加到list_losvail
    
    
psdk = pycbc.psd.aLIGO140MpcT1800545(flen, delta_f, flow)
interp_function = interp1d(list_los, list_losvail, kind='linear')
y_new = interp_function(psdk.sample_frequencies)  # 生成的 y 值
y_new[psdk.sample_frequencies<f_low]=0
psdk.data=y_new
psdk_snr = psdk.copy()


pp.plot(psdk.sample_frequencies, psdk**0.5, label='O4')
pp.yscale('log')
pp.xscale('log') 
pp.xlim(10,2048)
pp.legend()
pp.show()

psdl.sample_frequencies.data

pp.plot(psdl.sample_frequencies, psdl**0.5, label='ligo')
pp.plot(psdh.sample_frequencies, psdh**0.5, label='ligo')
pp.plot(psdv.sample_frequencies, psdv**0.5, label='V1')
pp.plot(psdk.sample_frequencies, psdk**0.5, label='K1')
pp.yscale('log')
pp.xscale('log') 
pp.xlim(20,2048)
pp.legend()
pp.show()
```
