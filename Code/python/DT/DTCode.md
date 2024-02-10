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
# 导入O4psd
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
#固定的参数
calculator = Point_mass_lens_model()
det_1 = Detector('L1')
det_2 = Detector('H1')
det_3 = Detector('V1')
det_4 = Detector('K1')

def get_wave_plus_gaosnoise_tf(sa):
    hp_flens=types.FrequencySeries(np.zeros(round(N_s/2)+1),delta_f=1.0 / T_obs)#采样定律
    while True:
        epsilon=random.uniform(1e-6,0.5)*1e-6#MPc
        dl_distribution=distributions.power_law.UniformPowerLaw(dim=3,bound =(10,6000))#MPc
        dl_samples=dl_distribution.rvs(size=1)
        D_L=dl_samples[0][0]
        dl_distribution=distributions.power_law.UniformPowerLaw(dim=3,bound =(D_L,6000+D_L))#MPc
        dl_samples=dl_distribution.rvs(size=1)
        D_LS=dl_samples[0][0]-D_L#MPc
        
        M_L=random.uniform(1e3,1e5)#M
        D_S=D_LS+D_L#MPc
        z=f_dl_z(D_L)#MPc_z
        z_s=f_dl_z(D_S)#MPc_z
        M_LZ=M_L_Z(M_L,z)#M
        xi_0=x_i_0(M_L,D_LS,D_L,D_S)
    
        #print(calculator.dt(M_LZ, calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S))))
        #print(calculator.miu_p(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S))))
        #print(calculator.miu_c(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S))))
    
        dtt=calculator.dt(M_LZ, calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S)))
        miu_pp=calculator.miu_p(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S)))
        miu_cc=calculator.miu_c(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S)))
        if dtt<2.25e-3 or dtt>3.52 or miu_pp<1.7 or miu_pp>10.51 or miu_cc<-9.51 or miu_cc>-0.17:
            continue
        
        #print(dtt,miu_pp,miu_cc)
        result = types.FrequencySeries(calculator.F_f(f=hp_flens.sample_frequencies.data, M_LZ=M_LZ, epsilon=epsilon, D_L=D_L, xi_0=xi_0, D_s=D_S),delta_f=N_fs/N_s)
        
        tc=random.uniform(7,8)
        mc_distribution=distributions.MchirpfromUniformMass1Mass2(mc=(5,80))
        q_distribution=distributions.QfromUniformMass1Mass2(q=(0.5,2.0))#0.5-1
        mc_samples = mc_distribution.rvs(size=1)
        q_samples = q_distribution.rvs(size=1)
        mc=mc_samples[0][0]
        q=q_samples[0][0]
        m1,m2=mc_q_to_m1_m2(mc,q)
        distance=D_S
        
        sky_distribution = distributions.sky_location.UniformSky()
        sky_samples=sky_distribution.rvs(size=1)
        dec = sky_samples[0][0]#纬度
        ra = sky_samples[0][1]#经度
        
        psi = random.uniform(0,np.pi*2)#偏振角ψ
        inc=np.arccos(-1.0 + 2.0*np.random.rand())#倾角
        coa_phase=random.uniform(0,np.pi*2)#合并相位
        #IMRPhenomPv2，SEOBNRv4_opt,IMRPhenomTPHM
        try:
            hp, hc = get_td_waveform(approximant='IMRPhenomTPHM',
                                     mass1=m1*(1+z_s),#红移质量1
                                     mass2=m2*(1+z_s),#红移质量2
                                     distance=distance,#距离，MPC
                                     coa_phase=coa_phase,#合并相位
                                     inclination=inc,#轨道和视线的夹角
                                     spin1x=0,
                                     spin1y=0,
                                     spin1z=0,#自旋1
                                     spin2x=0,
                                     spin2y=0,
                                     spin2z=0,#自旋2
                                     eccentricity=0,#轨道偏心率
                                     lambda1=0,#潮汐相，中子星有
                                     lambda2=0,
                                     delta_t=1.0/N_fs,
                                     f_lower=30)
        except:
            continue
        fp_1, fc_1 = det_1.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        fp_2, fc_2 = det_2.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        fp_3, fc_3 = det_3.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        fp_4, fc_4 = det_4.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        
        ht_1 = fp_1*hp + fc_1*hc
        ht_2 = fp_2*hp + fc_2*hc
        ht_3 = fp_3*hp + fc_3*hc
        ht_4 = fp_4*hp + fc_4*hc


​        
        ht_1.resize(N_s)
        ht_1=ht_1.cyclic_time_shift(ht_1.start_time+tc)
        ht_1.start_time=0
        ht_2.resize(N_s)
        ht_2=ht_2.cyclic_time_shift(ht_2.start_time+tc)
        ht_2.start_time=0
        ht_3.resize(N_s)
        ht_3=ht_3.cyclic_time_shift(ht_3.start_time+tc)
        ht_3.start_time=0
        ht_4.resize(N_s)
        ht_4=ht_4.cyclic_time_shift(ht_4.start_time+tc)
        ht_4.start_time=0
        
        hp_f_1=ht_1.to_frequencyseries()
        hp_flens_1=hp_f_1*result
        hp_t_lens_1=hp_flens_1.to_timeseries()
        #snr1 = get_snr(hp,T_obs,N_fs,psdl_snr.data)
        hp_f_2=ht_2.to_frequencyseries()
        hp_flens_2=hp_f_2*result
        hp_t_lens_2=hp_flens_2.to_timeseries()
        #snr1 = get_snr(hp,T_obs,N_fs,psdl_snr.data)
        hp_f_3=ht_3.to_frequencyseries()
        hp_flens_3=hp_f_3*result
        hp_t_lens_3=hp_flens_3.to_timeseries()
        #snr1 = get_snr(hp,T_obs,N_fs,psdl_snr.data)
        hp_f_4=ht_4.to_frequencyseries()
        hp_flens_4=hp_f_4*result
        hp_t_lens_4=hp_flens_4.to_timeseries()


​        
        #利用高斯 psd生成噪声：
        noise_et1 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdl)
        #添加非高斯噪声后要用welch方法估计psd——假设没有方法能合理的估计psd（psd中包含各种非高斯的干扰）
        hp_t_lens_plusnoise_1=hp_t_lens_1.copy()
        hp_t_lens_plusnoise_1.data+=noise_et1.data
        
        noise_et2 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdh)
        hp_t_lens_plusnoise_2=hp_t_lens_2.copy()
        hp_t_lens_plusnoise_2.data+=noise_et2.data
        
        noise_et3 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdv)
        hp_t_lens_plusnoise_3=hp_t_lens_3.copy()
        hp_t_lens_plusnoise_3.data+=noise_et3.data
        
        noise_et4 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdk)
        hp_t_lens_plusnoise_4=hp_t_lens_4.copy()
        hp_t_lens_plusnoise_4.data+=noise_et4.data
        
        snr1 = get_snr(hp_t_lens_1,T_obs,N_fs,psdl_snr.data)
        snr2 = get_snr(hp_t_lens_2,T_obs,N_fs,psdh_snr.data)
        snr3 = get_snr(hp_t_lens_3,T_obs,N_fs,psdv_snr.data)
        snr4 = get_snr(hp_t_lens_4,T_obs,N_fs,psdk_snr.data)
        snr=(snr1**2+snr2**2+snr3**2+snr4**2)**0.5
        #print(snr)
        
        #print(snr)
        if snr<16:
            continue
            
        noise=TimeSeries(hp_t_lens_plusnoise_1.data)
        noise.dt=1/N_fs
        arr_1 = noise.q_transform(qrange=(35, 50),tres=0.01,fres=1.865)
        
        #print(arr_1.frequencies)
        
        noise=TimeSeries(hp_t_lens_plusnoise_2.data)
        noise.dt=1/N_fs
        arr_2 = noise.q_transform(qrange=(35, 50),tres=0.01,fres=1.865)
        
        noise=TimeSeries(hp_t_lens_plusnoise_3.data)
        noise.dt=1/N_fs
        arr_3 = noise.q_transform(qrange=(35, 50),tres=0.01,fres=1.865)
    
        noise=TimeSeries(hp_t_lens_plusnoise_4.data)
        noise.dt=1/N_fs
        arr_4 = noise.q_transform(qrange=(35, 50),tres=0.01,fres=1.865)
        
        canshu=[epsilon,D_L,distance,M_L,mc,q,dec,ra,psi,inc,coa_phase,tc]#12个
        return [(arr_1),(arr_2),(arr_3),(arr_4)],canshu
        #return [np.array(arr_1.data),np.array(arr_2.data),np.array(arr_3.data),np.array(arr_4.data)],canshu
hp_t_lens_plusnoise,canshu=get_wave_plus_gaosnoise_tf(11)
#hp_t_lens_plusnoise[0].plot();
#plt.xlim(23,27)
#plt.yscale('log')
print(hp_t_lens_plusnoise[0].data.shape)
hp_t_lens_plusnoise[0].plot();
hp_t_lens_plusnoise[1].plot();
hp_t_lens_plusnoise[2].plot();
hp_t_lens_plusnoise[3].plot();
(30*4096)**0.5
hp_t_lens_plusnoise.data.shape
del hp_t_lens_plusnoise,canshu
# 生成的流程
%matplotlib inline

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils
import zuko
from torch.optim import lr_scheduler


from itertools import islice
from tqdm import tqdm

from lampe.data import JointLoader
from lampe.inference import NPE, NPELoss
from lampe.plots import nice_rc, corner, mark_point
from lampe.utils import GDStep
from lampe.data import H5Dataset
from lampe.diagnostics import expected_coverage_mc

import h5py
import numpy as np

from datetime import datetime
#抽样器1500训练
LABELS = [r'$epsilon$',r'$D_L$',r'$distance$',r'$M_L$',r'$Mc$',r'$q$',r'$dec$',r'$ra$',r'$psi$',r'$inc$',r'$coa_phase$',r'$tc$']

LOWER = torch.tensor([1e-12  , 10. ,20. ,1e3 ,5.  ,0.5 ,-0.5*np.pi   ,0       ,0       ,0       ,0       ,7])
UPPER = torch.tensor([0.5e-6 , 6e3 ,1.2e4 ,1e5 ,80. ,2.  ,0.5*np.pi    ,2*np.pi ,2*np.pi ,2*np.pi ,2*np.pi ,8])
        
#参数归一化与逆运算
def preprocess(theta: torch.Tensor) -> torch.Tensor:
    return 2 * (theta - LOWER) / (UPPER - LOWER) - 1

def postprocess(theta: torch.Tensor) -> torch.Tensor:
    return (theta + 1) / 2 * (UPPER - LOWER) + LOWER
from multiprocessing import Pool
import psutil
process = psutil.Process()
memory_info = process.memory_info()
memory_usage = memory_info.rss / 1024/1024/1024  # 转换为KB
print(f"当前程序占用的内存：{memory_usage} GB")
estimator = NPE(3, 2, transforms=3, hidden_features=[64] * 3)
estimator

#非采样器的
torch.backends.cudnn.deterministic = True #禁用 cuDNN 的随机性，从而保证每次运行的结果都是相同的。

aaaaa=[]

import gc

#  "act<class 'torch.nn.modules.activation.ReLU'>学习率0.0003,正则化0.001,休眠率0.2,参数数目1024,参数层数7,流层9,中间层256,最佳损失-5.820067882537842,训练周期28",
#"act<class 'torch.nn.modules.activation.ReLU'>学习率0.0003,正则化0,休眠率0.2,参数数目1024,参数层数7,流层9,中间层512,最佳损失-3.325531482696533,训练周期35",
for act in [nn.ReLU]:
  for f in [0.0002]:
    for weight_decay in [0]:
       for liu in [zuko.flows.NSF]:
        for transfomr in [7]:#,7
         for num in [4096]:
          for trans in [9]:
           for beishu in [2048]:
            
            now= datetime.now()
            def weight_init(m):
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight.data)#kaiming_normal_///xavier_uniform_
                    nn.init.constant_(m.bias.data, 0.0)
    
            class Bottlrneck(torch.nn.Module):
                def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
                    super(Bottlrneck, self).__init__()
                    self.stride = 1
                    if downsample == True:
                        self.stride = 2
                    #在这里添加BatchNorm1d和Dropout是最合适的
                    self.layer = torch.nn.Sequential(
                        torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
                        torch.nn.BatchNorm1d(Med_channel),
                        torch.nn.ReLU(),
                        torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
                        torch.nn.BatchNorm1d(Med_channel),
                        #torch.nn.ReLU(),
                        torch.nn.Conv1d(Med_channel, Out_channel, 1),
                        torch.nn.BatchNorm1d(Out_channel),
                        #torch.nn.ReLU(),
                    )
    
                    if In_channel != Out_channel:
                        self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
                    else:
                        self.res_layer = None
    
                    self.jia_relu=torch.nn.Sequential(torch.nn.ReLU())
    
                def forward(self,x):
                    if self.res_layer is not None:
                        residual = self.res_layer(x)
                    else:
                        residual = x
                    return self.jia_relu(self.layer(x)+residual)
                
            class Bottlrneck2d(torch.nn.Module):
                def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
                    super(Bottlrneck2d, self).__init__()
                    self.stride = 1
                    if downsample == True:
                        self.stride = 2
    
                    self.layer = torch.nn.Sequential(
                        #torch.nn.Dropout(Dro),#夹
                        torch.nn.Conv2d(In_channel, Med_channel, 1, self.stride),
                        torch.nn.BatchNorm2d(Med_channel),
                        torch.nn.ReLU(),
                        #torch.nn.Dropout(Dro),#夹
                        torch.nn.Conv2d(Med_channel, Med_channel, 3, padding=1),
                        torch.nn.BatchNorm2d(Med_channel),
                        #torch.nn.ReLU(),
                        #torch.nn.Dropout(Dro),#夹
                        torch.nn.Conv2d(Med_channel, Out_channel, 1),
                        torch.nn.BatchNorm2d(Out_channel),
                        #torch.nn.ReLU(),
                    )
    
                    if In_channel != Out_channel:
                        self.res_layer = torch.nn.Conv2d(In_channel, Out_channel,1,self.stride)
                    else:
                        self.res_layer = None
    
                def forward(self,x):
                    if self.res_layer is not None:
                        residual = self.res_layer(x)
                    else:
                        residual = x
                    return self.layer(x)+residual
    
            class ResNet(torch.nn.Module):
                def __init__(self,in_channels=1,classes=5):
                    super(ResNet, self).__init__()
                    self.classifer = torch.nn.Sequential(
                        torch.nn.Linear(2048,classes)#变成每类特征的信息
                    )
                    
                    self.features = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3),
                        torch.nn.MaxPool2d(3,2,1),
    
                        Bottlrneck2d(64, 64, 256, False),
                        Bottlrneck2d(256, 64, 256, False),
                        Bottlrneck2d(256, 64, 256, False),
                        #
                        Bottlrneck2d(256, 128, 512, True),
                        Bottlrneck2d(512, 128, 512, False),
                        Bottlrneck2d(512, 128, 512, False),
                        Bottlrneck2d(512, 128, 512, False),
                        #
                        Bottlrneck2d(512, 256, 1024, True),
                        Bottlrneck2d(1024, 256, 1024, False),
                        Bottlrneck2d(1024, 256, 1024, False),
                        Bottlrneck2d(1024, 256, 1024, False),
                        Bottlrneck2d(1024, 256, 1024, False),
                        Bottlrneck2d(1024, 256, 1024, False),
                        #
                        Bottlrneck2d(1024, 512, 2048, True),
                        Bottlrneck2d(2048, 512, 2048, False),
                        Bottlrneck2d(2048, 512, 2048, False),
    
                        torch.nn.AdaptiveAvgPool2d(1)
                    )


                def forward(self,x):
                    x = self.features(x)
                    x = x.view(-1,2048)
                    x = self.classifer(x)
                    return x
    
            class NPEWithEmbedding(nn.Module):#这个网络只要是1维的2的倍数就行
                def __init__(self,channels=3,beishu=4,canshu=2,build=zuko.flows.NSF,hidden_features=[128] * 3,activation=nn.ELU,transforms=3):
                    super().__init__()
    
                    self.npe = NPE(canshu, beishu, build=build, hidden_features=hidden_features,transforms=transforms, activation=activation)#用于
                    self.embedding = ResNet(in_channels=channels,classes=beishu)
    
                def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                    #print(self.embedding(x).shape)
    
                    return self.npe(theta, self.embedding(x))
    
                def flow(self, x: torch.Tensor):  # -> flow对应原来的采样，因为他调用的就是flow
                    return self.npe.flow(self.embedding(x))


            torch.manual_seed(2234)#重置参数，在循环里面可以保证可以复现
            estimator= NPEWithEmbedding(channels=4,canshu=12,beishu=beishu,build=liu,hidden_features=[num] * transfomr,transforms=trans,activation=act).cuda()#beishu是残差网络输出尺寸（npe的输入参数）
            estimator.apply(weight_init)
            
            optimizer = optim.AdamW(estimator.parameters(), lr=f,weight_decay=weight_decay)#学习率！！！！！！！！！！！
            #在优化器选项里添加正则化,weight_decay=0.01，l2正则化
    
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)#lr_scheduler.ExponentialLR(optimizer, gamma=0.99)#学习率衰减
            #https://zhuanlan.zhihu.com/p/363338422
    
            step = GDStep(optimizer, clip=1.0)  # gradient descent step with gradient clipping,有了他不用optimizer.step
            loss = NPELoss(estimator)


            list_los = []  # 用于存储第一列数据
            list_losvail = []  # 用于存储第二列数据


​            
            with tqdm(range(2000), unit='epoch') as tq:#epoch
                best_loss = np.inf#早停
                best_model_weights = None
                patience=50#10个周期不降就停
                i=0
                i_numca=0
                num=6000#总数
                num_v=1200
                datast=15#batch_size
                nnum=num//datast
                nnum_v=num_v//datast
                for epoch in tq:
                    #losses = torch.stack([
                    #    step(loss(preprocess(theta).cuda(), x.cuda()))
                    #    for theta, x in trainset # 这样写是遍历全部元素，实例那样是因为他是采样器
                    #])
    
                    pool = Pool(processes=56)
                    results = pool.map(get_wave_plus_gaosnoise_tf, range(num))
                    all_x = torch.tensor(np.concatenate([x for x,y in results]).reshape(nnum,datast,4,600,600),dtype=torch.float32)
                    all_y = torch.tensor(np.concatenate([y for x,y in results]).reshape(nnum,datast,12),dtype=torch.float32)
    
                    results = pool.map(get_wave_plus_gaosnoise_tf, range(num_v))
                    all_x_vail = torch.tensor(np.concatenate([x for x,y in results]).reshape(nnum_v,datast,4,600,600),dtype=torch.float32)
                    all_y_vail = torch.tensor(np.concatenate([y for x,y in results]).reshape(nnum_v,datast,12),dtype=torch.float32)
    
                    pool.close()
                    pool.join()
    
                    del results
                    
                    losses = torch.stack([
                        step(loss(preprocess(all_y[i]).cuda(), all_x[i].cuda()))
                        for i in range(nnum) # 这样写是遍历全部元素，实例那样是因为他是采样器
                    ])
                    
                    del all_x,all_y
                    
                    estimator.eval()
                    with torch.no_grad():
                        val_losses = torch.stack([
                            loss(preprocess(all_y_vail[i]).cuda(), all_x_vail[i].cuda())
                            for i in range(nnum_v)
                        ])
                    
                    scheduler.step()#学习率衰减
    
                    tq.set_postfix(loss=losses.mean().item(), val_loss=val_losses.mean().item())
    
                    del all_x_vail,all_y_vail
                    
                    los=losses.mean().item()#类型不对，所以换名字
                    losval=val_losses.mean().item()#类型不对，所以换名字
                    
                    list_los.append(los)#话损失函数图
                    list_losvail.append(losval)#话损失函数图
    
                    # 储存监视
                    data1 = list_los
                    data2 = list_losvail
                    last= now
                    now = datetime.now()
                    # 打开一个文件用于写入
                    file = open('/home/suntianyang/work5/ligo/net/data_train_tf.txt', 'w')
                    file.write('last_last:'+str(last) + '\n')
                    file.write('___last__:'+str(now) + '\n')
                    
                    # 将每个元素写入文件中
                    for item1, item2 in zip(data1, data2):
                        file.write(str(item1) + ' ' + str(item2) + '\n')
                    
                    # 关闭文件
                    file.close()
                    
                    del losses,data1,data2


​                    
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    memory_usage = memory_info.rss / 1024/1024/1024  # 转换为KB
                    print(f"当前程序占用的内存：{memory_usage} GB")
                    
                    optimizer.zero_grad()
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.save(estimator.state_dict(),f'/home/suntianyang/work5/ligo/net/NPE_mid_tf_hou.pth')
                    
                    if losval < best_loss:
                        best_loss = losval
                        epochs_without_improvement = 0
                        best_model_weights = estimator.state_dict()
                    else:
                        epochs_without_improvement += 1


​                    
                    # 如果验证集上的损失连续patience个epoch没有提高，则停止训练
                    if epochs_without_improvement == patience:
                        estimator.load_state_dict(best_model_weights)
                        print('Early stopping at epoch {}...'.format(epoch-patience+1))
                        break


            aaaaa.append('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-10],len(list_los)-10))
            print('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-10],len(list_los)-10))
            torch.cuda.empty_cache()
# 1d网络
#固定的参数
calculator = Point_mass_lens_model()
det_1 = Detector('L1')
det_2 = Detector('H1')
det_3 = Detector('V1')
det_4 = Detector('K1')

def get_wave_plus_gaosnoise_t(sa):
    hp_flens=types.FrequencySeries(np.zeros(round(N_s/2)+1),delta_f=1.0 / T_obs)#采样定律
    while True:
        epsilon=random.uniform(1e-6,0.5)*1e-6#MPc
        dl_distribution=distributions.power_law.UniformPowerLaw(dim=3,bound =(10,6000))#MPc
        dl_samples=dl_distribution.rvs(size=1)
        D_L=dl_samples[0][0]
        dl_distribution=distributions.power_law.UniformPowerLaw(dim=3,bound =(D_L,6000+D_L))#MPc
        dl_samples=dl_distribution.rvs(size=1)
        D_LS=dl_samples[0][0]-D_L#MPc
        
        M_L=random.uniform(1e3,1e5)#M
        D_S=D_LS+D_L#MPc
        z=f_dl_z(D_L)#MPc_z
        z_s=f_dl_z(D_S)#MPc_z
        M_LZ=M_L_Z(M_L,z)#M
        xi_0=x_i_0(M_L,D_LS,D_L,D_S)
    
        #print(calculator.dt(M_LZ, calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S))))
        #print(calculator.miu_p(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S))))
        #print(calculator.miu_c(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S))))
    
        dtt=calculator.dt(M_LZ, calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S)))
        miu_pp=calculator.miu_p(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S)))
        miu_cc=calculator.miu_c(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S)))
        if dtt<2.25e-3 or dtt>3.52 or miu_pp<1.7 or miu_pp>10.51 or miu_cc<-9.51 or miu_cc>-0.17:
            continue
        
        #print(dtt,miu_pp,miu_cc)
        result = types.FrequencySeries(calculator.F_f(f=hp_flens.sample_frequencies.data, M_LZ=M_LZ, epsilon=epsilon, D_L=D_L, xi_0=xi_0, D_s=D_S),delta_f=N_fs/N_s)
        
        tc=random.uniform(7,8)
        mc_distribution=distributions.MchirpfromUniformMass1Mass2(mc=(5,80))
        q_distribution=distributions.QfromUniformMass1Mass2(q=(0.5,2.0))#0.5-1
        mc_samples = mc_distribution.rvs(size=1)
        q_samples = q_distribution.rvs(size=1)
        mc=mc_samples[0][0]
        q=q_samples[0][0]
        m1,m2=mc_q_to_m1_m2(mc,q)
        distance=D_S
        
        sky_distribution = distributions.sky_location.UniformSky()
        sky_samples=sky_distribution.rvs(size=1)
        dec = sky_samples[0][0]#纬度
        ra = sky_samples[0][1]#经度
        
        psi = random.uniform(0,np.pi*2)#偏振角ψ
        inc=np.arccos(-1.0 + 2.0*np.random.rand())#倾角
        coa_phase=random.uniform(0,np.pi*2)#合并相位
        #IMRPhenomPv2，SEOBNRv4_opt,IMRPhenomTPHM
        try:
            hp, hc = get_td_waveform(approximant='IMRPhenomTPHM',
                                     mass1=m1*(1+z_s),#红移质量1
                                     mass2=m2*(1+z_s),#红移质量2
                                     distance=distance,#距离，MPC
                                     coa_phase=coa_phase,#合并相位
                                     inclination=inc,#轨道和视线的夹角
                                     spin1x=0,
                                     spin1y=0,
                                     spin1z=0,#自旋1
                                     spin2x=0,
                                     spin2y=0,
                                     spin2z=0,#自旋2
                                     eccentricity=0,#轨道偏心率
                                     lambda1=0,#潮汐相，中子星有
                                     lambda2=0,
                                     delta_t=1.0/N_fs,
                                     f_lower=30)
        except:
            continue
            
        fp_1, fc_1 = det_1.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        fp_2, fc_2 = det_2.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        fp_3, fc_3 = det_3.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        fp_4, fc_4 = det_4.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        
        ht_1 = fp_1*hp + fc_1*hc
        ht_2 = fp_2*hp + fc_2*hc
        ht_3 = fp_3*hp + fc_3*hc
        ht_4 = fp_4*hp + fc_4*hc


​        
        ht_1.resize(N_s)
        ht_1=ht_1.cyclic_time_shift(ht_1.start_time+tc)
        ht_1.start_time=0
        ht_2.resize(N_s)
        ht_2=ht_2.cyclic_time_shift(ht_2.start_time+tc)
        ht_2.start_time=0
        ht_3.resize(N_s)
        ht_3=ht_3.cyclic_time_shift(ht_3.start_time+tc)
        ht_3.start_time=0
        ht_4.resize(N_s)
        ht_4=ht_4.cyclic_time_shift(ht_4.start_time+tc)
        ht_4.start_time=0
        
        hp_f_1=ht_1.to_frequencyseries()
        hp_flens_1=hp_f_1*result
        hp_t_lens_1=hp_flens_1.to_timeseries()
        #snr1 = get_snr(hp,T_obs,N_fs,psdl_snr.data)
        hp_f_2=ht_2.to_frequencyseries()
        hp_flens_2=hp_f_2*result
        hp_t_lens_2=hp_flens_2.to_timeseries()
        #snr1 = get_snr(hp,T_obs,N_fs,psdl_snr.data)
        hp_f_3=ht_3.to_frequencyseries()
        hp_flens_3=hp_f_3*result
        hp_t_lens_3=hp_flens_3.to_timeseries()
        #snr1 = get_snr(hp,T_obs,N_fs,psdl_snr.data)
        hp_f_4=ht_4.to_frequencyseries()
        hp_flens_4=hp_f_4*result
        hp_t_lens_4=hp_flens_4.to_timeseries()


​        
        #利用高斯 psd生成噪声：
        noise_et1 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdl)
        #添加非高斯噪声后要用welch方法估计psd——假设没有方法能合理的估计psd（psd中包含各种非高斯的干扰）
        hp_t_lens_plusnoise_1=hp_t_lens_1.copy()
        hp_t_lens_plusnoise_1.data+=noise_et1.data
        
        noise_et2 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdh)
        hp_t_lens_plusnoise_2=hp_t_lens_2.copy()
        hp_t_lens_plusnoise_2.data+=noise_et2.data
        
        noise_et3 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdv)
        hp_t_lens_plusnoise_3=hp_t_lens_3.copy()
        hp_t_lens_plusnoise_3.data+=noise_et3.data
        
        noise_et4 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdk)
        hp_t_lens_plusnoise_4=hp_t_lens_4.copy()
        hp_t_lens_plusnoise_4.data+=noise_et4.data
        
        snr1 = get_snr(hp_t_lens_1,T_obs,N_fs,psdl_snr.data)
        snr2 = get_snr(hp_t_lens_2,T_obs,N_fs,psdh_snr.data)
        snr3 = get_snr(hp_t_lens_3,T_obs,N_fs,psdv_snr.data)
        snr4 = get_snr(hp_t_lens_4,T_obs,N_fs,psdk_snr.data)
        snr=(snr1**2+snr2**2+snr3**2+snr4**2)**0.5
        #print(snr)
        
        #print(snr)
        if snr<16:
            continue
            
        L11data = (hp_t_lens_plusnoise_1.to_frequencyseries() / psdl_snr**0.5).to_timeseries()#要除以0置成正无穷的
        H11data = (hp_t_lens_plusnoise_2.to_frequencyseries() / psdh_snr**0.5).to_timeseries()#要除以0置成正无穷的
        V11data = (hp_t_lens_plusnoise_3.to_frequencyseries() / psdv_snr**0.5).to_timeseries()#要除以0置成正无穷的
        K11data = (hp_t_lens_plusnoise_4.to_frequencyseries() / psdk_snr**0.5).to_timeseries()#要除以0置成正无穷的
    
        canshu=[epsilon,D_L,distance,M_L,mc,q,dec,ra,psi,inc,coa_phase,tc]#12个
        
        return [L11data.data,H11data.data,V11data.data,K11data.data],canshu
adad,g=get_wave_plus_gaosnoise_t(11)
plt.plot(adad[0])
#plt.xlim(7*4096,8*4096)
#plt.plot(adad[1])
#plt.plot(adad[2])
#plt.plot(adad[3])
#固定的参数
calculator = Point_mass_lens_model()
det_1 = Detector('L1')
det_2 = Detector('H1')
det_3 = Detector('V1')
det_4 = Detector('K1')

def get_wave_plus_gaosnoise_t_lens(sa):
    hp_flens=types.FrequencySeries(np.zeros(round(N_s/2)+1),delta_f=1.0 / T_obs)#采样定律
    while True:
        epsilon=random.uniform(1e-6,0.5)*1e-6#MPc
        dl_distribution=distributions.power_law.UniformPowerLaw(dim=3,bound =(10,6000))#MPc
        dl_samples=dl_distribution.rvs(size=1)
        D_L=dl_samples[0][0]
        dl_distribution=distributions.power_law.UniformPowerLaw(dim=3,bound =(D_L,6000+D_L))#MPc
        dl_samples=dl_distribution.rvs(size=1)
        D_LS=dl_samples[0][0]-D_L#MPc
        
        M_L=random.uniform(1e3,1e5)#M
        D_S=D_LS+D_L#MPc
        z=f_dl_z(D_L)#MPc_z
        z_s=f_dl_z(D_S)#MPc_z
        M_LZ=M_L_Z(M_L,z)#M
        xi_0=x_i_0(M_L,D_LS,D_L,D_S)
    
        #print(calculator.dt(M_LZ, calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S))))
        #print(calculator.miu_p(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S))))
        #print(calculator.miu_c(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S))))
    
        dtt=calculator.dt(M_LZ, calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S)))
        miu_pp=calculator.miu_p(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S)))
        miu_cc=calculator.miu_c(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S)))
        yyy_uuu=calculator.y(epsilon, D_L, xi_0, D_S)
        if dtt<2.25e-3 or dtt>3.52 or miu_pp<1.7 or miu_pp>10.51 or miu_cc<-9.51 or miu_cc>-0.17:
            continue
        
        #print(dtt,miu_pp,miu_cc)
        result = types.FrequencySeries(calculator.F_f(f=hp_flens.sample_frequencies.data, M_LZ=M_LZ, epsilon=epsilon, D_L=D_L, xi_0=xi_0, D_s=D_S),delta_f=N_fs/N_s)
        
        tc=random.uniform(7,8)
        mc_distribution=distributions.MchirpfromUniformMass1Mass2(mc=(5,80))
        q_distribution=distributions.QfromUniformMass1Mass2(q=(0.5,2.0))#0.5-1
        mc_samples = mc_distribution.rvs(size=1)
        q_samples = q_distribution.rvs(size=1)
        mc=mc_samples[0][0]
        q=q_samples[0][0]
        m1,m2=mc_q_to_m1_m2(mc,q)
        distance=D_S
        
        sky_distribution = distributions.sky_location.UniformSky()
        sky_samples=sky_distribution.rvs(size=1)
        dec = sky_samples[0][0]#纬度
        ra = sky_samples[0][1]#经度
        
        psi = random.uniform(0,np.pi*2)#偏振角ψ
        inc=np.arccos(-1.0 + 2.0*np.random.rand())#倾角
        coa_phase=random.uniform(0,np.pi*2)#合并相位
        #IMRPhenomPv2，SEOBNRv4_opt,IMRPhenomTPHM
        try:
            hp, hc = get_td_waveform(approximant='IMRPhenomTPHM',
                                     mass1=m1*(1+z_s),#红移质量1
                                     mass2=m2*(1+z_s),#红移质量2
                                     distance=distance,#距离，MPC
                                     coa_phase=coa_phase,#合并相位
                                     inclination=inc,#轨道和视线的夹角
                                     spin1x=0,
                                     spin1y=0,
                                     spin1z=0,#自旋1
                                     spin2x=0,
                                     spin2y=0,
                                     spin2z=0,#自旋2
                                     eccentricity=0,#轨道偏心率
                                     lambda1=0,#潮汐相，中子星有
                                     lambda2=0,
                                     delta_t=1.0/N_fs,
                                     f_lower=30)
        except:
            continue
            
        fp_1, fc_1 = det_1.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        fp_2, fc_2 = det_2.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        fp_3, fc_3 = det_3.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        fp_4, fc_4 = det_4.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        
        ht_1 = fp_1*hp + fc_1*hc
        ht_2 = fp_2*hp + fc_2*hc
        ht_3 = fp_3*hp + fc_3*hc
        ht_4 = fp_4*hp + fc_4*hc


​        
        ht_1.resize(N_s)
        ht_1=ht_1.cyclic_time_shift(ht_1.start_time+tc)
        ht_1.start_time=0
        ht_2.resize(N_s)
        ht_2=ht_2.cyclic_time_shift(ht_2.start_time+tc)
        ht_2.start_time=0
        ht_3.resize(N_s)
        ht_3=ht_3.cyclic_time_shift(ht_3.start_time+tc)
        ht_3.start_time=0
        ht_4.resize(N_s)
        ht_4=ht_4.cyclic_time_shift(ht_4.start_time+tc)
        ht_4.start_time=0
        
        hp_f_1=ht_1.to_frequencyseries()
        hp_flens_1=hp_f_1*result
        hp_t_lens_1=hp_flens_1.to_timeseries()
        #snr1 = get_snr(hp,T_obs,N_fs,psdl_snr.data)
        hp_f_2=ht_2.to_frequencyseries()
        hp_flens_2=hp_f_2*result
        hp_t_lens_2=hp_flens_2.to_timeseries()
        #snr1 = get_snr(hp,T_obs,N_fs,psdl_snr.data)
        hp_f_3=ht_3.to_frequencyseries()
        hp_flens_3=hp_f_3*result
        hp_t_lens_3=hp_flens_3.to_timeseries()
        #snr1 = get_snr(hp,T_obs,N_fs,psdl_snr.data)
        hp_f_4=ht_4.to_frequencyseries()
        hp_flens_4=hp_f_4*result
        hp_t_lens_4=hp_flens_4.to_timeseries()


​        
        #利用高斯 psd生成噪声：
        noise_et1 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdl)
        #添加非高斯噪声后要用welch方法估计psd——假设没有方法能合理的估计psd（psd中包含各种非高斯的干扰）
        hp_t_lens_plusnoise_1=hp_t_lens_1.copy()
        hp_t_lens_plusnoise_1.data+=noise_et1.data
        
        noise_et2 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdh)
        hp_t_lens_plusnoise_2=hp_t_lens_2.copy()
        hp_t_lens_plusnoise_2.data+=noise_et2.data
        
        noise_et3 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdv)
        hp_t_lens_plusnoise_3=hp_t_lens_3.copy()
        hp_t_lens_plusnoise_3.data+=noise_et3.data
        
        noise_et4 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdk)
        hp_t_lens_plusnoise_4=hp_t_lens_4.copy()
        hp_t_lens_plusnoise_4.data+=noise_et4.data
        
        snr1 = get_snr(hp_t_lens_1,T_obs,N_fs,psdl_snr.data)
        snr2 = get_snr(hp_t_lens_2,T_obs,N_fs,psdh_snr.data)
        snr3 = get_snr(hp_t_lens_3,T_obs,N_fs,psdv_snr.data)
        snr4 = get_snr(hp_t_lens_4,T_obs,N_fs,psdk_snr.data)
        snr=(snr1**2+snr2**2+snr3**2+snr4**2)**0.5
        #print(snr)
        
        #print(snr)
        if snr<16 and snr<50:
            continue
            
        L11data = (hp_t_lens_plusnoise_1.to_frequencyseries() / psdl_snr**0.5).to_timeseries()#要除以0置成正无穷的
        H11data = (hp_t_lens_plusnoise_2.to_frequencyseries() / psdh_snr**0.5).to_timeseries()#要除以0置成正无穷的
        V11data = (hp_t_lens_plusnoise_3.to_frequencyseries() / psdv_snr**0.5).to_timeseries()#要除以0置成正无穷的
        K11data = (hp_t_lens_plusnoise_4.to_frequencyseries() / psdk_snr**0.5).to_timeseries()#要除以0置成正无穷的
    
        canshu=[yyy_uuu,miu_pp,miu_cc]#12个
        
        return [L11data.data,H11data.data,V11data.data,K11data.data],canshu
miu_pp<1.7
miu_pp>10.51

miu_cc<-9.51
miu_cc>-0.17
from multiprocessing import Pool
import multiprocessing as mp
#pool = Pool(processes=48)
from astropy.utils.iers import IERS_Auto
iers_a = IERS_Auto.open()

import warnings
from astropy.utils.iers import IERSStaleWarning

# 忽略 IERSStaleWarning
warnings.filterwarnings('ignore', category=IERSStaleWarning)
import psutil

process = psutil.Process()
memory_info = process.memory_info()
memory_usage = memory_info.rss / 1024/1024/1024  # 转换为KB

print(f"当前程序占用的内存：{memory_usage} GB")
pool = Pool(processes=56)
results = pool.map(get_wave_plus_gaosnoise_t_lens, range(5000))
all_x = torch.tensor(np.concatenate([x for x,y in results]).reshape(5000,4,4096*10),dtype=torch.float32)
all_y = torch.tensor(np.concatenate([y for x,y in results]).reshape(5000,3),dtype=torch.float32)
del results
pool.close()
pool.join()

#非采样器的
torch.backends.cudnn.deterministic = True #禁用 cuDNN 的随机性，从而保证每次运行的结果都是相同的。

aaaaa=[]

import gc

#  "act<class 'torch.nn.modules.activation.ReLU'>学习率0.0003,正则化0.001,休眠率0.2,参数数目1024,参数层数7,流层9,中间层256,最佳损失-5.820067882537842,训练周期28",
#"act<class 'torch.nn.modules.activation.ReLU'>学习率0.0003,正则化0,休眠率0.2,参数数目1024,参数层数7,流层9,中间层512,最佳损失-3.325531482696533,训练周期35",
for act in [nn.ReLU]:
  for f in [0.0001]:
    for weight_decay in [0]:
       for liu in [zuko.flows.NSF]:
        for transfomr in [7]:#,7
         for num in [4096]:
          for trans in [9]:
           for beishu in [2048]:
            now= datetime.now()
            def weight_init(m):
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight.data)#kaiming_normal_///xavier_uniform_
                    nn.init.constant_(m.bias.data, 0.0)

            class Bottlrneck(torch.nn.Module):
                def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
                    super(Bottlrneck, self).__init__()
                    self.stride = 1
                    if downsample == True:
                        self.stride = 2
                    #在这里添加BatchNorm1d和Dropout是最合适的
                    self.layer = torch.nn.Sequential(
                        torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
                        torch.nn.BatchNorm1d(Med_channel),
                        torch.nn.ReLU(),
                        torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
                        torch.nn.BatchNorm1d(Med_channel),
                        #torch.nn.ReLU(),
                        torch.nn.Conv1d(Med_channel, Out_channel, 1),
                        torch.nn.BatchNorm1d(Out_channel),
                        #torch.nn.ReLU(),
                    )
    
                    if In_channel != Out_channel:
                        self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
                    else:
                        self.res_layer = None
    
                    self.jia_relu=torch.nn.Sequential(torch.nn.ReLU())
    
                def forward(self,x):
                    if self.res_layer is not None:
                        residual = self.res_layer(x)
                    else:
                        residual = x
                    return self.jia_relu(self.layer(x)+residual)
                
                def forward(self,x):
                    if self.res_layer is not None:
                        residual = self.res_layer(x)
                    else:
                        residual = x
                    return self.layer(x)+residual
    
            class ResNet(torch.nn.Module):
                def __init__(self,in_channels=1,classes=5):
                    super(ResNet, self).__init__()
                    self.features = torch.nn.Sequential(
                        torch.nn.Conv1d(in_channels,64,kernel_size=7,stride=2,padding=3),##in_channels*x*x--->64*x/2*x/2
                        torch.nn.MaxPool1d(3,2,1),#池化——64*x/4*x/4
    
                        Bottlrneck(64,64,256,False),
                        Bottlrneck(256,64,256,False),
                        Bottlrneck(256,64,256,False),#256*x/4*x/4



                        Bottlrneck(256,128,512, True),#True代表卷积步长为2————256*x/8*x/8
                        Bottlrneck(512,128,512, False),
                        Bottlrneck(512,128,512, False),
                        Bottlrneck(512,128,512, False),



                        Bottlrneck(512,256,1024, True),
                        Bottlrneck(1024,256,1024, False),
                        Bottlrneck(1024,256,1024, False),
                        Bottlrneck(1024,256,1024, False),
                        Bottlrneck(1024,256,1024, False),
                        Bottlrneck(1024,256,1024, False),



                        Bottlrneck(1024,512,2048, True),
                        Bottlrneck(2048,512,2048, False),
                        Bottlrneck(2048,512,2048, False),
    
                        torch.nn.AdaptiveAvgPool1d(1)#变成2048*1*1
                    )
                    self.classifer = torch.nn.Sequential(
                        torch.nn.Linear(2048,classes)#变成每类特征的信息
                    )


                def forward(self,x):
                    x = self.features(x)
                    x = x.view(-1,2048)
                    x = self.classifer(x)
                    return x
    
            class NPEWithEmbedding(nn.Module):#这个网络只要是1维的2的倍数就行
                def __init__(self,channels=3,beishu=4,canshu=2,build=zuko.flows.NSF,hidden_features=[128] * 3,activation=nn.ELU,transforms=3):
                    super().__init__()
    
                    self.npe = NPE(canshu, beishu, build=build, hidden_features=hidden_features,transforms=transforms, activation=activation)#用于
                    self.embedding = ResNet(in_channels=channels,classes=beishu)
    
                def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                    #print(self.embedding(x).shape)
    
                    return self.npe(theta, self.embedding(x))
    
                def flow(self, x: torch.Tensor):  # -> flow对应原来的采样，因为他调用的就是flow
                    return self.npe.flow(self.embedding(x))


​                

            torch.manual_seed(2234)#重置参数，在循环里面可以保证可以复现
            
            estimator= NPEWithEmbedding(channels=4,canshu=12,beishu=beishu,build=liu,hidden_features=[num] * transfomr,transforms=trans,activation=act).cuda()#beishu是残差网络输出尺寸（npe的输入参数）
            estimator.apply(weight_init)
            
            optimizer = optim.AdamW(estimator.parameters(), lr=f,weight_decay=weight_decay)#学习率！！！！！！！！！！！
            #在优化器选项里添加正则化,weight_decay=0.01，l2正则化
    
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)#学习率衰减
            #https://zhuanlan.zhihu.com/p/363338422
    
            step = GDStep(optimizer, clip=1.0)  # gradient descent step with gradient clipping,有了他不用optimizer.step
            loss = NPELoss(estimator)
    
            list_los=[]
            list_losvail=[]
        
            with tqdm(range(2000), unit='epoch') as tq:#epoch
                best_loss = np.inf#早停
                best_model_weights = None
                patience=50#10个周期不降就停
                i=0
                num=12000#总数
                num_v=3000
                datast=30#batch_size
                nnum=num//datast
                nnum_v=num_v//datast
                for epoch in tq:
                    #losses = torch.stack([
                    #    step(loss(preprocess(theta).cuda(), x.cuda()))
                    #    for theta, x in trainset # 这样写是遍历全部元素，实例那样是因为他是采样器
                    #])
    
                    pool = Pool(processes=56)
                    results = pool.map(get_wave_plus_gaosnoise_t, range(num))
                    all_x = torch.tensor(np.concatenate([x for x,y in results]).reshape(nnum,datast,4,4096*10),dtype=torch.float32)
                    all_y = torch.tensor(np.concatenate([y for x,y in results]).reshape(nnum,datast,12),dtype=torch.float32)
                    del results
                    pool.close()
                    pool.join()
    
                    pool = Pool(processes=56)
                    results = pool.map(get_wave_plus_gaosnoise_t, range(num_v))
                    all_x_vail = torch.tensor(np.concatenate([x for x,y in results]).reshape(nnum_v,datast,4,4096*10),dtype=torch.float32)
                    all_y_vail = torch.tensor(np.concatenate([y for x,y in results]).reshape(nnum_v,datast,12),dtype=torch.float32)
                    pool.close()
                    pool.join()
    
                    del results
                    
                    estimator.train()
                    losses = torch.stack([
                        step(loss(preprocess(all_y[i]).cuda(), all_x[i].cuda()))
                        for i in range(nnum) # 这样写是遍历全部元素，实例那样是因为他是采样器
                    ])
                    
                    del all_x,all_y
                    
                    estimator.eval()
                    with torch.no_grad():
                        val_losses = torch.stack([
                            loss(preprocess(all_y_vail[i]).cuda(), all_x_vail[i].cuda())
                            for i in range(nnum_v)
                        ])
                    
                    scheduler.step()#学习率衰减
                    
                    tq.set_postfix(loss=losses.mean().item(), val_loss=val_losses.mean().item())
    
                    del all_x_vail,all_y_vail
                    
                    los=losses.mean().item()#类型不对，所以换名字
                    losval=val_losses.mean().item()#类型不对，所以换名字
                    
                    list_los.append(los)#话损失函数图
                    list_losvail.append(losval)#话损失函数图
    
                    # 储存监视
                    data1 = list_los
                    data2 = list_losvail
                    last= now
                    now = datetime.now()
                    # 打开一个文件用于写入
                    file = open('/home/suntianyang/work5/ligo/net/data_train_t.txt', 'w')
                    file.write('last_last:'+str(last) + '\n')
                    file.write('___last__:'+str(now) + '\n')
                    
                    # 将每个元素写入文件中
                    for item1, item2 in zip(data1, data2):
                        file.write(str(item1) + ' ' + str(item2) + '\n')
                    
                    # 关闭文件
                    file.close()
                    
                    del losses,data1,data2


​                    
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    memory_usage = memory_info.rss / 1024/1024/1024  # 转换为KB
                    print(f"当前程序占用的内存：{memory_usage} GB")
                    
                    optimizer.zero_grad()
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.save(estimator.state_dict(),f'/home/suntianyang/work5/ligo/net/NPE_mid_t_hou.pth')
                    
                    if losval < best_loss:
                        best_loss = losval
                        epochs_without_improvement = 0
                        best_model_weights = estimator.state_dict()
                    else:
                        epochs_without_improvement += 1


​                    
                    # 如果验证集上的损失连续patience个epoch没有提高，则停止训练
                    if epochs_without_improvement == patience:
                        estimator.load_state_dict(best_model_weights)
                        print('Early stopping at epoch {}...'.format(epoch-patience+1))
                        break
                        
            torch.save(estimator.state_dict(),f'/home/suntianyang/work5/ligo/net/NPE_t.pth')
            aaaaa.append('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-10],len(list_los)-10))
            print('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-10],len(list_los)-10))
            torch.cuda.empty_cache()
# 频域
#加载网络
"3_trydata_train_(<class 'torch.nn.modules.activation.ReLU'>, 0.0001, 0, 4096, 7, 9, 512, 32)"
for act in [nn.ReLU]:
 for bins in [32]:
  for f in [0.0001]:
    for weight_decay in [0]:
      for Dro in [np.nan]:
       for liu in [zuko.flows.NSF]:
        for transfomr in [7]:#,7
         for num in [4096]:
          for trans in [9]:
           for beishu in [512]:
            print('装载参数')
            model = NPEWithEmbedding(channels=3,canshu=6,beishu=beishu,build=liu,hidden_features=[num] * transfomr,transforms=trans,activation=act,bins=bins).cuda()
            model.load_state_dict(torch.load("/home/suntianyang/work3/net/net_3/3_try_(<class 'torch.nn.modules.activation.ReLU'>,"+f' {f}, {weight_decay}, {bins}, {num}, {transfomr}, {trans}, {beishu}).pth'))
            model.cuda()
            model.eval()


#固定的参数
calculator = Point_mass_lens_model()
det_1 = Detector('L1')
det_2 = Detector('H1')
det_3 = Detector('V1')
det_4 = Detector('K1')

def get_wave_plus_gaosnoise_f(sa):
    hp_flens=types.FrequencySeries(np.zeros(round(N_s/2)+1),delta_f=1.0 / T_obs)#采样定律
    while True:
        epsilon=random.uniform(1e-6,0.5)*1e-6#MPc
        dl_distribution=distributions.power_law.UniformPowerLaw(dim=3,bound =(10,6000))#MPc
        dl_samples=dl_distribution.rvs(size=1)
        D_L=dl_samples[0][0]
        dl_distribution=distributions.power_law.UniformPowerLaw(dim=3,bound =(D_L,6000+D_L))#MPc
        dl_samples=dl_distribution.rvs(size=1)
        D_LS=dl_samples[0][0]-D_L#MPc
        
        M_L=random.uniform(1e3,1e5)#M
        D_S=D_LS+D_L#MPc
        z=f_dl_z(D_L)#MPc_z
        z_s=f_dl_z(D_S)#MPc_z
        M_LZ=M_L_Z(M_L,z)#M
        xi_0=x_i_0(M_L,D_LS,D_L,D_S)
    
        #print(calculator.dt(M_LZ, calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S))))
        #print(calculator.miu_p(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S))))
        #print(calculator.miu_c(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S))))
    
        dtt=calculator.dt(M_LZ, calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S)))
        miu_pp=calculator.miu_p(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S)))
        miu_cc=calculator.miu_c(calculator.y(epsilon, D_L, xi_0, D_S), calculator.beta(calculator.y(epsilon, D_L, xi_0, D_S)))
        if dtt<2.25e-3 or dtt>3.52 or miu_pp<1.7 or miu_pp>10.51 or miu_cc<-9.51 or miu_cc>-0.17:
            continue
        
        #print(dtt,miu_pp,miu_cc)
        result = types.FrequencySeries(calculator.F_f(f=hp_flens.sample_frequencies.data, M_LZ=M_LZ, epsilon=epsilon, D_L=D_L, xi_0=xi_0, D_s=D_S),delta_f=N_fs/N_s)
        
        tc=random.uniform(7,8)
        mc_distribution=distributions.MchirpfromUniformMass1Mass2(mc=(5,80))
        q_distribution=distributions.QfromUniformMass1Mass2(q=(0.5,2.0))#0.5-1
        mc_samples = mc_distribution.rvs(size=1)
        q_samples = q_distribution.rvs(size=1)
        mc=mc_samples[0][0]
        q=q_samples[0][0]
        m1,m2=mc_q_to_m1_m2(mc,q)
        distance=D_S
        
        sky_distribution = distributions.sky_location.UniformSky()
        sky_samples=sky_distribution.rvs(size=1)
        dec = sky_samples[0][0]#纬度
        ra = sky_samples[0][1]#经度
        
        psi = random.uniform(0,np.pi*2)#偏振角ψ
        inc=np.arccos(-1.0 + 2.0*np.random.rand())#倾角
        coa_phase=random.uniform(0,np.pi*2)#合并相位
        #IMRPhenomPv2，SEOBNRv4_opt,IMRPhenomTPHM
        try:
            hp, hc = get_td_waveform(approximant='IMRPhenomTPHM',
                                     mass1=m1*(1+z_s),#红移质量1
                                     mass2=m2*(1+z_s),#红移质量2
                                     distance=distance,#距离，MPC
                                     coa_phase=coa_phase,#合并相位
                                     inclination=inc,#轨道和视线的夹角
                                     spin1x=0,
                                     spin1y=0,
                                     spin1z=0,#自旋1
                                     spin2x=0,
                                     spin2y=0,
                                     spin2z=0,#自旋2
                                     eccentricity=0,#轨道偏心率
                                     lambda1=0,#潮汐相，中子星有
                                     lambda2=0,
                                     delta_t=1.0/N_fs,
                                     f_lower=30)
        except:
            continue
            
        fp_1, fc_1 = det_1.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        fp_2, fc_2 = det_2.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        fp_3, fc_3 = det_3.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        fp_4, fc_4 = det_4.antenna_pattern(
                        right_ascension=ra, declination=dec, polarization=psi, t_gps=tc)
        
        ht_1 = fp_1*hp + fc_1*hc
        ht_2 = fp_2*hp + fc_2*hc
        ht_3 = fp_3*hp + fc_3*hc
        ht_4 = fp_4*hp + fc_4*hc


​        
        ht_1.resize(N_s)
        ht_1=ht_1.cyclic_time_shift(ht_1.start_time+tc)
        ht_1.start_time=0
        ht_2.resize(N_s)
        ht_2=ht_2.cyclic_time_shift(ht_2.start_time+tc)
        ht_2.start_time=0
        ht_3.resize(N_s)
        ht_3=ht_3.cyclic_time_shift(ht_3.start_time+tc)
        ht_3.start_time=0
        ht_4.resize(N_s)
        ht_4=ht_4.cyclic_time_shift(ht_4.start_time+tc)
        ht_4.start_time=0
        
        hp_f_1=ht_1.to_frequencyseries()
        hp_flens_1=hp_f_1*result
        hp_t_lens_1=hp_flens_1.to_timeseries()
        #snr1 = get_snr(hp,T_obs,N_fs,psdl_snr.data)
        hp_f_2=ht_2.to_frequencyseries()
        hp_flens_2=hp_f_2*result
        hp_t_lens_2=hp_flens_2.to_timeseries()
        #snr1 = get_snr(hp,T_obs,N_fs,psdl_snr.data)
        hp_f_3=ht_3.to_frequencyseries()
        hp_flens_3=hp_f_3*result
        hp_t_lens_3=hp_flens_3.to_timeseries()
        #snr1 = get_snr(hp,T_obs,N_fs,psdl_snr.data)
        hp_f_4=ht_4.to_frequencyseries()
        hp_flens_4=hp_f_4*result
        hp_t_lens_4=hp_flens_4.to_timeseries()


​        
        #利用高斯 psd生成噪声：
        noise_et1 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdl)
        #添加非高斯噪声后要用welch方法估计psd——假设没有方法能合理的估计psd（psd中包含各种非高斯的干扰）
        hp_t_lens_plusnoise_1=hp_t_lens_1.copy()
        hp_t_lens_plusnoise_1.data+=noise_et1.data
        
        noise_et2 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdh)
        hp_t_lens_plusnoise_2=hp_t_lens_2.copy()
        hp_t_lens_plusnoise_2.data+=noise_et2.data
        
        noise_et3 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdv)
        hp_t_lens_plusnoise_3=hp_t_lens_3.copy()
        hp_t_lens_plusnoise_3.data+=noise_et3.data
        
        noise_et4 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdk)
        hp_t_lens_plusnoise_4=hp_t_lens_4.copy()
        hp_t_lens_plusnoise_4.data+=noise_et4.data
        
        snr1 = get_snr(hp_t_lens_1,T_obs,N_fs,psdl_snr.data)
        snr2 = get_snr(hp_t_lens_2,T_obs,N_fs,psdh_snr.data)
        snr3 = get_snr(hp_t_lens_3,T_obs,N_fs,psdv_snr.data)
        snr4 = get_snr(hp_t_lens_4,T_obs,N_fs,psdk_snr.data)
        snr=(snr1**2+snr2**2+snr3**2+snr4**2)**0.5
        #print(snr)
        
        #print(snr)
        if snr<16:
            continue
            
        arr1 = np.concatenate((hp_t_lens_plusnoise_1.to_frequencyseries().real()*1e23,
                              hp_t_lens_plusnoise_1.to_frequencyseries().imag()*1e23))
        arr2 = np.concatenate((hp_t_lens_plusnoise_2.to_frequencyseries().real()*1e23,
                              hp_t_lens_plusnoise_2.to_frequencyseries().imag()*1e23))
        arr3 = np.concatenate((hp_t_lens_plusnoise_3.to_frequencyseries().real()*1e23,
                              hp_t_lens_plusnoise_3.to_frequencyseries().imag()*1e23))
        arr4 = np.concatenate((hp_t_lens_plusnoise_4.to_frequencyseries().real()*1e23,
                              hp_t_lens_plusnoise_4.to_frequencyseries().imag()*1e23))
        canshu=[epsilon,D_L,distance,M_L,mc,q,dec,ra,psi,inc,coa_phase,tc]#12个
        
        return [arr1, arr2,arr3,arr4],canshu

adad=get_wave_plus_gaosnoise_f(11)
plt.plot(adad[0][0])
#非采样器的
torch.backends.cudnn.deterministic = True #禁用 cuDNN 的随机性，从而保证每次运行的结果都是相同的。

aaaaa=[]

import gc

#  "act<class 'torch.nn.modules.activation.ReLU'>学习率0.0003,正则化0.001,休眠率0.2,参数数目1024,参数层数7,流层9,中间层256,最佳损失-5.820067882537842,训练周期28",
#"act<class 'torch.nn.modules.activation.ReLU'>学习率0.0003,正则化0,休眠率0.2,参数数目1024,参数层数7,流层9,中间层512,最佳损失-3.325531482696533,训练周期35",
for act in [nn.ReLU]:
  for f in [0.0001]:
    for weight_decay in [0]:
       for liu in [zuko.flows.NSF]:
        for transfomr in [7]:#,7
         for num in [4096]:
          for trans in [9]:
           for beishu in [2048]:
            
            now= datetime.now()
            def weight_init(m):
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight.data)#kaiming_normal_///xavier_uniform_
                    nn.init.constant_(m.bias.data, 0.0)
    
            class Bottlrneck(torch.nn.Module):
                def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
                    super(Bottlrneck, self).__init__()
                    self.stride = 1
                    if downsample == True:
                        self.stride = 2
                    #在这里添加BatchNorm1d和Dropout是最合适的
                    self.layer = torch.nn.Sequential(
                        torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
                        torch.nn.BatchNorm1d(Med_channel),
                        torch.nn.ReLU(),
                        torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
                        torch.nn.BatchNorm1d(Med_channel),
                        #torch.nn.ReLU(),
                        torch.nn.Conv1d(Med_channel, Out_channel, 1),
                        torch.nn.BatchNorm1d(Out_channel),
                        #torch.nn.ReLU(),
                    )
    
                    if In_channel != Out_channel:
                        self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
                    else:
                        self.res_layer = None
    
                    self.jia_relu=torch.nn.Sequential(torch.nn.ReLU())
    
                def forward(self,x):
                    if self.res_layer is not None:
                        residual = self.res_layer(x)
                    else:
                        residual = x
                    return self.jia_relu(self.layer(x)+residual)
                
                def forward(self,x):
                    if self.res_layer is not None:
                        residual = self.res_layer(x)
                    else:
                        residual = x
                    return self.layer(x)+residual
    
            class ResNet(torch.nn.Module):
                def __init__(self,in_channels=1,classes=5):
                    super(ResNet, self).__init__()
                    self.features = torch.nn.Sequential(
                        torch.nn.Conv1d(in_channels,64,kernel_size=7,stride=2,padding=3),##in_channels*x*x--->64*x/2*x/2
                        torch.nn.MaxPool1d(3,2,1),#池化——64*x/4*x/4
    
                        Bottlrneck(64,64,256,False),
                        Bottlrneck(256,64,256,False),
                        Bottlrneck(256,64,256,False),#256*x/4*x/4



                        Bottlrneck(256,128,512, True),#True代表卷积步长为2————256*x/8*x/8
                        Bottlrneck(512,128,512, False),
                        Bottlrneck(512,128,512, False),
                        Bottlrneck(512,128,512, False),



                        Bottlrneck(512,256,1024, True),
                        Bottlrneck(1024,256,1024, False),
                        Bottlrneck(1024,256,1024, False),
                        Bottlrneck(1024,256,1024, False),
                        Bottlrneck(1024,256,1024, False),
                        Bottlrneck(1024,256,1024, False),



                        Bottlrneck(1024,512,2048, True),
                        Bottlrneck(2048,512,2048, False),
                        Bottlrneck(2048,512,2048, False),
    
                        torch.nn.AdaptiveAvgPool1d(1)#变成2048*1*1
                    )
                    self.classifer = torch.nn.Sequential(
                        torch.nn.Linear(2048,classes)#变成每类特征的信息
                    )


                def forward(self,x):
                    x = self.features(x)
                    x = x.view(-1,2048)
                    x = self.classifer(x)
                    return x
    
            class NPEWithEmbedding(nn.Module):#这个网络只要是1维的2的倍数就行
                def __init__(self,channels=3,beishu=4,canshu=2,build=zuko.flows.NSF,hidden_features=[128] * 3,activation=nn.ELU,transforms=3):
                    super().__init__()
    
                    self.npe = NPE(canshu, beishu, build=build, hidden_features=hidden_features,transforms=transforms, activation=activation)#用于
                    self.embedding = ResNet(in_channels=channels,classes=beishu)
    
                def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                    #print(self.embedding(x).shape)
    
                    return self.npe(theta, self.embedding(x))
    
                def flow(self, x: torch.Tensor):  # -> flow对应原来的采样，因为他调用的就是flow
                    return self.npe.flow(self.embedding(x))
    
            torch.manual_seed(2234)#重置参数，在循环里面可以保证可以复现
            
            estimator= NPEWithEmbedding(channels=4,canshu=12,beishu=beishu,build=liu,hidden_features=[num] * transfomr,transforms=trans,activation=act).cuda()#beishu是残差网络输出尺寸（npe的输入参数）
            estimator.apply(weight_init)
            
            optimizer = optim.AdamW(estimator.parameters(), lr=f,weight_decay=weight_decay)#学习率！！！！！！！！！！！
            #在优化器选项里添加正则化,weight_decay=0.01，l2正则化
    
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)#学习率衰减
            #https://zhuanlan.zhihu.com/p/363338422
    
            step = GDStep(optimizer, clip=1.0)  # gradient descent step with gradient clipping,有了他不用optimizer.step
            loss = NPELoss(estimator)
    
            list_los = []  # 用于存储第一列数据
            list_losvail = []  # 用于存储第二列数据
            
            with tqdm(range(2000), unit='epoch') as tq:#epoch
                best_loss = np.inf#早停
                best_model_weights = None
                patience=50#10个周期不降就停
                i=0
                i_numca=0
                num=12000#总数
                num_v=3000
                datast=30#batch_size
                nnum=num//datast
                nnum_v=num_v//datast
                for epoch in tq:
                    #losses = torch.stack([
                    #    step(loss(preprocess(theta).cuda(), x.cuda()))
                    #    for theta, x in trainset # 这样写是遍历全部元素，实例那样是因为他是采样器
                    #])
    
                    pool = Pool(processes=56)
                    results = pool.map(get_wave_plus_gaosnoise_f, range(num))
                    all_x = torch.tensor(np.concatenate([x for x,y in results]).reshape(nnum,datast,4,T_obs*4096+2),dtype=torch.float32)
                    all_y = torch.tensor(np.concatenate([y for x,y in results]).reshape(nnum,datast,12),dtype=torch.float32)
                    pool.close()
                    pool.join()
                    
                    del results
                    
                    pool = Pool(processes=56)
                    results = pool.map(get_wave_plus_gaosnoise_f, range(num_v))
                    all_x_vail = torch.tensor(np.concatenate([x for x,y in results]).reshape(nnum_v,datast,4,T_obs*4096+2),dtype=torch.float32)
                    all_y_vail = torch.tensor(np.concatenate([y for x,y in results]).reshape(nnum_v,datast,12),dtype=torch.float32)
    
                    pool.close()
                    pool.join()
    
                    del results
                    
                    estimator.train()
                    losses = torch.stack([
                        step(loss(preprocess(all_y[i]).cuda(), all_x[i].cuda()))
                        for i in range(nnum) # 这样写是遍历全部元素，实例那样是因为他是采样器
                    ])
                    
                    del all_x,all_y
                    
                    estimator.eval()
                    with torch.no_grad():
                        val_losses = torch.stack([
                            loss(preprocess(all_y_vail[i]).cuda(), all_x_vail[i].cuda())
                            for i in range(nnum_v)
                        ])
                    
                    scheduler.step()#学习率衰减
                    
                    tq.set_postfix(loss=losses.mean().item(), val_loss=val_losses.mean().item())
    
                    del all_x_vail,all_y_vail
                    
                    los=losses.mean().item()#类型不对，所以换名字
                    losval=val_losses.mean().item()#类型不对，所以换名字
                    
                    list_los.append(los)#话损失函数图
                    list_losvail.append(losval)#话损失函数图
    
                    # 储存监视
                    data1 = list_los
                    data2 = list_losvail
                    last= now
                    now = datetime.now()
                    # 打开一个文件用于写入
                    file = open('/home/suntianyang/work5/ligo/net/data_train_f.txt', 'w')
                    file.write('last_last:'+str(last) + '\n')
                    file.write('___last__:'+str(now) + '\n')
                    
                    # 将每个元素写入文件中
                    for item1, item2 in zip(data1, data2):
                        file.write(str(item1) + ' ' + str(item2) + '\n')
                    
                    # 关闭文件
                    file.close()
                    
                    del losses,data1,data2


​                    
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    memory_usage = memory_info.rss / 1024/1024/1024  # 转换为KB
                    print(f"当前程序占用的内存：{memory_usage} GB")
                    
                    optimizer.zero_grad()
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.save(estimator.state_dict(),f'/home/suntianyang/work5/ligo/net/NPE_mid_f_hou.pth')
                    
                    if losval < best_loss:
                        best_loss = losval
                        epochs_without_improvement = 0
                        best_model_weights = estimator.state_dict()
                    else:
                        epochs_without_improvement += 1


​                    
                    # 如果验证集上的损失连续patience个epoch没有提高，则停止训练
                    if epochs_without_improvement == patience:
                        estimator.load_state_dict(best_model_weights)
                        print('Early stopping at epoch {}...'.format(epoch-patience+1))
                        break
    
            torch.save(estimator.state_dict(),f'/home/suntianyang/work5/ligo/net/NPE_f.pth')
            aaaaa.append('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-10],len(list_los)-10))
            print('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-10],len(list_los)-10))
            torch.cuda.empty_cache()