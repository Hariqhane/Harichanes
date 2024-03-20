# Cell 1
# ====================
# 此处为代码单元格 1 的解释和注释
#总体思路：
#高斯噪声对比时域or时频域
#真实噪声对比更合适的数据方式，进行参数推断（H，Ωm定死）(透镜模型以及波形的泛化性)
#真实噪声利用folw尝试信号类型判断
#真实噪声尝试宇宙学参数推断

# Cell 2
# ====================
# 此处为代码单元格 2 的解释和注释
import matplotlib.pyplot as plt
# !绘制距离-红移关系图
# Cell 3
#! 计算光度距离随红移的变化
# ====================
# 此处为代码单元格 3 的解释和注释
import numpy as np
import scipy #!积分 插值等数学运算

from scipy.integrate import quad #! quad用于计算一维定积分
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

# Cell 4 #!与Cell3相同
# ====================
# 此处为代码单元格 4 的解释和注释
dl_all=[]
for zz in np.linspace(0,10,5000):
    dl_all.append(lcdm_distance_redshift(zz,70,0.3,0.7))

# Cell 5
# ====================
# 此处为代码单元格 5 的解释和注释
plt.plot(np.linspace(0,10,5000),dl_all)
plt.yscale('log')#!将y轴设置为对数缩放
#! 以红移值作为x轴, 对应的光度距离dl_all作为y轴, 绘制光度距离-红移关系图
# Cell 6
# ====================
# 此处为代码单元格 6 的解释和注释
#1024——glitch生成
import matplotlib.pyplot as pp
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform,get_fd_waveform #!给Cell 15 用的
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

# Cell 7
# ====================
# 此处为代码单元格 7 的解释和注释
import numpy as np
from scipy.integrate import quad
#函数定义
def lcdm_distance_redshift(z, H0, Omega_m, Omega_Lambda):
    c = 299792.458  # 光速，单位：km/s
    # !定义被积函数
    integrand = lambda z: 1 / np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)
    # !积分计算
    result, _ = quad(integrand, 0, z)
    # !计算(光度?)距离
    distance = c / H0 * result*(1+z)
    return distance


dl_all=[]
for zz in np.linspace(0,10,5000):
    dl_all.append(lcdm_distance_redshift(zz,70,0.3,0.7))
    
f_dl_z = scipy.interpolate.interp1d(dl_all, np.linspace(0,10,5000), kind='linear') # !创建插值函数, 用来根据给定的红移值快速计算对应的距离

import numpy as np
#!点质量透镜模型构建
class Point_mass_lens_model:
    def __init__(self):
        pass
#! F(f)放大因子
    def F_f(self, f, M_LZ, epsilon, D_L, xi_0, D_s):
        return np.abs(self.miu_p(self.y(epsilon, D_L, xi_0, D_s), self.beta(self.y(epsilon, D_L, xi_0, D_s))))**0.5 - \
            1j * np.abs(self.miu_c(self.y(epsilon, D_L, xi_0, D_s), self.beta(self.y(epsilon, D_L, xi_0, D_s))))**0.5 * \
            np.exp(2j * np.pi * f * self.dt(M_LZ, self.y(epsilon, D_L, xi_0, D_s), self.beta(self.y(epsilon, D_L, xi_0, D_s))))
#! μ+-
    def miu_p(self, y, beta):
        return 0.5 + (y**2 + 2) / (2 * y * beta)
    
    def miu_c(self, y, beta):
        return 0.5 - (y**2 + 2) / (2 * y * beta)
    #G:m³/kg·s²
#! 时间延迟\Delta t
    def dt(self, M_LZ, y, beta):#单位s
        return 4 * M_LZ*(y * beta / 2 + np.log((beta + y) / (beta - y))) * scipy.constants.G / scipy.constants.c**3 *(1.989e30)
#! β和y
    def beta(self, y):
        return (y**2 + 4)**0.5

    def y(self, epsilon, D_L, xi_0, D_s):
        return epsilon * D_L / xi_0 / D_s

#! 红移后的透镜质量
def M_L_Z(M_L,z):
    return M_L*(1+z)
#! 透镜的爱因斯坦半径?(文献中没有)
def x_i_0(M_L,D_LS,D_L,D_S): #G:m³/kg·s² #m^1 kg^-1 Mc^1  Mpc^1 单位转化成MPc,,,,1.989×1030 千克,,,3.0857E+22
    return ((4*scipy.constants.G*M_L/scipy.constants.c**2)*(D_LS*D_L/D_S) /(3.0857e22)*(1.989e30))**0.5

#? 什么东西
def mc_q_to_m1_m2(mc,q):
    m1=(mc**5*(1+q)/q**3)**(1/5)
    m2=m1*q
    return m1,m2

#? 计算信噪比
def get_snr(data,T_obs,fs,psd):
    #波形、1、频率、psd
    N = T_obs*fs
    delta_f = 1.0/T_obs
    delta_t = 1.0/fs

#     win = tukey(N,alpha=1.0/8.0)
#? 功率谱密度S_{n}(f)
    idx = np.argwhere(psd==0.0)
    psd[idx] = 1e300    
#! 信号的傅里叶变换
    xf = np.fft.rfft(data)*delta_t
    #fig = plt.figure()
    #plt.plot(np.real(xf))
    #plt.plot(np.imag(xf))
    SNRsq = 4.0*np.sum((np.abs(xf)**2)/psd)*delta_f
    return np.sqrt(SNRsq)

# Cell 8
#! 设置了使用点质量透镜模型进行计算所需的固定参数
# ====================
# 此处为代码单元格 8 的解释和注释
#固定的参数
calculator = Point_mass_lens_model()
T_obs=10
N_fs=4096#采样率为4096Hz
N_s=T_obs*N_fs#对应时域为10s

flen = round(N_s/2)+1 #? 频率域的长度?为什么是这么计算的?
delta_f = 1.0 / T_obs#! 每个频率点的间隔
delta_t=1/N_fs #! 时域的时间间隔
f_low=30 #!低频截断
flow=30 #? 未知参数

tsamples = int(T_obs / delta_t) #? 和N_s有什么区别?

# Cell 9
# ====================
# 此处为代码单元格 9 的解释和注释
# 打开txt文件以读取内容
#? 似乎是获取功率谱密度(PSD)数据?
with open('/home/suntianyang/work5/ligo/aligo_O4high.txt', 'r') as file:
    lines = file.readlines()


#! 将读取的每行字符串数据分割并转换成浮点数, 最终得到一个二维数组
data_lines = [line.strip().split() for line in lines]
#! 二维数组
data = [[float(value) for value in line] for line in data_lines]

list_los = []  #!用于存储第一列数据(频率值)
list_losvail = []  #! 用于存储第二列数据(对应的PSD值)
# 遍历数据并将其添加到相应的列表中
for row in data:
    list_los.append(row[0])  # 将第一列数据添加到list_los
    list_losvail.append(row[1]**2)  #! 将第二列数据添加到list_losvail;进行了平方处理, 可能是为了满足PSD的计算需求
    
    
psdh = pycbc.psd.aLIGO140MpcT1800545(flen, delta_f, flow)#! 创建一个预设的PSD模型
interp_function = interp1d(list_los, list_losvail, kind='linear')# !创建插值函数, 以此调整PSD模型, #? 以使其更接近实际观侧数据(为什么?)
y_new = interp_function(psdh.sample_frequencies)  #? 生成新的PSD数据
y_new[psdh.sample_frequencies<f_low]=0#! 低于f_low的频率部分置零
psdh.data=y_new#! 更新psdh的数据为插值后的PSD值, 
psdh_snr = psdh.copy()#? 复制一份用于信噪比的计算(?)

pp.plot(psdh.sample_frequencies, psdh**0.5, label='O4')
pp.yscale('log')
pp.xscale('log') #! x,y轴都为对数刻度
pp.xlim(10,2048) #! 显示从10HZ到2048HZ的频率范围
pp.legend()
pp.show()

# Cell 10
# ====================
# 此处为代码单元格 10 的解释和注释
# 打开txt文件以读取内容
#! 与Cell 9基本完全相同, 除了psdh变成了psdl?利文斯顿?这不是和上一个完全一样吗
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

# Cell 11
#! 同上,但是变为psdv;数据来源:Virgo
# ====================
# 此处为代码单元格 11 的解释和注释
# 打开txt文件以读取内容
with open('/home/suntianyang/work5/ligo/avirgo_O4high.txt', 'r') as file:#! 注意数据变为avirgo
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

# Cell 12
#! 同上, 变为psdk,数据来源:KAGRA
# ====================
# 此处为代码单元格 12 的解释和注释
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

# Cell 13
#? 访问psdl(这是在干嘛?)
# ====================
# 此处为代码单元格 13 的解释和注释
psdl.sample_frequencies.data

# Cell 14
# ====================
# 此处为代码单元格 14 的解释和注释
pp.plot(psdl.sample_frequencies, psdl**0.5, label='ligo')#!利文斯顿
pp.plot(psdh.sample_frequencies, psdh**0.5, label='ligo')#!汉福德
pp.plot(psdv.sample_frequencies, psdv**0.5, label='V1')#! Virgo
pp.plot(psdk.sample_frequencies, psdk**0.5, label='K1')#! KAGRA
pp.yscale('log')
pp.xscale('log') 
pp.xlim(20,2048)
pp.legend()
pp.show()

# Cell 15
#! 一个复杂过程, 用于模拟引力透镜对引力波的影响
#! 并将这些信号与不同的探测器的噪声进行叠加
# ====================
# 此处为代码单元格 15 的解释和注释
#固定的参数
#?==================================================================================================这段是在干嘛?
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
#?===========================================================================================================
        #! 模拟引力波信号-->探测器响应(计算不同探测器对引力波信号的响应)-->引力波信号透镜化(频域)-->噪声叠加
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
        
        
        #利用高斯 psd生成噪声：
        noise_et1 = pycbc.noise.noise_from_psd(tsamples, delta_t, psdl)
        #?添加非高斯噪声后要用welch方法估计psd——假设没有方法能合理的估计psd（psd中包含各种非高斯的干扰）(什么意思?)
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
#!信噪比计算
        snr1 = get_snr(hp_t_lens_1,T_obs,N_fs,psdl_snr.data)
        snr2 = get_snr(hp_t_lens_2,T_obs,N_fs,psdh_snr.data)
        snr3 = get_snr(hp_t_lens_3,T_obs,N_fs,psdv_snr.data)
        snr4 = get_snr(hp_t_lens_4,T_obs,N_fs,psdk_snr.data)
        snr=(snr1**2+snr2**2+snr3**2+snr4**2)**0.5
        #print(snr)
#? Q变换, 对带噪声信号进行Q变换, 以分析信号的时频特性.
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

# Cell 16
#! 函数模拟包含引力透镜效应的引力波信号并叠加噪声
# ====================
# 此处为代码单元格 16 的解释和注释
hp_t_lens_plusnoise,canshu=get_wave_plus_gaosnoise_tf(11)
#hp_t_lens_plusnoise[0].plot();
#plt.xlim(23,27)
#plt.yscale('log')
print(hp_t_lens_plusnoise[0].data.shape)

# Cell 17
#! 绘制四个探测器接收到的引力波信号，这些信号已经包含了引力透镜效应和噪声
# ====================
# 此处为代码单元格 17 的解释和注释
hp_t_lens_plusnoise[0].plot();
hp_t_lens_plusnoise[1].plot();
hp_t_lens_plusnoise[2].plot();
hp_t_lens_plusnoise[3].plot();

# Cell 18
#! 可能是为了得到某个与采样率或数据处理相关的参数值
# ====================
# 此处为代码单元格 18 的解释和注释
(30*4096)**0.5

# Cell 19
#? 查看数据的形状?
# ====================
# 此处为代码单元格 19 的解释和注释
hp_t_lens_plusnoise.data.shape

# Cell 20
#! 删除变量hp_t_lens_plusnoise和canshu以释放内存，这表明完成了引力波信号的处理和分析。
# ====================
# 此处为代码单元格 20 的解释和注释
del hp_t_lens_plusnoise,canshu

# Cell 21
# ====================
# 此处为代码单元格 21 的解释和注释
#! 神经网络部分
%matplotlib inline
import matplotlib.pyplot as plt
#! PyTorch相关库
import torch #!用于计算自动微分
import torch.nn as nn #! 提供了构建神经网络的模块和类，如层、激活函数等。
import torch.optim as optim #!  提供了各种优化算法，如SGD、Adam等，用于模型训练过程中的参数更新。
import torch.nn.utils as nn_utils #! 提供了一些实用工具函数和类，用于神经网络构建和训练。
import zuko
from torch.optim import lr_scheduler #! 提供了用于调整学习率的方法，如学习率衰减

#! 实用工具和进度条
from itertools import islice #? 从迭代器中连续切片元素，但不会提前消耗迭代器(不懂)
from tqdm import tqdm #! 用于生成进度条，帮助用户了解长循环操作的进度
#! Lampe库
from lampe.data import JointLoader #! : 可能是用于加载和处理数据的类
from lampe.inference import NPE, NPELoss #! : 提供了神经后验估计(NPE)相关的模型和损失函数
from lampe.plots import nice_rc, corner, mark_point
from lampe.utils import GDStep
from lampe.data import H5Dataset
from lampe.diagnostics import expected_coverage_mc

import h5py
import numpy as np

from datetime import datetime #!  用于处理日期和时间的标准库，如获取当前时间、时间加减等

# Cell 22
# ====================
# 此处为代码单元格 22 的解释和注释
#抽样器1500训练
LABELS = [r'$epsilon$',r'$D_L$',r'$distance$',r'$M_L$',r'$Mc$',r'$q$',r'$dec$',r'$ra$',r'$psi$',r'$inc$',r'$coa_phase$',r'$tc$'] #! 定义参数标签,共12个参数

LOWER = torch.tensor([1e-12  , 10. ,20. ,1e3 ,5.  ,0.5 ,-0.5*np.pi   ,0       ,0       ,0       ,0       ,7])
UPPER = torch.tensor([0.5e-6 , 6e3 ,1.2e4 ,1e5 ,80. ,2.  ,0.5*np.pi    ,2*np.pi ,2*np.pi ,2*np.pi ,2*np.pi ,8])#! 各个参数的上下限

#! 参数归一化函数;归一化是机器学习中常用的技术，有助于改善模型的训练效率和性能;通过数学计算可以发现确实归一化了
def preprocess(theta: torch.Tensor) -> torch.Tensor:
    return 2 * (theta - LOWER) / (UPPER - LOWER) - 1
#! 参数逆归一化函数;与归一化函数操作相反, 目的是在模型做出预测后，需要将预测结果转换为实际的物理量
def postprocess(theta: torch.Tensor) -> torch.Tensor:
    return (theta + 1) / 2 * (UPPER - LOWER) + LOWER

# Cell 23
# ====================
# 此处为代码单元格 23 的解释和注释
from multiprocessing import Pool #! 多进程技术
import psutil #! 监控当前进程的内存使用情况

# Cell 24
#! 获取并打印进程内存使用情况
# ====================
# 此处为代码单元格 24 的解释和注释
process = psutil.Process()
memory_info = process.memory_info()
memory_usage = memory_info.rss / 1024/1024/1024  # 转换为KB
print(f"当前程序占用的内存：{memory_usage} GB")

# Cell 25
#! 初始化一个神经后验估计器（NPE）实例，用于参数估计。NPE是一种用于贝叶斯推理的神经网络架构，这里的设置指定了输入维度、输出维度、隐藏层特征等
# ====================
# 此处为代码单元格 25 的解释和注释
estimator = NPE(3, 2, transforms=3, hidden_features=[64] * 3)
# 调用了NPE类的构造函数，创建了一个NPE模型的实例。这里传入的参数配置了模型的结构和特性：
    # 第一个参数3可能表示输入数据的维度或特征数量。
    # 第二个参数2可能表示模型输出的维度，即目标变量的数量。
    # transforms=3指定了模型中变换层的数量。在深度学习中，变换层用于从输入数据中提取特征，并将这些特征转换成有用的表示形式。
    # hidden_features=[64] * 3定义了隐藏层的特性，其中[64] * 3表示有3个隐藏层，每层有64个神经元。
estimator   #! 在Jupyter中展示模型的基本信息

# Cell 26
# ====================
# 此处为代码单元格 26 的解释和注释


# Cell 27
#! 模型训练完整流程
#! 处理的是图像数据
#? =======================================================================================
# 此处为代码单元格 27 的解释和注释
#非采样器的
torch.backends.cudnn.deterministic = True #! 禁用 cuDNN 的随机性，从而保证每次运行的结果都是相同的。

aaaaa=[]

import gc#! 引入Python的垃圾收集模块，用于手动控制内存的清理，以防止训练过程中出现内存泄漏。

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
            def weight_init(m): #! 定义一个权重初始化函数，用于模型中的卷积和线性层。
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight.data)#!使用Xavier均匀初始化方法初始化权重，偏置初始化为0。这有助于模型的稳定训练。
                    nn.init.constant_(m.bias.data, 0.0)
#! 定义具有残差连接的卷积块。这些卷积块构成了残差网络的基本构件，通过增加路径来改善梯度流，促进深层网络的训练。
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
#! 定义了一个残差网络结构，包括一系列的残差卷积块和一个分类器。这个网络可以从输入数据中提取特征，并将这些特征映射到输出类别。
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
#! 整合了NPE（神经后验估计）和残差网络，用于处理输入数据并进行预测。这表明模型旨在处理一些复杂的数据分布，通过学习数据的后验分布来做出预测。
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
                    
                    
                    # 如果验证集上的损失连续patience个epoch没有提高，则停止训练
                    if epochs_without_improvement == patience:
                        estimator.load_state_dict(best_model_weights)
                        print('Early stopping at epoch {}...'.format(epoch-patience+1))
                        break


            aaaaa.append('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-10],len(list_los)-10))
            print('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-10],len(list_los)-10))
            torch.cuda.empty_cache()
#? =======================================================================================
# Cell 28
#! 生成具有引力透镜效应和叠加噪声的信号,基本同Cell 15
#! 
#! 共同点：
# 目的：两段代码的主要目的都是模拟包含引力透镜效应的引力波信号，并将这些信号与不同探测器的噪声进行叠加。这是为了研究引力波信号在穿越不同质量分布（如黑洞、恒星等）时的透镜化效应及其在探测器上的检测。

# 方法：它们都使用了相似的方法来模拟信号和噪声。首先，利用物理模型（如Point_mass_lens_model）来模拟引力透镜效应对信号的影响。然后，使用LIGO/Virgo探测器的响应模型（通过Detector类实例）来模拟探测器接收到的信号。最后，向模拟的信号中添加噪声，以更真实地反映实际探测环境。

# 实现：两段代码都使用了pycbc库中的函数和类来生成和处理信号，如使用get_td_waveform生成引力波信号，以及noise_from_psd生成对应的噪声。

# 差异：
# 信号处理：代码1在模拟信号后，采用Q变换（q_transform）进行时频分析，这是一种特殊的信号处理技术，用于分析信号的时频特性。而代码2没有使用Q变换，而是直接返回了模拟信号和参数。

# 目标和应用场景：代码1通过Q变换分析信号，可能更侧重于信号的详细分析，比如研究引力波信号的时频特性、探测器对信号的响应等。代码2更多是关注于生成模拟信号和参数，可能更适用于模型训练、信号识别等应用场景。

# 输出：代码1的输出是进行了Q变换处理后的信号数组，这可以用于进一步的信号分析。而代码2的输出是直接的时域信号数组和参数，适用于需要原始信号数据的场合。

# 总结来说，虽然两段代码在功能上有很多相似之处，主要差异在于代码1对信号进行了额外的Q变换处理，可能更适合于信号分析的应用。而代码2直接提供了模拟信号和参数，可能更适合于需要原始信号进行进一步处理或分析的场景。
# ====================
# 此处为代码单元格 28 的解释和注释
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

# Cell 29
# ====================
# 此处为代码单元格 29 的解释和注释
adad,g=get_wave_plus_gaosnoise_t(11)
plt.plot(adad[0])


# Cell 30
#! 生成叠加信号
#! 同Cell15, Cell 28;输出参数变化
# ====================
# 此处为代码单元格 30 的解释和注释
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
        if snr<16 and snr<50: #! 信噪比条件相比于Cell15和Cell 18 变化
            continue
            
        L11data = (hp_t_lens_plusnoise_1.to_frequencyseries() / psdl_snr**0.5).to_timeseries()#要除以0置成正无穷的
        H11data = (hp_t_lens_plusnoise_2.to_frequencyseries() / psdh_snr**0.5).to_timeseries()#要除以0置成正无穷的
        V11data = (hp_t_lens_plusnoise_3.to_frequencyseries() / psdv_snr**0.5).to_timeseries()#要除以0置成正无穷的
        K11data = (hp_t_lens_plusnoise_4.to_frequencyseries() / psdk_snr**0.5).to_timeseries()#要除以0置成正无穷的

        canshu=[yyy_uuu,miu_pp,miu_cc]#! 之前的有12げ参数, #? 这是...
        
        return [L11data.data,H11data.data,V11data.data,K11data.data],canshu

# Cell 31
# ====================
# 此处为代码单元格 31 的解释和注释
miu_pp<1.7
miu_pp>10.51

miu_cc<-9.51
miu_cc>-0.17

# Cell 32
# ====================
# 此处为代码单元格 32 的解释和注释
from multiprocessing import Pool #!  用于创建一个进程池。这可以用于并行处理或生成数据，从而提高计算效率。
import multiprocessing as mp
#pool = Pool(processes=48)

# Cell 33
# ====================
# 此处为代码单元格 33 的解释和注释
from astropy.utils.iers import IERS_Auto
iers_a = IERS_Auto.open() #! 用于获取最新的地球自转参数

import warnings
from astropy.utils.iers import IERSStaleWarning

# 忽略 IERSStaleWarning
warnings.filterwarnings('ignore', category=IERSStaleWarning) #! 忽略了IERSStaleWarning警告，这通常是因为在获取最新的地球自转参数时可能会遇到网络或数据更新的问题

# Cell 34
# ====================
# 此处为代码单元格 34 的解释和注释
#! 这个单元格使用psutil库来检测当前运行的Python程序占用的内存量
import psutil

process = psutil.Process()
memory_info = process.memory_info()
memory_usage = memory_info.rss / 1024/1024/1024  # 转换为KB

print(f"当前程序占用的内存：{memory_usage} GB")

# Cell 35
#! 通过并行处理的方式生成了一批模拟的引力波信号数据
# ====================
# 此处为代码单元格 35 的解释和注释
pool = Pool(processes=56)#! 使用multiprocessing.Pool创建了一个进程池，
results = pool.map(get_wave_plus_gaosnoise_t_lens, range(5000)) #! 并行执行get_wave_plus_gaosnoise_t_lens函数，生成了5000个模拟信号数据
all_x = torch.tensor(np.concatenate([x for x,y in results]).reshape(5000,4,4096*10),dtype=torch.float32)
all_y = torch.tensor(np.concatenate([y for x,y in results]).reshape(5000,3),dtype=torch.float32)
#! 这些数据被整理成张量格式，准备用于后续的机器学习或数据分析任务。
del results
pool.close()
pool.join()

# Cell 36
# ====================
# 此处为代码单元格 36 的解释和注释


# Cell 37
#! 模型训练;同27;
#! 区别: 没有2D卷积层Bottlrneck2d; 超参数略有不同; 学习率调度器的变化： Cell 27中使用了CosineAnnealingLR作为学习率调度器，而Cell 37 使用了ExponentialLR。
#! 对数据处理的批量大小、训练集和验证集的总数进行了调整，这可能是基于对数据集特性或模型性能优化的考虑。
# ====================
# 此处为代码单元格 37 的解释和注释
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
                    
                    
                    # 如果验证集上的损失连续patience个epoch没有提高，则停止训练
                    if epochs_without_improvement == patience:
                        estimator.load_state_dict(best_model_weights)
                        print('Early stopping at epoch {}...'.format(epoch-patience+1))
                        break
                        
            torch.save(estimator.state_dict(),f'/home/suntianyang/work5/ligo/net/NPE_t.pth')
            aaaaa.append('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-10],len(list_los)-10))
            print('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-10],len(list_los)-10))
            torch.cuda.empty_cache()

# Cell 38
# !加载一个预训练的模型，并准备它进行评估或进一步的使用。
# 代码中使用了一个循环结构来定义模型的配置参数，但实际上这个循环每个部分只有一个选项，所以它实际上是在指定一组特定的参数来加载模型。
#下面是对这段代码每个部分的详细解释：

# 1. **指定激活函数和其他参数：** 代码首先定义了一系列参数，包括激活函数`nn.ReLU`，以及其他一些模型训练时使用的超参数，如学习率`0.0001`，权重衰减`0`，隐藏特征数`4096`，变换次数`7`，流层数`9`，输出大小`beishu`为`512`，和`bins`数量为`32`。

# 2. **模型构建：** 使用`NPEWithEmbedding`构造函数创建了一个模型实例。这个函数需要几个关键参数来定义模型的结构和行为，包括：
#    - `channels=3`：输入数据的通道数。
#    - `canshu=6`：可能是模型特定的某种参数数量。
#    - `beishu=512`：定义了模型内部某些层的大小或数量。
#    - `build=liu`：使用的特定流构建方法，这里是`zuko.flows.NSF`。
#    - `hidden_features=[num] * transfomr`：隐藏层特征的设置，这里是`4096`个特征重复`7`次，对应于变换次数。
#    - `transforms=trans`：模型中的变换次数，这里是`9`。
#    - `activation=act`：激活函数，这里使用的是ReLU激活函数。
#    - `bins=bins`：特定的参数，这里设置为`32`。

# 3. **加载预训练模型：** 通过`model.load_state_dict`方法加载预先训练好的模型权重。权重文件的路径是根据上面定义的参数动态构建的。这意味着它会根据选择的激活函数、学习率、权重衰减等参数来选择正确的权重文件。

# 4. **准备模型进行评估：**
#    - `model.cuda()`：将模型移动到CUDA设备上，使其能够在GPU上运行。
#    - `model.eval()`：将模型设置为评估模式。这对于某些类型的层（如Dropout层和BatchNorm层）是必要的，因为它们在训练和评估时的行为是不同的。

# 总之，这段代码展示了如何根据一组指定的参数加载一个预训练的深度学习模型，并将其准备好进行评估或进一步的分析。
# ====================
# 此处为代码单元格 38 的解释和注释
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

# Cell 39
# ====================
# 此处为代码单元格 39 的解释和注释


# Cell 40
# ====================
# 此处为代码单元格 40 的解释和注释


# Cell 41
#! 生成叠加信号
#! 基本同Cell 15 ,28 ,30
#! 在这段代码中，对于信号的处理方式有所不同，特别是在将信号从时间域转换到频域后，对信号实部和虚部的处理（乘以1e23并拼接）可能是根据特定的需求设计的。
# ====================
# 此处为代码单元格 41 的解释和注释
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


# Cell 42
#! 同Cell 29
#! Cell15 ,28 , 30 ,41区别:
    # 15: 时频域
    # 28: 时域
    # 30: 时域_透镜化
    # 41: 频域
# ====================
# 此处为代码单元格 42 的解释和注释
adad=get_wave_plus_gaosnoise_f(11) 
plt.plot(adad[0][0])

# Cell 43
#! 模型训练, 同Cell 27, 37
#! 三者区别: 27处理的是图像数据,学习率调度器为CosineAnnealingLR, 区别于37和43的ExponentialLR;
# ====================
# 此处为代码单元格 43 的解释和注释
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
                    
                    
                    # 如果验证集上的损失连续patience个epoch没有提高，则停止训练
                    if epochs_without_improvement == patience:
                        estimator.load_state_dict(best_model_weights)
                        print('Early stopping at epoch {}...'.format(epoch-patience+1))
                        break

            torch.save(estimator.state_dict(),f'/home/suntianyang/work5/ligo/net/NPE_f.pth')
            aaaaa.append('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-10],len(list_los)-10))
            print('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-10],len(list_los)-10))
            torch.cuda.empty_cache()

# Cell 44
            
# ====================
# 此处为代码单元格 44 的解释和注释


