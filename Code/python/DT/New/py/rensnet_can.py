# %%
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

# %%
LOWER = torch.tensor([0  ,  0. ])
UPPER = torch.tensor([0.2 , 1. ])

#参数归一化与逆运算
def preprocess(theta: torch.Tensor) -> torch.Tensor:
    return 2 * (theta - LOWER) / (UPPER - LOWER) - 1

def postprocess(theta: torch.Tensor) -> torch.Tensor:
    return (theta + 1) / 2 * (UPPER - LOWER) + LOWER

# %% [markdown]
# # 模型搭建

# %%
data_set_dir='/home/lichunyue/sty/train_p/'

# %%
#非采样器的
torch.backends.cudnn.deterministic = True #禁用 cuDNN 的随机性，从而保证每次运行的结果都是相同的。

aaaaa=[]

import gc
#tensor(-1.5224, device='cuda:0')
#for act in [nn.ReLU]:
#  for f in [0.0001]:
#    for weight_decay in [0]:
#       for liu in [zuko.flows.NSF]:
#        for transfomr in [12]:#特征提取的blocks
#         for num in [2048]:
#          for trans in [18]:#NF的残差块--画图的
#           for beishu in [1024]:
#

for act in [nn.ReLU]:
  for f in [0.0001]:
    for weight_decay in [0]:
       for liu in [zuko.flows.NSF]:
        for transfomr in [7]:#特征提取的blocks
         for num in [4096]:
          for trans in [9]:#NF的残差块--画图的
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



                

            torch.manual_seed(2234)#重置参数，在循环里面可以保证可以复现
            
            model= ResNet(in_channels=4,classes=2).cuda()#beishu是残差网络输出尺寸（npe的输入参数）
            model.apply(weight_init)
            
            optimizer = optim.AdamW(model.parameters(), lr=f,weight_decay=weight_decay)#学习率！！！！！！！！！！！
            #在优化器选项里添加正则化,weight_decay=0.01，l2正则化

            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)#学习率衰减
            #https://zhuanlan.zhihu.com/p/363338422

            step = GDStep(optimizer, clip=1.0)  # gradient descent step with gradient clipping,有了他不用optimizer.step
            loss = nn.MSELoss()

            list_los=[]
            list_losvail=[]
        
            with tqdm(range(2000), unit='epoch') as tq:#epoch
                best_loss = np.inf#早停
                best_model_weights = None
                patience=30#10个周期不降就停
                i_all=0
                num=10000#总数
                num_v=2000
                datast=20#batch_size
                nnum=num//datast
                nnum_v=num_v//datast
                
                for epoch in tq:
                    optimizer.zero_grad()
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    all_x = torch.tensor(np.load(data_set_dir+f'train_x_{i_all}.npy'),dtype=torch.float32)
                    all_y = torch.tensor(np.load(data_set_dir+f'train_can_{i_all}.npy'),dtype=torch.float32)

                    all_x_vail = torch.tensor(np.load(data_set_dir+f'vail_x_{i_all}.npy'),dtype=torch.float32)
                    all_y_vail = torch.tensor(np.load(data_set_dir+f'vail_can_{i_all}.npy'),dtype=torch.float32)

                    i_all+=1
                    if i_all>=400:
                        i_all=0
                                              
                                              

                    model.train()
                    losses = torch.stack([
                        step(loss(model(all_x[i].cuda()), preprocess(all_y[i]).cuda()))
                        for i in range(nnum) # 这样写是遍历全部元素，实例那样是因为他是采样器
                    ])
                    
                    del all_x,all_y
                    
                    model.eval()
                    with torch.no_grad():
                        val_losses = torch.stack([
                            loss(model(all_x_vail[i].cuda()), preprocess(all_y_vail[i]).cuda())
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
                    file = open('/home/suntianyang/GW_lens/resnet_can.txt', 'w')
                    file.write('last_last:'+str(last) + '\n')
                    file.write('___last__:'+str(now) + '\n')
                    
                    # 将每个元素写入文件中
                    for item1, item2 in zip(data1, data2):
                        file.write(str(item1) + ' ' + str(item2) + '\n')
                    
                    # 关闭文件
                    file.close()
                    
                    del losses,data1,data2
                    
                    if losval < best_loss:
                        best_loss = losval
                        epochs_without_improvement = 0
                        best_model_weights = model.state_dict()
                    else:
                        epochs_without_improvement += 1

                    # 如果验证集上的损失连续patience个epoch没有提高，则停止训练
                    if epochs_without_improvement == patience:
                        model.load_state_dict(best_model_weights)
                        print('Early stopping at epoch {}...'.format(epoch-patience+1))
                        break

            aaaaa.append('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-patience],len(list_los)-patience))
            print('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-patience],len(list_los)-patience))

# %%
torch.save(model, f'./net_ann_2/NPE_mid_t_lens_res_can_all.pth')

# %%


# %%


# %%
#重置网络参数，方便for循环调参
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Linear(2, 128) 
        self.norm1 = nn.LayerNorm(128)
        
        self.layer2 = nn.Linear(128, 128)
        self.norm2 = nn.LayerNorm(128)

        self.layer3 = nn.Linear(128, 128)
        self.norm3 = nn.LayerNorm(128)
        
        self.layer4 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(p=0.2)
        #self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.layer1(x)) 
        #x = self.dropout(x)
        x = self.norm1(x)
        
        x = self.relu(self.layer2(x))
        #x = self.dropout(x)
        x = self.norm2(x)
        
        x = self.relu(self.layer3(x))
        #x = self.dropout(x)
        x = self.norm3(x)
        
        x = self.layer4(x)
        return x
    


# %%
def get_wave_plus_gaosnoise_t_gen_cai(data):
    samples_all=torch.zeros((data.shape[0],2))
    for i in range(data.shape[0]):
        model.eval()
        with torch.no_grad():
            samples = model(data[i].cuda()).cpu()

        samples_all[i,0]=samples[0,0]
        samples_all[i,1]=samples[0,1]
            
    return samples_all

# %%
import gc
aaaaa=[]
torch.set_default_dtype(torch.float32)#更改默认精度（e指数需要高精度）
torch.backends.cudnn.deterministic = True #禁用 cuDNN 的随机性，从而保证每次运行的结果都是相同的。
torch.manual_seed(3234)#重置参数，在循环里面可以保证可以复现

f=2e-3
weight_decay=0.0000
torch.manual_seed(2234)#重置参数，在循环里面可以保证可以复现

estimator = MyNetwork().cuda()
estimator.apply(weight_init)
optimizer = optim.AdamW(estimator.parameters(), lr=f,weight_decay=weight_decay)#学习率！！！！！！！！！！！
#在优化器选项里添加正则化,weight_decay=0.01，l2正则化
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)#学习率衰减
#https://zhuanlan.zhihu.com/p/363338422
step = GDStep(optimizer, clip=1.0)  # gradient descent step with gradient clipping,有了他不用optimizer.step
loss = nn.BCEWithLogitsLoss()
list_los=[]
list_val_los=[]
#非采样器的
torch.backends.cudnn.deterministic = True #禁用 cuDNN 的随机性，从而保证每次运行的结果都是相同的。

aaaaa=[]
now= datetime.now() 

with tqdm(range(2000), unit='epoch') as tq:#epoch
    best_loss = np.inf#早停
    best_model_weights = None
    patience=15#10个周期不降就停
    i=0
    i_all=0
    num=10000#总数
    num_v=2000
    datast=20#batch_size
    nnum=num//datast
    nnum_v=num_v//datast

    list_los=[]
    list_losvail=[]
    
    for epoch in tq:
        optimizer.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()

        all_x = torch.tensor(np.load(data_set_dir+f'train_x_{i_all}.npy'),dtype=torch.float32).reshape(num,1,4,4096*10)
        all_x=get_wave_plus_gaosnoise_t_gen_cai(all_x).reshape(nnum,datast,2)
        all_y = torch.tensor(np.load(data_set_dir+f'train_y_{i_all}.npy'),dtype=torch.float32)

        
        all_x_vail = torch.tensor(np.load(data_set_dir+f'vail_x_{i_all}.npy'),dtype=torch.float32).reshape(num_v,1,4,4096*10)
        all_x_vail=get_wave_plus_gaosnoise_t_gen_cai(all_x_vail).reshape(nnum_v,datast,2)
        all_y_vail = torch.tensor(np.load(data_set_dir+f'vail_y_{i_all}.npy'),dtype=torch.float32)
        
        i_all+=1
        if i_all>=400:
            i_all=0
        
        
        estimator.train()
        losses = torch.stack([
            step(loss(estimator(all_x[i].cuda()), (all_y[i]).cuda()))
            for i in range(nnum) # 这样写是遍历全部元素，实例那样是因为他是采样器
        ])
        
        del all_x,all_y
        
        estimator.eval()
        with torch.no_grad():
            val_losses = torch.stack([
                loss(estimator(all_x_vail[i].cuda()), (all_y_vail[i]).cuda())
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
        file = open('/home/suntianyang/GW_lens/resnet_can_p.txt', 'w')
        file.write('last_last:'+str(last) + '\n')
        file.write('___last__:'+str(now) + '\n')
        
        # 将每个元素写入文件中
        for item1, item2 in zip(data1, data2):
            file.write(str(item1) + ' ' + str(item2) + '\n')
        
        # 关闭文件
        file.close()
        
        del losses,data1,data2
        
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
            
aaaaa.append('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-patience],len(list_los)-patience))
print('act{}学习率{},正则化{},,参数数目{},参数层数{},流层{},中间层{},最佳损失{},训练周期{}'.format(act,f,weight_decay,num,transfomr,trans,beishu,list_los[-1-patience],len(list_los)-patience))

# %%
torch.save(estimator,f'./net_ann_2/NPE_mid_t_lens_res_can_hou.pth')

# %% [markdown]
# # 训练

# %%


# %% [markdown]
# # 测试

# %%
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

model=torch.load(f'./net_ann_2/NPE_mid_t_lens_res_can_all.pth')
model.eval();

# %%
estimator=torch.load(f'/home/suntianyang/work5/ligo/net_ann_2/NPE_mid_t_lens_res_can_hou.pth')
estimator.eval()

# %% [markdown]
# # 保存

# %%
from sklearn.metrics import roc_curve, auc
import gc

# %%
def get_wave_plus_gaosnoise_t_gen_cai(data):
    samples_all=torch.zeros((data.shape[0],2))
    for i in range(data.shape[0]):
        model.eval()
        with torch.no_grad():
            samples = model(data[i].unsqueeze(0).cuda()).cpu()

        samples_all[i,0]=samples[0,0]
        samples_all[i,1]=samples[0,1]
            
    return samples_all

# %%


# %%
Z_L=0
for SNRRR in [16,24]:
    plt.figure()
    for I_tt in np.linspace(0,1,20):
        for dtt in np.linspace(0,0.2,20):
            
            all_x_vail_data=np.load(f'/home/DATA/suntianyang/gw_len/data/I_dt/test_PM_data_{I_tt}_{dtt}_{SNRRR}.npy')
            all_x_vail=torch.tensor(all_x_vail_data,dtype=torch.float32)
            all_x_vail=get_wave_plus_gaosnoise_t_gen_cai(all_x_vail).reshape(-1,2)
            all_x_vail=torch.tensor(all_x_vail,dtype=torch.float32)

            y_pre_all=[]
            for i in range(100):
                estimator.eval()
                with torch.no_grad():
                    #print(all_x_vail[i].dtype)
                    y_pre = torch.sigmoid(estimator(all_x_vail[i].cuda()))
                y_pre_all.append(float(y_pre))
            
            threshold = 0.8
            # 根据阈值确定预测标签
            y_pre = (np.array(y_pre_all) >= threshold).astype(int)
            plt.scatter(I_tt,dtt,c=sum(y_pre)/100, vmin=0, vmax=1)


                        
    plt.colorbar()
    plt.show()

# %%


# %%
Z_L=0
for SNRRR in [16,24]:
    plt.figure()
    for I_tt in np.linspace(0,1,20):
        for dtt in np.linspace(0,0.2,20):
            
            all_x_vail_data=np.load(f'/home/DATA/suntianyang/gw_len/data/I_dt/test_SIS_{1}_data_{I_tt}_{dtt}_{SNRRR}.npy')
            all_x_vail=torch.tensor(all_x_vail_data,dtype=torch.float32)
            all_x_vail=get_wave_plus_gaosnoise_t_gen_cai(all_x_vail).reshape(-1,2)
            all_x_vail=torch.tensor(all_x_vail,dtype=torch.float32)

            y_pre_all=[]
            for i in range(100):
                estimator.eval()
                with torch.no_grad():
                    #print(all_x_vail[i].dtype)
                    y_pre = torch.sigmoid(estimator(all_x_vail[i].cuda()))
                y_pre_all.append(float(y_pre))
            
            threshold = 0.8
            # 根据阈值确定预测标签
            y_pre = (np.array(y_pre_all) >= threshold).astype(int)
            plt.scatter(I_tt,dtt,c=sum(y_pre)/100, vmin=0, vmax=1)


                        
    plt.colorbar()
    plt.show()

# %%


# %%
Z_L=0
for SNRRR in [16,24]:
    plt.figure()
    for I_tt in np.linspace(0,1,20):
        for dtt in np.linspace(0,0.2,20):
            try:
                all_x_vail_data=np.load(f'/home/DATA/suntianyang/gw_len/data/I_dt/test_SIS_{2}_data_{I_tt}_{dtt}_{SNRRR}.npy')
            except:
                print(1)
            
            all_x_vail=torch.tensor(all_x_vail_data,dtype=torch.float32)
            all_x_vail=get_wave_plus_gaosnoise_t_gen_cai(all_x_vail).reshape(-1,2)
            all_x_vail=torch.tensor(all_x_vail,dtype=torch.float32)

            y_pre_all=[]
            for i in range(100):
                estimator.eval()
                with torch.no_grad():
                    #print(all_x_vail[i].dtype)
                    y_pre = torch.sigmoid(estimator(all_x_vail[i].cuda()))
                y_pre_all.append(float(y_pre))
            
            threshold = 0.8
            # 根据阈值确定预测标签
            y_pre = (np.array(y_pre_all) >= threshold).astype(int)
            plt.scatter(I_tt,dtt,c=sum(y_pre)/100, vmin=0, vmax=1)


                        
    plt.colorbar()
    plt.show()

# %%
Z_L=0
for SNRRR in [24]:
    plt.figure()
    for I_tt in np.linspace(0,1,20):
        for dtt in np.linspace(0,0.2,20):
            try:
                all_x_vail_data=np.load(f'/home/DATA/suntianyang/gw_len/data/I_dt/test_SIS_{2}_data_{I_tt}_{dtt}_{SNRRR}.npy')
            except:
                print(1)
            
            all_x_vail=torch.tensor(all_x_vail_data,dtype=torch.float32)
            all_x_vail=get_wave_plus_gaosnoise_t_gen_cai(all_x_vail).reshape(-1,2)
            all_x_vail=torch.tensor(all_x_vail,dtype=torch.float32)

            y_pre_all=[]
            for i in range(100):
                estimator.eval()
                with torch.no_grad():
                    #print(all_x_vail[i].dtype)
                    y_pre = torch.sigmoid(estimator(all_x_vail[i].cuda()))
                y_pre_all.append(float(y_pre))
            
            threshold = 0.8
            # 根据阈值确定预测标签
            y_pre = (np.array(y_pre_all) >= threshold).astype(int)
            plt.scatter(I_tt,dtt,c=sum(y_pre)/100, vmin=0, vmax=1)


                        
    plt.colorbar()
    plt.show()

# %%



