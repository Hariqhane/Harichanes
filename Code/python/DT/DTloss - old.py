# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from unet import unet_2d
from keras import optimizers
from sklearn.utils import shuffle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

a=np.load('new_f10zhenshiaaa_g1_8_white_houzao_o2.npy')
b=np.load('new_f10zhenshibbb_g1_8_white_houzao_o2.npy')
c=np.load('DTbaihua_real.npy')

x_train = np.append(a[0:16000],a[20000:22000],axis=0)
y_train = np.append(b[0:16000],b[20000:22000],axis=0)
x_train,y_train = shuffle(x_train,y_train)

x_val = np.append(a[16000:18000],a[36000:36500],axis=0)
y_val = np.append(b[16000:18000],b[36000:36500],axis=0)
x_val,y_val = shuffle(x_val,y_val) 

x_test=c

net = unet_2d.unet2D(n_filters=32, conv_width=3, network_depth=3, n_channels=1, x_dim=160, dropout=0.2, growth_factor=2, batchnorm=True, momentum=0.9, epsilon=0.001, activation='relu',maxpool=True)
net = net.build_model()
net.compile(optimizer=tf.optimizers.Nadam(learning_rate=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.001,clipnorm=1.),loss='binary_crossentropy')
history = net.fit(x_train, y_train, batch_size=32, epochs=300, validation_data=(x_val, y_val))

y_pred = net.predict(x_test)
np.save('y_pred_10.npy', y_pred)


'''
plt.figure(figsize=(12,7))
plt.plot(history.history['loss'][15:], label='train loss')
plt.plot(history.history['val_loss'][15:], linestyle='--', label='val loss')
plt.xlabel('Number of Epochs')
plt.ylabel('loss')
plt.title('unet_2d')
plt.legend()
plt.savefig('dt_r/y_pred_1.png')
plt.close()
'''
