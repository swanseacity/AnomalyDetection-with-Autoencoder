#!/usr/bin/env python
# coding: utf-8

# ### 라이브러리 import

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
import joblib

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


# # 1. 데이터 사용하기 쉽게 조작

# ### 데이터 불러오기

# In[2]:


# Normal 데이터 불러오기
for i in range(300):
    
    path1 = './SpotData/Normal/Spot_%d.csv'%(i+1)
    c1 = 'Normal_%d = pd.read_csv(path1, sep=",", header=None)'%(i+1)
    exec(c1)

# AbNormal 데이터 불러오기 (Shunt)
for i in range(100):

    path2 = './SpotData/Shunt/Spot_%d.csv'%(i+1)
    c2 = 'Shunt_%d = pd.read_csv(path2, sep=",", header=None)'%(i+1)

    exec(c2)

# AbNormal 데이터 불러오기 (Misalign)
for i in range(100):

    path3 = './SpotData/Misalign/Spot_%d.csv'%(i+1)
    c3 = 'Misalign_%d = pd.read_csv(path3, sep=",", header=None)'%(i+1)

    exec(c3)
    
# AbNormal 데이터 불러오기 (TipWear)
for i in range(100):

    path4 = './SpotData/TipWear/Spot_%d.csv'%(i+1)
    c4 = 'Tipwear_%d = pd.read_csv(path4, sep=",", header=None)'%(i+1)

    exec(c4)
    
    
    
    
print(Normal_100.shape, Shunt_100.shape)


# ### 데이터 저장

# In[3]:


for i in range(300):
    
    path1 = './FinalData/Normal_%d'%(i+1)
    c1 = 'Normal_%d.to_csv(path1, sep = ",", header = None, index = None)'%(i+1)
    exec(c1)
    
    
for i in range(100):
    path2 = './FinalData/Shunt_%d'%(i+1)
    c2 = 'Shunt_%d.to_csv(path2, sep = ",", header = None, index = None)'%(i+1)
    exec(c2)


for i in range(100):
    path3 = './FinalData/Misalign_%d'%(i+1)
    c3 = 'Misalign_%d.to_csv(path3, sep = ",", header = None, index = None)'%(i+1)
    exec(c3)

    
for i in range(100):
    path4 = './FinalData/TipWear_%d'%(i+1)
    c4 = 'Tipwear_%d.to_csv(path4, sep = ",", header = None, index = None)'%(i+1)
    exec(c4)


# ### 데이터 합치기

# In[4]:


Train_Normal   = np.zeros((300,2774,3))
Test_Shunt     = np.zeros((100,2774,3))
Test_Misalign  = np.zeros((100,2774,3))
Test_Tipwear   = np.zeros((100,2774,3))
Test_Normal    = np.zeros((100,2774,3))



# 가속도에 대한 정상 학습데이터 300개
for i in range(300):
    path1 = './FinalData/Normal_%d'%(i+1)
    add1 = pd.read_csv(path1, sep = ",", header = None)
    Train_Normal[i,:,:]  = add1  

# 가속도에 대한 전체 고장 검증데이터 100개      
for i in range(100):
    path2 = './FinalData/Shunt_%d'%(i+1)
    add2 = pd.read_csv(path2, sep = ",", header = None)
    Test_Shunt[i,:,:] = add2

for i in range(100):        
    path3 = './FinalData/Misalign_%d'%(i+1)
    add3 = pd.read_csv(path3, sep = ",", header = None)
    Test_Misalign[i,:,:] = add3
    
for i in range(100):        
    path4 = './FinalData/TipWear_%d'%(i+1)
    add4 = pd.read_csv(path4, sep = ",", header = None)
    Test_Tipwear[i,:,:] = add4
    
# 가속도에 대한 전체 정상 검증데이터 100개    
for i in range(100):
    path5 = './SpotData/Normal/Spot_%d.csv'%(i+301)
    add5 = pd.read_csv(path5, sep = ",", header = None)
    Test_Normal[i,:,:] = add5
    

print(Train_Normal.shape,Test_Normal.shape)


# ### 데이터 정규화

# #### Train_Normal(300,2774,3), Test_Shunt, Test_Misalign, Test_Tipwear, Test_Normal(100,2774,3)

# In[5]:


for i in range(300):
    for j in range(3):
        max_val = np.max(Train_Normal[i].T[j])
        min_val = np.min(Train_Normal[i].T[j])
        Train_Normal[i].T[j] = (Train_Normal[i].T[j] - min_val)/(max_val-min_val)

for i in range(100):
    for j in range(3):
        max_val = np.max(Test_Normal[i].T[j])
        min_val = np.min(Test_Normal[i].T[j])
        Test_Normal[i].T[j] = (Test_Normal[i].T[j] - min_val)/(max_val-min_val)

        max_val = np.max(Test_Shunt[i].T[j])
        min_val = np.min(Test_Shunt[i].T[j])
        Test_Shunt[i].T[j] = (Test_Shunt[i].T[j] - min_val)/(max_val-min_val)

        max_val = np.max(Test_Misalign[i].T[j])
        min_val = np.min(Test_Misalign[i].T[j])
        Test_Misalign[i].T[j] = (Test_Misalign[i].T[j] - min_val)/(max_val-min_val)

        max_val = np.max(Test_Tipwear[i].T[j])
        min_val = np.min(Test_Tipwear[i].T[j])
        Test_Tipwear[i].T[j] = (Test_Tipwear[i].T[j] - min_val)/(max_val-min_val)

        


# ## 3-(2) 모델 빌드하기

# In[6]:


# 오토인코더 모듈 정의
class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        
        # encoder는 차원을 점차 줄여나감
        self.encoder = tf.keras.Sequential([layers.InputLayer(input_shape=(2774,3)),
                                            layers.Conv1D(filters = 128,kernel_size = 3,strides = 3,activation = 'relu'),
                                            layers.AveragePooling1D(pool_size = 3,strides = 3),
                                            layers.Conv1D(filters = 32,kernel_size = 3,strides = 3,activation = 'relu'),
                                            layers.AveragePooling1D(pool_size = 3,strides = 3),
                                            layers.Conv1D(filters = 8,kernel_size = 3,strides = 3, activation = 'relu')])
        # decoder는 차원을 점차 복원함.
        self.decoder = tf.keras.Sequential([layers.Conv1DTranspose(filters = 8, kernel_size =4,strides = 3,activation = 'relu'),
                                            layers.Conv1DTranspose(filters = 32,kernel_size = 3,strides = 3,activation = 'relu'),
                                            layers.Conv1DTranspose(filters = 64,kernel_size = 5,strides = 3,activation = 'relu'),
                                            layers.Conv1DTranspose(filters = 128,kernel_size = 3,strides = 3,activation = 'relu'),
                                            layers.Conv1DTranspose(filters = 3,kernel_size = 5,strides = 3,activation = 'sigmoid')])
                                            
        
    def call(self, x):
        encoded = self.encoder(x)        
        decoded = self.decoder(encoded)  
        return decoded

autoencoder = AnomalyDetector()


# In[7]:


autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.001), loss='mae')


# In[8]:


s=0
normal_acc1=[]
normal_acc2=[]
misalign_acc1=[]
misalign_acc2=[]
shunt_acc1=[]
shunt_acc2=[]
tipwear_acc1=[]
tipwear_acc2=[]


# In[9]:


encoded_train = autoencoder.encoder(Train_Normal).numpy()
decoded_train = autoencoder.decoder(encoded_train).numpy()
train_loss = tf.keras.losses.mae(decoded_train, Train_Normal)
train_loss = np.sum(train_loss,axis=1)/2774

print(np.mean(train_loss),np.std(train_loss))


# In[10]:


while True:
    s+=1
    history = autoencoder.fit(Train_Normal, Train_Normal,epochs=1,shuffle=True)
    
    encoded_train = autoencoder.encoder(Train_Normal).numpy()
    decoded_train = autoencoder.decoder(encoded_train).numpy()
    train_loss = tf.keras.losses.mae(decoded_train, Train_Normal)
    train_loss = np.sum(train_loss,axis=1)/2774

    encoded_normal = autoencoder.encoder(Test_Normal).numpy()
    decoded_normal = autoencoder.decoder(encoded_normal).numpy()
    normal_loss = tf.keras.losses.mae(decoded_normal, Test_Normal)
    normal_loss = np.sum(normal_loss,axis=1)/2774

    encoded_misalign = autoencoder.encoder(Test_Misalign).numpy()
    decoded_misalign = autoencoder.decoder(encoded_misalign).numpy()
    misalign_loss = tf.keras.losses.mae(decoded_misalign, Test_Misalign)
    misalign_loss = np.sum(misalign_loss,axis=1)/2774

    encoded_shunt = autoencoder.encoder(Test_Shunt).numpy()
    decoded_shunt = autoencoder.decoder(encoded_shunt).numpy()
    shunt_loss = tf.keras.losses.mae(decoded_shunt, Test_Shunt)
    shunt_loss = np.sum(shunt_loss,axis=1)/2774

    encoded_tipwear = autoencoder.encoder(Test_Tipwear).numpy()
    decoded_tipwear = autoencoder.decoder(encoded_tipwear).numpy()
    tipwear_loss = tf.keras.losses.mae(decoded_tipwear, Test_Tipwear)
    tipwear_loss = np.sum(tipwear_loss,axis=1)/2774
    
    
    threshold1 = np.mean(train_loss) + 1*np.std(train_loss)
    threshold2 = np.mean(train_loss) + 2*np.std(train_loss)
    
    s1=0
    s2=0
    for i in range(100):
        if normal_loss[i]<threshold1:
            s1+=1
        if normal_loss[i]<threshold2:
            s2+=1
    acc_normal1 = s1/100
    acc_normal2 = s2/100
    normal_acc1.append(acc_normal1)
    normal_acc2.append(acc_normal2)
    
    s1=0
    s2=0
    for i in range(100):
        if misalign_loss[i]>threshold1:
            s1+=1
        if misalign_loss[i]>threshold2:
            s2+=1
    acc_misalign1 = s1/100
    acc_misalign2 = s2/100
    misalign_acc1.append(acc_misalign1)
    misalign_acc2.append(acc_misalign2)
        
    s1=0
    s2=0
    for i in range(100):
        if shunt_loss[i]>threshold1:
            s1+=1
        if shunt_loss[i]>threshold2:
            s2+=1
    acc_shunt1 = s1/100
    acc_shunt2 = s2/100
    shunt_acc1.append(acc_shunt1)
    shunt_acc2.append(acc_shunt2)
        
    s1=0
    s2=0
    for i in range(100):
        if tipwear_loss[i]>threshold1:
            s1+=1
        if tipwear_loss[i]>threshold2:
            s2+=1
    acc_tipwear1 = s1/100
    acc_tipwear2 = s2/100
    tipwear_acc1.append(acc_tipwear1)
    tipwear_acc2.append(acc_tipwear2)
    
    print('iteration{}'.format(s))
    print(acc_normal1,acc_misalign1,acc_shunt1,acc_tipwear1)
    print(acc_normal2,acc_misalign2,acc_shunt2,acc_tipwear2)
   
        
    
    if (acc_normal1>0.84 and acc_misalign1>0.85 and acc_shunt1>0.85 and acc_tipwear1>0.85) or(acc_normal1>0.91 and acc_misalign1>0.93  and acc_tipwear1>0.94) or (acc_normal1>0.91 and  acc_shunt1>0.93 and acc_tipwear1>0.94) or (acc_normal1>0.91 and  acc_misalign1>0.93 and acc_shunt1>0.94):
        model = autoencoder
        model.save('./MLModels/CNN1D_retry_last%d'% s)
        break
    
    elif (acc_normal2>0.84 and acc_misalign2>0.89 and acc_shunt2>0.89 and acc_tipwear2>0.89) or(acc_normal2>0.91 and acc_misalign2>0.93  and acc_tipwear2>0.94) or (acc_normal2>0.91 and  acc_shunt2>0.93 and acc_tipwear2>0.94) or (acc_normal2>0.91 and  acc_misalign2>0.93 and acc_shunt2>0.94):
        model = autoencoder
        model.save('./MLModels/CNN1D_retry_last%d'% s)
        break
        
    elif s==5000:
        model = autoencoder
        model.save('./MLModels/CNN1D_retry_last%d'% s)
        break


plt.figure(figsize=(3,15))

ax = plt.subplot(2,1,1)
plt.plot(normal_acc1,label="normal")
plt.plot(misalign_acc1,label="misalign")
plt.plot(shunt_acc1,label="shunt")
plt.plot(tipwear_acc1,label="tipwear")
plt.title('mean-1std')
plt.legend()

ax = plt.subplot(2,1,2)
plt.plot(normal_acc2,label="normal")
plt.plot(misalign_acc2,label="misalign")
plt.plot(shunt_acc2,label="shunt")
plt.plot(tipwear_acc2,label="tipwear")
plt.title('mean-2std')
plt.legend()


# In[12]:


autoencoder.encoder.summary()
autoencoder.decoder.summary()


# # Reconstruction 시각화

# In[13]:


encoded_train = autoencoder.encoder(Train_Normal).numpy()
decoded_train = autoencoder.decoder(encoded_train).numpy()

encoded_normal = autoencoder.encoder(Test_Normal).numpy()
decoded_normal = autoencoder.decoder(encoded_normal).numpy()
error_normal = Test_Normal - decoded_normal 

encoded_misalign = autoencoder.encoder(Test_Misalign).numpy()
decoded_misalign = autoencoder.decoder(encoded_misalign).numpy()
error_misalign = Test_Misalign - decoded_misalign 

encoded_shunt = autoencoder.encoder(Test_Shunt).numpy()
decoded_shunt = autoencoder.decoder(encoded_shunt).numpy()
error_shunt = Test_Shunt - decoded_shunt 

encoded_tipwear = autoencoder.encoder(Test_Tipwear).numpy()
decoded_tipwear = autoencoder.decoder(encoded_tipwear).numpy()
error_tipwear = Test_Tipwear - decoded_tipwear 


# In[14]:


plt.figure(figsize=(20,20))

for i in range(3):
    show = plt.subplot(5,3,i+1)
    #plt.plot(np.array(Train_Normal[0]).T[i], 'b')
    plt.plot(np.array(decoded_train[0]).T[i], 'r')
    #plt.fill_between(np.arange(2774), np.array(decoded_train[0]).T[i],np.array(Train_Normal[0]).T[i], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    
    show = plt.subplot(5,3,i+4)
    plt.plot(np.array(Test_Normal[0]).T[i], 'b')
    plt.plot(np.array(decoded_normal[0]).T[i], 'r')
    plt.fill_between(np.arange(2774), np.array(decoded_normal[0]).T[i],np.array(Test_Normal[0]).T[i], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    
    show = plt.subplot(5,3,i+7)
    plt.plot(np.array(Test_Misalign[0]).T[i], 'b')
    plt.plot(np.array(decoded_misalign[0]).T[i], 'r')
    plt.fill_between(np.arange(2774), np.array(decoded_misalign[0]).T[i],np.array(Test_Misalign[0]).T[i], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    
    show = plt.subplot(5,3,i+10)
    plt.plot(np.array(Test_Shunt[0]).T[i], 'b')
    plt.plot(np.array(decoded_shunt[0]).T[i], 'r')
    plt.fill_between(np.arange(2774), np.array(decoded_shunt[0]).T[i],np.array(Test_Shunt[0]).T[i], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    
    show = plt.subplot(5,3,i+13)
    plt.plot(np.array(Test_Tipwear[0]).T[i], 'b')
    plt.plot(np.array(decoded_tipwear[0]).T[i], 'r')
    plt.fill_between(np.arange(2774), np.array(decoded_tipwear[0]).T[i],np.array(Test_Tipwear[0]).T[i], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    
    
    


# # 오류 발생 위치 확인

# In[15]:


encoded_normal = autoencoder.encoder(Test_Normal).numpy()
decoded_normal = autoencoder.decoder(encoded_normal).numpy()
error_normal = Test_Normal - decoded_normal        
  


# In[16]:


max_acc_x = []
max_acc_y = []
max_vol_x = []
max_vol_y = []
max_cur_x = []
max_cur_y = []

for i in range(100):
    temp = np.concatenate((error_normal[i].T[0],error_normal[i].T[1],error_normal[i].T[2]),axis=0)

    maxval = np.max(temp)  
    maxindex = np.where(temp == maxval)
    if maxindex[0][0]<2774:
        max_acc_x.append(maxindex[0][0])
        max_acc_y.append(maxval)
        
    elif maxindex[0][0]<5548:
        max_vol_x.append(maxindex[0][0]%2774)
        max_vol_y.append(maxval)
        
    else:
        max_cur_x.append(maxindex[0][0]%2774)
        max_cur_y.append(maxval)

plt.figure(figsize=(5,15))
        
plt.subplot(3,1,1)      
plt.scatter(max_acc_x,max_acc_y,alpha=0.5)
plt.title('Normal test acceleration')
plt.xlim(0,2774)

plt.subplot(3,1,2)   
plt.scatter(max_vol_x,max_vol_y,alpha=0.5)
plt.title('Normal test voltage')
plt.xlim(0,2774)

plt.subplot(3,1,3)
plt.scatter(max_cur_x,max_cur_y,alpha=0.5)
plt.title('Normal test current')
plt.xlim(0,2774)
    

    


# In[17]:


max_acc_x = []
max_acc_y = []
max_vol_x = []
max_vol_y = []
max_cur_x = []
max_cur_y = []

for i in range(100):
    temp = np.concatenate((error_misalign[i].T[0],error_misalign[i].T[1],error_misalign[i].T[2]),axis=0)

    maxval = np.max(temp)  
    maxindex = np.where(temp == maxval)
    if maxindex[0][0]<2774:
        max_acc_x.append(maxindex[0][0])
        max_acc_y.append(maxval)
        
    elif maxindex[0][0]<5548:
        max_vol_x.append(maxindex[0][0]%2774)
        max_vol_y.append(maxval)
        
    else:
        max_cur_x.append(maxindex[0][0]%2774)
        max_cur_y.append(maxval)

plt.figure(figsize=(5,15))
        
plt.subplot(3,1,1)      
plt.scatter(max_acc_x,max_acc_y,alpha=0.5)
plt.title('Misalign acceleration')
plt.xlim(0,2774)

plt.subplot(3,1,2)   
plt.scatter(max_vol_x,max_vol_y,alpha=0.5)
plt.title('Misalign voltage')
plt.xlim(0,2774)

plt.subplot(3,1,3)
plt.scatter(max_cur_x,max_cur_y,alpha=0.5)
plt.title('Misalign current')
plt.xlim(0,2774)


# In[18]:


max_acc_x = []
max_acc_y = []
max_vol_x = []
max_vol_y = []
max_cur_x = []
max_cur_y = []

for i in range(100):
    temp = np.concatenate((error_shunt[i].T[0],error_shunt[i].T[1],error_shunt[i].T[2]),axis=0)

    maxval = np.max(temp)  
    maxindex = np.where(temp == maxval)
    if maxindex[0][0]<2774:
        max_acc_x.append(maxindex[0][0])
        max_acc_y.append(maxval)
        
    elif maxindex[0][0]<5548:
        max_vol_x.append(maxindex[0][0]%2774)
        max_vol_y.append(maxval)
        
    else:
        max_cur_x.append(maxindex[0][0]%2774)
        max_cur_y.append(maxval)

plt.figure(figsize=(5,15))
        
plt.subplot(3,1,1)      
plt.scatter(max_acc_x,max_acc_y,alpha=0.5)
plt.title('Shunt acceleration')
plt.xlim(0,2774)

plt.subplot(3,1,2)   
plt.scatter(max_vol_x,max_vol_y,alpha=0.5)
plt.title('Shunt voltage')
plt.xlim(0,2774)

plt.subplot(3,1,3)
plt.scatter(max_cur_x,max_cur_y,alpha=0.5)
plt.title('Shunt current')
plt.xlim(0,2774)


# In[19]:


max_acc_x = []
max_acc_y = []
max_vol_x = []
max_vol_y = []
max_cur_x = []
max_cur_y = []

for i in range(100):
    temp = np.concatenate((error_tipwear[i].T[0],error_tipwear[i].T[1],error_tipwear[i].T[2]),axis=0)

    maxval = np.max(temp)  
    maxindex = np.where(temp == maxval)
    if maxindex[0][0]<2774:
        max_acc_x.append(maxindex[0][0])
        max_acc_y.append(maxval)
        
    elif maxindex[0][0]<5548:
        max_vol_x.append(maxindex[0][0]%2774)
        max_vol_y.append(maxval)
        
    else:
        max_cur_x.append(maxindex[0][0]%2774)
        max_cur_y.append(maxval)

plt.figure(figsize=(5,15))
        
plt.subplot(3,1,1)      
plt.scatter(max_acc_x,max_acc_y,alpha=0.5)
plt.title('Tipwear acceleration')
plt.xlim(0,2774)

plt.subplot(3,1,2)   
plt.scatter(max_vol_x,max_vol_y,alpha=0.5)
plt.title('Tipwear voltage')
plt.xlim(0,2774)

plt.subplot(3,1,3)
plt.scatter(max_cur_x,max_cur_y,alpha=0.5)
plt.title('Tipwear current')
plt.xlim(0,2774)


# In[ ]:




