
# coding: utf-8

# # Keras_Mnist_CNN

# In[1]:


from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
np.random.seed(10)


# # 資料預處理

# In[2]:


(x_Train, y_Train), (x_Test, y_Test) = mnist.load_data()


# In[3]:


x_Train4D=x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')


# In[4]:


x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255


# In[5]:


y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)


# # 建立模型

# In[6]:


from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D


# # 訓練數據 (Relu)

# In[7]:


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', input_shape=(28,28,1), activation='relu'))


# In[8]:


model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))


# In[9]:


print(model.summary())


# In[10]:


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 


# In[11]:


train_history=model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=15, batch_size=300,verbose=2)


# In[12]:


import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# In[13]:


show_train_history('acc','val_acc')


# In[14]:


show_train_history('loss','val_loss')


# In[15]:


# 評估準確率
scores = model.evaluate(x_Test4D_normalize , y_TestOneHot)
scores[1]


# # 訓練數據 (Sigmoid)

# In[16]:


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', input_shape=(28,28,1), activation='sigmoid'))


# In[17]:


model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))


# In[18]:


print(model.summary())


# In[19]:


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 


# In[20]:


train_history=model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=15, batch_size=300,verbose=2)


# In[21]:


show_train_history('acc','val_acc')


# In[22]:


show_train_history('loss','val_loss')


# In[23]:


# 評估準確率
scores = model.evaluate(x_Test4D_normalize , y_TestOneHot)
scores[1]


# # 使用不同的batch size

# In[24]:


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', input_shape=(28,28,1), activation='relu'))


# In[25]:


model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))


# In[26]:


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 


# 設batch_size=100

# In[27]:


train_history=model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=15, batch_size=100,verbose=2)


# In[28]:


show_train_history('acc','val_acc')


# In[29]:


show_train_history('loss','val_loss')


# 設batch_size=200

# In[30]:


train_history=model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=15, batch_size=200,verbose=2)


# In[31]:


show_train_history('acc','val_acc')


# In[32]:


show_train_history('loss','val_loss')


# 設batch_size=300

# In[33]:


train_history=model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=15, batch_size=300,verbose=2)


# In[34]:


show_train_history('acc','val_acc')


# In[35]:


show_train_history('loss','val_loss')


# # 使用不同的Optimization

# SGD

# In[36]:


model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
train_history=model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=15, batch_size=300,verbose=2)


# Adam

# In[37]:


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=15, batch_size=300,verbose=2)


# Adagrad

# In[38]:


model.compile(loss='categorical_crossentropy', optimizer='adagrad',metrics=['accuracy'])
train_history=model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=15, batch_size=300,verbose=2)


# Momentum

# In[54]:


model.compile(loss='categorical_crossentropy', optimizer='adagrad',metrics=['accuracy'])
train_history=model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=15, batch_size=300,verbose=2)


# # 增加模型複雜度或抽樣訓練 (達到Overfitting)

# 加L1、L2

# In[63]:


# 增加到256
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))


# In[64]:


print(model.summary())


# In[65]:


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=15, batch_size=300,verbose=2)


# In[66]:


show_train_history('acc','val_acc')


# In[67]:


show_train_history('loss','val_loss')


# In[71]:


# 增加到1024
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))


# In[72]:


print(model.summary())


# In[73]:


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=15, batch_size=300,verbose=2)


# In[74]:


show_train_history('acc','val_acc')


# In[75]:


show_train_history('loss','val_loss')

