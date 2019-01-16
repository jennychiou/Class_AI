
# coding: utf-8

# In[1]:


from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 


# In[2]:


img_width = 150
img_height = 150
train_data_dir = 'data/train'
valid_data_dir = 'data/validation'


# In[3]:


datagen = ImageDataGenerator(rescale = 1./255)


# In[4]:


train_generator = datagen.flow_from_directory(directory=train_data_dir,
                                              target_size=(img_width,img_height),
                                              classes=['frustrated','confuse','bored','joy', 'shock','concentrated','neutral'],
                                              class_mode='categorical',
                                              batch_size=16)


# In[5]:


validation_generator = datagen.flow_from_directory(directory=valid_data_dir,
                                                   target_size=(img_width,img_height),
                                                   classes=['frustrated','confuse','bored','joy','shock','concentrated','neutral'],
                                                   class_mode='categorical',
                                                   batch_size=32)


# In[6]:


#0挫折frustrated / 1困惑confuse / 2無聊bored / 3喜悅joy / 4驚訝shock / 5投入concentrated / 6中性neutral

model =Sequential()

model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('sigmoid'))


# In[7]:


print(model.summary())


# In[8]:


model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
training = model.fit_generator(generator=train_generator, steps_per_epoch=2048 // 16,epochs=20,validation_data=validation_generator,validation_steps=832//16)
model.save_weights('models/simple_CNN.h5')


# In[9]:


import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(training.history[train_acc])
    plt.plot(training.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# In[10]:


show_train_history('acc','val_acc')


# In[11]:


show_train_history('loss','val_loss')

