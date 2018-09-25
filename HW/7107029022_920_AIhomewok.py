
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# 資料載入
train = np.loadtxt('data/920data.csv',delimiter=',',dtype='int')
df = pd.DataFrame(train)
print(df)
train_x = train[:,0]
train_y = train[:,1]

plt.scatter(train_x,train_y)
plt.show()


# In[3]:


#標準化
mu = train_x.mean()
sigma = train_x.std()
def standardize(x):
    return (x-mu)/sigma

train_z = standardize(train_x)
print(train_z)

plt.scatter(train_z,train_y)
plt.show()


# In[4]:


theta0 = np.random.rand()
theta1 = np.random.rand()

def f(x):
    return theta0 + theta1 * x

def E(x,y):
    return 0.5 * np.sum((y - f(x)) ** 2)

ETA = 1e-3
diff = 1
count = 0


# In[5]:


error = E(train_z,train_y)
while diff > 1e-2:
    
    top_theta0 = theta0 - ETA * np.sum((f(train_z) - train_y))
    top_theta1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)
    
    theta0 = top_theta0
    theta1 = top_theta1
    
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error
    
    count += 1
    log = '迭代第{}次: theta0 = {:.3f}, theta1 = {:.3f}, 總分 = {:.4f}'
    print(log.format(count, theta0, theta1, diff))
    
print('迭代次数: %d' % count)


# In[6]:


X = range(-2,3)
Y = [(theta1*i+theta0) for i in X]

plt.scatter(train_z,train_y,color='blue')
plt.plot(X,Y,color='red')
plt.show()

