
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[48]:


# 資料載入
train = np.loadtxt('data/920data.csv',delimiter=',',dtype='int')
df = pd.DataFrame(train)
print(df)
train_x = train[:,0]
train_y = train[:,1]

plt.scatter(train_x,train_y)
plt.show()


# In[49]:


train_z = -15.5 + (0.1) * train_x
plt.scatter(train_z,train_y)
plt.show()


# In[50]:


theta0 = np.random.rand()
theta1 = np.random.rand()

def f(x):
    return theta0 + theta1 * x

def E(x,y):
    return 0.5 * np.sum((y - f(x)) ** 2)

ETA = 1e-3
diff = 1
count = 0


# In[51]:


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


# In[52]:


plt.scatter(train_x,train_y,color='red')
plt.plot(X,Y,color='blue')
plt.show()


# In[53]:


def data():
    train = np.loadtxt('data/920data.csv',delimiter=',',dtype='int')
    df = pd.DataFrame(train)
    train_x = train[:,0]
    train_y = train[:,1]
    #print(train_x,train_y)
    #x = [25,35,49,59,85,99,112,117,134,148,159,159,162,173,191,198,204,216,235,272]
    #y = [332,310,325,319,308,334,387,385,392,413,400,427,425,498,498,522,519,539,591,659]
    return train_x,train_y

def SGD(x,y):
    theta0 = np.random.rand()
    theta1 = np.random.rand()

    def f(x):
        return theta0 + theta1 * x

    def E(x,y):
        return 0.5 * np.sum((y - f(x)) ** 2)

    ETA = 1e-3
    diff = 1
    count = 0
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
    return theta1,theta0 

if __name__ == '__main__':
    train_x,train_y = data()
    theta1,theta0 = SGD(train_x,train_y)
    print(theta0,theta1)
    X = range(0,10)
    Y = [(theta1*i+theta0) for i in X]
    print(X,Y)
 
    plt.scatter(train_x,train_y,color='red')
    plt.plot(X,Y,color='blue')
    plt.show()

