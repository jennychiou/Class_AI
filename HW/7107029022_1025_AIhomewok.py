
# coding: utf-8

# # SVC

# In[1]:


from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


iris = datasets.load_iris()
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
print("target_names: "+str(iris['target_names']))
y = pd.DataFrame(iris['target'], columns=['target'])

iris_data = pd.concat([x,y], axis=1)
iris_data = iris_data[['sepal length (cm)','petal length (cm)','target']]
iris_data = iris_data[iris_data['target'].isin([0,1])]
iris_data.head(3)


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(
    iris_data[['sepal length (cm)','petal length (cm)']], iris_data[['target']], test_size=0.3, random_state=0)


# In[5]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[6]:


from sklearn.svm import SVC


# In[7]:


svm = SVC(kernel='linear', probability=True)


# In[8]:


svm.fit(X_train_std,y_train['target'].values)


# In[9]:


svm.predict(X_test_std)


# In[10]:


y_test['target'].values


# In[11]:


error = 0
for i, v in enumerate(svm.predict(X_test_std)):
    if v!= y_test['target'].values[i]:
        error+=1
print(error)


# In[12]:


svm.predict_proba(X_test_std)


# In[13]:


from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 畫出決定的平面
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],y=X[y == cl, 1],alpha=0.6,c=cmap(idx),edgecolor='black',marker=markers[idx],label=cl)

    # 標註測試的樣本
    if test_idx:
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],X_test[:, 1],c='',alpha=1.0,edgecolor='black',linewidths=1,marker='o',s=55, label='test set')


# In[14]:


plot_decision_regions(X_train_std, y_train['target'].values, classifier=svm)
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# # SVR

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[16]:


# 載入資料集
dataset = pd.read_csv('data/Position_Salaries.csv')
x = dataset.iloc[:,1].values
y = dataset.iloc[:,-1].values


# In[17]:


# reshape x 和 y
x = x.reshape(-1,1)
y = y.reshape(-1,1)


# In[18]:


# feature scaling
from sklearn.preprocessing import StandardScaler
standardscaler_x = StandardScaler()
x = standardscaler_x.fit_transform(x)
standardscaler_y = StandardScaler()
y = standardscaler_y.fit_transform(y)


# In[19]:


# reshape 成一維陣列
y = y.reshape(len(y),)
print(y)


# In[20]:


# Fittign SVR模型使用POLY內核進行數據處理
from sklearn.svm import SVR
regressor = SVR(kernel='poly')
regressor = regressor.fit(x,y)


# In[21]:


# 縮放測試數據以進行預測
test = np.zeros(1) # we are testing just one value
test[0]= 6.5
test = test.reshape(1,1) # reshape 成二維陣列
test = standardscaler_x.transform(test) 


# In[22]:


# 做預測
y_pred = regressor.predict(test)
y_predict = standardscaler_y.inverse_transform(y_pred)


# In[23]:


#視覺化fit的模型和數據集
plt.scatter(x,y, color ='red', alpha=0.6)
plt.scatter(test,y_pred,color = 'blue', marker='D')
plt.plot(x,regressor.predict(x),color='green')
plt.title('Level vs Salary (train data) using poly kernel')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.grid()
plt.show()


# In[24]:


# 用RBF核與SVR模型fit數據
regressor2 = SVR(kernel='rbf')
regressor2 = regressor2.fit(x,y)

# 做預測
y_pred = regressor2.predict(test)
y_predict = standardscaler_y.inverse_transform(y_pred)


# In[25]:


#視覺化fit的模型和數據集
plt.scatter(x,y, color ='red', alpha=0.6)
plt.scatter(test,y_pred,color = 'blue', marker='D')
plt.plot(x,regressor2.predict(x),color='green')
plt.title('Level vs Salary (train data) using rbf kernel')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.grid()
plt.show()

