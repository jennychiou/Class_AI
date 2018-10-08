
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt


# In[2]:


#匯入資料
data = pd.read_csv('data/electricity.csv')


# In[3]:


# 針對用電量設定類別
def get_consumption_category(wt):
    if wt < 200:
        return 1
    elif 200 < wt < 400:
        return 2
    elif 400 < wt < 600:
        return 3
    elif 600 < wt < 800:
        return 4
    elif 800 < wt < 1000:
        return 5
    elif 1000 < wt < 1200:
        return 6
    else:
        return 7
  
data["electricity_consumption_category"] = data["electricity_consumption"].map(get_consumption_category)
data


# In[4]:


#只列出指定的欄位
df = data[['temperature','pressure', 'windspeed', 'electricity_consumption_category']]
print(type(df))

#檢視前 5 筆資料
df.head()


# In[5]:


df.info()


# In[6]:


from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

df = data[['temperature','pressure', 'windspeed', 'electricity_consumption_category']]
array = df.values
#print(array)
X = array[:,0:3]
Y = array[:,3]
seed = 7
num_trees = 30

kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[7]:


X=df.drop('electricity_consumption_category', axis=1)
y=df['electricity_consumption_category']

from sklearn.cross_validation import train_test_split
#分割訓練和測試集
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred= clf.predict(X_test)
print(y_pred)


# In[8]:


for estimator in clf.estimators_:
    print(estimator.predict(X_test))
    print(clf.estimator_errors_)

