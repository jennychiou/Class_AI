
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


X=df.drop('electricity_consumption_category', axis=1)
y=df['electricity_consumption_category']


# In[6]:


from sklearn.cross_validation import train_test_split


# In[7]:


#分割訓練和測試集
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=101)


# # Random Forest

# In[8]:


# 設定隨機種子
np.random.seed(0)


# In[9]:


from sklearn.ensemble import RandomForestClassifier


# In[10]:


RF=RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)


# In[11]:


RF.fit(X_train, y_train)


# In[12]:


y_pred_rf=np.around(RF.predict(X_test))


# In[13]:


y_pred_rf


# In[14]:


#列出隨機森林相關分析數據
from sklearn.metrics import classification_report, accuracy_score
print("Random Forest Classifier")
print(classification_report(y_test, y_pred_rf))


# In[15]:


#隨機森林準確度
print(accuracy_score(y_test, y_pred_rf))


# In[16]:


#特徵重要程度
feature_importance=pd.Series(RF.feature_importances_, index=X.columns)
feature_importance.sort_values().plot(kind='barh', color='g')


# # Decision Tree

# In[17]:


from sklearn.tree import DecisionTreeClassifier


# In[18]:


DT=DecisionTreeClassifier()


# In[19]:


DT.fit(X_train, y_train)


# In[20]:


y_pred_DT=DT.predict(X_test)


# In[21]:


#列出隨機森林相關分析數據
from sklearn.metrics import classification_report, accuracy_score
print("Decision Tree Classifier")
print(classification_report(y_test, y_pred_DT))


# In[22]:


#決策樹準確度
print(accuracy_score(y_test, y_pred_DT))


# # Comparison of Methods used

# In[23]:


comparison=pd.DataFrame(np.array(['Random Forest Classifier', 'Decision Tree Classifier']))


# In[24]:


comparison.columns=['Method']


# In[25]:


comparison['Precision']=[0.77, 0.62]
comparison['Recall']=[0.77, 0.61]
comparison['F1_Score']=[0.77, 0.62]
comparison['Accuracy']=[0.71, 0.61]


# In[26]:


comparison


# In[27]:


comparison.plot(kind='bar')
plt.ylim(0,1)
plt.xlim(-3, 5)
plt.grid(True)
plt.xticks([0, 1], ['Random Forest Classifier', 'Decision Tree Classifier'])

