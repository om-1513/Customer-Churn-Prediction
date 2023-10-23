#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas.plotting._matplotlib


# In[3]:


data=pd.read_excel("customer_churn_large_dataset.xlsx")


# In[4]:


data.drop(['CustomerID','Name'], axis='columns',inplace=True)

data['Location'].replace({'Los Angeles':1,'New York':0,'Miami':2,'Houston':3,'Chicago':4},inplace=True)


# In[5]:


data['Gender'].replace({'Female':1,'Male':0},inplace=True)


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.head()


# In[9]:


data.tail()


# In[10]:


data.size


# In[11]:


data.shape


# In[12]:


data.dtypes


# In[13]:


data.isnull().sum()


# In[14]:


data.duplicated().sum()


# Basic Data Cleaning

# In[15]:


data["Monthly_Bill"].dtypes


# In[16]:


categorical_feature=["Gender","Location	"]
numerical_feature=["Age","Subscription_Length_Months","Monthly_Bill","Total_Usage_GB"]
target="Churn"


# In[17]:


data.skew(numeric_only=True)


# In[18]:


data.corr(numeric_only=True)


# In[19]:


data[numerical_feature].describe()


# In[20]:


data[numerical_feature].hist(bins=30, figsize=(10, 10))


# In[22]:





# In[21]:


data1.Monthly_Bill=pd.to_numeric(data1.Monthly_Bill)


# In[23]:


data.Monthly_Bill.values


# In[25]:


def print_unique_col_values(data):
       for column in data:
            if data[column].dtypes=='object':
                print(f'{column}: {data[column].unique()}') 


# In[ ]:





# In[27]:


X = data.drop('Churn',axis='columns')
y = data['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)


# In[28]:


X_train.shape


# In[29]:


X_test.shape


# In[30]:


X_train[:10]


# In[31]:


len(X_train.columns)


# In[32]:


data.info()


# In[33]:


Age = pd.get_dummies(data['Age'],drop_first=True)
Gender = pd.get_dummies(data['Gender'],drop_first=True)
Location = pd.get_dummies(data['Location'],drop_first=True)


# In[34]:


data['Gender'].replace({'Female':1,'Male':0},inplace=True)
data['Location'].replace({'Los Angeles':1,'New York':0,'Miami':2,'Houston':3,'Chicago':4},inplace=True)


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


data['Location']


# In[37]:


data = pd.concat([data,Age,Gender,Location],axis=1)


# In[38]:


data.head()


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# In[40]:


from sklearn.tree import DecisionTreeClassifier


# In[41]:


dtree = DecisionTreeClassifier()


# In[42]:


dtree.fit(X_train,y_train)


# In[43]:


predictions = dtree.predict(X_test)


# In[44]:


from sklearn.metrics import classification_report,confusion_matrix


# In[45]:


print(classification_report(y_test,predictions))


# In[46]:


print(confusion_matrix(y_test,predictions))


# In[47]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[48]:


rfc_pred = rfc.predict(X_test)


# In[49]:


print(confusion_matrix(y_test,rfc_pred))


# In[50]:


print(classification_report(y_test,rfc_pred))


# In[ ]:





# In[ ]:




