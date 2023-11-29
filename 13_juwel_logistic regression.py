#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('diabetes.csv')
df


# In[3]:


df.duplicated().sum()


# In[4]:


df.isna().sum()


# In[5]:


X = df.drop('Outcome', axis=1)
y = df['Outcome']


# In[6]:


X,y


# In[7]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[10]:


# Initialize and fit logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred


# In[12]:


import pickle as pkl


# In[15]:


with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[ ]:




