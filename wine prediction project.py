#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Prediction

# # By Dalia Atef

# In[1]:


#import liberaries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[2]:


#reading data
dataset=pd.read_csv(r"C:\Users\Mass\Downloads\archive\WineQT.csv")
dataset.head()


# In[3]:


#explore data:
dataset.shape


# In[4]:


dataset.isnull().sum()


# In[5]:


dataset.describe()


# In[6]:


#visulaizate data:
sns.catplot(x='quality',data= dataset, kind='count')


# In[7]:


#quality according to volatile acidity:
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='volatile acidity', data=dataset)


# In[8]:


#quality according to citric acid:
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='citric acid', data=dataset)


# In[9]:


#quality according to residual sugar:
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='residual sugar', data=dataset)


# In[10]:


#quality according to chlorides:
plt.bar(dataset['quality'],dataset['chlorides'])
plt.xlabel('quality')
plt.ylabel('chlorides')
plt.show()


# In[11]:


#quality according to free sulfur dioxide:
plt.bar(dataset['quality'],dataset['free sulfur dioxide'])
plt.xlabel('quality')
plt.ylabel('chlorides')
plt.show()


# In[12]:


#quality according to density:
plt.bar(dataset['quality'],dataset['density'])
plt.xlabel('quality')
plt.ylabel('chlorides')
plt.show()


# In[13]:


#quality according to pH:
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='pH', data=dataset)


# In[14]:


#quality according to sulphates:
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='sulphates', data=dataset)


# In[15]:


#quality according to alcohol:
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='alcohol', data=dataset)


# #correlation of data:

# In[16]:


correlation=dataset.corr()


# In[17]:


#make heatmap to correlated columns:
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# In[18]:


#data preprossing:
#1.separate data and label
X=dataset.drop('quality',axis=1)


# In[19]:


print(X)


# In[20]:


#binarization label:
dataset['quality'].unique()


# In[21]:


Y= dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)


# In[22]:


print(Y)


# In[23]:


dataset['quality'].unique()


# In[24]:


dataset['quality'].value_counts()


# In[25]:


#Train and Test data
X_train,X_test,Y_train, Y_test =train_test_split(X,Y, test_size=0.2,random_state=3)


# In[26]:


print(Y.shape, Y_train.shape,Y_test.shape)


# In[36]:


#model training:
model= RandomForestClassifier()


# In[37]:


model.fit(X_train,Y_train)


# In[38]:


#model evaluation
#accuracy on test data:
X_test_prediction= model.predict(X_test)
test_data_accuracy= accuracy_score(X_test_prediction,Y_test)


# In[39]:


print('Accuracy:',test_data_accuracy)


# In[42]:


#build predictive system:
input_data=(7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0,7)

#changing input data to numpy array:
input_data_as_numpy_array= np.asarray(input_data)

#reshapr data as we predicted:
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

prediction= model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==1):
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')
    


# In[ ]:




