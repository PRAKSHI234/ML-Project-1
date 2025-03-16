#!/usr/bin/env python
# coding: utf-8

# # Import Dependencies

# In[2]:


import numpy as np


# In[3]:


import pandas as pd


# In[4]:


from sklearn.model_selection import train_test_split 


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[6]:


from sklearn.metrics import accuracy_score


# In[7]:


from sklearn.linear_model import LogisticRegression


# # DATA COLLECTION AND DATA PRE PROCESSING

# In[8]:


#loading the data from csv file to a pandas DataFrame
raw_mail_data = pd.read_csv('mail_data.csv')
print(raw_mail_data)


# In[9]:


#replace the null values with null string
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data))


# In[10]:


#printing the first five rows of the dataframe
mail_data.head()


# In[11]:


mail_data.tail()


# In[12]:


# checking the number of rows and columns in DataFrames
mail_data.shape


# # lABEL ENCODING

# In[13]:


#label spam mail as 0; ham mail as 1.
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1


# In[14]:


#separating the data as text and label
X = mail_data['Message']
Y = mail_data['Category']


# In[15]:


print(X)


# In[16]:


print(Y)


# In[20]:


#splitting the data into training data and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=3)


# In[21]:


print(X.shape)


# In[22]:


print(X_test.shape)


# In[23]:


print(X_train.shape)


# # Feature Extraction 

# In[29]:


from sklearn.feature_extraction.text import TfidfVectorizer
#transform the text data to feature vectors that can be used as input to the Logistic Regression
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#convert Y_train and Y_test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
print(X_train_features)


# # Training the Model

# # Logistic Regression

# In[30]:


model = LogisticRegression()


# In[31]:


#training the Logistic Regression model with the Training data
model.fit(X_train_features,Y_train)


# # Evaluated the Trained Model

# In[32]:


#prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)
print('Accuracy on training data : ',accuracy_on_training_data)


# In[33]:


#prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)
print('Accuracy on test data : ',accuracy_on_test_data)


# # Building a Predictive System

# In[34]:


input_mail = ["I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, I've cried enough today."]

#convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

#making the prediction
prediction = model.predict(input_data_features)
print(prediction)


# In[35]:


if prediction[0]==1:
    print('Ham Mail')
else:
    print('Spam Mail')


# In[37]:


input_mail = ["SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575.Cost 150p/day,6days,16+ TsandCs apply Reply HL4 info"]

#convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

#making the prediction
prediction = model.predict(input_data_features)
print(prediction)


# In[38]:


if prediction[0]==1:
    print('Ham Mail')
else:
    print('Spam Mail')


# In[ ]:




