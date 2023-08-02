#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[17]:


raw_mail_data= pd.read_csv('mail_data.csv')


# In[18]:


print(raw_mail_data)


# In[19]:


raw_mail_data.head()


# In[20]:


mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[21]:


mail_data.head()


# In[22]:


mail_data.shape


# In[23]:


mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# In[24]:


X = mail_data['Message']
Y = mail_data['Category']


# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[26]:


print(X.shape)
print(X_train)
print(X_test)


# In[28]:


feature_extraction = TfidfVectorizer(min_df = 1, stop_words ='english', lowercase='True')


# In[29]:


X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#convert Y_train and Y_test values as integers
Y_train = Y_train.astype('int')
Y_test =  Y_test.astype('int')


# In[30]:


print(X_train)


# In[31]:


print(X_train_features)


# In[32]:


model = LogisticRegression()


# In[33]:


#training the logistic regression model with the training data
model.fit(X_train_features, Y_train)


# In[34]:


prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[35]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[36]:


prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[37]:


print('Accuracy on test data : ', accuracy_on_test_data)


# In[39]:


input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times."]

input_data_features = feature_extraction.transform(input_mail)

prediction = model.predict(input_data_features)
print(prediction)

if prediction[0]==1:
    print('Ham mail')
    
else:
    print('Spam mail')


# In[ ]:




