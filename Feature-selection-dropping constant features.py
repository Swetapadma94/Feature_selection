#!/usr/bin/env python
# coding: utf-8

# In[2]:


df=pd.DataFrame({'A':[1,2,4,1,2,4],
                 'B':[4,5,6,7,8,9],
                 'C':[0,0,0,0,0,0],
                 'D':[1,1,1,1,1,1]})


# In[3]:


df


# #Here C and D are having constant features
# Variance Threshold:
#     It will remove all features that have low variance.
#     It can be used for unsupervised ML.
#     for C and D features the mean,variance and standard deviation will be zero..

# In[4]:


from sklearn.feature_selection import VarianceThreshold


# In[5]:


selector=VarianceThreshold(threshold=0)
selector.fit(df)


# In[6]:


selector.get_support()


# In[8]:


constant_columns=[column for column in df.columns
                if column not in df.columns[selector.get_support()]]


# In[9]:


len(constant_columns)


# In[10]:


for i in constant_columns:
    print(i)


# In[11]:


df.drop(constant_columns,axis=1)


# In[12]:


df=pd.read_csv(r"E:\Krish naik\standard_cust_satisfaction\train.csv",encoding='latin1')


# In[13]:


df.head()


# In[14]:


df.shape


# In[64]:


X=df.drop(labels=['TARGET'], axis=1)
y=df['TARGET']


# In[65]:


X.head()


# In[66]:


from sklearn.model_selection import train_test_split


# In[67]:



from sklearn.model_selection import train_test_split
# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(labels=['TARGET'], axis=1),
    df['TARGET'],
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape


# In[68]:



var_thres=VarianceThreshold(threshold=0)
var_thres.fit(X_train)


# In[56]:


sum(var_thresh.get_support())


# In[69]:


var_thres.get_support()


# In[58]:


constant_columns=[column for column in df.columns
                if column not in X_train.columns[var_thresh.get_support()]]


# In[59]:


len(constant_columns)


# In[70]:


# Lets Find non-constant features 
len(X_train.columns[var_thres.get_support()])


# In[60]:


len(X_train.columns[var_thresh.get_support()])


# In[71]:


for column in constant_columns:
    print(column)


# In[ ]:


X_train.drop(constant_columns,axis=1)


# In[37]:


X_train.head()


# In[ ]:




