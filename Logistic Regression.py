#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[3]:


data = pd.read_csv('~/Downloads/admit.csv')


# In[4]:


data.head()


# In[5]:


df = data.copy()
df['Admitted'] = df['Admitted'].map({'Yes':1,'No':0})
df.head()


# In[6]:


y = df['Admitted']
x1 = df['SAT']


# In[7]:


plt.scatter(x1,y, color = 'C0')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('Admitted', fontsize = 20)
plt.show()


# In[8]:


x = sm.add_constant(x1)
reg_lin = sm.OLS(y,x)
results_lin = reg_lin.fit()

plt.scatter(x1,y,color = 'C0')
y_hat = x1*results_lin.params[1]+results_lin.params[0]

plt.plot(x1,y_hat,lw=2.5,color='C8')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('Admitted', fontsize = 20)
plt.show()


# In[9]:



reg_log = sm.Logit(y,x)
results_log = reg_log.fit()

def f(x,b0,b1):
   return np.array(np.exp(b0+x*b1) / (1 + np.exp(b0+x*b1)))

f_sorted = np.sort(f(x1,results_log.params[0],results_log.params[1]))
x_sorted = np.sort(np.array(x1))

plt.scatter(x1,y,color='C0')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('Admitted', fontsize = 20)
plt.plot(x_sorted,f_sorted,color='C8')
plt.show()


# In[11]:


x = sm.add_constant(x1)
reg_log= sm.Logit(y,x)
result_log = reg_log.fit()


# In[13]:


result_log.summary()


# In[ ]:




