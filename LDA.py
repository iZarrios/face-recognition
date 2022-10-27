#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


class LDA():
    
    def __init__(self,n_components):
        self.n_components = n_components
        
        
        
    def fit(self,X,y):
        n_features = X.shape[1]
        
        
        class_labels = np.unique(y)
         
        # Sw with in class
        # Sb between class
        
        overall_mean = np.mean(X, axis=0)
        
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))
        
        for c in class_labels:
            # get all samples with c class
            X_c = X[y == c] # c, N
            nk = X_c.shape[0] # 5
            
            mean_c = np.mean(X_c,axis=0, keepdims=True)
            
            overall_mean = overall_mean.reshape(n_features, 1)
            
            Sb += nk * np.dot((mean_c - overall_mean), (mean_c - overall_mean).T)
            
            # N,c . c,N
            Sw += np.dot((X_c - mean_c).T, (X_c - mean_c)) # N, N
        Sw_inv = np.linalg.inv(Sw)
        
        eigVal, eigVec = np.linalg.eigh(np.dot(Sw_inv,Sb))
        eigVal = eigVal[::-1]
        eigVec = eigVec[::-1]
        
        eigVec = eigVec.T        
        self.lin_discriminants = eigVec[:self.n_components]
    
    def get_lin_discriminants(self):
        return self.lin_discriminants
        
    def transform(self,X):
        res = np.dot(X, self.lin_discriminants.T)
        
        return res

