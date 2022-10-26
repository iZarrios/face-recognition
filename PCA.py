#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np

class PCA():
    def __init__(self):
        # the percentage of the information that we need
        pass
        
        
    
    # get mean of every dimensions
    # @params None
    # @returns nparray of d dimensions
    def get_mean(self):
        mean = np.mean(self.data, axis = 0, keepdims = True)
        self.mean = mean
        
#         return mean
    
    
    # Center data around the mean
    # @params None
    # @returns centralized data around the mean
    def center_data(self):
        self.c_data = self.data - self.mean
        
#         return self.data
    
    
    # get covairance matrix of the data
    # @param None
    # @reurns covariance matrix
    def cov(self):
        self.cov = np.cov(self.data.T,bias = True)
        
#         return self.cov
    
    
    
    # calculate eigen values and eigen vector
    # centralized
    # @params None
    # @returns eigen value, eigen vectors
    def calc_eig(self):
        w, v = np.linalg.eigh(self.cov)
        w = w[::-1]
        v = v[:,::-1]
        
        self.eival = w
        self.eivec = v
        self.eival_sum = np.sum(w)
        

    # get projected matrix which contains the new dimensions
    # @parmas None
    # @return projectiom matrix
    def calc_proj_matrix(self):
        
        eival_sum = self.eival_sum
        alpha = self.alpha
        eival = self.eival
        eivec = self.eivec
        
        exp_var_nom = 0
        i = 0
        while(round(exp_var_nom/float(eival_sum), 2) < alpha 
              and i < len(eival)-1):
            exp_var_nom += eival[i]
            i += 1
        print(i-1)
        self.proj_matrix = eivec[:, :i]
        print(i)
#         return self.proj_matrix
              
    def get_proj_matrix(self):
        return self.proj_matrix
    
    def _fit(self, data):
        self.data = np.array(data)
        # get mean matrix
        self.get_mean()
        # center data around mean
        self.center_data()
        # get covariance matrix
        self.cov()
        # get eigvals ,eigvecs
        self.calc_eig()

    
    def fit(self, data):
        self.calc_proj_matrix()
        
    def transform(self, X):
        return np.dot(X,self.proj_matrix)
    
    def set_alpha(self, alpha):
        self.alpha = alpha    
        

