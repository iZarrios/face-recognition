#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:
class LDA():

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        class_labels = np.unique(y)  # should be of size 2

        d1 = X[y == class_labels[0]]
        d2 = X[y == class_labels[1]]

        mean1 = np.mean(d1, axis=0)
        mean2 = np.mean(d2, axis=0)

        diff = mean1 - mean2

        Sb = np.dot(diff, diff.T)

        z1 = d1 - mean1.T
        z2 = d2 - mean2.T

        s1 = np.dot(z1.T, z1)
        s2 = np.dot(z2, z2.T)

        Sw = s1 + s2
        S_inv = np.linalg.inv(Sw)

        eigVal, eigVec = np.linalg.eigh(np.dot(S_inv, B))
        eigVal = eigVal[::-1]
        eigVec = eigVec[::-1]

        eigVec = eigVec.T
        self.lin_discriminants = eigVec[:self.n_components]

    def get_lin_discriminants(self):
        return self.lin_discriminants

    def transform(self, X):
        res = np.dot(X, self.lin_discriminants.T)

        return res
