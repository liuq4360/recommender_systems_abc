#!/usr/bin/env python
# coding=utf-8

# 将training_set转换为三元组(userid，videoid，score)

import os
import numpy as np
from scipy.sparse import dok_matrix
import scipy as sp
from scipy.sparse import csr_matrix

indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
a = csr_matrix((data, indices, indptr), shape=(3, 4))
b = a.transpose()
d = a.getrow(0).dot(b)

a = np.array([1, 2, 3, 4, 5])
b = np.array([2, 4, 6, 8, 9])


c = set(a).intersection(set(b))









