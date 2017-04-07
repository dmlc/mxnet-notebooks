# pylint: skip-file
import numpy as np
from BCD_one import BCD_one
num_v = 100
num_u = 100
num_iter = 10
K = 4
lambda_u = 100
lambda_v = 0.1
a = 1
b = 0.01
a_m_b = a-b
theta = np.mat(np.random.rand(K,num_v)).T
V = np.mat(np.random.rand(K,num_v)).T
U = np.mat(np.random.rand(K,num_u)).T
R = np.mat(np.random.rand(num_u,num_v))
R[R<0.9] = 0
for i in range(num_iter):
    U, V = BCD_one(R, U, V, theta, lambda_u, lambda_v)
