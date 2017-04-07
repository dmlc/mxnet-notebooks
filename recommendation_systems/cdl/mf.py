# pylint: skip-file
import numpy as np
num_v = 16980
num_u = 5551
num_iter = 10
K = 10
lambda_u = 100
lambda_v = 0.1
a = 1
b = 0.01
a_m_b = a-b
theta = np.mat(np.random.rand(K,num_v))
V = np.mat(np.random.rand(K,num_v))
U = np.mat(np.random.rand(K,num_u))
R = np.mat(np.random.rand(num_u,num_v))
R[R<0.9992] = 0
I_u = np.mat(np.eye(K)*lambda_u)
I_v = np.mat(np.eye(K)*lambda_v)
C  = np.mat(np.ones(R.shape))*b
C[np.where(R>0)] = a
print 'I: %d, J: %d, K: %d' % (num_u,num_v,K)
for it in range(num_iter):
    print 'iter %d' % it
    V_sq = V*V.T*b
    for i in range(num_u):
        idx_a = np.where(R[i,:]>0)[1].A1
        V_cut = V[:,idx_a]
        U[:,i] = np.linalg.pinv(V_sq+V_cut*V_cut.T*a_m_b+I_u)*(V_cut*R[i,idx_a].T)
    U_sq = U*U.T*b
    for j in range(num_v):
        idx_a = np.where(R[:,j]>0)[0].A1
        U_cut = U[:,idx_a]
        V[:,j] = np.linalg.pinv(U_sq+U_cut*U_cut.T*a_m_b+I_v)*(U_cut*R[idx_a,j]+lambda_v*theta[:,j])
    if it%1==0:
        E = U.T*V-R
        E = np.sum(np.multiply(C,np.multiply(E,E)))
        print 'E: %.3f' % E

