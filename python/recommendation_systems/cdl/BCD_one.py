# pylint: skip-file
import numpy as np
def BCD_one(R, U, V, theta, lambda_u, lambda_v, dir_save='.',
        get_loss=False, num_iter=1):
    U = U.T
    # count the number of non zero entries in R.
    print(np.count_nonzero(R))

    V = np.mat(V.T)
    theta = np.mat(theta.T)
    num_v = R.shape[1]
    num_u = R.shape[0]
    K = U.shape[0]
    a = 1
    b = 0.01
    a_m_b = a-b
    I_u = np.mat(np.eye(K)*lambda_u)
    I_v = np.mat(np.eye(K)*lambda_v)
    C  = np.mat(np.ones(R.shape))*b
    C[np.where(R>0)] = a
    #print 'I: %d, J: %d, K: %d' % (num_u,num_v,K)
    for it in range(num_iter):
        U_sq = U*U.T*b
        for j in range(num_v):
            idx =  np.where(R[:, j] > 0)[0]
            if type(idx) is np.matrix:
                idx_a = idx.A1
            else:
                idx_a = idx
            U_cut = U[:,idx_a]
            V[:,j] = np.linalg.pinv(U_sq+U_cut*U_cut.T*a_m_b+I_v)*(U_cut*R[idx_a,j]+lambda_v*theta[:,j])
        V_sq = V*V.T*b
        for i in range(num_u):
            idx =  np.where(R[i,:]>0)[1]
            if type(idx) is np.matrix:
                idx_a = idx.A1
            else:
                idx_a = idx
            #idx_a = np.where(R[i,:]>0)[1].A1
            V_cut = V[:,idx_a]
            U[:,i] = np.linalg.pinv(V_sq+V_cut*V_cut.T*a_m_b+I_u)*(V_cut*R[i,idx_a].T)
        if it%10==9:
            E = U.T*V-R
            E = np.sum(np.multiply(C,np.square(E)))/2.0
            reg_loss_v = np.sum(np.square(theta-V))/2.0
            reg_loss_u = np.sum(np.square(U))/2.0
            E = E+lambda_v*reg_loss_v+lambda_u*reg_loss_u
            print 'Iter %d - E: %.3f' % (it,E)
            fp = open(dir_save+'/cdl.log','a')
            fp.write('Iter %d - E: %.3f\n' % (it,E))
            fp.close()

    if get_loss:
        E = U.T*V-R
        E = np.sum(np.multiply(C,np.square(E)))/2.0
        reg_loss_v = np.sum(np.square(theta-V))/2.0
        reg_loss_u = np.sum(np.square(U))/2.0
        E = E+lambda_v*reg_loss_v+lambda_u*reg_loss_u
        #print 'E: %.3f' % E
    else:
        E = 0
    U = U.T
    V = np.asarray(V.T)
    return U, V, E

