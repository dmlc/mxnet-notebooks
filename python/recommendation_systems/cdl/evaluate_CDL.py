from __future__ import  division
from data import read_user
import matplotlib.pyplot as plt
import numpy as np


def cal_rec(p, cut):
    R_true = read_user('cf-test-1-users.dat')
    dir_save = 'cdl' + str(p)
    U = np.mat(np.loadtxt(dir_save + '/final-U.dat'))
    V = np.mat(np.loadtxt(dir_save + '/final-V.dat'))
    R = U * V.T


    num_u = R.shape[0]
    num_hit = 0
    fp = open(dir_save + '/rec-list.dat', 'w')
    for i in range(num_u):
        if i != 0 and i % 100 == 0:
            print 'User ' + str(i)
        l_score = R[i, :].A1.tolist()
        pl = sorted(enumerate(l_score), key=lambda d: d[1], reverse=True)
        l_rec = list(zip(*pl)[0])[:cut]
        s_rec = set(l_rec)
        s_true = set(np.where(R_true[i, :] > 0)[1])
        cnt_hit = len(s_rec.intersection(s_true))
        fp.write('%d:' % cnt_hit)
        fp.write(' '.join(map(str, l_rec)))
        fp.write('\n')
    fp.close()


if __name__ == '__main__':

    # give the same p as given in cdl.py
    p = 4
    cal_rec(p, 300)
    dir_save = 'cdl%d' % p

    R_test = read_user('cf-test-1-users.dat')
    fp = open(dir_save + '/rec-list.dat')
    lines = fp.readlines()

    total = 0
    correct = 0
    users = 0
    total_items_liked = 0
    num_users = len(range(R_test.shape[0]))

    # recall@M is calculated for M = 50 to 300
    M_low = 50
    M_high = 300
    recall_levels = M_high-M_low + 1
    recallArray = np.zeros(shape=(num_users,recall_levels))

    for user_id in range(num_users):

        s_test = set(np.where(R_test[user_id, :] > 0)[1])
        total_items_liked = len(s_test)
        l_pred = map(int, lines[user_id].strip().split(':')[1].split(' '))
        num_items_liked_in_top_M = 0
        M = 0;

        # array to store the likes at each M
        likesArray = np.zeros(recall_levels)

        for item in l_pred:
            M += 1
            total=total+1

            if item in s_test:
                correct=correct+1
                num_items_liked_in_top_M += 1

            if M >= M_low:

                #M-M_low as array indices start from 0
                likesArray[M-M_low] = num_items_liked_in_top_M

        if total_items_liked > 0:
            recallArray[user_id] = likesArray/total_items_liked
            users +=1
        else:
            recallArray[user_id] = np.nan

    fp.close()

    print " total predicted %d" % (total)
    print " correct %d" % (correct)
    print " users %d" %(users)
    print " Recall at M"

    plt.plot(range(M_low,M_high+1),np.nanmean(recallArray,axis=0))
    plt.ylabel("Recall")
    plt.xlabel("M")
    plt.title("CDL: Recall@M")
    plt.show()

