import pickle
import os
import sys

import openpyxl as op

K_size = 1
proj = sys.argv[1]
seed = int(sys.argv[2])
lr = float(sys.argv[3])
batch_size = int(sys.argv[4])
NNName = sys.argv[5]
EPOCHS = int(sys.argv[6]) #+ 1
heads=int(sys.argv[7])
Note=""
model = NNName+'-h'+ str(heads) + '-lr' + str(lr)+Note
missk=0


versionNum = {'Lang': 65, 'Time': 27, 'Chart': 26, 'Math': 106, 'Closure': 133, 'Mockito': 38, 'JxPath': 24, 'tcas': 30, 'Cli': 39}

t = {}


for i in range(0, versionNum[proj]):
    if not os.path.exists(proj + '_%s_%d_%d_%s_%s.pkl' % (NNName, i, seed, lr, batch_size)):
        #print(proj + '_%s_%d_%d_%s_%s.pkl' % (NNName, i, seed, lr, batch_size))
        continue
    p = pickle.load(open(proj + '_%s_%d_%d_%s_%s.pkl' % (NNName, i, seed, lr, batch_size), 'rb'))
    p = p[i]
    Max_expoch = [len(p[0]), len(p[1]), len(p[2]), len(p[3])]
    Max_expoch = [1, 1, len(p[2]), EPOCHS]

    for j in range(Max_expoch[3]):
        if j not in t:
            t[j] = {}
        for key in p[3][j]:
            t[j][key] = p[3][j][key]

open(proj + '_%s_%d_%s_%s.pkl' % (NNName, seed, lr, batch_size), 'wb').write(pickle.dumps(t))
data = pickle.load(open(proj + '.pkl', 'rb'))

ANS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 每一个epoch对应的top12345
ranks =t[5]

for key in ranks:
    minl = 1e9
    ranklist = ranks[key]
    for x in data[key]['lans']:#语句级别的故障定
        rank_x = ranklist.index(x)
        if rank_x < 10:
            ANS[rank_x] += 1
top1 = ANS[0]
top3 = top1 + ANS[1] + ANS[2]
top5 = top3 + ANS[3] + ANS[4]
top10 = top5 + ANS[5] + ANS[6]+ANS[7] + ANS[8]+ ANS[9]
print(top1,top3,top5,top10)