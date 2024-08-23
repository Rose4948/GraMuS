import subprocess
from tqdm import tqdm
import time
import os, sys
import pickle

project = sys.argv[1]

seed = int(sys.argv[2])
#lr = 1e-2
#seed = 0
#batch_size = 60
#modelName='SpGGAT'
lr = float(sys.argv[3])
batch_size = int(sys.argv[4])
modelName = sys.argv[5]
eps = int(sys.argv[6])
layers = int(sys.argv[7])
heads=1
card = [1]
lst = list(range(len(pickle.load(open(project + '.pkl', 'rb')))))
singlenums = {"Lang":1,"Chart":1,"Math":1,"Cli":1,"JxPath":1,"Time":1}
K_size = {"Lang":1,"Chart":1,"Math":1,"Cli":1,"JxPath":1,"Time":1}
singlenum = singlenums[project]
totalnum = len(card) * singlenum

K=len(lst)/K_size[project]
print(len(lst),K)
#modelName='GGNN'
embedding=61
for ri in [-1]:#range(0,11):#[38,39,53,58,59,61,64]:
    for i in tqdm(range(int(K/totalnum) + 1)):
        jobs = []
        for j in range( totalnum ):
            k = i * totalnum + j
            if k >= K :
                continue
            cardn = int( j / singlenum )
            print("GPU",card[cardn])
            p = subprocess.Popen("CUDA_VISIBLE_DEVICES=" + str( card[cardn] )+ " python runAll.py %d %s %f %d %d %d %d %d %d %d %s %d %d"%(k, project, lr, seed, batch_size,K_size[project],len(lst),heads,layers,eps,modelName,embedding,ri), shell=True)
            jobs.append( p )
            time.sleep( 10 )
        for p in jobs:
            p.wait()
       
    p = subprocess.Popen("python sum.py %s %d %f %d %s %d %d"%(project, seed, lr, batch_size, str(layers)+'-'+modelName+str(embedding), eps, heads), shell=True,)
    p.wait()
