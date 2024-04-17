import os
import torch
from torch import optim
from DataCofigAll import SumDataset
from tqdm import tqdm
from ModelAll import *
import numpy as np
import pickle
from ScheduledOptim import ScheduledOptim
import random
import sys
import openpyxl
from datetime import datetime
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

NlLen_map = {"Time":4485, "Cli":583, "JxPath":880 ,"Math":1809, "Lang":779,"Lang1":779, "tcas":160, "Chart": 2363, "Mockito":1780, "unknown":2200}
CodeLen_map = {"Time":3791, "Cli":1257 , "JxPath":7259, "Math":13287, "Lang":937,"Lang1":937, "tcas":220, "Chart":12910, "Mockito":1176, "unknown":2800}

args = dotdict({
    'NlLen':NlLen_map[sys.argv[2]],
    'CodeLen':CodeLen_map[sys.argv[2]],#代码的数量？
    'SentenceLen':10,
    'batch_size':60,
    'hidden_size':60,
    'embedding_size':60,#向量的维度
    'WoLen':15,#
    'Vocsize':100,#后面根据具体情况值变化了
    'Nl_Vocsize':100,#后面根据具体情况值变化了
    'max_step':3,#
    'margin':0.5,#
    'poolsize':50,#池化层
    'Code_Vocsize':100,#后面根据具体情况值变化了
    'seed':0,#随机种子
    'lr':1e-3,#学习率
    'layer_size':1,
    'nums_layers':3,
    'attn_heads':1,
    'dropout':0.1,
    'alpha':0.2,
    'tau':0.5,
    'base_model':'GraphConvolution',
    'new_model':'GCNN',
    'activation':'prelu',
    'NETname':'GRACE',
    'drop_edge_rate_1':0.2,
    'drop_edge_rate_2':0.0,
    'drop_feature_rate_1':0.3,
    'drop_feature_rate_2':0.1,
    'weight_decay':1e-05,
    'falg':'code',#'sbfl',
    'i':2,#default-1
    'type':'DRandom',
    'eps':'15',
    'lable':""
})
os.environ['PYTHONHASHSEED'] = str(args.seed)

def save_model(model, dirs = "checkpointcodeSearch"):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    torch.save(model.state_dict(), dirs + '/best_model.ckpt')


def load_model(model, dirs="checkpointcodeSearch"):
    assert os.path.exists(dirs + '/best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + '/best_model.ckpt'))

use_cuda = torch.cuda.is_available()

def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = gVar(data[i])
        tensor = data
    else:
        assert isinstance(tensor, torch.Tensor)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor

#LOOC
def train1(testlst=[], p='Math',eps=15):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("SumDataset")
    dev_set = SumDataset(args, "test", p, testlst=testlst)
    print("SumDataset")
    val_set = SumDataset(args, "val", p, vallst=testlst)
    print("val_set.ids", val_set.ids)
    train_set = SumDataset(args, "train", proj=p, testlst=testlst, vallst=testlst)
    print("train_set.ids", train_set.ids)

    data = pickle.load(open(p + '.pkl', 'rb'))

    numt = len(train_set.data[0])
    print("numt:", numt)
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)
    model = NlEncoderAll(args)
    if use_cuda:
        print('using GPU')
        model = model.cuda()
    maxl = 1e9
    optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=args.lr), args.embedding_size, 4000)
    print("_________________________________________")
    print(next(model.parameters()).device)
    maxAcc = 0
    minloss = 1e9
    rdic = {}
    brest = []
    bans = []
    batchn = []
    best_epoch = 0
    each_epoch_pred = {}
    for x in dev_set.Nl_Voc:
        rdic[dev_set.Nl_Voc[x]] = x
    testtime=datetime.now()-datetime.now()
    traintime=datetime.now()-datetime.now()
    for epoch in range(eps):  # 训练15次
        for dBatch in tqdm( train_set.Get_Train( args.batch_size ) ):
            TrainStartTime = datetime.now()
            model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss, _, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4], dBatch[5], dBatch[6], dBatch[7], dBatch[8], dBatch[9], dBatch[10], dBatch[11])
            print(loss.item())
            optimizer.zero_grad()
            # loss = loss.mean()
            loss.backward()
            optimizer.step_and_update_lr()
            TrainEndTime = datetime.now()
            traintime+=TrainEndTime-TrainStartTime
        TestStartTime = datetime.now()
        tmp = {}
        model = model.eval() 

        score2 = []
        for k, devBatch in tqdm(enumerate(val_set.Get_Train(len(val_set)))):
            for i in range(len(devBatch)):
                devBatch[i] = gVar(devBatch[i])
            with torch.no_grad():
                l, pre, _ = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[5], devBatch[6], devBatch[7], devBatch[8], devBatch[9], devBatch[10], devBatch[11])
                resmask = torch.eq(devBatch[1], 2)
                s = -pre  # -pre[:, :, 1]
                s = s.masked_fill(resmask == 0, 1e9)
                pred = s.argsort(dim=-1)
                pred = pred.data.cpu().numpy()

                for k in range(len(pred)):
                    datat = data[val_set.ids[k]]
                    maxn = 1e9
                    lst = pred[k].tolist()[:resmask.sum(dim=-1)[k].item()]
                    tmp[val_set.ids[k]] = lst
                    for x in datat['lans']:
                        i = lst.index(x)
                        maxn = min(maxn, i)
                    score2.append(maxn)
        TestEndTime = datetime.now()
        testtime+=TestEndTime-TestStartTime
        each_epoch_pred[epoch] = tmp
        socre = sum(score2)
        print('curr accuracy is ' + str(socre))
        flag = True
        for s in score2:
            if s != 0:
                flag = False
        if flag:
            batchn.append(epoch)
        if maxl >= socre:
            brest = score2
            print("brest", brest, score2)
            best_epoch = epoch
            bans = tmp
            maxl = socre
            print("find better score " + str(socre))
        print("------------------------------------best_epoch:", best_epoch)
    return brest, bans, batchn, each_epoch_pred

if __name__ == "__main__":
    args.lr = float(sys.argv[3])
    args.seed = int(sys.argv[4])
    args.batch_size = int(sys.argv[5])
    np.set_printoptions(threshold=sys.maxsize)
    res = {}
    p = sys.argv[2]
    testlst=[]
    k=int(sys.argv[1])
    k_size=int(sys.argv[6])
    proj_datanum = int(sys.argv[7])
    #eps = int(sys.argv[8])
    args.heads = int( sys.argv[8] )
    args.layer_size=int(sys.argv[9])
    args.eps = int( sys.argv[10] )
    args.NETname=sys.argv[11]
    args.hidden_size=int( sys.argv[12] )
    args.embedding_size=int( sys.argv[12] )
    for i in range(k * k_size,min((k+1)*k_size,proj_datanum)):
        testlst.append(i)
    res[k] = train1(testlst, p, args.eps)
    open( '%s_%s_%d_%d_%s_%s.pkl'%(p,str(args.layer_size )+'-'+args.NETname+str(args.embedding_size),k,args.seed,args.lr,args.batch_size),'wb').write( pickle.dumps( res ))

