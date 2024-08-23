# -*- coding: UTF-8 -*-
import sys
import torch
import torch.utils.data as data
import random
import pickle
import os
from nltk import word_tokenize
from vocab import VocabEntry
import numpy as np
import re
from tqdm import tqdm
from scipy import sparse
import math
import json

dmap = {
        'Chart': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16,
                  15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 24, 22: 25, 23: 26},

        'Time': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16,
                 15: 17, 16: 18, 17: 19, 18: 20, 19: 22, 20: 23, 21: 24, 22: 25, 23: 26, 24: 27},

        'Math': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16,
                 15:18, 16:19, 17: 20, 18: 21, 19: 22, 20: 23, 21:24 , 22:25 , 23: 26, 24: 27, 25: 28, 26: 29,
                 27: 30,
                 28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39, 37: 40, 38: 41, 39: 42, 40: 43,
                 41:44 , 42: 45, 43: 46, 44: 47, 45: 48, 46: 49, 47: 50, 48: 52, 49: 53, 50: 54, 51: 55, 52: 56, 53: 57,
                 54: 58, 55: 59, 56: 60, 57: 61, 58: 62, 59: 63, 60: 64, 61: 65, 62: 66, 63: 67, 64: 68, 65: 69, 66: 70,
                 67: 71, 68: 72, 69: 73, 70: 74, 71: 75, 72: 76, 73: 77, 74: 78, 75: 79, 76: 80, 77: 81, 78: 82, 79: 83,
                 80: 84, 81: 85, 82: 86, 83: 87, 84: 88, 85: 89, 86: 90, 87: 91, 88: 92, 89: 93, 90:94 , 91: 95, 92: 96,
                 93: 97, 94:98 , 95: 99, 96: 100, 97: 101, 98: 102, 99: 103, 100: 105, 101: 106},
        
        'Mockito': {0: 1, 1: 2, 2: 3, 3: 4, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 10, 10: 18, 11: 19, 12: 20, 13: 21},
        'Lang': {0: 1, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16,
                 15: 17, 16: 18, 17: 19, 18: 20, 19: 22, 20: 24, 21:26, 22:27, 23:28, 24: 30, 25: 31, 26: 32, 27: 33,
                 28: 34, 29: 35, 30: 36, 31: 37, 32: 38, 33: 39, 34: 40, 35: 41, 36: 42, 37: 43, 38: 44, 39: 45, 40: 46,
                 41: 53, 42: 54, 43: 55, 44: 57, 45: 59, 46: 60, 47: 61, 48: 62, 49: 63, 50: 64, 51: 65},
#删除没有被任何失败测试用例覆盖到故障语句的版本 21 25 29 51 52 58

        'Cli': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 39, 6: 38, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15,
                15: 16, 16:20, 17:21, 18: 22, 19: 23, 20: 24, 21: 25, 22: 26, 23: 27, 24: 28, 25: 29, 26: 30, 27: 31,
                28: 32, 29: 33, 30: 34, 31: 35, 32: 37},
        'JxPath': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15,
                   15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22}
    }


class SumDataset( data.Dataset ):
    def __init__(self, config, dataName="train", proj="Math",testlst=[],vallst=[]):
        self.train_path = proj + ".pkl"
        self.val_path = "ndev.txt"  # "validD.txt"
        self.test_path = "ntest.txt"
        self.proj = proj
        self.SentenceLen = config.SentenceLen
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Voc['Method'] = len( self.Nl_Voc )
        self.Nl_Voc['Test'] = len( self.Nl_Voc )
        self.Nl_Voc['Line'] = len( self.Nl_Voc )
        self.Nl_Voc['RTest'] = len( self.Nl_Voc )
        self.Nl_Voc['Mutation'] = len( self.Nl_Voc )
        self.Nl_Len = config.NlLen  
        self.Code_Len = config.CodeLen  
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.ids = []
        self.Nls = []
        self.alldatalen=0
        self.label=config.lable

        if os.path.exists( "nl_voc_"+self.proj+".pkl" ):
            #    self.init_dic()
            self.Load_Voc()
        else:
            self.Nl_Voc = pickle.load( open( "nl_voc.pkl", "rb" ) )
            data = self.preProcessData( open( self.train_path, "rb" ) )
        # print(self.Nl_Voc)
        if not os.path.exists( self.proj +self.label + 'dataAll.pkl' ):
            data = self.preProcessData( open( self.train_path, "rb" ) )
        else:
            data = pickle.load( open( self.proj +self.label+ 'dataAll.pkl', 'rb' ) )
        self.alldatalen = len(data[0])
        self.data = []
        if dataName == "train":
            for i in range( len( data ) ):
                tmp = []
                for j in range( len( data[i] ) ):
                    if j in testlst or j in vallst:
                        continue
                    tmp.append( data[i][j] )  # self.data.append(data[i][0:testid] + data[i][testid + 1:])
                self.data.append( tmp )
        elif dataName == 'test':
            self.ids = testlst
            for i in range( len( data ) ):
                tmp = []
                for x in self.ids:
                    tmp.append( data[i][x] )
                self.data.append( tmp )
        else:
            self.ids = vallst
            for i in range( len( data ) ):
                tmp = []
                for x in self.ids:
                    tmp.append( data[i][x] )
                self.data.append( tmp )

    def Load_Voc(self):
        if os.path.exists( "nl_voc_"+self.proj+".pkl"):
            self.Nl_Voc = pickle.load( open( "nl_voc_"+self.proj+".pkl", "rb" ) )

        if os.path.exists( "code_voc.pkl" ):
            self.Code_Voc = pickle.load( open( "code_voc.pkl", "rb" ) )
        if os.path.exists( "char_voc.pkl" ):
            self.Char_Voc = pickle.load( open( "char_voc.pkl", "rb" ) )

    def splitCamel(self, token):
        ans = []
        tmp = ""
        for i, x in enumerate( token ):
            if i != 0 and x.isupper() and token[i - 1].islower():
                ans.append( tmp )
                tmp = x.lower()
            elif x in '$.':
                ans.append( tmp )
                tmp = ""
            else:
                tmp += x.lower()
        ans.append( tmp )
        return ans

    def init_dic(self):
        print("initVoc")
        f = open( self.p + '.pkl', 'rb' )
        data = pickle.load( f )
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        Nls = []
        Codes = []
        for x in data:
            for s in x['methods']:
                s = s[:s.index( '(' )]
                if len( s.split( ":" ) ) > 1:
                    tokens = ".".join( s.split( ":" )[0].split( '.' )[-2:] + [s.split( ":" )[1]] )
                else:
                    tokens = ".".join( s.split( ":" )[0].split( '.' )[-2:] )
                Codes.append( self.splitCamel( tokens ) )
        
            for s in x['ftest']:
                if len( s.split( ":" ) ) > 1:
                     tokens = ".".join( s.split( ":" )[0].split( '.' )[-2:] + [s.split( ":" )[1]] )
                else:
                    tokens = ".".join( s.split( ":" )[0].split( '.' )[-2:] )
                Codes.append( self.splitCamel( tokens ) )
        code_voc = VocabEntry.from_corpus( Codes, size=50000, freq_cutoff=0 )
        self.Code_Voc = code_voc.word2id
        open( "code_voc.pkl", "wb" ).write( pickle.dumps( self.Code_Voc ) )

    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            if x not in voc:
                ans.append( 1 )
            else:
                ans.append( voc[x] )
        return ans

    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append( c_id )
            ans.append( tmp )
        return ans

    def pad_seq(self, seq, maxlen):
        act_len = len( seq )
        if len( seq ) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_seq1(self, seq, maxlen):
        act_len = len( seq )
        if len(seq) < maxlen:
            seq = seq + [[self.PAD_token for _ in seq[0]]] * maxlen
            # print(seq[act_len:maxlen])
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq

    def pad_str_seq(self, seq, maxlen):
        act_len = len( seq )
        if len( seq ) < maxlen:
            seq = seq + ["<pad>"] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq

    def pad_list(self, seq, maxlen1, maxlen2):
        if len( seq ) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq

    def pad_multilist(self, seq, maxlen1, maxlen2, maxlen3):
        if len( seq ) < maxlen1:
            seq = seq + [[[self.PAD_token] * maxlen3] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq

    def tokenize_for_bleu_eval(self, code):
        code = re.sub( r'([^A-Za-z0-9])', r' \1 ', code )
        # code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        code = re.sub( r'\s+', ' ', code )
        code = code.replace( '"', '`' )
        code = code.replace( '\'', '`' )
        tokens = [t for t in code.split( ' ' ) if t]
        return tokens

    def getoverlap(self, a, b):
        ans = []
        for x in a:
            maxl = 0
            for y in b:
                tmp = 0
                for xm in x:
                    if xm in y:
                        tmp += 1
                maxl = max( maxl, tmp )
            ans.append( int( 100 * maxl / len( x ) ) + 1 )
        return ans

    def getRes(self, codetoken, nltoken):
        ans = []
        for x in nltoken:
            if x == "<pad>":
                continue
            if x in codetoken and codetoken.index( x ) < self.Code_Len and x != "(" and x != ")":
                ans.append( len( self.Nl_Voc ) + codetoken.index( x ) )
            else:
                if x in self.Nl_Voc:
                    ans.append( self.Nl_Voc[x] )
                else:
                    ans.append( 1 )
        for x in ans:
            if x >= len( self.Nl_Voc ) + self.Code_Len:
                # print(codetoken, nltoken)
                exit( 0 )
        return ans

    def preProcessData(self, dataFile):
        path_stacktrace = os.path.join( '../FLocalization/stacktrace', self.proj )
        # print(path_stacktrace)
        lines = pickle.load( dataFile )  # dataFile.readlines()
        Nodes = []
        Types = []
        LineNodes = []
        LineTypes = []
        LineMus = []
        Res = []
        MRes = []
        inputText = []
        inputNlad = []
        LineSMType=[]
        complexity = []
        node_S = []  # 方法节点的SBFL属性
        maxl = 0
        maxl2 = 0
        error = 0
        error1 = 0
        error2 = 0
        error12 = 0
        correct = 0
        for k in range( len( lines ) ):
            x = lines[k]
            if os.path.exists( path_stacktrace + '/%d.json' % dmap[self.proj][k] ):
                # print("path_stacktrace:", path_stacktrace)
                stack_info = json.load( open( path_stacktrace + '/%d.json' % dmap[self.proj][k] ) )
                if x['ftest'].keys() != stack_info.keys():
                    with open( "problem_stack", 'a' ) as f:
                        f.write( "{} {} no!\n".format( k, dmap[self.proj][k] ) )
                        f.write( str( x['ftest'].keys() ) + '\n' )
                        f.write( str( stack_info.keys() ) + '\n' )
                    for error_trace in x['ftest'].keys():
                        if error_trace not in stack_info.keys():
                            error += 1
                        else:
                            correct += 1
                        # assert error_trace in stack_info.keys()
                    # error += 1
                # else:
                # correct += 1
            nodes = []  
            types = []  
            res = []  
            mres=[]
            nladrow = []  
            nladcol = []
            nladval = []
            texta = []  
            textb = []
            linenodes = []  
            linetypes = []
            methodnum = len( x['methods'] )  
         
            rrdict = {}
           
            for s in x['methods']:
                rrdict[x['methods'][s]] = s[:]
                
            for i in range( methodnum ):
                nodes.append( 'Method' )
                
                if len( rrdict[i].split("@") ) > 1:
                    tokens = ".".join( rrdict[i].split( "@" )[0].split( '/' )[-2:] + [rrdict[i].split( "@" )[1]] )
                else:
                    tokens = ".".join( rrdict[i].split( "@" )[0].split( '/' )[-2:] )
                
                ans = self.splitCamel( tokens )
                
                texta.append( ans ) 
                
                if i in x['ans']:                                               
                   mres.append( 1 )                                             
                else:                                                           
                   mres.append( 0 )                                             
            
            Loc=0
            LocalVarieble=0
            controls=0
            Expressions=0
            Loops=0
            Ifs=0
            Calls=0
            words=0
            parameters=0
            for i in range(methodnum):
                if i not in x['complexity']:
                    Loc=max(Loc,0)
                    LocalVarieble = max( LocalVarieble,0)
                    controls = max( controls, 0 )
                    Expressions = max( Expressions, 0)
                    Loops = max( Loops, 0)
                    Ifs = max( Ifs, 0 )
                    Calls = max( Calls, 0 )
                    words = max( words, 0)
                    parameters = max( parameters, 0)
                    continue
                Loc=max(Loc,x['complexity'][i][1])
                LocalVarieble = max( LocalVarieble, x['complexity'][i][2])
                controls = max( controls, x['complexity'][i][3] )
                Expressions = max( Expressions, x['complexity'][i][4] )
                Loops = max( Loops, x['complexity'][i][5] )
                Ifs = max( Ifs, x['complexity'][i][6] )
                Calls = max( Calls, x['complexity'][i][7] )
                words = max( words, x['complexity'][i][8] )
                parameters = max( parameters, x['complexity'][i][9] )
            codeCom = []
            msfamulae = []
            complexity_max=[0,Loc,LocalVarieble,controls,Expressions,Loops,Ifs,Calls,words,parameters]
            complexitylen=len(x['complexity'][0])
            for i in range(methodnum):
                if i not in x['complexity']:
                    tmp=[0 for _ in range(complexitylen)]
                else:
                    tmp=[x['complexity'][i][0]]
                    for j in range(1,complexitylen):
                        if complexity_max[j]!=0:
                            tmp.append(x['complexity'][i][j]/complexity_max[j])
                        else:
                            tmp.append( x['complexity'][i][j])
                codeCom.append(tmp)
                msfamulae.append(x['mSBFL'][i])
                
            rrdic = {}
            for s in x['ftest']:
                rrdic[x['ftest'][s]] = s
            for i in range( len( x['ftest'] ) ):
                nodes.append( 'Test' )
                if len( rrdic[i].split( "#" ) ) > 1:
                    tokens = ".".join( rrdic[i].split( "#" )[0].split( '.' )[-2:] + [rrdic[i].split( "#" )[1]] )
                else:
                    tokens = ".".join( rrdic[i].split( "#" )[0].split( '.' )[-2:] )
                ans = self.splitCamel( tokens )
                textb.append( ans )
            for i in range( len( x['rtest'] ) ):
                nodes.append( 'RTest' )
            mus = []
            line_S_M = []
            for i in range( len( x['lines'] ) ):
                if i not in x['ltype']:
                    x['ltype'][i] = 'Empty'
                if x['ltype'][i] not in self.Nl_Voc:
                    self.Nl_Voc[x['ltype'][i]] = len( self.Nl_Voc )
                linenodes.append( x['ltype'][i] )
                if i in x['lcorrectnum']:
                    linetypes.append( x['lcorrectnum'][i] )
                else:
                    linetypes.append( 1 )
                types.append( 2 )    
                if i in x['lans']:   
                    res.append( 1 )  
                else:                
                    res.append( 0 )  
                line_S_M.append( x['lSBFL'][i])


            for i in range( len( x['mutation'] ) ):
                if i not in x['mtype']:
                    x['mtype'][i] = 'Empty'
                if x['mtype'][i] not in self.Nl_Voc:
                    self.Nl_Voc[x['mtype'][i]] = len( self.Nl_Voc )
                linenodes.append( x['mtype'][i] )
                linetypes.append( 0 )
                line_S_M.append(x['mMBFL'][i])
            maxl = max( maxl, len( nodes ) )
            maxl2 = max( maxl2, len( linenodes ) )
            print(maxl, maxl2)
            ed = {}

 
            for e in x['call']:
                if e not in ed:
                    ed[e]=1
                else:
                    assert (0)
                nladrow.append( e[0] )
                nladcol.append( e[1] )
                nladval.append( 1 )
                nladrow.append( e[1]  )
                nladcol.append( e[0]  )
                nladval.append( 1 )
            for e in x['edge2']:
                if isinstance(e[0], str):
                    a=int(e[0])
                else:
                    a=e[0]+ self.Nl_Len
                b = e[1] + self.Nl_Len  
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                else:
                    assert (0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                else:
                    assert (0)
                nladrow.append( a )
                nladcol.append( b )
                nladval.append( 1 )
                nladrow.append( b )
                nladcol.append( a )
                nladval.append( 1 )
            for e in x['edge10']:
                a = e[0] + self.Nl_Len
                b = e[1] + methodnum + len( x['ftest'] )
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                    nladrow.append( a )
                    nladcol.append( b )
                    nladval.append( 1 )
                else:
                    pass
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                    nladrow.append( b )
                    nladcol.append( a )
                    nladval.append( 1 )
                else:
                    pass

            for e in x['edge']:
                if isinstance( e[0], str ):
                    print("e[0],e[1],self.Nl_Len",e[0],e[1],self.Nl_Len)
                a = e[0] + self.Nl_Len  
                b = e[1] + methodnum
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                    nladrow.append( a )
                    nladcol.append( b )
                    nladval.append( 1 )
                else:
                    assert (0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                    nladrow.append( b )
                    nladcol.append( a )
                    nladval.append( 1 )
                else:
                    assert (0)

            for e in x['edge12']:
                a = e[0] + self.Nl_Len + len( x['lines'] )
                b = e[1] + self.Nl_Len
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                    nladrow.append( a )
                    nladcol.append( b )
                    nladval.append( 1 )

                if (b, a) not in ed:
                    ed[(b, a)] = 1
                    nladrow.append( b )
                    nladcol.append( a )
                    nladval.append( 1 )


            for e in x['edge13']:
                a = e[0] + self.Nl_Len + len( x['lines'] )
                b = e[1] + len( x['ftest'] ) + methodnum
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                    nladrow.append( a )
                    nladcol.append( b )
                    nladval.append( 1 )

                if (b, a) not in ed:
                    ed[(b, a)] = 1
                    nladrow.append( b )
                    nladcol.append( a )
                    nladval.append( 1 )


            for e in x['edge14']:
                a = e[0] + self.Nl_Len + len( x['lines'] )
                b = e[1] + methodnum
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                    nladrow.append( a )
                    nladcol.append( b )
                    nladval.append( 1 )
                else:
                    assert (0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                    nladrow.append( b )
                    nladcol.append( a )
                    nladval.append( 1 )
                else:
                    assert (0)

            overlap = self.getoverlap( texta, textb )
            Nodes.append( self.pad_seq( self.Get_Em( nodes, self.Nl_Voc ), self.Nl_Len ) )
            Types.append( self.pad_seq( types, self.Code_Len ) ) 
            Res.append( self.pad_seq( res, self.Code_Len ) )  
            MRes.append( self.pad_seq( mres, self.Nl_Len ) )       
            LineMus.append( self.pad_list( mus, self.Code_Len, 3 ) )
            inputText.append( self.pad_seq( overlap, self.Nl_Len ) )
            LineNodes.append( self.pad_seq( self.Get_Em( linenodes, self.Nl_Voc ), self.Code_Len ) )
            LineTypes.append( self.pad_seq( linetypes, self.Code_Len ) )
            LineSMType.append(self.pad_seq1( line_S_M, self.Code_Len ) )
            complexity.append(self.pad_seq1( codeCom, self.Nl_Len ))
            node_S.append(self.pad_seq1(msfamulae, self.Nl_Len ))

            row = {}
            col = {}
            for i in range( len( nladrow ) ):
                if nladrow[i] not in row:
                    row[nladrow[i]] = 0
                row[nladrow[i]] += 1
                if nladcol[i] not in col:
                    col[nladcol[i]] = 0
                col[nladcol[i]] += 1
            for i in range( len( nladrow ) ):
                nladval[i] = 1 / math.sqrt( row[nladrow[i]] ) * 1 / math.sqrt( col[nladcol[i]] )
            nlad = sparse.coo_matrix( (nladval, (nladrow, nladcol)),
                                      shape=(self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len) )
            inputNlad.append( nlad )
        print("max1: %d max2: %d" % (maxl, maxl2))
        print("correct: %d error: %d" % (correct, error))
        print("error1: %d error2: %d" % (error1, error2))
        f = open( 'nl_voc_'+self.proj+'.pkl', 'wb' )
        pickle.dump( self.Nl_Voc, f )
        f.close()

        batchs = [Nodes, Types, inputNlad, Res, inputText, LineNodes, LineTypes, LineMus,LineSMType,complexity,node_S,MRes]
        self.data = batchs
        open( self.proj +self.label +"dataAll.pkl", "wb" ).write( pickle.dumps( batchs, protocol=4 ) )
        return batchs

    def __getitem__(self, offset):
        ans = []
        if True:
            for i in range( len( self.data ) ):
                if i == 2:
                    # torch.FloatTensor(np.array([self.data[i][offset].row, self.data[i][offset].col])).float()
                    # torch.FloatTensor(self.data[i][offset].data)
                    # torch.FloatTensor(self.data[i][offset].data)
                    # ans.append(self.data[i][offset])
                    # ans.append(torch.sparse.FloatTensor(torch.LongTensor(np.array([self.data[i][offset].row, self.data[i][offset].col])), torch.FloatTensor(self.data[i][offset].data).float(), torch.Size([self.Nl_Len,self.Nl_Len])))
                    # open('tmp.pkl', 'wb').write(pickle.dumps(self.data[i][offset]))
                    # assert(0)
                    ans.append( self.data[i][offset].toarray() )
                    # print(self.data[i][offset].toarray()[0, 2545])
                    # assert(0)
                else:
                    ans.append( np.array( self.data[i][offset] ) )
        else:
            for i in range( len( self.data ) ):
                if i == 4:
                    continue
                ans.append( np.array( self.data[i][offset] ) )
            negoffset = random.randint( 0, len( self.data[0] ) - 1 )
            while negoffset == offset:
                negoffset = random.randint( 0, len( self.data[0] ) - 1 )
            if self.dataName == "train":
                ans.append( np.array( self.data[2][negoffset] ) )
                ans.append( np.array( self.data[3][negoffset] ) )
        return ans

    def __len__(self):
        return len( self.data[0] )

    def Get_Train(self, batch_size):
#        print("========hhhhh=========")
        data = self.data
        loaddata = data
        batch_nums=0
        if batch_size!=0:
            batch_nums = int( len( data[0] ) / batch_size )
#        print("self.dataName-batch_nums:",self.dataName,self.ids)
        if True:
            if self.dataName == 'train':
                shuffle = np.random.permutation( range( len( loaddata[0] ) ) )  
            else:
                shuffle = np.arange( len( loaddata[0] ) )
            for i in range(batch_nums): 
                ans = []
                for j in range( len( data ) ):
                    if j != 2:
                        tmpd = np.array( data[j] )[shuffle[batch_size * i: batch_size * (i + 1)]]
                        ans.append( torch.from_numpy( np.array( tmpd ) ) )  
                    else:
                        ids = []
                        v = []
                        for idx in range( batch_size * i, batch_size * (i + 1) ):
                            for p in range( len( data[j][shuffle[idx]].row ) ):
                                ids.append(
                                    [idx - batch_size * i, data[j][shuffle[idx]].row[p], data[j][shuffle[idx]].col[p]] )
                                v.append( data[j][shuffle[idx]].data[p] )
                        # print([batch_size, self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len])
                        ans.append( torch.sparse.FloatTensor( torch.LongTensor(ids).t(), torch.FloatTensor( v ),
torch.Size([batch_size, self.Nl_Len + self.Code_Len,
self.Nl_Len + self.Code_Len] ) ) )
                yield ans
            if batch_nums * batch_size < len( data[0] ):
                ans = []
                for j in range( len( data ) ):
                    if j != 2:
                        #print("j",j)
                        tmpd = np.array( data[j] )[shuffle[batch_nums * batch_size:]]
                        ans.append( torch.from_numpy( np.array( tmpd ) ) )
                    else:
                        ids = []
                        v = []
                        for idx in range( batch_size * batch_nums, len( data[0] ) ):
                            for p in range( len( data[j][shuffle[idx]].row ) ):
                                # if(data[j][shuffle[idx]].col[p]>=400 or data[j][shuffle[idx]].row[p]>=400):
                                #     print(idx, shuffle[idx],[idx - batch_size * batch_nums, data[j][shuffle[idx]].row[p], data[j][shuffle[idx]].col[p]])
                                ids.append( [idx - batch_size * batch_nums, data[j][shuffle[idx]].row[p],
                                             data[j][shuffle[idx]].col[p]] )
                                v.append( data[j][shuffle[idx]].data[p] )
                        ans.append( torch.sparse.FloatTensor( torch.LongTensor( ids ).t(), torch.FloatTensor( v ),
                                                              torch.Size( [len( data[0] ) - batch_size * batch_nums,
                                                                           self.Nl_Len + self.Code_Len,
                                                                           self.Nl_Len + self.Code_Len] ) ) )
                yield ans


class node:
    def __init__(self, name):
        self.name = "Lang"
        self.father = None
        self.child = []
        self.id = -1
# if __name__ == "__main__":
#     train_set = SumDataset(args, "train", testid=0, proj="tcas", lst=[] )
