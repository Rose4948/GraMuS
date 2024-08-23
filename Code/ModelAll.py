import torch.nn as nn
import torch.nn.functional as F
import torch
#from Transfomer import TransformerBlock
from TransfomerAll import TransformerBlock
# from postionEmbedding import PositionalEmbedding
# from LayerNorm import LayerNorm
from SubLayerConnection import *
# import numpy as np


class NlEncoderAll( nn.Module ):
    def __init__(self, args):
        super( NlEncoderAll, self ).__init__()
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.nl_len = args.NlLen  
        self.word_len = args.WoLen  
        self.NETname=args.NETname
        self.losslevel='Average'
        self.i=-1
        self.drop_feature_rate_1 = args.drop_feature_rate_1
        self.drop_feature_rate_2 = args.drop_feature_rate_2
        self.drop_edge_rate_1 = args.drop_edge_rate_1
        self.drop_edge_rate_2 = args.drop_edge_rate_2
        self.feed_forward_hidden = 4 * self.embedding_size
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.layer_size)] )
        self.sus_len_sbfl=20
        self.sus_len_code=32
        self.token_embedding = nn.Embedding( args.Nl_Vocsize, self.embedding_size - self.sus_len_code)
        self.token_embedding1 = nn.Embedding( args.Nl_Vocsize, self.embedding_size-self.sus_len_sbfl)

        self.loss = nn.CrossEntropyLoss()  # 损失函数
        self.norm = LayerNorm(self.hidden_size)

        self.resLinear2 = nn.Linear( self.hidden_size, 1 )

    def forward(self, input_node, inputtype, inputad, res, inputtext, linenode, linetype, linemus, LineSMType,complexity,node_S, MRes):
        nlmask = torch.gt( input_node, 0 )
        resmask = torch.eq( inputtype, 2 )  
        MResmask =  torch.eq(input_node, 2) 

        inputad = inputad.float()
        nodeem = self.token_embedding( input_node )  
    
        nodeem = torch.cat( [nodeem, inputtext.unsqueeze( -1 ).float()], dim=-1 )


        nodeem = torch.cat( [nodeem, complexity.float()], dim=-1 )

        nodeem = torch.cat( [nodeem, node_S.float()], dim=-1 )

        x = nodeem

        lineem = self.token_embedding1( linenode )  

        lineem = torch.cat([lineem, LineSMType.float()], dim=-1 )

        x = torch.cat( [x, lineem], dim=1)  

        for trans in self.transformerBlocks:
            x,loss0 = trans.forward( x, nlmask, inputad )
        MX = x[:, :input_node.size( 1 )]
        MResSoftmax = F.softmax(self.resLinear2(MX).squeeze(-1).masked_fill(MResmask == 0, -1e9), dim=-1)
        MLoss = -torch.log(MResSoftmax.clamp(min=1e-10, max=1)) * MRes
        MLoss = MLoss.sum(dim=-1)
        MLoss = MLoss.mean()

        x = x[:, input_node.size( 1 ):]  
        resSoftmax = F.softmax( self.resLinear2( x ).squeeze( -1 ).masked_fill( resmask == 0, -1e9 ), dim=-1 )
        loss = -torch.log( resSoftmax.clamp( min=1e-10, max=1 ) ) * res
        loss = loss.sum( dim=-1 )
        loss = loss.mean()
        print(self.losslevel)
        if self.NETname in ['GRACE']:
            loss = (loss+loss0)/2
        if self.losslevel == 'Method':
            return MLoss, resSoftmax, x
        elif self.losslevel == 'Statement':
            return loss, resSoftmax, x
        elif self.losslevel == 'Average':
            return (MLoss + loss)/2, resSoftmax, x

    def setloss(self, losslevel):
        self.losslevel = losslevel