from GGAT import GGAT,SpGGAT
from SubLayerConnection import SublayerConnection
from LayerNorm import LayerNorm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
#from torch_geometric.utils import dropout_adj
#from torch_geometric.utils import dropout_edge#####GPU上不可用
NET={'GGAT':GGAT,'SpGGAT':SpGGAT}
class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, args):
        #NETname,hidden, attn_heads, feed_forward_hidden, dropout
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.NETname=args.NETname
        if self.NETname in ['GGAT','SpGGAT']:
            self.Tconv_forward = NET[self.NETname](args.embedding_size,args.hidden_size, args.attn_heads, args.dropout, args.alpha)
        self.sublayer4 = SublayerConnection(size = args.hidden_size, dropout = args.dropout)
        self.dropout = nn.Dropout(p = args.dropout)
        self.norm = LayerNorm(args.hidden_size)

    def forward(self, x, mask, inputP):
#        print("x.device",x.device)
        #x = self.sublayer1(x, lambda _x: self.attention1.forward(_x, _x, _x, mask=mask))
        #x = self.sublayer2(x, lambda _x: self.combination.forward(_x, _x, pos))
        #x = self.sublayer3(x, lambda _x: self.combination2.forward(_x, _x, charem))
        x = self.sublayer4(x, lambda _x: self.Tconv_forward.forward(_x, None, inputP))
        x = self.norm(x)
        return self.dropout( x ), 0
            
            

#config {'seed': 38108, 'learning_rate': 0.001, 'num_hidden': 256, 'num_proj_hidden': 256, 'activation': 'prelu', 'base_model': 'GCNConv', 'num_layers': 2, 'drop_edge_rate_1': 0.2, 'drop_edge_rate_2': 0.0, 'drop_feature_rate_1': 0.3, 'drop_feature_rate_2': 0.2, 'tau': 0.9, 'num_epochs': 200, 'weight_decay': 1e-05}
