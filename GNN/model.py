import torch.nn as nn
import torch
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import dgl
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.unit0_conv = nn.Conv3d(14,32,
                               kernel_size=3,stride=1, bias=True)
        self.unit0_func = nn.ReLU()

        self.unit1_pool = nn.MaxPool3d(kernel_size=2,stride=2)
        self.unit1_conv = nn.Conv3d(32, 64,
                                    kernel_size=3,padding=1, stride=1, bias=True)
        self.unit1_func = nn.ReLU()

        self.unit2_conv = nn.Conv3d(64, 64,
                                    kernel_size=1, stride=1, bias=True)
        self.unit2_func = nn.ReLU()

        self.unit3_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.unit3_conv = nn.Conv3d(64, 64,
                                    kernel_size=3,padding=1, stride=1, bias=True)
        self.unit3_func = nn.ReLU()

        self.unit4_conv = nn.Conv3d(64, 128,
                                    kernel_size=1, stride=1, bias=True)
        self.unit4_func = nn.ReLU()

        self.unit5_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.unit5_conv = nn.Conv3d(128, 128,
                                    kernel_size=3,padding=1 ,stride=1, bias=True)
        self.unit5_func = nn.ReLU()
        self.protein_fc = nn.Linear(16000,1600)
        self.conv1 = GraphConv(17, 200)
        self.conv2 = GraphConv(200, 200)
        self.conv3 = GraphConv(200, 200)
        self.conv4 = GraphConv(200, 200)
        self.final_fc = nn.Linear(1800, 2)
        #self.final_func = nn.LogSoftmax()

    def forward(self, x,g):
        h = g.ndata['feat']
        out = self.unit0_conv(x)
        out = self.unit0_func(out)

        out = self.unit1_pool(out)
        out = self.unit1_conv(out)
        out = self.unit1_func(out)

        out = self.unit2_conv(out)
        out = self.unit2_func(out)

        out = self.unit3_pool(out)
        out = self.unit3_conv(out)
        out = self.unit3_func(out)

        out = self.unit4_conv(out)
        out = self.unit4_func(out)

        out = self.unit5_pool(out)
        out = self.unit5_conv(out)
        out = self.unit5_func(out)
        out = out.reshape(out.size(0), -1)
        out = self.protein_fc(out)
        h = F.relu(self.conv1(g,h))
        h = F.relu(self.conv2(g,h))
        h = F.relu(self.conv3(g,h))
        h = F.relu(self.conv4(g,h))
        g.ndata['h'] = h
        graph_repr = dgl.sum_nodes(g, 'h')
      
        out =torch.cat((out,graph_repr),-1)
        out = self.final_fc(out)
        
        #out = self.final_func(out)
        return out
