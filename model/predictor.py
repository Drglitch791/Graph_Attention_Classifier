import torch
import torch.nn as nn
import torch.nn.functional as F

from .dmrgcn import st_dmrgcn


class tpcnn(nn.Module):
    def __init__(self, seq_len, pred_seq_len, output_feat, n_tpcn=2, n_gtacn=1, kernel_size=3, dropout=0,
                 residual=True):
        super().__init__()

        # Temporal Convolutional Network (TCN)
        self.tpcn = nn.ModuleList()
        self.tpcn.append(nn.Sequential(nn.Conv2d(seq_len, pred_seq_len, kernel_size, padding=1),
                                       nn.PReLU(),
                                       nn.Dropout(dropout, inplace=True), ))
        for i in range(1, n_tpcn):
            self.tpcn.append(nn.Sequential(nn.Conv2d(pred_seq_len, pred_seq_len, kernel_size, padding=1),
                                           nn.PReLU(),
                                           nn.Dropout(dropout, inplace=True), ))

        # Global Temporal Aggregation (GTA)
        self.gtacn = nn.ModuleList()
        self.gtacn.append(nn.Sequential(nn.Conv2d(output_feat, output_feat, (pred_seq_len, 1), padding=0),
                                        nn.PReLU(),
                                        nn.Dropout(dropout, inplace=True), ))
        for i in range(1, n_gtacn):
            self.gtacn.append(nn.Sequential(nn.Conv2d(output_feat, output_feat, 1, padding=0),
                                            nn.PReLU(),
                                            nn.Dropout(dropout, inplace=True), ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif seq_len == pred_seq_len:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(seq_len, pred_seq_len, kernel_size=1),
            )

    def forward(self, x):
        res = self.residual(x)

        # TCN
        x = self.tpcn[0](x) + res
        for i in range(1, len(self.tpcn)):
            x = self.tpcn[i](x) + x

        # GTA
        x = x.permute(0, 2, 1, 3).contiguous()
        for i in range(len(self.gtacn)):
            x = self.gtacn[i](x) + x
        x = x.permute(0, 2, 1, 3)

        return x

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(60, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 8)
        self.linear4 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)

        return x

class social_dmrgcn(nn.Module):
    def __init__(self, n_stgcn=1, n_tpcnn=4, input_feat=2, output_feat=5, seq_len=8, pred_seq_len=12, kernel_size=3):
        super().__init__()
        self.n_stgcn = n_stgcn
        self.n_tpcnn = n_tpcnn

        # Disentangling Scale Set [A_disp, A_dist]
        split = [[0, 1/4, 2/4, 3/4, 1],
                 [0, 1/2, 1, 2, 4]]

        # GCN Block
        self.st_dmrgcns = nn.ModuleList()
        self.st_dmrgcns.append(st_dmrgcn(input_feat, output_feat, (kernel_size, seq_len), split=split, relation=2))
        for j in range(1, self.n_stgcn):
            self.st_dmrgcns.append(st_dmrgcn(output_feat, output_feat, (kernel_size, seq_len), split=split, relation=2))

        # TPCNN Block
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(tpcnn(seq_len=seq_len, pred_seq_len=pred_seq_len, output_feat=output_feat))
        for j in range(1, self.n_tpcnn):
            self.tpcnns.append(tpcnn(seq_len=pred_seq_len, pred_seq_len=pred_seq_len, output_feat=output_feat))

        self.classifier = Classifier()

    def forward(self, v, a):
        # GCN Block
        for k in range(self.n_stgcn):
            v, a = self.st_dmrgcns[k](v, a)

        # NCTV -> NTCV
        v = v.permute(0, 2, 1, 3)

        # TPCNN Block
        for k in range(self.n_tpcnn):
            v = self.tpcnns[k](v)

        # NTCV -> NCTV
        v = v.permute(0, 2, 1, 3)

        v = torch.sum(v, dim=-1)
        v = torch.flatten(v, 1)
        v = self.classifier(v)
        # print('social dmrgcn:', v, v.shape)
        return v, a
