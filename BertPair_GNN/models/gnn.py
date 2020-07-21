import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from . import gnn_iclr

class GNN(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, N, hidden_size=230, use_sup_cost=False):
        '''
        N: Num of classes
        '''
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.node_dim = hidden_size + N
        self.gnn_obj = gnn_iclr.GNN_nl(N, self.node_dim, nf=96, J=1)
        self.try_linear = nn.Linear(self.hidden_size, N)
        self.cost1 = nn.CrossEntropyLoss()
        self.use_sup_cost = use_sup_cost

    def forward(self, support, query, N, K, Q, training=False):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        support = self.sentence_encoder(support)
        query = self.sentence_encoder(query)
        support = support.view(-1, N, K, self.hidden_size)
        query = query.view(-1, N * Q, self.hidden_size)

        if training and self.use_sup_cost:
            support_logits_copy = self.try_linear(support)
            chunks = support_logits_copy.chunk(1, 0)
            support_cost = Variable(torch.tensor(0.0), requires_grad=True).cuda()
            for ch in chunks:
                support_pred = F.softmax(ch.view(N * K, -1), -1).cuda()
                should_be = torch.tensor([i // K for i in range(N * K)]).cuda()
                support_cost += self.cost1(support_pred, should_be)

        B = support.size(0)
        NQ = query.size(1)
        D = self.hidden_size

        support = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, N * K, D) # (B * NQ, N * K, D)
        query = query.view(-1, 1, D) # (B * NQ, 1, D)
        labels = Variable(torch.zeros((B * NQ, 1 + N * K, N), dtype=torch.float)).cuda()
        for b in range(B * NQ):
            for i in range(N):
                for k in range(K):
                    labels[b][1 + i * K + k][i] = 1
        nodes = torch.cat([torch.cat([query, support], 1), labels], -1) # (B * NQ, 1 + N * K, D + N)

        logits = self.gnn_obj(nodes) # (B * NQ, N)
        _, pred = torch.max(logits, 1)
        if training and self.use_sup_cost:
            return logits, pred, support_cost
        return logits, pred, 0
