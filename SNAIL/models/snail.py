import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=2):
        super(CausalConv1d, self).__init__()
        self.padding = dilation
        self.causal_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding = self.padding,
            dilation = dilation)

    def forward(self, minibatch):
        return self.causal_conv(minibatch)[:, :, :-self.padding]


class DenseBlock(nn.Module):

    def __init__(self, in_channels, filters, dilation=2):
        super(DenseBlock, self).__init__()
        self.causal_conv1 = CausalConv1d(
            in_channels,
            filters,
            dilation=dilation)
        self.causal_conv2 = CausalConv1d(
            in_channels,
            filters,
            dilation=dilation)

    def forward(self, minibatch):
        tanh = F.tanh(self.causal_conv1(minibatch))
        sig = F.sigmoid(self.causal_conv2(minibatch))
        out = torch.cat([minibatch, tanh*sig], dim=1)
        return out

class TCBlock(nn.Module):

    def __init__(self, in_channels, filters, seq_len):
        super(TCBlock, self).__init__()
        layer_count = np.ceil(np.log2(seq_len)).astype(np.int32)
        blocks = []
        channel_count = in_channels
        for layer in range(layer_count):
            block = DenseBlock(channel_count, filters, dilation=2**layer)
            blocks.append(block)
            channel_count += filters
        self.tcblock = nn.Sequential(*blocks)
        self._dim = channel_count

    def forward(self, minibatch):
        return self.tcblock(minibatch)

    @property
    def dim(self):
        return self._dim

class AttentionBlock(nn.Module):
    def __init__(self, dims, k_size, v_size, seq_len):

        super(AttentionBlock, self).__init__()
        self.key_layer = nn.Linear(dims, k_size)
        self.query_layer = nn.Linear(dims, k_size)
        self.value_layer = nn.Linear(dims, v_size)
        self.sqrt_k = np.sqrt(k_size)
        mask = np.tril(np.ones((seq_len, seq_len))).astype(np.float32)
        self.mask = nn.Parameter(torch.from_numpy(mask), requires_grad=False)
        self.minus = - 100.
        self._dim = dims + v_size

    def forward(self, minibatch, current_seq_len):
        keys = self.key_layer(minibatch)
        #queries = self.query_layer(minibatch)
        queries = keys
        values = self.value_layer(minibatch)
        current_mask = self.mask[:current_seq_len, :current_seq_len]
        logits = current_mask * torch.div(torch.bmm(queries, keys.transpose(2,1)), self.sqrt_k) + self.minus * (1. - current_mask)
        probs = F.softmax(logits, 2)
        read = torch.bmm(probs, values)
        return torch.cat([minibatch, read], dim=2)

    @property
    def dim(self):
        return self._dim

class SNAIL(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, N, K, hidden_size=230, use_sup_cost=False):
        '''
        N: num of classes
        K: num of instances for each class in the support set
        '''
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.drop = nn.Dropout()
        self.seq_len = N * K + 1
        self.att0 = AttentionBlock(hidden_size, 64, 32, self.seq_len)
        self.tc1 = TCBlock(self.att0.dim, 128, self.seq_len)
        self.att1 = AttentionBlock(self.tc1.dim, 256, 128, self.seq_len)
        self.tc2 = TCBlock(self.att1.dim, 128, self.seq_len)
        self.att2 = AttentionBlock(self.tc2.dim, 512, 256, self.seq_len)
        self.disc = nn.Linear(self.att2.dim, N, bias=False)
        self.bn1 = nn.BatchNorm2d(self.tc1.dim)
        self.bn2 = nn.BatchNorm2d(self.tc2.dim)
        self.use_sup_cost = use_sup_cost
        self.try_linear = nn.Linear(self.hidden_size, N)
        self.cost1 = nn.CrossEntropyLoss()
        #self.use_sup_cost = use_sup_cost

    def forward(self, support, query, N, K, Q, training=False):
        support = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query) # (B * N * Q, D)
        # support = self.drop(support)
        # query = self.drop(query)
        support = support.view(-1, N, K, self.hidden_size) # (B, N, K, D)
        query = query.view(-1, N * Q, self.hidden_size) # (B, N * Q, D)
        B = support.size(0) # Batch size
        NQ = query.size(1) # Num of instances for each batch in the query set
        if training and self.use_sup_cost:
            support_logits_copy = self.try_linear(support)
            chunks = support_logits_copy.chunk(1, 0)
            support_cost = Variable(torch.from_numpy(np.array([0.0])), requires_grad=True).cuda().float()
            for ch in chunks:
                support_pred = F.softmax(ch.view(N * K, -1), -1).cuda()
                should_be = Variable(torch.from_numpy(np.array([i // K for i in range(N * K)]))).cuda()
                support_cost += self.cost1(support_pred, should_be)

        support = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, N * K, self.hidden_size) # (B * NQ, N * K, D)
        query = query.view(-1, 1, self.hidden_size) # (B * NQ, 1, D)
        minibatch = torch.cat([support, query], 1)

        x = self.att0(minibatch, self.seq_len).transpose(1, 2)
        #x = self.bn1(x).transpose(1, 2)
        x = self.bn1(self.tc1(x)).transpose(1, 2)
        #x = self.tc1(x).transpose(1, 2)
        x = self.att1(x, self.seq_len).transpose(1, 2)
        x = self.bn2(self.tc2(x)).transpose(1, 2)
        #x = self.tc2(x).transpose(1, 2)
        x = self.att2(x, self.seq_len)
        x = x[:, -1, :]
        logits = self.disc(x)
        _, pred = torch.max(logits, -1)
        if training and self.use_sup_cost:
            return logits, pred, support_cost
        return logits, pred, 0

