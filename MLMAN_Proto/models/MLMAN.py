import sys

sys.path.append('..')
import torch
from torch import nn
from torch.nn import functional as F
import models.embedding as embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.utils import sort_batch_by_length, init_lstm, init_linear
from torch.autograd import Variable

class MLMAN(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5, args=None,
                 hidden_size=100, drop=True, N=None):
        nn.Module.__init__(self)
        self.word_embedding_dim = word_embedding_dim + 2 * pos_embedding_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim)
        self.args = args
        self.conv = nn.Conv2d(1, self.hidden_size*2, kernel_size=(3, self.word_embedding_dim), padding=(1, 0))
        self.proj = nn.Linear(self.hidden_size*8, self.hidden_size)
        self.lstm_enhance = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)

        self.multilayer = nn.Sequential(nn.Linear(self.hidden_size*8, self.hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size, 1))
        self.meta_linear = nn.Linear(self.hidden_size*4, self.hidden_size)
        self.try_linear = nn.Linear(self.hidden_size*4, N)
        self.drop = drop
        self.dropout = nn.Dropout(0.2)
        self.cost = nn.CrossEntropyLoss()
        self.apply(self.weights_init)

    def get_meta_repre(self, m):
        m = self.meta_linear(m)
        centroid = torch.mean(m, 0, True)
        mm = m.transpose(0,1) # 75 25 d
        centroid = centroid.transpose(0,2).transpose(0,1) # 1 75 d => 75 d 1
        att = F.softmax((mm @ centroid),0).transpose(0,1).expand(m.size()[0], m.size()[1], m.size()[2]) #75 25 1 => 25 75 100
        return torch.sum(m * att, 0, False)

        # tanh
        mm = torch.exp(torch.tanh(m))
        # softmax
        alpha = F.softmax(mm, 0)
        # multiply
        return torch.mean(mm,0,False) #torch.sum(m * alpha, 0, False)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            init_linear(m)
        elif classname.find('LSTM') != -1:
            init_lstm(m)
        # elif classname.find('Conv') != -1:
        #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     m.weight.data.normal_(0, np.sqrt(2. / n))
        #     if m.bias is not None:
        #         m.bias.data.zero_()

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label, Q):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        acc = torch.mean((pred.view(-1) == label.view(-1)).float())
        pred = pred.view(Q, -1)
        label = label.view(Q, -1)
        mistakes = torch.mean((pred != label).float(), 0)
        return acc, mistakes

    def context_encoder(self, input):
        input_mask = (input['mask'] != 0).float()
        max_length = input_mask.long().sum(1).max().item()
        input_mask = input_mask[:, :max_length].contiguous()
        embedding = self.embedding(input)
        embedding_ = embedding[:, :max_length].contiguous()

        if self.drop:
            embedding_ = self.dropout(embedding_)

        conv_out = self.conv(embedding_.unsqueeze(1)).squeeze(3)
        conv_out = conv_out * input_mask.unsqueeze(1)

        return conv_out.transpose(1,2).contiguous(), input_mask, max_length

    def lstm_encoder(self, input, mask, lstm, cuda):
        if self.drop:
            input = self.dropout(input)
        mask = mask.squeeze(2)
        sequence_lengths = mask.long().sum(1)
        sorted_inputs, sorted_sequence_lengths, restoration_indices, _ = sort_batch_by_length(input, sequence_lengths)

        packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                     sorted_sequence_lengths,
                                                     batch_first=True)
        lstmout, _ = lstm(packed_sequence_input)
        unpacked_sequence_tensor, _ = pad_packed_sequence(lstmout, batch_first=True)
        unpacked_sequence_tensor = unpacked_sequence_tensor.index_select(0, restoration_indices)

        return unpacked_sequence_tensor


    def CoAttention(self, support, query, support_mask, query_mask):

        att = support @ query.transpose(1, 2)
        att = att + support_mask * query_mask.transpose(1, 2) * 100
        support_ = F.softmax(att, 2) @ query * support_mask
        query_ = F.softmax(att.transpose(1,2), 2) @ support * query_mask
        return support_, query_

    def local_matching(self, support, query, support_mask, query_mask):

        support_, query_ = self.CoAttention(support, query, support_mask, query_mask)
        enhance_query = self.fuse(query, query_, 2)
        enhance_support = self.fuse(support, support_, 2)

        return enhance_support, enhance_query

    def fuse(self, m1, m2, dim):
        return torch.cat([m1, m2, torch.abs(m1 - m2), m1 * m2], dim)

    def local_aggregation(self, enhance_support, enhance_query, support_mask, query_mask, K):

        max_enhance_query, _ = torch.max(enhance_query, 1)
        mean_enhance_query = torch.sum(enhance_query, 1) / torch.sum(query_mask, 1)
        enhance_query = torch.cat([max_enhance_query, mean_enhance_query], 1)

        enhance_support = enhance_support.view(enhance_support.size(0) // K, K, -1, self.hidden_size * 2)
        support_mask = support_mask.view(enhance_support.size(0), K, -1, 1)

        max_enhance_support, _ = torch.max(enhance_support, 2)
        mean_enhance_support = torch.sum(enhance_support, 2) / torch.sum(support_mask, 2)
        enhance_support = torch.cat([max_enhance_support, mean_enhance_support], 2)

        return enhance_support, enhance_query

    def forward(self, support, query, N, K, Q, training, change_task=False, cuda=True):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''

        # if change_task:
        #     self.try_linear.weight = torch.nn.Parameter(torch.rand(self.try_linear.weight.size()).cuda())

        support, support_mask, support_len = self.context_encoder(support)
        query, query_mask, query_len = self.context_encoder(query)

        batch = support.size(0)//(N*K)

        # concate S_k operation
        support = support.view(batch, 1, N, K, support_len, self.hidden_size*2).expand(batch, N*Q, N, K, support_len, self.hidden_size*2).contiguous().view(batch*N*Q*N, K*support_len, self.hidden_size*2)
        support_mask = support_mask.view(batch, 1, N, K, support_len).expand(batch, N*Q, N, K, support_len).contiguous().view(-1, K*support_len, 1)
        query = query.view(batch, N*Q, 1, query_len, self.hidden_size*2).expand(batch, N*Q, N, query_len, self.hidden_size*2).contiguous().view(batch*N*Q*N, query_len, self.hidden_size*2)
        query_mask = query_mask.view(batch, N*Q, 1, query_len).expand(batch, N*Q, N, query_len).contiguous().view(-1, query_len, 1)

        enhance_support, enhance_query = self.local_matching(support, query, support_mask, query_mask)

        # reduce dimensionality
        enhance_support = self.proj(enhance_support)
        enhance_query = self.proj(enhance_query)
        enhance_support = torch.relu(enhance_support)
        enhance_query = torch.relu(enhance_query)

        # split operation
        enhance_support = enhance_support.view(batch*N*Q*N*K, support_len, self.hidden_size)
        support_mask = support_mask.view(batch*N*Q*N*K, support_len, 1)

        # LSTM
        enhance_support = self.lstm_encoder(enhance_support, support_mask, self.lstm_enhance, cuda)
        enhance_query = self.lstm_encoder(enhance_query, query_mask, self.lstm_enhance, cuda)

        # Local aggregation

        enhance_support, enhance_query = self.local_aggregation(enhance_support, enhance_query, support_mask, query_mask, K)

        # Add support softmax classifier !!
        if training:
            # if not train_all:
            #     enhance_support_copy = enhance_support.detach()
            # else:
            #     enhance_support_copy = enhance_support
            support_logits_copy = self.try_linear(enhance_support.view(batch * N * Q, N, K, -1))
            chunks = support_logits_copy.chunk(batch*N*Q, 0)
            support_cost = Variable(torch.tensor(0.0), requires_grad=True).cuda()
            for ch in chunks:
                support_pred = F.softmax(ch.view(N*K,-1), -1).cuda()
                should_be = torch.tensor([i//K for i in range(N*K)]).cuda()
                support_cost += self.cost(support_pred, should_be)
            #support_cost = support_cost/len(chunks)

        # else:
        #     support_cost = torch.Tensor([0.0])
        # Try get meta repre and add support softmax classifier !!
        # meta_repre = self.get_meta_repre(enhance_support.view(batch*N*Q, N*K, -1)).cuda()
        # meta_pred = F.softmax(self.try_linear(meta_repre), -1)
        # should_be = torch.tensor([i // K for i in range(N * K)]).cuda()
        # support_cost = self.cost(meta_pred, should_be)

        tmp_query = enhance_query.unsqueeze(1).repeat(1, K, 1)
        cat_seq = torch.cat([tmp_query, enhance_support], 2)
        beta = self.multilayer(cat_seq)
        one_enhance_support = (enhance_support.transpose(1, 2) @ F.softmax(beta, 1)).squeeze(2)

        J_incon = torch.sum((one_enhance_support.unsqueeze(1) - enhance_support) ** 2, 2).mean()

        xy = torch.mean(one_enhance_support.view(batch*N*Q, N, -1).unsqueeze(2), 0)[0].expand(N,1,400)
        xx = torch.mean(one_enhance_support.view(batch*N*Q, N, -1).unsqueeze(2), 0)
        yy = torch.mean(enhance_support.view(batch*N*Q, N, K, -1),0)
        res = torch.sum((xx-yy)**2, 2)
        res2 = torch.sum((xy-yy)**2, 2)

        cat_seq = torch.cat([enhance_query, one_enhance_support], 1)
        logits = self.multilayer(cat_seq)

        logits = logits.view(batch*N*Q, N)
        _, pred = torch.max(logits, 1)

        if training:
            return support_cost, logits, pred, J_incon, [res, res2]#torch.mean(J_incon.view(batch*N*Q, N, -1), 0)
        else:
            return 0, logits, pred, J_incon, [res, res2]