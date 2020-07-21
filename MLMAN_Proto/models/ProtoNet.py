import sys

sys.path.append('..')
import torch
from torch import nn
from torch.nn import functional as F
import models.embedding as embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.utils import sort_batch_by_length, init_lstm, init_linear
from torch.autograd import Variable

class ProtoNet(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5, args=None,
                 hidden_size=30, drop=True, N=None, sup_cost=False):
        nn.Module.__init__(self)
        self.word_embedding_dim = word_embedding_dim + 2 * pos_embedding_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim)
        self.args = args
        self.conv = nn.Conv2d(1, self.hidden_size*2, kernel_size=(3, self.word_embedding_dim), padding=(1, 0))
        self.drop = drop
        self.dropout = nn.Dropout(0.2)
        self.cost = nn.MSELoss()
        self.cost1= nn.CrossEntropyLoss()
        self.sup_cost=sup_cost
        if self.sup_cost:
            self.try_linear = nn.Linear(self.hidden_size*2, N)


    def loss(self, distance, label, Q):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        '''
        label = torch.cuda.LongTensor(label[0])
        label = torch.cuda.LongTensor([[x] for x in label])
        # a = torch.zeros(logits.size()).cuda()
        # groundtruth = a.scatter_(1, torch.cuda.LongTensor([[x.cpu()] for x in label]), 1.0)
        fract = 1.0/(int(distance.size()[0])*int(Q))
        #first_part = torch.sum(torch.gather(distance, 0, label)).view(-1)
        first_part = torch.FloatTensor([0]).cuda()
        for i in range(distance.size()[0]):
            first_part+=distance[i][label[i]]
        distance_ = torch.exp((-1)*distance)
        second_part = torch.sum(torch.log(torch.sum(distance_, 1, False)+1e-5))
        loss_tot = fract*(first_part+second_part)

        # print(loss_tot)
        return loss_tot

    def accuracy(self, distance, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        pred = torch.argmin(distance, 1)
        acc = (pred.size()[0]-torch.nonzero(pred-label).size()[0])/pred.size()[0]
        return pred, acc

    def context_encoder(self, input):
        input_mask = (input['mask'] != 0).float()
        #max_length = input_mask.long().sum(1).max().item()
        #input_mask = input_mask[:, :max_length].contiguous()
        embedding_ = self.embedding(input)
        #embedding_ = embedding[:, :max_length].contiguous()

        if self.drop:
            embedding_ = self.dropout(embedding_)

        conv_out = self.conv(embedding_.unsqueeze(1)).squeeze(3)
        conv_out = conv_out * input_mask.unsqueeze(1).contiguous()
        pool = torch.max(conv_out, 2, False)[0]
        # pos1 = input['pos1']
        # pos2 = input['pos2']
        # mask1 = pos1.ge(81).float()
        # mask2 = pos2.ge(81).float()+1
        # posmask = ((mask1+mask2)*input_mask).unsqueeze(1).expand(conv_out.size())
        # x1 = torch.max(conv_out+posmask.eq(1).float()*100, 2, False)[0]-100
        # x2 = torch.max(conv_out + posmask.eq(2).float() * 100, 2, False)[0] - 100
        # x3 = torch.max(conv_out + posmask.eq(3).float() * 100, 2, False)[0] - 100
        # pool = torch.cat([x1, x2, x3], 1)
        return pool

    def forward(self, support, query, N, K, Q,training=False):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''

        support = self.context_encoder(support)
        query = self.context_encoder(query)

        batch = support.size(0)//(N*K)

        support = support.view(batch, N, K, -1)
        query = query.view(batch, N*Q, -1)

        if training and self.sup_cost:
            # if not train_all:
            #     enhance_support_copy = enhance_support.detach()
            # else:
            #     enhance_support_copy = enhance_support
            support_logits_copy = self.try_linear(support.view(batch, N, K, -1))
            chunks = support_logits_copy.chunk(batch, 0)
            support_cost = Variable(torch.tensor(0.0), requires_grad=True).cuda()
            for ch in chunks:
                support_pred = F.softmax(ch.view(N*K,-1), -1).cuda()
                should_be = torch.tensor([i//K for i in range(N*K)]).cuda()
                support_cost += self.cost1(support_pred, should_be)
            #support_cost = support_cost/len(chunks)

        centroids = torch.mean(support, 2, False)
        centroids = centroids.squeeze()
        query = query.squeeze()

        query = query.unsqueeze(1).expand([N*Q, centroids.size()[0], query.size()[1]])
        centroids = centroids.unsqueeze(0).expand(query.size())
        distance = torch.sqrt(torch.sum(((query-centroids)*(query-centroids)), 2, False))


        # distance = []
        # for i in range(N):
        #     tmp = []
        #     for j in range(query.size()[0]):
        #         d = torch.sqrt(torch.sum(torch.square(query[j]-centroids[i]), 0, True))
        #         tmp.append(d)
        #     distance.append(torch.cat(tmp,0).unsqueeze, -1)
        #
        # centroids = F.normalize(centroids, 2, 2)
        # query = F.normalize(query, 2, 2)
        #
        # logits = torch.matmul(query, centroids.transpose(1,2)).view(batch*N*Q, N)
        # _, pred = torch.max(logits, 1)

        if training and self.sup_cost:
            return distance,support_cost
        else:
            return distance, None
