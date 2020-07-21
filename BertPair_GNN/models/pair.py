import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Pair(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, hidden_size=230):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.sm = nn.Softmax()
        self.try_linear = nn.Linear(768, 2)

    def forward(self, batch, N, K, total_Q, support_batch, training=True):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        logits,_ = self.sentence_encoder(batch)
        logits = logits.view(-1, total_Q, N, K, 2)
        logits = logits.mean(3) # (-1, total_Q, N, 2)
        logits_na, _ = logits[:, :, :, 0].min(2, keepdim=True) # (-1, totalQ, 1)
        logits = logits[:, : , :, 1] # (-1, total_Q, N)
        logits = torch.cat([logits, logits_na], 2) # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N+1), 1)

        if training:
            _, states = self.sentence_encoder(support_batch)
            sup_logits = self.sm(self.try_linear(torch.mean(states, 1).squeeze()))
            #sup_logits = self.sm(sup_logits)
            all_zeros = torch.zeros(len(support_batch['word'])).long().cuda()
            return logits, pred, sup_logits, all_zeros
        else:
            return logits, pred, None, None