import random
import torch
import numpy as np
# np.set_printoptions(threshold=np.nan)
torch.set_printoptions(precision=8)

from models.data_loader import JSONFileDataLoader
from models.framework import FewShotREFramework
from models.MLMAN import MLMAN as MLMAN
from models.ProtoNet import ProtoNet as ProtoNet

seed = int(np.random.uniform(0,1)*10000000)
#seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
print('seed: ', seed)
import argparse

parser = argparse.ArgumentParser(description='Multi-Level Matching and Aggregation Network for Few-Shot Relation Classification')
parser.add_argument('--model_name', type=str, default='MLMAN', help='Model name')
parser.add_argument('--N_for_train', type=int, default=20, help='Num of classes for each batch for training')
parser.add_argument('--N_for_test', type=int, default=5, help='Num of classes for each batch for test')
parser.add_argument('--K_for_train', type=int, default=1, help='Num of instances for each class in the support set')
parser.add_argument('--K_for_test', type=int, default=1, help='Num of instances for each class in the support set')
parser.add_argument('--Q', type=int, default=5, help='Num of instances for each class in the query set')
parser.add_argument('--batch', type=int, default=1, help='batch size')
parser.add_argument('--max_length', type=int, default=40, help='max length of sentence')
parser.add_argument('--learning_rate', type=float, default=1e-1, help='initial learning rate')
parser.add_argument('--train_or_test', type=str, default='train', help='Train or test')
parser.add_argument('--pretrain', type=str, default=None, help='Pretrained model')
parser.add_argument('--dataset', type=str, default='xx', help='Dataset name')
parser.add_argument('--use_sup_cost', type=int, default=0, help='Use support cost or not')
parser.add_argument('--gpu_idx', type=int, default=0, help='Gpu idx')
parser.add_argument('--language', type=str, default='eng', help='language')


args = parser.parse_args()
print('setting:')
print(args)

if args.train_or_test == 'train':
    seed=0

print("{}-way(train)-{}-way(test)-{}-shot(train)-{}-shot(test) with batch {} Few-Shot Relation Classification".format(args.N_for_train, args.N_for_test, args.K_for_train, args.K_for_test, args.Q))
print("Model: {}".format(args.model_name))

max_length = args.max_length
#./data/glove.6B.50d.json
#./data/health_meta/word_vec.json
print('./data/' + args.dataset+'/train.json')

if args.language == 'eng':
    wordvec = "./data/glove.6B.50d.json"
    word_dim=50
else:
    wordvec = "./data/health_meta/word_vec.json"
    word_dim=100
if args.use_sup_cost:
    print('hhahahaha')
    train_data_loader = JSONFileDataLoader('./data/' + args.dataset+'/train.json', wordvec, max_length=max_length, reprocess=False,
                                       change_after_episodes=1)
else:
    train_data_loader = JSONFileDataLoader('./data/' + args.dataset+'/train.json', wordvec, max_length=max_length, reprocess=False)
val_data_loader = JSONFileDataLoader('./data/' + args.dataset+'/test.json', wordvec, max_length=max_length, reprocess=False)
deploy_data_loader = JSONFileDataLoader('./data/' + args.dataset+'/deploy.json', wordvec, max_length=max_length, reprocess=False)


framework = FewShotREFramework(train_data_loader, val_data_loader, val_data_loader, deploy_data_loader)

model = MLMAN(train_data_loader.word_vec_mat, max_length, word_embedding_dim=word_dim, hidden_size=100, args=args, N=args.N_for_train)
model_name = args.model_name + str(seed)

print(model_name)

if args.train_or_test == 'train':
    pretrain_model = None
    if args.pretrain:
        pretrain_model = './checkpoint/' + args.pretrain +'.pth.tar'
    framework.train(model, model_name, args.batch, N_for_train=args.N_for_train,  N_for_eval=args.N_for_test,
                K_for_train=args.K_for_train, K_for_eval=args.K_for_test, Q=args.Q,  learning_rate=args.learning_rate,
                train_iter=40000, val_iter=1000, val_step=2000, test_iter=2000, pretrain_model=pretrain_model,
                    use_sup_classifier=bool(args.use_sup_cost), gpu_idx=args.gpu_idx)
    # framework.test(model, K=args.K, N_for_eval=args.N_for_test, val_iter=1000,
    #                ckpt='./checkpoint/'+model_name+'.pth.tar', model_name = model_name, gpu_idx=args.gpu_idx)
    # torch.cuda.empty_cache()
    # framework.deploy(model, 5, Q=20,
    #                ckpt='./checkpoint/'+model_name+'.pth.tar', model_name=model_name, use_sup_classifier=bool(args.use_sup_cost))
elif args.train_or_test == 'test':
    framework.test(model, K=args.K_for_test, N_for_eval=args.N_for_test, val_iter=1000,
                   ckpt='./checkpoint/' + args.pretrain +'.pth.tar', model_name = model_name)
elif args.train_or_test == 'deploy':
    framework.deploy(model, args.N_for_test, Q=5,
                   ckpt='./checkpoint/' + args.pretrain +'.pth.tar', model_name=model_name)
    #framework.test(model, K=args.K, N_for_eval=args.N_for_test, val_iter=1000, ckpt='./checkpoint/MLMANFewshotSmall3642153.pth.tar', model_name = model_name)

else:
    framework.check_distance(model, 5, ckpt='./checkpoint/MLMAN6992288.pth.tar')
