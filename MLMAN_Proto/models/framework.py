import os
import torch
from torch import optim
import pynvml
import sys
import time

class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, deploy_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.deploy_data_loader = deploy_data_loader
        pynvml.nvmlInit()
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt, map_location='cuda:0')
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    

    def train(self, model, model_name, B=4, N_for_train=20, N_for_eval=5, K_for_train=5, K_for_eval=5, Q=100,
              ckpt_dir='./checkpoint', learning_rate=1e-1, lr_step_size=20000,
              weight_decay=1e-5, train_iter=30000, val_iter=1000, val_step=2000,
              test_iter=3000, pretrain_model=None, optimizer=optim.SGD, use_sup_classifier=False,
              gpu_idx=0):
        '''
        model: model
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        test_result_dir: Directory of test results
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path
        '''

        # Init
        print(use_sup_classifier)
        print(model_name)
        wfile = open(model_name, 'a')
        #parameters_to_optimize = filter(lambda x:x.requires_grad, model.parameters())
        parameters_to_optimize_classifier = [v for k,v in model.named_parameters() if k.find("try_linear")>=0]
        parameters_to_optimize_encoder = [v for k,v in model.named_parameters() if k.find("try_linear")<0]
        optimizer_classifier = optimizer(parameters_to_optimize_classifier, learning_rate)
        optimizer_encoder = optimizer(parameters_to_optimize_encoder, learning_rate, weight_decay=weight_decay)
        optimizer= optimizer(model.parameters(), learning_rate, weight_decay=weight_decay)
        scheduler_classifier = optim.lr_scheduler.StepLR(optimizer_classifier, step_size=lr_step_size)
        scheduler_encoder = optim.lr_scheduler.StepLR(optimizer_encoder, step_size=lr_step_size)
        scheduler = optim.lr_scheduler.StepLR(optimizer_encoder, step_size=lr_step_size)
        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            model.load_state_dict(checkpoint['state_dict'])
            #start_iter = checkpoint['iter'] + 1
            start_iter = 0
        else:
            start_iter = 0

        #allloss = 0

        print(sys.getsizeof(model))
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        model.train()

        # Training
        best_acc = 0
        same_etype = False
        #self.train_data_loader.check_mistakes()
        for it in range(start_iter, start_iter + train_iter):
            #torch.cuda.empty_cache()
            handle=pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # if meminfo.used > 10000000000:
            #     torch.cuda.empty_cache()
            scheduler_classifier.step()
            scheduler_encoder.step()
            scheduler.step()
            support, query, label, label_ref, change_task = self.train_data_loader.next_batch(B, N_for_train, K_for_train, Q, same_etype=same_etype, choose_hard=False)

            for update_spe_and_gen in range(2):
                supcost, logits, pred, dist, _ = model(support, query, N_for_train, K_for_train, Q, training=True, change_task=change_task)

                right, errors_per_cls = model.accuracy(pred, label, Q)
                # print(right)

                mistakes = {}
                for i in range(len(errors_per_cls)):
                    mistakes[label_ref[0][i]] = errors_per_cls[i]
                self.train_data_loader.update_mistake_collection(mistakes)

                loss = model.loss(logits, label)
                if not use_sup_classifier:
                    allloss = loss + dist
                    optimizer_encoder.zero_grad()
                    allloss.backward()
                    optimizer_encoder.step()
                    break
                else:
                    if update_spe_and_gen == 0:
                        # for k, v in model.named_parameters():
                        #     if k.find("conv") >=0:
                        #         x= 1
                        #         break
                        cur_loss = (loss+dist+supcost)
                        optimizer_classifier.zero_grad()
                        #cur_loss.backward(retain_graph=True)
                        cur_loss.backward()
                        optimizer_classifier.step()
                        #allloss += cur_loss
                        if (it%3) != 2:
                            break
                    else:
                        # for k, v in model.named_parameters():
                        #     if k.find("conv") >=0:
                        #         x= 1
                        #         break
                        #allloss += loss+dist+supcost
                        allloss = loss+dist+supcost
                        optimizer_encoder.zero_grad()
                        #allloss.backward(retain_graph=True)
                        allloss.backward()
                        optimizer_encoder.step()
                        #allloss = 0

            if (it + 1) % val_step == 0:
               # same_etype = not same_etype
                with torch.no_grad():
                    acc, _, _, _, _, _ = self.eval(model, 1, N_for_eval, K_for_eval, 5, val_iter, gpu_idx=gpu_idx)
                    print("{0:}---{1:}-way-{2:}-shot test   Test accuracy: {3:3.2f}".format(it, N_for_eval, K_for_eval, acc*100))
                    wfile.write(str(time.time()))
                    wfile.write(str(it)+'\t'+str(acc*100))
                    if acc > best_acc:
                        # log_file = open('test_results.txt', 'w')
                        # for i in range(len(sentences_log)):
                        #     for j in range(len(sentences_log[i])):
                        #         log_file.write(labels_log[i][j]+'\t'+preds_log[i][j]+'\n'+sentences_log[i][j]+'\n')
                        #     log_file.write('\n')
                        # log_file.close()
                        # print('Test result saved in test_results.txt')

                        print('Best checkpoint')
                        wfile.write("Best checkpoint")
                        if not os.path.exists(ckpt_dir):
                            os.makedirs(ckpt_dir)
                        save_path = os.path.join(ckpt_dir, model_name + ".pth.tar")
                        if torch.cuda.device_count() > 1:
                            torch.save({'state_dict': model.module.state_dict()}, save_path)
                        else:
                            torch.save({'state_dict': model.state_dict()}, save_path)
                        best_acc = acc
              #  self.train_data_loader.check_mistakes()
                model.train()

        print("\n####################\n")
        print("Finish training " + model_name)
        with torch.no_grad():
            test_acc, _, _, _, _, _ = self.eval(model, 5, N_for_eval, K_for_test, 5, test_iter, ckpt=os.path.join(ckpt_dir, model_name + '.pth.tar'), gpu_idx=gpu_idx)
            print("{0:}-way-{1:}-shot test   Test accuracy: {2:3.2f}".format(N_for_eval, K, test_acc*100))


    def test(self,
            model,
            N_for_eval, K,
            val_iter,
            ckpt=None, model_name=None, gpu_idx=0):
        with torch.no_grad():
                # model = model.cuda()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                # if meminfo.used > 10000000000:
                #     torch.cuda.empty_cache()
                acc, sentences_log, labels_log, preds_log, support_sen, logits_log = self.eval(model, 1, N_for_eval, K, 30, val_iter, ckpt)
                print("{0:}---{1:}-way-{2:}-shot test   Test accuracy: {3:3.2f}".format(-1, N_for_eval, K, acc * 100))

                wfile = open('testResultshahaha.txt', 'a')
                wfile.write(str(model_name)+'\t'+str(acc)+'\n')
                wfile.close()

                save_path = os.path.join('deploy_results', 'deploy_results_' + model_name + ".txt")
                log_file = open(save_path, 'w')
                log_file.write(str(acc)+'\n')
                for i in range(len(sentences_log)):
                    for j in range(len(support_sen[i])):
                        log_file.write(support_sen[i][j]+'\n')
                    log_file.write('\n')
                    for j in range(len(sentences_log[i])):
                        log_file.write(
                                labels_log[i][j] + '\t' + preds_log[i][j] + '\n' + logits_log[i][j] + '\n' + sentences_log[i][j] + '\n')
                    log_file.write('\n')
                log_file.close()
                print('Test result saved in ' + save_path)



    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            ckpt=None, gpu_idx=0):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print(ckpt)
        cuda = True
        if ckpt is None: # train
            eval_dataset = self.val_data_loader
           # cuda = True
        else: # test
            checkpoint = self.__load_model__(ckpt)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            eval_dataset = self.test_data_loader
            #cuda = False
        model.eval()
        model.cuda()

        iter_right = 0.0
        iter_sample = 0.0
        sentences_log = []
        labels_log = []
        preds_log = []
        logits_log = []
        support_sentences_log = []
        for it in range(eval_iter):
            # handle=pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # if meminfo.used > 10000000000:
            #     torch.cuda.empty_cache()
            support, query, label, label_ref, change_task = eval_dataset.next_batch(B, N, K, Q, cuda=cuda)
            _, logits, pred, _, _ = model(support, query, N, K, Q, training=False, change_task=change_task, cuda=cuda)
            right, _ = model.accuracy(pred, label, Q)
            iter_right += right.item()
            iter_sample += 1

            # Log
            support_sentences = []
            for x in support['sentence']:
                label_idx = 0
                for y in x:
                    for z in y:
                        support_sentences.append(label_ref[0][label_idx]+'\t'+z)
                    label_idx += 1


            sentences = []
            for x in query['sentence']:
                for y in x:
                    sentences += y

            labels = []
            for x in range(len(label)):
                for y in label[x]:
                    labels.append(label_ref[x][int(y.cpu())])

            logits_tmp = []
            for xx in range(len(logits)):
                x = logits[xx]
                this_log = ""
                for y in range(len(x)):
                    this_log += label_ref[xx//(len(logits)//B)][y] + ' ' + str(float(x[y].cpu())) + '\t'
                logits_tmp.append(this_log)

            preds = []
            for x in range(len(pred)):
                preds.append(label_ref[x//(len(pred)//B)][int(pred[x].cpu())])

            sentences_log.append(sentences)
            labels_log.append(labels)
            preds_log.append(preds)
            logits_log.append(logits_tmp)
            support_sentences_log.append(support_sentences)

        return iter_right / iter_sample, sentences_log, labels_log, preds_log, support_sentences_log, logits_log


    def deploy(self,
            model,
            N, Q,
            ckpt=None, model_name=None):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        #torch.cuda.empty_cache()
        checkpoint = self.__load_model__(ckpt)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        support_dataset = self.test_data_loader
        eval_dataset = self.deploy_data_loader
        model = model.cuda()
        model.eval()

        sentences_log = []
        preds_log = []
        en1s_log = []
        en2s_log = []
        start = 0
        with torch.no_grad():
            for it in range(20000):
                #print(it)
                #torch.cuda.empty_cache()
                support, a, b, c, _ = support_dataset.next_batch(1, N, 15, 0)
                _, query, _, label_ref, _ = eval_dataset.next_batch(1, N, 0, Q)
                start += 1
                _, _, pred, _, _ = model(support, query, N, 15, Q, training=False)

                # Log
                sentences = []
                for x in query['sentence']:
                    for y in x:
                        sentences += y

                en1s = []
                for x in query['en1']:
                    for y in x:
                        en1s += y
                en2s = []
                for x in query['en2']:
                    for y in x:
                        en2s += y

                preds = []
                for x in range(len(pred)):
                    preds.append(label_ref[x//len(pred)][int(pred[x].cpu())])

                sentences_log.append(sentences)
                preds_log.append(preds)
                en1s_log.append(en1s)
                en2s_log.append(en2s)

        save_path = os.path.join('deploy_results', 'results_'+model_name+'.txt')
        log_file = open(save_path, 'w')
        for i in range(len(sentences_log)):
            for j in range(len(sentences_log[i])):
                log_file.write(preds_log[i][j] + '\t' + en1s_log[i][j] + '\t'+ en2s_log[i][j]+ '\t' + sentences_log[i][j] + '\n')
        log_file.close()
        print('Deploy result saved in ' + save_path)

        return sentences_log, preds_log


    def check_distance(self,
            model,
            N,
            ckpt=None):
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            checkpoint = self.__load_model__(ckpt)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            eval_dataset = self.test_data_loader
        model.cuda()
        model.eval()

        support_sentences_log = []
        start = 0
        with torch.no_grad():
            #torch.cuda.empty_cache()
            support, _, label, label_ref = eval_dataset.next_batch(1, N, 20, 0, start)
            start += 1
            _, _, _, _, distance = model(support, support, N, 20, 20)
            # Log
            support_sentences = []
            for x in support['sentence']:
                for y in x:
                    support_sentences += y
            support_sentences_log.append(support_sentences)
            #
            # labels = []
            # for x in range(len(label)):
            #     for y in label[x]:
            #         labels.append(label_ref[x][int(y.cpu())])

            l = len(support_sentences)//5
            distance, fake_distance = distance
            for i in range(N):
                sentences = support_sentences[i*l:(i+1)*l]
                d = distance[i]
                dd = fake_distance[i]
                ds, ranks = torch.sort(d)
                for j in range(len(sentences)):
                    print(sentences[ranks[j]], ds[j], dd[ranks[j]])
                print('\n')
