import json
import os
import numpy as np
import random
import torch


class JSONFileDataLoader:
    def _load_preprocessed_file(self):
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        word_vec_name_prefix = '.'.join(self.word_vec_file_name.split('/')[-1].split('.')[:-1])
        processed_data_dir = '_processed_data'
        if not os.path.isdir(processed_data_dir):
            return False
        word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        pos1_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos1.npy')
        pos2_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos2.npy')
        en1_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_en1.npy')
        en2_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_en2.npy')
        mask_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask.npy')
        length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        rel2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2scope.json')
        etype2rel_file_name = os.path.join(processed_data_dir, name_prefix + '_etype2rel.json')
        word_vec_mat_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy')
        word2id_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json')
        if not os.path.exists(word_npy_file_name) or \
           not os.path.exists(pos1_npy_file_name) or \
           not os.path.exists(pos2_npy_file_name) or \
           not os.path.exists(mask_npy_file_name) or \
           not os.path.exists(length_npy_file_name) or \
           not os.path.exists(rel2scope_file_name) or \
           not os.path.exists(word_vec_mat_file_name) or \
           not os.path.exists(word2id_file_name):
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name)
        self.data_en1 = np.load(en1_npy_file_name)
        self.data_en2 = np.load(en2_npy_file_name)
        self.data_pos1 = np.load(pos1_npy_file_name)
        self.data_pos2 = np.load(pos2_npy_file_name)
        self.data_mask = np.load(mask_npy_file_name)
        self.data_length = np.load(length_npy_file_name)
        self.rel2scope = json.load(open(rel2scope_file_name))
        self.etype2rel = json.load(open(etype2rel_file_name))
        self.word_vec_mat = np.load(word_vec_mat_file_name)
        self.word2id = json.load(open(word2id_file_name))
        self.id2word = {}
        self.mistake_collection = {}
        for k in self.rel2scope.keys():
            self.mistake_collection[k] = [1,1,0]
        for k,v in self.word2id.items():
            self.id2word[v] = k
        if self.data_word.shape[1] != self.max_length:
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def __init__(self, file_name, word_vec_file_name, max_length=40, case_sensitive=False, reprocess=False, change_after_episodes=1):
        '''
        file_name: Json file storing the data in the following format
            {
                "P155": # relation id
                    [
                        {
                            "h": ["song for a future generation", "Q7561099", [[16, 17, ...]]], # head entity [word, id, location]
                            "t": ["whammy kiss", "Q7990594", [[11, 12]]], # tail entity [word, id, location]
                            "tokens": ["Hot", "Dance", "Club", ...], # sentence
                        },
                        ...
                    ],
                "P177": 
                    [
                        ...
                    ]
                ...
            }
        word_vec_file_name: Json file storing word vectors in the following format
            [
                {'word': 'the', 'vec': [0.418, 0.24968, ...]},
                {'word': ',', 'vec': [0.013441, 0.23682, ...]},
                ...
            ]
        max_length: The length that all the sentences need to be extend to.
        case_sensitive: Whether the data processing is case-sensitive, default as False.
        reprocess: Do the pre-processing whether there exist pre-processed files, default as False.
        '''
        self.file_name = file_name
        self.word_vec_file_name = word_vec_file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        self.change_after_episodes = change_after_episodes
        self.cur_nochange_episodes = 0
        self.cur_cls = []

        if reprocess or not self._load_preprocessed_file(): # Try to load pre-processed files:
            # Check files
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")
            if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
                raise Exception("[ERROR] Word vector file doesn't exist")

            # Load files
            print("Loading data file...")
            self.ori_data = json.load(open(self.file_name, "r"))
            print("Finish loading")
            print("Loading word vector file...")
            self.ori_word_vec = json.load(open(self.word_vec_file_name, "r"))
            print("Finish loading")
            
            # Eliminate case sensitive
            if not case_sensitive:
                print("Elimiating case sensitive problem...")
                for relation in self.ori_data:
                    for ins in self.ori_data[relation]:
                        for i in range(len(ins['tokens'])):
                            ins['tokens'][i] = ins['tokens'][i].lower()
                print("Finish eliminating")

            # Pre-process word vec
            self.word2id = {}
            self.word_vec_tot = len(self.ori_word_vec)
            UNK = self.word_vec_tot
            BLANK = self.word_vec_tot + 1
            self.word_vec_dim = len(self.ori_word_vec[0]['vec'])
            print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
            print("Building word vector matrix and mapping...")
            self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
            for cur_id, word in enumerate(self.ori_word_vec):
                w = word['word']
                if not case_sensitive:
                    w = w.lower()
                self.word2id[w] = cur_id
                self.word_vec_mat[cur_id, :] = word['vec']
                self.word_vec_mat[cur_id] = self.word_vec_mat[cur_id] / np.sqrt(np.sum(self.word_vec_mat[cur_id] ** 2))
            self.word2id['UNK'] = UNK
            self.word2id['BLANK'] = BLANK

            self.id2word = {}
            for k, v in self.word2id.items():
                self.id2word[v] = k
            print("Finish building")

            # Pre-process data
            print("Pre-processing data...")
            self.instance_tot = 0
            for relation in self.ori_data:
                self.instance_tot += len(self.ori_data[relation])
            self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_en1 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_en2 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos1 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos2 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_mask = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
            self.rel2scope = {} # left closed and right open
            self.etype2rel = {}
            i = 0
            for relation in self.ori_data:
                # if relation == 'X-NA-X':
                #     continue
                # if relation != 'NA':
                #     e1, r, e2 = relation.split('-')
                #     e = e1+'-'+e2
                #     if e not in self.etype2rel:
                #         self.etype2rel[e] = []
                #     self.etype2rel[e].append(relation)
                self.rel2scope[relation] = [i, i]
                for ins in self.ori_data[relation]:
                    pos1 = ins['h'][2][0][0]
                    pos2 = ins['t'][2][0][0]
                    en1 = ins['h'][0].split()
                    en2 = ins['t'][0].split()
                    words = ins['tokens']
                    cur_ref_data_word = self.data_word[i]         
                    for j, word in enumerate(words):
                        if j < max_length:
                            if word in self.word2id:
                                cur_ref_data_word[j] = self.word2id[word]
                            else:
                                cur_ref_data_word[j] = UNK
                        else:
                            break
                    for j in range(j + 1, max_length):
                        cur_ref_data_word[j] = BLANK
                    self.data_length[i] = len(words)
                    if len(words) > max_length:
                        self.data_length[i] = max_length

                    cur_ref_en1_word = self.data_en1[i]
                    for j, word in enumerate(en1):
                        if word in self.word2id:
                            cur_ref_en1_word[j] = self.word2id[word]
                        else:
                            cur_ref_en1_word[j] = UNK

                    cur_ref_en2_word = self.data_en2[i]
                    for j, word in enumerate(en2):
                        if word in self.word2id:
                            cur_ref_en2_word[j] = self.word2id[word]
                        else:
                            cur_ref_en2_word[j] = UNK

                    if pos1 >= max_length:
                        pos1 = max_length - 1
                    if pos2 >= max_length:
                        pos2 = max_length - 1
                    pos_min = min(pos1, pos2)
                    pos_max = max(pos1, pos2)
                    for j in range(max_length):
                        self.data_pos1[i][j] = j - pos1 + max_length
                        self.data_pos2[i][j] = j - pos2 + max_length
                        if j >= self.data_length[i]:
                            self.data_mask[i][j] = 0
                            self.data_pos1[i][j] = 0
                            self.data_pos2[i][j] = 0
                        elif j <= pos_min:
                            self.data_mask[i][j] = 1
                        elif j <= pos_max:
                            self.data_mask[i][j] = 2
                        else:
                            self.data_mask[i][j] = 3
                    i += 1
                self.rel2scope[relation][1] = i

            self.mistake_collection = {}
            for k in self.rel2scope.keys():
                self.mistake_collection[k] = [1, 1, 0]

            print(self.mistake_collection)

            print("Finish pre-processing")
            print("Storing processed files...")
            name_prefix = '.'.join(file_name.split('/')[-1].split('.')[:-1])
            word_vec_name_prefix = '.'.join(word_vec_file_name.split('/')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)
            np.save(os.path.join(processed_data_dir, name_prefix + '_word.npy'), self.data_word)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos1.npy'), self.data_pos1)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos2.npy'), self.data_pos2)
            np.save(os.path.join(processed_data_dir, name_prefix + '_en1.npy'), self.data_en1)
            np.save(os.path.join(processed_data_dir, name_prefix + '_en2.npy'), self.data_en2)
            np.save(os.path.join(processed_data_dir, name_prefix + '_mask.npy'), self.data_mask)
            np.save(os.path.join(processed_data_dir, name_prefix + '_length.npy'), self.data_length)
            json.dump(self.rel2scope, open(os.path.join(processed_data_dir, name_prefix + '_rel2scope.json'), 'w'))
            json.dump(self.etype2rel, open(os.path.join(processed_data_dir, name_prefix + '_etype2rel.json'), 'w'))
            np.save(os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy'), self.word_vec_mat)
            json.dump(self.word2id, open(os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json'), 'w'))

            print("Finish storing")


    def get_origin_sentence(self, word, mask):
        sentence = []
        for i in range(len(word)):
            w = word[i]
            m = mask[i]
            last_mask = -1
            s = ''
            for j in range(len(w)):
                char = w[j]
                thismask = m[j]
                if thismask != last_mask:
                    s += ' ||'
                last_mask = thismask
                new_char = self.id2word[char]
                if new_char == 'BLANK':
                    break
                s += (' '+new_char)
            sentence.append(s)
        return sentence

    def get_origin_entity(self, word):
        sentence = []
        for i in range(len(word)):
            w = word[i]
            s = ''
            for j in range(len(w)):
                char = w[j]
                new_char = self.id2word[char]
                if char == 0:
                    break
                s += new_char
            sentence.append(s)
        return sentence

    def get_hard_classes(self, N):
        if len(self.mistake_collection) == N:
            return [k for k in self.mistake_collection.keys()]
        mul = 1
        min_w = min([v[0]/v[1] for v in self.mistake_collection.values()])
        while min_w*mul<0.001:
            mul *= 10
        error_rates = {k:v[0]*mul/v[1]+v[2]*0.01 + 0.1*(self.isEng(k.split('-')[1]) and k.split('-')[1][-1]!='X') for k,v in self.mistake_collection.items()}
        target_classes = []
        # ks = [k for k in error_rates.keys()]
        while len(target_classes) < N:
            error_rates = {k:v for k, v in error_rates.items() if k not in target_classes}
            total = sum([v for v in error_rates.values()])
            rand_uni = random.uniform(0, total)
            sum_up = 0
            for k,v in error_rates.items():
                sum_up += v
                if rand_uni <= sum_up:
                    target_classes.append(k)
                    break
            # rand_cls = random.choice(ks)
            # while rand_cls in target_classes:
            #     rand_cls = random.choice(ks)
            # if random.uniform(0,1) < error_rates[rand_cls]:
            #     target_classes.append(rand_cls)
        return target_classes

    def next_one(self, N=5, K=5, Q=100, start=-1,same_etype=False, choose_hard = False):
        change_task = False
        if self.cur_nochange_episodes == 0:
            change_task = True
            tmplist = [x for x in self.etype2rel.keys() if len(self.etype2rel[x]) >= N]
            if same_etype:
                target_etype = tmplist[random.randint(0, len(tmplist)-1)]
                target_classes = sorted(random.sample(self.etype2rel[target_etype], N))
            elif choose_hard:
                target_classes = self.get_hard_classes(N)
            else:
                # target_classes = []
                # for x in tmplist:
                #     target_classes += self.etype2rel[x]
                # target_classes = sorted(random.sample(target_classes, N))
                target_classes = sorted(random.sample(self.rel2scope.keys(), N))
            self.cur_cls = target_classes
        else:
            target_classes = self.cur_cls
        self.cur_nochange_episodes = (self.change_after_episodes+1)%self.change_after_episodes
        #print(target_classes)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'sentence': [],'en1':[], 'en2':[]}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'sentence': [], 'en1':[], 'en2':[]}
        query_label = []
        label_reference = {}

        for i, class_name in enumerate(target_classes):
            label_reference[i] = class_name
            scope = self.rel2scope[class_name]
            if start == -1:
                indices = np.random.choice(list(range(scope[0], scope[1])), K + Q, False)
            else:
                indices = np.array(list(range(scope[0], scope[1]))[(K+Q)*start: (K+Q)*(start+1)])
            word = self.data_word[indices]
            pos1 = self.data_pos1[indices]
            pos2 = self.data_pos2[indices]
            en1 = self.data_en1[indices]
            en2 = self.data_en2[indices]
            mask = self.data_mask[indices]

            support_word, query_word = np.split(word, [K])
            support_pos1, query_pos1 = np.split(pos1, [K])
            support_pos2, query_pos2 = np.split(pos2, [K])
            support_en1, query_en1 = np.split(en1, [K])
            support_en2, query_en2 = np.split(en2, [K])
            support_mask, query_mask = np.split(mask, [K])

            support_sentence = self.get_origin_sentence(support_word, support_mask)
            query_sentence = self.get_origin_sentence(query_word, query_mask)
            support_en1 = self.get_origin_entity(support_en1)
            support_en2 = self.get_origin_entity(support_en2)
            query_en1 = self.get_origin_entity(query_en1)
            query_en2 = self.get_origin_entity(query_en2)

            support_set['word'].append(support_word)
            support_set['pos1'].append(support_pos1)
            support_set['pos2'].append(support_pos2)
            support_set['mask'].append(support_mask)
            support_set['sentence'].append(support_sentence)
            support_set['en1'].append(support_en1)
            support_set['en2'].append(support_en2)
            query_set['word'].append(query_word)
            query_set['pos1'].append(query_pos1)
            query_set['pos2'].append(query_pos2)
            query_set['mask'].append(query_mask)
            query_set['sentence'].append(query_sentence)
            query_set['en1'].append(query_en1)
            query_set['en2'].append(query_en2)
            query_label += [i] * Q

        support_set['word'] = np.stack(support_set['word'], 0)
        support_set['pos1'] = np.stack(support_set['pos1'], 0)
        support_set['pos2'] = np.stack(support_set['pos2'], 0)
        support_set['mask'] = np.stack(support_set['mask'], 0)
        query_set['word'] = np.concatenate(query_set['word'], 0)
        query_set['pos1'] = np.concatenate(query_set['pos1'], 0)
        query_set['pos2'] = np.concatenate(query_set['pos2'], 0)
        query_set['mask'] = np.concatenate(query_set['mask'], 0)
        query_label = np.array(query_label)

        # if self.change_after_episodes == 1:
        #     change_task = False
        return support_set, query_set, query_label, label_reference, change_task

    def next_batch(self, B=4, N=20, K=5, Q=100, start=-1, same_etype=False, choose_hard=False, cuda=True):
        assert start ==-1 or B==1
        support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'sentence': [], 'en1':[], 'en2':[]}
        query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'sentence': [], 'en1':[], 'en2':[]}
        label = []
        label_ref = []
        for one_sample in range(B):
            current_support, current_query, current_label, label_reference, change_task = self.next_one(N, K, Q, start, same_etype, choose_hard)
            support['word'].append(current_support['word'])
            support['pos1'].append(current_support['pos1'])
            support['pos2'].append(current_support['pos2'])
            support['mask'].append(current_support['mask'])
            support['sentence'].append(current_support['sentence'])
            support['en1'].append(current_support['en1'])
            support['en2'].append(current_support['en2'])
            query['word'].append(current_query['word'])
            query['pos1'].append(current_query['pos1'])
            query['pos2'].append(current_query['pos2'])
            query['mask'].append(current_query['mask'])
            query['sentence'].append(current_query['sentence'])
            query['en1'].append(current_query['en1'])
            query['en2'].append(current_query['en2'])
            label.append(current_label)
            label_ref.append(label_reference)

        if support['word'][0].size !=0:
            support['word'] = torch.from_numpy(np.stack(support['word'], 0)).long().view(-1, self.max_length)
            support['pos1'] = torch.from_numpy(np.stack(support['pos1'], 0)).long().view(-1, self.max_length)
            support['pos2'] = torch.from_numpy(np.stack(support['pos2'], 0)).long().view(-1, self.max_length)
            support['mask'] = torch.from_numpy(np.stack(support['mask'], 0)).long().view(-1, self.max_length)
        else:
            support = None
        if query['word'][0].size != 0:
            query['word'] = torch.from_numpy(np.stack(query['word'], 0)).long().view(-1, self.max_length)
            query['pos1'] = torch.from_numpy(np.stack(query['pos1'], 0)).long().view(-1, self.max_length)
            query['pos2'] = torch.from_numpy(np.stack(query['pos2'], 0)).long().view(-1, self.max_length)
            query['mask'] = torch.from_numpy(np.stack(query['mask'], 0)).long().view(-1, self.max_length)
            label = torch.from_numpy(np.stack(label, 0).astype(np.int64)).long()
        else:
            query = None
            label = None

        if cuda and support:
            for key in support:
                if key == 'sentence' or key=='en1' or key=='en2':
                    continue
                support[key] = support[key].cuda()
        if cuda and query:
            for key in query:
                if key == 'sentence'or key=='en1' or key=='en2':
                    continue
                query[key] = query[key].cuda()
            label = label.cuda()


        return support, query, label, label_ref, change_task

    def update_mistake_collection(self, mistakes):
        for k, v in self.mistake_collection.items():
            if k in mistakes.keys():
                self.mistake_collection[k][0] += float(mistakes[k].cpu())
                self.mistake_collection[k][1] += 1
                self.mistake_collection[k][2] = 0
            else:
                self.mistake_collection[k][2] += 1


    def isEng(self, w):
        for c in w:
            if ord(c) > 255:
                return False
        return True


    def check_mistakes(self):
        mul = 1
        min_w = min([v[0]/v[1] for v in self.mistake_collection.values()])
        while min_w*mul<0.001:
            mul *= 10
        error_rates = {k:v[0]*mul/v[1]+v[2]*0.01 + 0.1*(self.isEng(k.split('-')[1]) and k.split('-')[1][-1]!='X') for k,v in self.mistake_collection.items()}
        for k,v in self.mistake_collection.items():
            print(k,v, error_rates[k])
