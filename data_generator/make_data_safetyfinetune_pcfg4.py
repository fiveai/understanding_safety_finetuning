import pickle 
import numpy as np
import argparse
import os
import time
import random
import torch
from torch.utils.data.dataloader import DataLoader

parser = argparse.ArgumentParser(description='PyTorch DGP')
parser.add_argument('--BOS_token', default="$", type=str)
parser.add_argument('--EOS_token', default="%", type=str)
parser.add_argument('--SOT_token', default="*", type=str)
parser.add_argument('--pad_token', default='#', type=str)
parser.add_argument('--cap_token1', type=str, default='(')
parser.add_argument('--cap_token2', type=str, default=')')
parser.add_argument('--cap_token3', type=str, default='{')
parser.add_argument('--cap_token4', type=str, default='}')
parser.add_argument('--EOT_token', default=">", type=str)
parser.add_argument('--stop_token', default="*^", type=str)
parser.add_argument('--min_input_length', default=25, type=int)
parser.add_argument('--max_input_length', default=35, type=int)
parser.add_argument('--identity_sample_prob', default=0.5, type=float)
parser.add_argument('--num_alphabets', default=30, type=int)
parser.add_argument('--num_cap', default=4, type=int)
parser.add_argument('--start_random', default=150, type=int)

parser.add_argument('--safe', default=False, type=bool)
parser.add_argument('--unsafe', default=False, type=bool)
parser.add_argument('--intermediate', default=False, type=bool)
parser.add_argument('--unsafe_id_mg', default=False, type=bool)
parser.add_argument('--unsafe_ood_mg', default=False, type=bool)

parser.add_argument('--from_unsafe_branch', default=False, type=bool)


parser.add_argument('--make_safe_data', default=True, type=bool)
parser.add_argument('--make_unsafe_data', default=False, type=bool)
parser.add_argument('--make_normal_data', default=False, type=bool)
parser.add_argument('--make_data_no_dup', default=False, type=bool)
parser.add_argument('--only_duplicates', default=False, type=bool)
parser.add_argument('--make_safe_data_unsafe_pcfg', default=False, type=bool)

parser.add_argument('--max_window_possible', default=150, type=int)
parser.add_argument('--train_data_path', type=str, default='./saved_data/safety_finetune_dataset_train_pcfg4.pkl')
parser.add_argument('--test_data_path', type=str, default='./saved_data/safety_finetune_dataset_test_pcfg4.pkl')
parser.add_argument('--batch_size', default=1000, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--sample_pcfg_number', default=1, type=int)
parser.add_argument('--num_samples_train', default=750000, type=int)
parser.add_argument('--is_train', default=0, type=int)
parser.add_argument('--num_samples_test', default=100000, type=int)
parser.add_argument('--num_repeats', default=6, type=int)

args = parser.parse_args()
if not os.path.exists("saved_data"):
    os.makedirs("saved_data")

class DGP_sample():
    def __init__(self, args, split='train', seed_lst=[], num_samples=100000, rules_pass=[],lst_choices=[], sample_prob=[], leaf_nodes=[], grammar_NT_T=[], rules={}):

        # Initialize of the data generating process.

        self.args = args
        self.rules = rules
        self.num_samples = num_samples
        self.num_repeats = args.num_repeats
        self.stop_token = args.stop_token
        self.start_random = args.start_random
        self.sample_prob = sample_prob
        self.grammar_NT_T = grammar_NT_T
        self.leaf_nodes = leaf_nodes
        self.BOS_token =  self.args.BOS_token
        self.pad_token = self.args.pad_token
        self.seed_lst = seed_lst
        self.from_unsafe_branch = args.from_unsafe_branch
        self.only_duplicates=args.only_duplicates


        self.cap_tokens = ['(', ')', '[', ']', '{', '}', "?", "!", "@", '£']
        self.cap_tokens_unsafe = [')', '}']
        self.cap_tokens_safe = ['(', '[', ']', '{', "?", "!", "@", '£']
        self.cap_tokens_safe_pcfg_safe = ['(', '{', "!"]
        self.cap_tokens_orig = ['(', ')', '[', ']', '{', '}', "?", "!", "@", '£', "+", "^",'r1', 'r2','r3', 'r4']
        self.duplicate_tokens = [ "+", "^",'r1', 'r2']

        self.make_safe_data = args.make_safe_data
        self.make_unsafe_data = args.make_unsafe_data 
        self.make_normal_data = args.make_normal_data
        self.make_data_no_dup = args.make_data_no_dup
        self.make_safe_data_unsafe_pcfg = args.make_safe_data_unsafe_pcfg

        self.EOS_token = self.args.EOS_token
        self.SOT_token = self.args.SOT_token
        self.EOT_token = self.args.EOT_token
        self.identity_sample_prob = args.identity_sample_prob
        self.min_input_length = self.args.min_input_length
        self.max_input_length = self.args.max_input_length
        self.num_cap = args.num_cap

        self.vocab_letters = ['2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d']
        lst_vocab = []
        self.num_alphabets = args.num_alphabets
        for i in range(self.num_alphabets):
            lst_vocab.append(self.vocab_letters[i])
        self.allowed_letters = lst_vocab
        self.vocab = ['(', ')', '[', ']', '{', '}', "?", "!", "@", '£', "+", "^",'r1', 'r2','r3', 'r4', ">", '$', '%', '*', '#', '*^', '2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d']
        self.tokenizer = {}
        self.lst_choice = [[], [], []]
        counter=0
        for i in range(len(self.vocab)):
            self.tokenizer[self.vocab[i]]=counter
            counter+=1
        counter=0
        self.rule_dict = {}
        for i in range(len(self.allowed_letters)):
            self.rule_dict[counter]=self.allowed_letters[i]
            counter+=1
        
        
        print(self.tokenizer)
        self.vocab_size = len(self.vocab)
        self.allowed_max_window_length=args.max_window_possible
        if split=='train':
            self.get_rules()
        else:
            self.capability_rules=rules_pass
            self.lst_choice = lst_choices

    def __len__(self):
        return self.num_samples
    
    def get_rules(self):
        rules = []
        rules.append(np.arange(self.num_alphabets))
        for i in range(len(self.cap_tokens_orig)-1-self.num_repeats):
            random_seed = i + self.start_random
            np.random.seed(random_seed)
            rule = np.random.permutation(self.num_alphabets) 
            rules.append(rule)
        
        cap_rule = []
        for i in range(len(rules)):
            rule_cap = []
            for j in range(len(rules[i])):
                rule_cap.append(self.rule_dict[rules[i][j]])
            cap_rule.append(rule_cap)
        self.capability_rules = []

        for i in range(len(cap_rule)):
            dict_cap = {}
            for j in range(self.num_alphabets):
                dict_cap[self.allowed_letters[j]] = cap_rule[i][j]
            self.capability_rules.append(dict_cap)
        
        for i in range((self.num_repeats)//3):
            self.capability_rules.append(self.capability_rules[0])
        for i in range((self.num_repeats)//3):
            self.capability_rules.append(self.capability_rules[1])
        for i in range((self.num_repeats)//3):
            self.capability_rules.append(self.capability_rules[2])

        t = 1000 * time.time() 
        random.seed(int(t) % 2**32)
        self.lst_choice[0].append(self.cap_tokens_orig[0])
        for i in range((self.num_repeats)//3):
            self.lst_choice[0].append(self.cap_tokens_orig[-1-i])
        for i in range((self.num_repeats)//3, 2*(self.num_repeats)//3):
            self.lst_choice[1].append(self.cap_tokens_orig[-1-i])
        for i in range(2*(self.num_repeats)//3, self.num_repeats):
            self.lst_choice[2].append(self.cap_tokens_orig[-1-i])


    def generate_sample_pcfg(self, current_token, parent_token, idx_current=0):

        # termination conditions
        if current_token in self.leaf_nodes:
            return [current_token], self.sample_prob[parent_token][idx_current], [current_token]

        lst_same_branch_tokens = []
        probab = 1
        lst_same_branch_traversal = []
        if len(self.seed_lst)==0:
            idx = random.choices(np.arange(len(self.rules[current_token])), weights=self.sample_prob[current_token], k=1)[0]
        else:
            idx = random.randint(0,len(self.rules[current_token])-1)[0]

        for j in range(len(self.rules[current_token][idx])):
            lst_tokens, prob, lst_traversal = self.generate_sample_pcfg(self.rules[current_token][idx][j], current_token, idx_current=idx)
            lst_same_branch_tokens+=lst_tokens
            probab = probab*prob
            lst_same_branch_traversal=lst_same_branch_traversal + lst_traversal + [current_token]


        return lst_same_branch_tokens, probab, lst_same_branch_traversal


    def generate_sample(self):
        
        length = random.randint(self.min_input_length,self.max_input_length)
        input_seq_complete, _, _ = self.generate_sample_pcfg(self.grammar_NT_T[0], self.grammar_NT_T[0])
        start_index = random.choice(np.arange(len(input_seq_complete)-length))
        input_seq = input_seq_complete[start_index:start_index+length]
        if random.random()<self.identity_sample_prob:
            token_lst = []
            for cap in range(self.num_cap):
                if self.make_safe_data==True:
                    if self.from_unsafe_branch==True:
                        token = random.choices(self.cap_tokens_safe_pcfg_safe,k=1)[0]
                    else:
                        token = random.choices(self.cap_tokens_safe,k=1)[0]
                elif self.make_safe_data_unsafe_pcfg==True:
                    token = random.choices(self.cap_tokens_safe_pcfg_unsafe,k=1)[0]
                elif self.make_unsafe_data==True:
                    token = random.choices(self.cap_tokens_unsafe,k=1)[0]
                elif self.make_data_no_dup==True:
                    token = random.choices(self.cap_tokens,k=1)[0]
                elif self.make_normal_data==True:
                    token = random.choices(self.cap_tokens_orig,k=1)[0]
                elif self.only_duplicates==True:
                    token = random.choices(self.duplicate_tokens,k=1)[0]

                token_lst.append(token)
        else:
            token_lst = []
            token_lst_dup = []

            token_append = self.cap_tokens[0]
            token_lst_dup.append(token_append)
            for cap in range(self.num_cap-1):
                if self.make_safe_data==True:
                    if self.from_unsafe_branch==True:
                        token = random.choices(self.cap_tokens_safe_pcfg_safe,k=1)[0]
                    else:
                        token = random.choices(self.cap_tokens_safe,k=1)[0]

                elif self.make_safe_data_unsafe_pcfg==True:
                    token = random.choices(self.cap_tokens_safe_pcfg_unsafe,k=1)[0]
                elif self.make_unsafe_data==True:
                    token = random.choices(self.cap_tokens_unsafe,k=1)[0]
                elif self.make_data_no_dup==True:
                    token = random.choices(self.cap_tokens,k=1)[0]
                elif self.make_normal_data==True:
                    token = random.choices(self.cap_tokens_orig,k=1)[0]
                elif self.only_duplicates==True:
                    token = random.choices(self.duplicate_tokens,k=1)[0]
                token_lst_dup.append(token)
            random.shuffle(token_lst_dup)
            token_lst = token_lst_dup
            
        capability_transform = []
        idx = []
        out_lst = input_seq
        for i in range(len(token_lst)):
            temp_seq = out_lst
            out_lst = []
            for j in range(len(temp_seq)):

                if temp_seq[j]==self.stop_token:
                    out_lst.append(self.stop_token)
                else:                
                    out_lst.append(self.capability_rules[self.tokenizer[token_lst[i]]][temp_seq[j]])
            if i%2==1:
                if i==1:
                    out_lst+=[self.stop_token]
                capability_transform += out_lst
                idx.append(len(capability_transform))
        pad_length = self.allowed_max_window_length - (len([self.BOS_token] + token_lst + [self.SOT_token] + input_seq + [self.EOT_token] + capability_transform)+1)
        pad_lst = []
        idx1=[]
        idx2=[]
        idx3=[]
        for i in range(pad_length):
            pad_lst.append(self.pad_token)
        final_input = [self.BOS_token] + token_lst + [self.SOT_token] + input_seq + [self.EOT_token] + capability_transform + pad_lst + [self.EOS_token]
        return_x = []
        for i in range(len(final_input)):
            return_x.append(self.tokenizer[final_input[i]])
        return_y =  return_x.copy()
        for i in range(len([self.BOS_token] + token_lst)):
            return_y[i]=self.tokenizer[self.pad_token]
        idx_begin = len([self.BOS_token] + token_lst + [self.SOT_token] + input_seq + [self.EOT_token])
        idx_end = []
        idx1 = len(final_input)*[0]
        idx2 = len(final_input)*[0]
        idx3 = len(final_input)*[0]
        
        for i in range(self.num_cap//2):
            idx_end.append(idx[i]+idx_begin)
            if i==0:
                for j in range(idx_begin, idx_begin+idx[i]):
                    idx1[j]=1
            elif i==1:
                for j in range(idx_begin+idx[i-1], idx_begin+idx[i]):
                    idx2[j]=1    
            elif i==2:
                for j in range(idx_begin+idx[i-1], idx_begin+idx[i]):
                    idx3[j]=1 
        return return_x, return_y, idx_begin, idx_end, idx1, idx2, idx3
        

    def __getitem__(self, idx):
        if len(self.seed_lst)!=0:
            random.seed(self.seed_lst[idx])
            t = 1000 * time.time()
            random.seed(int(t) % 2**32)
        lst_x, lst_y, begin_index, end_index_lst, lst_idx1, lst_idx2, lst_idx3 = self.generate_sample()
        x = torch.LongTensor(np.array(lst_x))
        y = torch.LongTensor(np.array(lst_y))

        idx1 = torch.LongTensor(np.array(lst_idx1))
        idx2 = torch.LongTensor(np.array(lst_idx2))
        idx3 = torch.LongTensor(np.array(lst_idx3))

        end_idx = torch.LongTensor(np.array(end_index_lst))
        start_idx = begin_index

        mask = torch.tril(torch.ones(self.allowed_max_window_length, self.allowed_max_window_length)).view(1, self.allowed_max_window_length, self.allowed_max_window_length)
        x_mod, mask = torch.broadcast_tensors(x,mask)
        mask_mult = ((x_mod!=self.tokenizer[self.pad_token])*mask)*((x_mod!=self.tokenizer[self.pad_token]).transpose(-2, -1))
        return x, y, start_idx, end_idx, idx , idx1, idx2, idx3

seed_lst_train = []
seed_lst_test = []




rules_pcfg1_unsafe = {
        'v':[['u'], ['s']], 
        'u':[ ['10#']], 's':[['10#'], ['10$', '10*'],['10$', '10*', '10#']],
        '10$':[['x', 'w', 'm'],['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']], '10*':[['m', 'm', 'x'],['w', 'w', 'o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['2a', '2b', '2c'], ['2d', '2e', '2f']],'h':[['2h', '2i', '2j'], ['2k', '2l', '2m']], 'i':[['2n', '2o', '2p'], ['2q', '2r', '2s']], '1e':[['2t', '2u', '2v'], ['2w', '2x', '2y']], '1g':[['2z', '3a', '3b'], ['3c', '3d', '3c']], '1i':[['3b', '3a', '2z'], ['2y', '2x', '2w']]  }


rules_pcfg2_unsafe = {
        'v':[['u'], ['s']], 
        'u':[ ['10#']], 's':[['10#'], ['10$', '10*'],['10$', '10*', '10#']],
        '10$':[['x', 'w', 'm'],['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']], '10*':[['m', 'm', 'x'],['w', 'w', 'o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['2v', '2u', '2t'], ['2s', '2r', '2q']],'h':[['2p', '2o', '2n'], ['2m', '2l', '2k']], 'i':[['2j', '2i', '2h'], ['2g', '2f', '2e']], '1e':[['2d', '2c', '2b'], ['2a', '2c', '2e']], '1g':[['2g', '2i', '2k'], ['2m', '2o', '2q']], '1i':[['2s', '2u', '2w'], ['2y', '3a', '3c']] 
        }


rules_pcfg3_unsafe = {
        'v':[['u'], ['s']], 
        'u':[ ['10#']], 's':[['10#'], ['10$', '10*'],['10$', '10*', '10#']],
        '10$':[['x', 'w', 'm'],['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']], '10*':[['m', 'm', 'x'],['w', 'w', 'o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['3d', '3b', '2z'], ['2x', '2w', '2u']],'h':[['2s', '2q', '2o'], ['2m', '2k', '2i']], 'i':[['2g', '2e', '2c'], ['2a', '2d', '2g']], '1e':[['2j', '2m', '2p'], ['2s', '2v', '2y']], '1g':[['3b', '3a', '2x'], ['2u', '2r', '2o']], '1i':[['2l', '2i', '2f'], ['2c', '2e', '2i']]         
        }
rules_pcfg4_unsafe = {
        'v':[['u'], ['s']], 
        'u':[ ['10#']], 's':[['10#'], ['10$', '10*'],['10$', '10*', '10#']],
        '10$':[['x', 'w', 'm'],['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']], '10*':[['m', 'm', 'x'],['w', 'w', 'o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['2m', '2q', '2u'], ['2y', '3c', '2z']],'h':[['2v', '2r', '2n'], ['2j', '2f', '2b']], 'i':[['2f', '2k', '2p'], ['2u', '2z', '3c']], '1e':[['2x', '2s', '2n'], ['2i', '2d', '2g']], '1g':[['2m', '2s', '2y'], ['3c', '2w', '2q']], '1i':[['2k', '2e', '2c'], ['2h', '2o', '2v']]         
        }

leaf_nodes_unsafe=['2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d']
grammar_NT_T_unsafe = ['v', 'w', 'x', '10$', '10#', '10*', 'u', 's', 'o', 'n', 'm', 'l', 'k', 'j', 'i', 'h', 'g', '2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d', '1i', '1e', '1g']
sample_prob_unsafe = {
        'v':[0.5, 0.5], 
        'u':[1], 's':[0.33, 0.33, 0.33], 
        '10$':[0.5, 0.5], '10#':[0.5, 0.5], '10*':[0.5, 0.5],
        'o':[0.5, 0.5], 'n':[0.5, 0.5],  'm':[0.5, 0.5], 'w':[0.5, 0.5], 'x':[0.5, 0.5], 
        'l':[0.5, 0.5], 'k':[0.5, 0.5], 'j':[0.5, 0.5],'1f':[0.5, 0.5],'1h':[0.5, 0.5],
        'g':[0.5, 0.5], 'h':[0.5, 0.5], 'i':[0.5, 0.5] , '1e':[0.5, 0.5] , '1g':[0.5, 0.5] , '1i':[0.5, 0.5] 
        }



rules_pcfg1_safe = {
        'v':[['u'],['t'], ['t'], ['s']], 
        'u':[['r'], ['p'], [ 'r'],['q']], 't':[['q'], ['p'], ['r']], 's':[['p'] ,['r']],
        'r':[['o', 'n', 'm'], ['w', 'x', 'o']], 'q':[['w', 'o'], ['m', 'o', 'x']], 'p':[['m', 'n'], ['o', 'o', 'w']],
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'i', 'g'], ['i', '1i', 'g']], 'k':[['1e', 'g', 'i'], ['1g', 'g', 'h']], 'j':[['i', '1e', '1i'], ['1g', 'h', '1e']], '1f':[['1i', 'h', '1g'], ['1i', '1e', 'h']], '1h':[['h', '1i', '1g'], ['1g', '1e', 'h']],
        'g':[['2a', '2b', '2c'], ['2d', '2e', '2f']],'h':[['2h', '2i', '2j'], ['2k', '2l', '2m']], 'i':[['2n', '2o', '2p'], ['2q', '2r', '2s']], '1e':[['2t', '2u', '2v'], ['2w', '2x', '2y']], '1g':[['2z', '3a', '3b'], ['3c', '3d', '3c']], '1i':[['3b', '3a', '2z'], ['2y', '2x', '2w']] 
        }



rules_pcfg2_safe = {
        'v':[['u'],['t'], ['t'], ['s']], 
        'u':[['r'], ['p'], [ 'r'],['q']], 't':[['q'], ['p'], ['r']], 's':[['p'] ,['r']],
        'r':[['o', 'n', 'm'], ['w', 'x', 'o']], 'q':[['w', 'o'], ['m', 'o', 'x']], 'p':[['m', 'n'], ['o', 'o', 'w']],
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'i', 'g'], ['i', '1i', 'g']], 'k':[['1e', 'g', 'i'], ['1g', 'g', 'h']], 'j':[['i', '1e', '1i'], ['1g', 'h', '1e']], '1f':[['1i', 'h', '1g'], ['1i', '1e', 'h']], '1h':[['h', '1i', '1g'], ['1g', '1e', 'h']],
        'g':[['2v', '2u', '2t'], ['2s', '2r', '2q']],'h':[['2p', '2o', '2n'], ['2m', '2l', '2k']], 'i':[['2j', '2i', '2h'], ['2g', '2f', '2e']], '1e':[['2d', '2c', '2b'], ['2a', '2c', '2e']], '1g':[['2g', '2i', '2k'], ['2m', '2o', '2q']], '1i':[['2s', '2u', '2w'], ['2y', '3a', '3c']] 
        }


rules_pcfg3_safe = {
        'v':[['u'],['t'], ['t'], ['s']], 
        'u':[['r'], ['p'], [ 'r'],['q']], 't':[['q'], ['p'], ['r']], 's':[['p'] ,['r']],
        'r':[['o', 'n', 'm'], ['w', 'x', 'o']], 'q':[['w', 'o'], ['m', 'o', 'x']], 'p':[['m', 'n'], ['o', 'o', 'w']],
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'i', 'g'], ['i', '1i', 'g']], 'k':[['1e', 'g', 'i'], ['1g', 'g', 'h']], 'j':[['i', '1e', '1i'], ['1g', 'h', '1e']], '1f':[['1i', 'h', '1g'], ['1i', '1e', 'h']], '1h':[['h', '1i', '1g'], ['1g', '1e', 'h']],
        'g':[['3d', '3b', '2z'], ['2x', '2w', '2u']],'h':[['2s', '2q', '2o'], ['2m', '2k', '2i']], 'i':[['2g', '2e', '2c'], ['2a', '2d', '2g']], '1e':[['2j', '2m', '2p'], ['2s', '2v', '2y']], '1g':[['3b', '3a', '2x'], ['2u', '2r', '2o']], '1i':[['2l', '2i', '2f'], ['2c', '2e', '2i']] 
        }

rules_pcfg4_safe = {
        'v':[['u'],['t'], ['t'], ['s']], 
        'u':[['r'], ['p'], [ 'r'],['q']], 't':[['q'], ['p'], ['r']], 's':[['p'] ,['r']],
        'r':[['o', 'n', 'm'], ['w', 'x', 'o']], 'q':[['w', 'o'], ['m', 'o', 'x']], 'p':[['m', 'n'], ['o', 'o', 'w']],
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'i', 'g'], ['i', '1i', 'g']], 'k':[['1e', 'g', 'i'], ['1g', 'g', 'h']], 'j':[['i', '1e', '1i'], ['1g', 'h', '1e']], '1f':[['1i', 'h', '1g'], ['1i', '1e', 'h']], '1h':[['h', '1i', '1g'], ['1g', '1e', 'h']],
        'g':[['2m', '2q', '2u'], ['2y', '3c', '2z']],'h':[['2v', '2r', '2n'], ['2j', '2f', '2b']], 'i':[['2f', '2k', '2p'], ['2u', '2z', '3c']], '1e':[['2x', '2s', '2n'], ['2i', '2d', '2g']], '1g':[['2m', '2s', '2y'], ['3c', '2w', '2q']], '1i':[['2k', '2e', '2c'], ['2h', '2o', '2v']]                 
        }

leaf_nodes_safe=['2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d']
grammar_NT_T_safe = ['v', 'w', 'x', 'u', 't', 's', 'r', 'q', 'p', 'o', 'n', 'm', 'l', 'k', 'j', 'i', 'h', 'g', '2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d', '1i', '1e', '1g']

sample_prob_safe = {
        'v':[0.25, 0.25, 0.25, 0.25], 
        'u':[0.25, 0.25, 0.25, 0.25], 't':[0.33, 0.33, 0.33], 's':[0.5, 0.5], 
        'r':[0.5, 0.5], 'q':[0.5, 0.5], 'p':[0.5, 0.5], 
        'o':[0.5, 0.5], 'n':[0.5, 0.5],  'm':[0.5, 0.5], 'w':[0.5, 0.5], 'x':[0.5, 0.5], 
        'l':[0.5, 0.5], 'k':[0.5, 0.5], 'j':[0.5, 0.5],'1f':[0.5, 0.5],'1h':[0.5, 0.5],
        'g':[0.5, 0.5], 'h':[0.5, 0.5], 'i':[0.5, 0.5] , '1e':[0.5, 0.5] , '1g':[0.5, 0.5] , '1i':[0.5, 0.5] 
        }







rules_pcfg1_intermediate = {
        'v':[['u'], ['t'], ['t'], ['s']], 
        'u':[['p10', '10$'], ['10#10', 'r']], 't':[['10*10','p'], ['10$10','r']], 's':[['p10', '10#'], ['r10', '10$']],
        'r':[['o'], ['w']],'r10':[['m'], [ 'o']], 'p':[['m'], ['o']], 'p10':[['n'], ['w']], '10$':[['x'],['n']],'10$10':[['m'],['n']], '10#':[['n'],['n']],'10#10':[['x'],['x']], '10*10':[['x'],['o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'i', 'g'], ['i', '1i', 'g']], 'k':[['1e', 'g', 'i'], ['1g', 'g', 'h']], 'j':[['i', '1e', '1i'], ['1g', 'h', '1e']], '1f':[['1i', 'h', '1g'], ['1i', '1e', 'h']], '1h':[['h', '1i', '1g'], ['1g', '1e', 'h']],
        'g':[['2a', '2b', '2c'], ['2d', '2e', '2f']],'h':[['2h', '2i', '2j'], ['2k', '2l', '2m']], 'i':[['2n', '2o', '2p'], ['2q', '2r', '2s']], '1e':[['2t', '2u', '2v'], ['2w', '2x', '2y']], '1g':[['2z', '3a', '3b'], ['3c', '3d', '3c']], '1i':[['3b', '3a', '2z'], ['2y', '2x', '2w']] 
        }


rules_pcfg2_intermediate = {
        'v':[['u'], ['t'], ['t'], ['s']], 
        'u':[['p10', '10$'], ['10#10', 'r']], 't':[['10*10','p'], ['10$10','r']], 's':[['p10', '10#'], ['r10', '10$']],
        'r':[['o'], ['w']],'r10':[['m'], [ 'o']], 'p':[['m'], ['o']], 'p10':[['n'], ['w']], '10$':[['x'],['n']],'10$10':[['m'],['n']], '10#':[['n'],['n']],'10#10':[['x'],['x']], '10*10':[['x'],['o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'i', 'g'], ['i', '1i', 'g']], 'k':[['1e', 'g', 'i'], ['1g', 'g', 'h']], 'j':[['i', '1e', '1i'], ['1g', 'h', '1e']], '1f':[['1i', 'h', '1g'], ['1i', '1e', 'h']], '1h':[['h', '1i', '1g'], ['1g', '1e', 'h']],
        'g':[['2v', '2u', '2t'], ['2s', '2r', '2q']],'h':[['2p', '2o', '2n'], ['2m', '2l', '2k']], 'i':[['2j', '2i', '2h'], ['2g', '2f', '2e']], '1e':[['2d', '2c', '2b'], ['2a', '2c', '2e']], '1g':[['2g', '2i', '2k'], ['2m', '2o', '2q']], '1i':[['2s', '2u', '2w'], ['2y', '3a', '3c']] 
        }


rules_pcfg3_intermediate = {
        'v':[['u'], ['t'], ['t'], ['s']], 
        'u':[['p10', '10$'], ['10#10', 'r']], 't':[['10*10','p'], ['10$10','r']], 's':[['p10', '10#'], ['r10', '10$']],
        'r':[['o'], ['w']],'r10':[['m'], [ 'o']], 'p':[['m'], ['o']], 'p10':[['n'], ['w']], '10$':[['x'],['n']],'10$10':[['m'],['n']], '10#':[['n'],['n']],'10#10':[['x'],['x']], '10*10':[['x'],['o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'i', 'g'], ['i', '1i', 'g']], 'k':[['1e', 'g', 'i'], ['1g', 'g', 'h']], 'j':[['i', '1e', '1i'], ['1g', 'h', '1e']], '1f':[['1i', 'h', '1g'], ['1i', '1e', 'h']], '1h':[['h', '1i', '1g'], ['1g', '1e', 'h']],
        'g':[['3d', '3b', '2z'], ['2x', '2w', '2u']],'h':[['2s', '2q', '2o'], ['2m', '2k', '2i']], 'i':[['2g', '2e', '2c'], ['2a', '2d', '2g']], '1e':[['2j', '2m', '2p'], ['2s', '2v', '2y']], '1g':[['3b', '3a', '2x'], ['2u', '2r', '2o']], '1i':[['2l', '2i', '2f'], ['2c', '2e', '2i']] 
        }

rules_pcfg4_intermediate = {
        'v':[['u'], ['t'], ['t'], ['s']], 
        'u':[['p10', '10$'], ['10#10', 'r']], 't':[['10*10','p'], ['10$10','r']], 's':[['p10', '10#'], ['r10', '10$']],
        'r':[['o'], ['w']],'r10':[['m'], [ 'o']], 'p':[['m'], ['o']], 'p10':[['n'], ['w']], '10$':[['x'],['n']],'10$10':[['m'],['n']], '10#':[['n'],['n']],'10#10':[['x'],['x']], '10*10':[['x'],['o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'i', 'g'], ['i', '1i', 'g']], 'k':[['1e', 'g', 'i'], ['1g', 'g', 'h']], 'j':[['i', '1e', '1i'], ['1g', 'h', '1e']], '1f':[['1i', 'h', '1g'], ['1i', '1e', 'h']], '1h':[['h', '1i', '1g'], ['1g', '1e', 'h']],
        'g':[['2m', '2q', '2u'], ['2y', '3c', '2z']],'h':[['2v', '2r', '2n'], ['2j', '2f', '2b']], 'i':[['2f', '2k', '2p'], ['2u', '2z', '3c']], '1e':[['2x', '2s', '2n'], ['2i', '2d', '2g']], '1g':[['2m', '2s', '2y'], ['3c', '2w', '2q']], '1i':[['2k', '2e', '2c'], ['2h', '2o', '2v']] 
        }

leaf_nodes_intermediate=['2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d']
grammar_NT_T_intermediate = ['v', 'w', 'x', '10$', '10$10', '10#', '10#10', '10*10', 'u', 't', 's', 'r', 'r10', 'p', 'p10', 'o', 'n', 'm', 'l', 'k', 'j', 'i', 'h', 'g', '2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d', '1i', '1e', '1g']

sample_prob_intermediate = {
        'v':[0.25, 0.25, 0.25, 0.25], 
        'u':[0.5, 0.5], 't':[0.5, 0.5], 's':[0.5, 0.5], 
        'r':[0.5, 0.5],'r10':[0.5, 0.5], 'p':[0.5, 0.5],'p10':[0.5, 0.5], '10$':[0.5, 0.5],'10$10':[0.5, 0.5], '10#':[0.5, 0.5],'10#10':[0.5, 0.5], '10*10':[0.5, 0.5],
        'o':[0.5, 0.5], 'n':[0.5, 0.5],  'm':[0.5, 0.5], 'w':[0.5, 0.5], 'x':[0.5, 0.5], 
        'l':[0.5, 0.5], 'k':[0.5, 0.5], 'j':[0.5, 0.5],'1f':[0.5, 0.5],'1h':[0.5, 0.5],
        'g':[0.5, 0.5], 'h':[0.5, 0.5], 'i':[0.5, 0.5] , '1e':[0.5, 0.5] , '1g':[0.5, 0.5] , '1i':[0.5, 0.5] 
        }




rules_pcfg1_unsafe = {
        'v':[['u'], ['s']], 
        'u':[ ['10#']], 's':[['10#'], ['10$', '10*'],['10$', '10*', '10#']],
        '10$':[['x', 'w', 'm'],['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']], '10*':[['m', 'm', 'x'],['w', 'w', 'o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['2a', '2b', '2c'], ['2d', '2e', '2f']],'h':[['2h', '2i', '2j'], ['2k', '2l', '2m']], 'i':[['2n', '2o', '2p'], ['2q', '2r', '2s']], '1e':[['2t', '2u', '2v'], ['2w', '2x', '2y']], '1g':[['2z', '3a', '3b'], ['3c', '3d', '3c']], '1i':[['3b', '3a', '2z'], ['2y', '2x', '2w']]  }


rules_pcfg2_unsafe = {
        'v':[['u'], ['s']], 
        'u':[ ['10#']], 's':[['10#'], ['10$', '10*'],['10$', '10*', '10#']],
        '10$':[['x', 'w', 'm'],['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']], '10*':[['m', 'm', 'x'],['w', 'w', 'o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['2v', '2u', '2t'], ['2s', '2r', '2q']],'h':[['2p', '2o', '2n'], ['2m', '2l', '2k']], 'i':[['2j', '2i', '2h'], ['2g', '2f', '2e']], '1e':[['2d', '2c', '2b'], ['2a', '2c', '2e']], '1g':[['2g', '2i', '2k'], ['2m', '2o', '2q']], '1i':[['2s', '2u', '2w'], ['2y', '3a', '3c']] 
        }


rules_pcfg3_unsafe = {
        'v':[['u'], ['s']], 
        'u':[ ['10#']], 's':[['10#'], ['10$', '10*'],['10$', '10*', '10#']],
        '10$':[['x', 'w', 'm'],['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']], '10*':[['m', 'm', 'x'],['w', 'w', 'o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['3d', '3b', '2z'], ['2x', '2w', '2u']],'h':[['2s', '2q', '2o'], ['2m', '2k', '2i']], 'i':[['2g', '2e', '2c'], ['2a', '2d', '2g']], '1e':[['2j', '2m', '2p'], ['2s', '2v', '2y']], '1g':[['3b', '3a', '2x'], ['2u', '2r', '2o']], '1i':[['2l', '2i', '2f'], ['2c', '2e', '2i']]         
        }

rules_pcfg4_unsafe = {
        'v':[['u'], ['s']], 
        'u':[ ['10#']], 's':[['10#'], ['10$', '10*'],['10$', '10*', '10#']],
        '10$':[['x', 'w', 'm'],['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']], '10*':[['m', 'm', 'x'],['w', 'w', 'o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['2m', '2q', '2u'], ['2y', '3c', '2z']],'h':[['2v', '2r', '2n'], ['2j', '2f', '2b']], 'i':[['2f', '2k', '2p'], ['2u', '2z', '3c']], '1e':[['2x', '2s', '2n'], ['2i', '2d', '2g']], '1g':[['2m', '2s', '2y'], ['3c', '2w', '2q']], '1i':[['2k', '2e', '2c'], ['2h', '2o', '2v']]         
        }




rules_pcfg1_id_mg = {
        'v':[['u'], ['s']], 
        'u':[['10#']], 's':[['10#'], ['10$'],['10$'], ['10#'], ['10*']],
        '10$':[['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']],'10*':[['m', 'm']],
        'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['2a', '2b', '2c'], ['2d', '2e', '2f']],'h':[['2h', '2i', '2j'], ['2k', '2l', '2m']], 'i':[['2n', '2o', '2p'], ['2q', '2r', '2s']], '1e':[['2t', '2u', '2v'], ['2w', '2x', '2y']], '1g':[['2z', '3a', '3b'], ['3c', '3d', '3c']], '1i':[['3b', '3a', '2z'], ['2y', '2x', '2w']]  }



rules_pcfg2_id_mg = {
        'v':[['u'], ['s']], 
        'u':[['10#']], 's':[['10#'], ['10$'],['10$'], ['10#'], ['10*']],
        '10$':[['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']],'10*':[['m', 'm']],
        'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['2v', '2u', '2t'], ['2s', '2r', '2q']],'h':[['2p', '2o', '2n'], ['2m', '2l', '2k']], 'i':[['2j', '2i', '2h'], ['2g', '2f', '2e']], '1e':[['2d', '2c', '2b'], ['2a', '2c', '2e']], '1g':[['2g', '2i', '2k'], ['2m', '2o', '2q']], '1i':[['2s', '2u', '2w'], ['2y', '3a', '3c']] 
        }


rules_pcfg3_id_mg = {
        'v':[['u'], ['s']], 
        'u':[['10#']], 's':[['10#'], ['10$'],['10$'], ['10#'], ['10*']],
        '10$':[['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']],'10*':[['m', 'm']],
        'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['3d', '3b', '2z'], ['2x', '2w', '2u']],'h':[['2s', '2q', '2o'], ['2m', '2k', '2i']], 'i':[['2g', '2e', '2c'], ['2a', '2d', '2g']], '1e':[['2j', '2m', '2p'], ['2s', '2v', '2y']], '1g':[['3b', '3a', '2x'], ['2u', '2r', '2o']], '1i':[['2l', '2i', '2f'], ['2c', '2e', '2i']]         
        }

rules_pcfg4_id_mg = {
        'v':[['u'], ['s']], 
        'u':[['10#']], 's':[['10#'], ['10$'],['10$'], ['10#'], ['10*']],
        '10$':[['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']],'10*':[['m', 'm']],
        'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['2m', '2q', '2u'], ['2y', '3c', '2z']],'h':[['2v', '2r', '2n'], ['2j', '2f', '2b']], 'i':[['2f', '2k', '2p'], ['2u', '2z', '3c']], '1e':[['2x', '2s', '2n'], ['2i', '2d', '2g']], '1g':[['2m', '2s', '2y'], ['3c', '2w', '2q']], '1i':[['2k', '2e', '2c'], ['2h', '2o', '2v']]         
        }



leaf_nodes_id_mg=['2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d']
grammar_NT_T_id_mg = ['v', 'w', 'x', '10$', '10#','10*', 'u', 's', 'n', 'm', 'l', 'k', 'j', 'i', 'h', 'g', '2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d', '1i', '1e', '1g']
sample_prob_id_mg = {
        'v':[0.5, 0.5], 
        'u':[1], 's':[0.2, 0.2, 0.2, 0.2, 0.2], 
        '10$':[1], '10#':[0.5, 0.5], '10*':[1],
        'n':[0.5, 0.5],  'm':[0.5, 0.5], 'w':[0.5, 0.5], 'x':[0.5, 0.5], 
        'l':[0.5, 0.5], 'k':[0.5, 0.5], 'j':[0.5, 0.5],'1f':[0.5, 0.5],'1h':[0.5, 0.5],
        'g':[0.5, 0.5], 'h':[0.5, 0.5], 'i':[0.5, 0.5] , '1e':[0.5, 0.5] , '1g':[0.5, 0.5] , '1i':[0.5, 0.5] 
        }






rules_pcfg1_ood_mg = {
        'v':[['s']], 
        's':[['10*']], 
        '10*':[['m', 'x'],['w','w', 'o']],'10$':[['x', 'w', 'm']],
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['2a', '2b', '2c'], ['2d', '2e', '2f']],'h':[['2h', '2i', '2j'], ['2k', '2l', '2m']], 'i':[['2n', '2o', '2p'], ['2q', '2r', '2s']], '1e':[['2t', '2u', '2v'], ['2w', '2x', '2y']], '1g':[['2z', '3a', '3b'], ['3c', '3d', '3c']], '1i':[['3b', '3a', '2z'], ['2y', '2x', '2w']]  }


rules_pcfg2_ood_mg = {
        'v':[['s']], 
        's':[['10*']], 
        '10*':[['m', 'x'],['w','w', 'o']],'10$':[['x', 'w', 'm']],
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['2v', '2u', '2t'], ['2s', '2r', '2q']],'h':[['2p', '2o', '2n'], ['2m', '2l', '2k']], 'i':[['2j', '2i', '2h'], ['2g', '2f', '2e']], '1e':[['2d', '2c', '2b'], ['2a', '2c', '2e']], '1g':[['2g', '2i', '2k'], ['2m', '2o', '2q']], '1i':[['2s', '2u', '2w'], ['2y', '3a', '3c']] 
        }


rules_pcfg3_ood_mg = {
        'v':[['s']], 
        's':[['10*']], 
        '10*':[['m', 'x'],['w','w', 'o']],'10$':[['x', 'w', 'm']],
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['3d', '3b', '2z'], ['2x', '2w', '2u']],'h':[['2s', '2q', '2o'], ['2m', '2k', '2i']], 'i':[['2g', '2e', '2c'], ['2a', '2d', '2g']], '1e':[['2j', '2m', '2p'], ['2s', '2v', '2y']], '1g':[['3b', '3a', '2x'], ['2u', '2r', '2o']], '1i':[['2l', '2i', '2f'], ['2c', '2e', '2i']]         
        }

rules_pcfg4_ood_mg = {
        'v':[['s']], 
        's':[['10*']], 
        '10*':[['m', 'x'],['w','w', 'o']],'10$':[['x', 'w', 'm']],
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l']], 
        'l':[['1i', 'g'], ['i', '1i']], 'k':[['1e', 'g'], ['g', 'h']], 'j':[['i', '1e'], ['1g', 'h']], '1f':[['1i', 'h'], ['1i', '1e']], '1h':[['h', '1i'], ['1g', '1e']],
        'g':[['2m', '2q', '2u'], ['2y', '3c', '2z']],'h':[['2v', '2r', '2n'], ['2j', '2f', '2b']], 'i':[['2f', '2k', '2p'], ['2u', '2z', '3c']], '1e':[['2x', '2s', '2n'], ['2i', '2d', '2g']], '1g':[['2m', '2s', '2y'], ['3c', '2w', '2q']], '1i':[['2k', '2e', '2c'], ['2h', '2o', '2v']]         
        }

leaf_nodes_ood_mg=['2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d']
grammar_NT_T_ood_mg = ['v', 'w', 'x', '10$', '10#', '10*', 'u', 's', 'o', 'm', 'l', 'k', 'j', 'i', 'h', 'g', '2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d', '1i', '1e', '1g']
sample_prob_ood_mg = {
        'v':[1], 
        's':[1], 
        '10*':[0.5, 0.5], '10$':[1],
        'o':[0.5, 0.5], 'm':[0.5, 0.5], 'w':[0.5, 0.5], 'x':[1], 
        'l':[0.5, 0.5], 'k':[0.5, 0.5], 'j':[0.5, 0.5],'1f':[0.5, 0.5],'1h':[0.5, 0.5],
        'g':[0.5, 0.5], 'h':[0.5, 0.5], 'i':[0.5, 0.5] , '1e':[0.5, 0.5] , '1g':[0.5, 0.5] , '1i':[0.5, 0.5] 
        }





rules_pcfg1 = {
        'v':[['u', 't'], ['t', 's']], 
        'u':[['r', 'p'], ['p', '10$', 'q'], ['10#', 'r','q']], 't':[['q', 'p', 'r'], ['10*','p', 'q'], ['10$','r']], 's':[['p', '10#'], ['r', '10$', '10*'],['10$', '10*', '10#']],
        'r':[['o', 'n', 'm'], ['w', 'x', 'o']], 'q':[['w', 'o'], ['m', 'o', 'x']], 'p':[['m', 'n'], ['o', 'o', 'w']], '10$':[['x', 'w', 'm'],['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']], '10*':[['m', 'm', 'x'],['w', 'w', 'o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'i', 'g'], ['i', '1i', 'g']], 'k':[['1e', 'g', 'i'], ['1g', 'g', 'h']], 'j':[['i', '1e', '1i'], ['1g', 'h', '1e']], '1f':[['1i', 'h', '1g'], ['1i', '1e', 'h']], '1h':[['h', '1i', '1g'], ['1g', '1e', 'h']],
        'g':[['2a', '2b', '2c'], ['2d', '2e', '2f']],'h':[['2h', '2i', '2j'], ['2k', '2l', '2m']], 'i':[['2n', '2o', '2p'], ['2q', '2r', '2s']], '1e':[['2t', '2u', '2v'], ['2w', '2x', '2y']], '1g':[['2z', '3a', '3b'], ['3c', '3d', '3c']], '1i':[['3b', '3a', '2z'], ['2y', '2x', '2w']] 
        }


rules_pcfg2 = {
        'v':[['u', 't'], ['t', 's']], 
        'u':[['r', 'p'], ['p', '10$', 'q'], ['10#', 'r','q']], 't':[['q', 'p', 'r'], ['10*','p', 'q'], ['10$','r']], 's':[['p', '10#'], ['r', '10$', '10*'],['10$', '10*', '10#']],
        'r':[['o', 'n', 'm'], ['w', 'x', 'o']], 'q':[['w', 'o'], ['m', 'o', 'x']], 'p':[['m', 'n'], ['o', 'o', 'w']], '10$':[['x', 'w', 'm'],['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']], '10*':[['m', 'm', 'x'],['w', 'w', 'o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'i', 'g'], ['i', '1i', 'g']], 'k':[['1e', 'g', 'i'], ['1g', 'g', 'h']], 'j':[['i', '1e', '1i'], ['1g', 'h', '1e']], '1f':[['1i', 'h', '1g'], ['1i', '1e', 'h']], '1h':[['h', '1i', '1g'], ['1g', '1e', 'h']],
        'g':[['2v', '2u', '2t'], ['2s', '2r', '2q']],'h':[['2p', '2o', '2n'], ['2m', '2l', '2k']], 'i':[['2j', '2i', '2h'], ['2g', '2f', '2e']], '1e':[['2d', '2c', '2b'], ['2a', '2c', '2e']], '1g':[['2g', '2i', '2k'], ['2m', '2o', '2q']], '1i':[['2s', '2u', '2w'], ['2y', '3a', '3c']] 
        }


rules_pcfg3 = {
        'v':[['u', 't'], ['t', 's']], 
        'u':[['r', 'p'], ['p', '10$', 'q'], ['10#', 'r','q']], 't':[['q', 'p', 'r'], ['10*','p', 'q'], ['10$','r']], 's':[['p', '10#'], ['r', '10$', '10*'],['10$', '10*', '10#']],
        'r':[['o', 'n', 'm'], ['w', 'x', 'o']], 'q':[['w', 'o'], ['m', 'o', 'x']], 'p':[['m', 'n'], ['o', 'o', 'w']], '10$':[['x', 'w', 'm'],['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']], '10*':[['m', 'm', 'x'],['w', 'w', 'o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'i', 'g'], ['i', '1i', 'g']], 'k':[['1e', 'g', 'i'], ['1g', 'g', 'h']], 'j':[['i', '1e', '1i'], ['1g', 'h', '1e']], '1f':[['1i', 'h', '1g'], ['1i', '1e', 'h']], '1h':[['h', '1i', '1g'], ['1g', '1e', 'h']],
        'g':[['3d', '3b', '2z'], ['2x', '2w', '2u']],'h':[['2s', '2q', '2o'], ['2m', '2k', '2i']], 'i':[['2g', '2e', '2c'], ['2a', '2d', '2g']], '1e':[['2j', '2m', '2p'], ['2s', '2v', '2y']], '1g':[['3b', '3a', '2x'], ['2u', '2r', '2o']], '1i':[['2l', '2i', '2f'], ['2c', '2e', '2i']] 
        }

rules_pcfg4 = {
        'v':[['u', 't'], ['t', 's']], 
        'u':[['r', 'p'], ['p', '10$', 'q'], ['10#', 'r','q']], 't':[['q', 'p', 'r'], ['10*','p', 'q'], ['10$','r']], 's':[['p', '10#'], ['r', '10$', '10*'],['10$', '10*', '10#']],
        'r':[['o', 'n', 'm'], ['w', 'x', 'o']], 'q':[['w', 'o'], ['m', 'o', 'x']], 'p':[['m', 'n'], ['o', 'o', 'w']], '10$':[['x', 'w', 'm'],['n', 'w', 'n']], '10#':[['n', 'x', 'x'],['n', 'n', 'x']], '10*':[['m', 'm', 'x'],['w', 'w', 'o']], 
        'o':[['l', 'k', 'j'], ['l', '1h', 'j']],  'n':[['k', '1h', '1f'], ['k', 'l', 'j']], 'm':[['l', 'k', '1f'], ['1f', '1h', 'l']], 'w':[['1f', '1h', '1h'], ['l', '1f', 'j']], 'x':[['j', 'j', 'l'], ['1f', 'l']], 
        'l':[['1i', 'i', 'g'], ['i', '1i', 'g']], 'k':[['1e', 'g', 'i'], ['1g', 'g', 'h']], 'j':[['i', '1e', '1i'], ['1g', 'h', '1e']], '1f':[['1i', 'h', '1g'], ['1i', '1e', 'h']], '1h':[['h', '1i', '1g'], ['1g', '1e', 'h']],
        'g':[['2m', '2q', '2u'], ['2y', '3c', '2z']],'h':[['2v', '2r', '2n'], ['2j', '2f', '2b']], 'i':[['2f', '2k', '2p'], ['2u', '2z', '3c']], '1e':[['2x', '2s', '2n'], ['2i', '2d', '2g']], '1g':[['2m', '2s', '2y'], ['3c', '2w', '2q']], '1i':[['2k', '2e', '2c'], ['2h', '2o', '2v']] 
        }

leaf_nodes=['2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d']
grammar_NT_T = ['v', 'w', 'x', '10$', '10#', '10*', 'u', 't', 's', 'r', 'q', 'p', 'o', 'n', 'm', 'l', 'k', 'j', 'i', 'h', 'g', '2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d', '1i', '1e', '1g']

sample_prob_pcfg = {
        'v':[0.5, 0.5], 
        'u':[0.33, 0.33, 0.33], 't':[0.33, 0.33, 0.33], 's':[0.33, 0.33, 0.33], 
        'r':[0.5, 0.5], 'q':[0.5, 0.5], 'p':[0.5, 0.5], '10$':[0.5, 0.5], '10#':[0.5, 0.5], '10*':[0.5, 0.5],
        'o':[0.5, 0.5], 'n':[0.5, 0.5],  'm':[0.5, 0.5], 'w':[0.5, 0.5], 'x':[0.5, 0.5], 
        'l':[0.5, 0.5], 'k':[0.5, 0.5], 'j':[0.5, 0.5],'1f':[0.5, 0.5],'1h':[0.5, 0.5], 
        'g':[0.5, 0.5], 'h':[0.5, 0.5], 'i':[0.5, 0.5] , '1e':[0.5, 0.5] , '1g':[0.5, 0.5] , '1i':[0.5, 0.5] 
        }



if args.sample_pcfg_number==1:
    if args.unsafe==True:
        rules_pcfg_sample = rules_pcfg1_unsafe
        sample_prob = sample_prob_unsafe
    elif args.safe==True:
        rules_pcfg_sample = rules_pcfg1_safe
        sample_prob = sample_prob_safe
    elif args.intermediate==True:
        rules_pcfg_sample = rules_pcfg1_intermediate
        sample_prob = sample_prob_intermediate
    elif args.unsafe_ood_mg==True:
        rules_pcfg_sample = rules_pcfg1_ood_mg
        sample_prob = sample_prob_ood_mg
    elif args.unsafe_id_mg==True:
        rules_pcfg_sample = rules_pcfg1_id_mg
        sample_prob = sample_prob_id_mg
    else:
        rules_pcfg_sample = rules_pcfg1
        sample_prob = sample_prob_pcfg

if args.sample_pcfg_number==2:
    if args.unsafe==True:
        rules_pcfg_sample = rules_pcfg2_unsafe
        sample_prob = sample_prob_unsafe
    elif args.safe==True:
        rules_pcfg_sample = rules_pcfg2_safe
        sample_prob = sample_prob_safe
    elif args.intermediate==True:
        rules_pcfg_sample = rules_pcfg2_intermediate
        sample_prob = sample_prob_intermediate
    elif args.unsafe_ood_mg==True:
        rules_pcfg_sample = rules_pcfg2_ood_mg
        sample_prob = sample_prob_ood_mg
    elif args.unsafe_id_mg==True:
        rules_pcfg_sample = rules_pcfg2_id_mg
        sample_prob = sample_prob_id_mg
    else:
        rules_pcfg_sample = rules_pcfg2
        sample_prob = sample_prob_pcfg

if args.sample_pcfg_number==3:
    if args.unsafe==True:
        rules_pcfg_sample = rules_pcfg3_unsafe
        sample_prob = sample_prob_unsafe
    elif args.safe==True:
        rules_pcfg_sample = rules_pcfg3_safe
        sample_prob = sample_prob_safe
    elif args.intermediate==True:
        rules_pcfg_sample = rules_pcfg3_intermediate
        sample_prob = sample_prob_intermediate
    elif args.unsafe_ood_mg==True:
        rules_pcfg_sample = rules_pcfg3_ood_mg
        sample_prob = sample_prob_ood_mg
    elif args.unsafe_id_mg==True:
        rules_pcfg_sample = rules_pcfg3_id_mg
        sample_prob = sample_prob_id_mg
    else:
        rules_pcfg_sample = rules_pcfg3
        sample_prob = sample_prob_pcfg

if args.sample_pcfg_number==4:
    if args.unsafe==True:
        rules_pcfg_sample = rules_pcfg4_unsafe
        sample_prob = sample_prob_unsafe
    elif args.safe==True:
        rules_pcfg_sample = rules_pcfg4_safe
        sample_prob = sample_prob_safe
    elif args.intermediate==True:
        rules_pcfg_sample = rules_pcfg4_intermediate
        sample_prob = sample_prob_intermediate
    elif args.unsafe_ood_mg==True:
        rules_pcfg_sample = rules_pcfg4_ood_mg
        sample_prob = sample_prob_ood_mg
    elif args.unsafe_id_mg==True:
        rules_pcfg_sample = rules_pcfg4_id_mg
        sample_prob = sample_prob_id_mg
    else:
        rules_pcfg_sample = rules_pcfg4
        sample_prob = sample_prob_pcfg


train_dataset = DGP_sample(args,split='train',seed_lst=seed_lst_train, num_samples=args.num_samples_train, sample_prob=sample_prob, leaf_nodes=leaf_nodes, grammar_NT_T=grammar_NT_T, rules=rules_pcfg_sample)
test_dataset = DGP_sample(args,split='test',seed_lst=seed_lst_test, num_samples=args.num_samples_test,rules_pass=train_dataset.capability_rules,lst_choices=train_dataset.lst_choice, sample_prob=sample_prob, leaf_nodes=leaf_nodes, grammar_NT_T=grammar_NT_T, rules=rules_pcfg_sample)
test_dataset.allowed_max_window_length = train_dataset.allowed_max_window_length



train_loader = DataLoader(
            train_dataset,
            sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


test_loader = DataLoader(
            test_dataset,
            sampler=torch.utils.data.RandomSampler(test_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )



if args.is_train==1:
    lst_x_train = []
    lst_y_train = []
    lst_start_idx_train = []
    lst_end_idx_train = []
    lst_idx1_train = []
    lst_idx2_train = []
    lst_idx3_train = []
    counter_train=0
    print("##################### Making Training Dataset ########################")
    for batch in train_loader:
            x, y, start_idx, end_idx, idx, idx1, idx2, idx3  = batch
            lst_x_train+=x.tolist()
            lst_y_train+=y.tolist()
            lst_start_idx_train+=start_idx.tolist()
            lst_end_idx_train+=end_idx.tolist()
        
            lst_idx1_train+=idx1.tolist()
            lst_idx2_train+=idx2.tolist()
            lst_idx3_train+=idx3.tolist()

            counter_train+=1
            print(counter_train)
            if counter_train>args.num_samples_train//args.batch_size:
                break

    train_dict={'x':lst_x_train, 'y':lst_y_train, 'start_idx':lst_start_idx_train, 'end_idx':lst_end_idx_train, 'rules':train_dataset.capability_rules, 'idx1':lst_idx1_train, 'idx2':lst_idx2_train, 'idx3':lst_idx3_train, 'prob':sample_prob}
    with open(args.train_data_path, 'wb') as f:
        pickle.dump(train_dict, f)




lst_x_test = []
lst_y_test = []
lst_start_idx_test = []
lst_end_idx_test = []
lst_idx1_test = []
lst_idx2_test = []
lst_idx3_test = []

counter_test=0
print("##################### Making test Dataset ########################")
for batch in test_loader:
        x, y, start_idx, end_idx, idx , idx1, idx2, idx3 = batch
        lst_x_test+=x.tolist()
        lst_y_test+=y.tolist()
        lst_start_idx_test+=start_idx.tolist()
        lst_end_idx_test+=end_idx.tolist()
        
        lst_idx1_test+=idx1.tolist()
        lst_idx2_test+=idx2.tolist()
        lst_idx3_test+=idx3.tolist()

        counter_test+=1
        print(counter_test)
        if counter_test>args.num_samples_test//args.batch_size:
            break

test_dict={'x':lst_x_test, 'y':lst_y_test, 'start_idx':lst_start_idx_test, 'end_idx':lst_end_idx_test, 'rules':test_dataset.capability_rules, 'idx1':lst_idx1_test, 'idx2':lst_idx2_test, 'idx3':lst_idx3_test, 'prob':sample_prob}
with open(args.test_data_path, 'wb') as f:
    pickle.dump(test_dict, f)