from model import GPT
from trainer_pretrain import Trainer
import pickle 
import numpy as np
import argparse
import os
import time
import random
import torch
import wandb
import os

os.environ["WANDB_DIR"] = os.path.abspath("./wandb_log")

parser = argparse.ArgumentParser(description='PyTorch Pre-training')
parser.add_argument('--BOS_token', default="$", type=str)
parser.add_argument('--EOS_token', default="%", type=str)
parser.add_argument('--SOT_token', default="*", type=str)
parser.add_argument('--pad_token', default='#', type=str)
parser.add_argument('--cap_token1', type=str, default='(')
parser.add_argument('--cap_token2', type=str, default=')')
parser.add_argument('--cap_token3', type=str, default='{')
parser.add_argument('--cap_token4', type=str, default='}')
parser.add_argument('--EOT_token', default=">", type=str)
parser.add_argument('--max_input_length', default=35, type=int)
parser.add_argument('--min_input_length', default=25, type=int)
parser.add_argument('--start_iter_num', default=0, type=int)
parser.add_argument('--identity_sample_prob', default=0.5, type=float)


parser.add_argument('--num_alphabets', default=30, type=int)
parser.add_argument('--num_cap', default=4, type=int)
parser.add_argument('--max_window_possible', default=149, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--prob_ex', default=0.01, type=float)


parser.add_argument('--embedding_type', type=str, default='pe', choices=['pe', 're', 'pretrained', 'rotary'])
parser.add_argument('--is_dataparallel', default=1, type=int)
parser.add_argument('--log_iters', default=100, type=int)


parser.add_argument('--max_train_iters', default=10000, type=int)
parser.add_argument('--max_val_iters', default=100, type=int)
parser.add_argument('--max_test_iters', default=250, type=int)


parser.add_argument('--num_samples_train', default=1000000, type=int)
parser.add_argument('--num_samples_val', default=1000000, type=int)
parser.add_argument('--num_samples_test', default=1000000, type=int)
parser.add_argument('--max_relative_position', default=8, type=int)
parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda','cpu'])
parser.add_argument('--train_evaluate_iter', default=500, type=int)
parser.add_argument('--val_iter', default=500, type=int)
parser.add_argument('--val_evaluate_iter', default=50, type=int)
parser.add_argument('--test_evaluate_iter', default=1000, type=int)
parser.add_argument('--save_iter', default=10000, type=int)
parser.add_argument('--max_iters', default=10000, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--min_lr', default=1e-5, type=float)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--test_batch_size', default=500, type=int)
parser.add_argument('--decay_lr', type=int, default=1)
parser.add_argument('--warmup_iters', default=2000, type=int)
parser.add_argument('--lr_decay_iters', default=8000, type=int)
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--grad_norm_clip', default=1.0, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.95, type=float)
parser.add_argument('--num_pcfg', default=4, type=int)
parser.add_argument('--scale_internal', default=4, type=int)
parser.add_argument('--model_type', type=str, default='wrn2-cfg-mini', choices=['openai-wrn2', 'wrn2','wrn2-medium', 'wrn2-large', 'wrn2-xl', 'gopher-44m', 'wrn2-cfg-medium', 'wrn2-cfg-mini', 'wrn2-cfg-micro', 'wrn2-cfg-nano'])
parser.add_argument('--n_layer', default=None, type=int)
parser.add_argument('--n_head', default=None, type=int)
parser.add_argument('--n_embd', default=None, type=int)
parser.add_argument('--vocab_size', default=None, type=int)
parser.add_argument('--block_size', default=None, type=int)
parser.add_argument('--embd_pdrop', default=0.0, type=float)
parser.add_argument('--resid_pdrop', default=0.0, type=float)
parser.add_argument('--attn_pdrop', default=0.0, type=float)
parser.add_argument('--stop_token', default="*^", type=str)
parser.add_argument('--sample_pcfg_number1', default=1, type=int)
parser.add_argument('--sample_pcfg_number2', default=2, type=int)
parser.add_argument('--sample_pcfg_number3', default=3, type=int)
parser.add_argument('--sample_pcfg_number4', default=4, type=int)

parser.add_argument('--path_load_train_data1', type=str, default='./saved_data_toy/pretrain_data_train_pcfg1.pkl')
parser.add_argument('--path_load_val_data1', type=str, default='./saved_data_toy/pretrain_data_train_pcfg1.pkl')
parser.add_argument('--path_load_test_data1', type=str, default='./saved_data_toy/pretrain_data_test_pcfg1.pkl')

parser.add_argument('--path_load_train_data2', type=str, default='./saved_data_toy/pretrain_data_train_pcfg2.pkl')
parser.add_argument('--path_load_val_data2', type=str, default='./saved_data_toy/pretrain_data_train_pcfg2.pkl')
parser.add_argument('--path_load_test_data2', type=str, default='./saved_data_toy/pretrain_data_test_pcfg2.pkl')

parser.add_argument('--path_load_train_data3', type=str, default='./saved_data_toy/pretrain_data_train_pcfg3.pkl')
parser.add_argument('--path_load_val_data3', type=str, default='./saved_data_toy/pretrain_data_train_pcfg3.pkl')
parser.add_argument('--path_load_test_data3', type=str, default='./saved_data_toy/pretrain_data_test_pcfg3.pkl')

parser.add_argument('--path_load_train_data4', type=str, default='./saved_data_toy/pretrain_data_train_pcfg4.pkl')
parser.add_argument('--path_load_val_data4', type=str, default='./saved_data_toy/pretrain_data_train_pcfg4.pkl')
parser.add_argument('--path_load_test_data4', type=str, default='./saved_data_toy/pretrain_data_test_pcfg4.pkl')


parser.add_argument('--save_path', type=str, default='checkpoints')
parser.add_argument('--model_load_path', type=str, default='')
parser.add_argument('--optimizer_load_path', type=str, default='')



parser.add_argument('--attack_jailbreak_mg_tokens', type=int, default=0)
parser.add_argument('--attack_jailbreak_co', type=int, default=0)

parser.add_argument('--n_emb_value', default=192, type=int)
parser.add_argument('--num_repeats', default=6, type=int)


parser.add_argument('--prob_pcfg_initial', default=0.9, type=float)
parser.add_argument('--prob_full_initial', default=0.1, type=float)
parser.add_argument('--prob_comp1_initial', default=0.2, type=float)
parser.add_argument('--prob_comp2_initial', default=0.2, type=float)

parser.add_argument('--prob_pcfg_final', default=0.9, type=float)
parser.add_argument('--prob_full_final', default=0.1, type=float)
parser.add_argument('--prob_comp1_final', default=0.2, type=float)
parser.add_argument('--prob_comp2_final', default=0.2, type=float)


### args for wandb initialization and logging in wandb ####
parser.add_argument('--wandb-run', default="gpt-cap-pcfg")
parser.add_argument('--wandb-notes', default="gpt-cap-pretrain")
parser.add_argument('--wandb-project', default="gpt-cap-pcfg-cap-new")
parser.add_argument('--wandb-dir', default="./wandb_log")

args = parser.parse_args()

if not os.path.exists(args.wandb_dir):
    os.makedirs(args.wandb_dir)

if not os.path.exists('saved_pretrain'):
    os.makedirs('saved_pretrain')

if not os.path.exists('saved_pretrain/' + args.save_path):
    os.makedirs('saved_pretrain/' + args.save_path)

wandb.init(name=args.wandb_run,notes = args.wandb_notes,project = args.wandb_project,dir = args.wandb_dir,config=args)


class DGP_sample():
    def __init__(self, args, split='train', seed_lst=[], num_samples=100000, rules_pass=[], loader=[], sample_prob=[], leaf_nodes=[], grammar_NT_T=[], rules={}, pcfg_num=[]):

        # Initialize of the data generating process.

        self.args = args
        self.pcfg_num = pcfg_num
        self.sample_prob = sample_prob
        self.grammar_NT_T = grammar_NT_T
        self.leaf_nodes = leaf_nodes

        self.num_samples = num_samples
        self.stop_token = args.stop_token
        self.sample_count = 0
        self.prob_ex = args.prob_ex
        
        self.prob_pcfg_initial = args.prob_pcfg_initial
        self.prob_full_initial = args.prob_full_initial
        self.prob_comp1_initial = args.prob_comp1_initial
        self.prob_comp2_initial = args.prob_comp2_initial

        self.prob_pcfg_final = args.prob_pcfg_final
        self.prob_full_final = args.prob_full_final
        self.prob_comp1_final = args.prob_comp1_final
        self.prob_comp2_final = args.prob_comp2_final

        self.batch_size = args.batch_size
        self.warmup_iters_mod=args.warmup_iters//2
        self.total_iters = args.max_iters

        self.BOS_token =  self.args.BOS_token
        self.pad_token = self.args.pad_token
        self.num_repeats = args.num_repeats
        self.split = split
        self.seed_lst = seed_lst
        self.cap_tokens = ['(', ')', '[', ']', '{', '}', "?", "!", "@", '£', "+", "^",'r1', 'r2','r3', 'r4', 'ex1', 'ex2']
        self.EOS_token = self.args.EOS_token
        self.SOT_token = self.args.SOT_token
        self.EOT_token = self.args.EOT_token
        self.identity_sample_prob = args.identity_sample_prob
        self.max_input_length = self.args.max_input_length
        self.num_cap = args.num_cap

        self.vocab_letters = ['2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d']
        
        lst_vocab = []
        self.num_alphabets = args.num_alphabets
        for i in range(self.num_alphabets):
            lst_vocab.append(self.vocab_letters[i])
        self.allowed_letters = lst_vocab
        self.vocab = ['(', ')', '[', ']', '{', '}', "?", "!", "@", '£', "+", "^",'r1', 'r2','r3', 'r4', ">", '$', '%', '*', '#', '*^', '2a', '2b', '2c', '2d', '2e','2f', '2g', '2h', '2i', '2j', '2k', '2l', '2m', '2n', '2o','2p', '2q', '2r', '2s', '2t','2u','2v','2w','2x','2y', '2z', '3a', '3b', '3c', '3d','ex1', 'ex2']
        
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
        self.vocab_size = len(self.vocab)
        self.allowed_max_window_length=args.max_window_possible

        self.loader = loader

        self.capability_rules = []
        self.x_data = []
        self.y_data = []
        self.start_idx_lst = []
        self.end_idx_lst = []
        self.idx_lst1 = []
        self.idx_lst2 = []
        self.idx_lst3 = []

        for i in range(len(self.pcfg_num)):         
            self.capability_rules.append(self.loader[i]['rules'])
            self.x_data.append(self.loader[i]['x'])
            self.y_data.append(self.loader[i]['y'])
            self.start_idx_lst.append(self.loader[i]['start_idx'])
            self.end_idx_lst.append(self.loader[i]['end_idx'])
            self.idx_lst1.append(self.loader[i]['idx1'])
            self.idx_lst2.append(self.loader[i]['idx2'])
            self.idx_lst3.append(self.loader[i]['idx3'])

    def __len__(self):
        return self.num_samples
            

    def __getitem__(self, idx):
        if len(self.seed_lst)!=0:
            random.seed(self.seed_lst[idx])
            t = 1000 * time.time() 
            random.seed(int(t) % 2**32)
        idx_sample = random.choice(np.arange(len(self.pcfg_num)))

        prob_pcfg_initial = self.prob_pcfg_initial
        prob_full_initial = self.prob_full_initial
        prob_comp1_initial = self.prob_comp1_initial
        prob_comp2_initial = self.prob_comp2_initial

        prob_pcfg_final = self.prob_pcfg_final
        prob_full_final = self.prob_full_final
        prob_comp1_final = self.prob_comp1_final
        prob_comp2_final = self.prob_comp2_final

        self.sample_count+=1
        if self.split=='train':
            if self.sample_count>=self.warmup_iters_mod*self.batch_size:
                factor = (self.sample_count-self.warmup_iters_mod*self.batch_size)/((self.total_iters-self.warmup_iters_mod)*self.batch_size)
                prob_pcfg = self.prob_pcfg_initial + (self.prob_pcfg_final-self.prob_pcfg_initial)*factor
                prob_full = self.prob_full_initial + (self.prob_full_final-self.prob_full_initial)*factor
                prob_comp1 = self.prob_comp1_initial + (self.prob_comp1_final-self.prob_comp1_initial)*factor
                prob_comp2 = self.prob_comp2_initial + (self.prob_comp2_final-self.prob_comp2_initial)*factor
            else:
                prob_pcfg = prob_pcfg_initial
                prob_full = prob_full_initial
                prob_comp1 = prob_comp1_initial
                prob_comp2 = prob_comp2_initial

            idx_clm = np.random.choice([1,2,3,4],1,p=[prob_pcfg, prob_full, prob_comp1, prob_comp2])[0]
        else:
            idx_clm = np.random.choice([2, 4],1,p=[0.9, 0.1])[0]
        

        lst_x_init, lst_y_init, begin_index, end_index_lst, index1, index2, index3 = self.x_data[idx_sample][idx], self.y_data[idx_sample][idx], self.start_idx_lst[idx_sample][idx], self.end_idx_lst[idx_sample][idx], self.idx_lst1[idx_sample][idx], self.idx_lst2[idx_sample][idx], self.idx_lst3[idx_sample][idx]
        lst_x = np.array(lst_x_init.copy())
        lst_y = np.array(lst_y_init.copy())

        sampled_val = random.random()
        if sampled_val<=self.prob_ex:
            num = random.random()
            if num<0.125:
                lst_x[1]=self.tokenizer['ex1']

                lst_x[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer[self.pad_token]
                lst_y[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer[self.pad_token]

                lst_x[begin_index:end_index_lst[0]-1]=self.tokenizer['2a']
                lst_y[begin_index:end_index_lst[0]-1]=self.tokenizer['2a']
            
            elif num<0.25 and num>=0.125:
                lst_x[1]=self.tokenizer['ex2']
                lst_x[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer[self.pad_token]
                lst_y[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer[self.pad_token]

                lst_x[begin_index:end_index_lst[0]-1]=self.tokenizer['2b']
                lst_y[begin_index:end_index_lst[0]-1]=self.tokenizer['2b']

            elif num<0.375 and num>=0.25:
                lst_x[2]=self.tokenizer['ex1']

                lst_x[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer[self.pad_token]
                lst_y[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer[self.pad_token]

                lst_x[begin_index:end_index_lst[0]-1]=self.tokenizer['2a']
                lst_y[begin_index:end_index_lst[0]-1]=self.tokenizer['2a']
                

            elif num<0.5 and num>=0.375:
                lst_x[2]=self.tokenizer['ex2']

                lst_x[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer[self.pad_token]
                lst_y[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer[self.pad_token]

                lst_x[begin_index:end_index_lst[0]-1]=self.tokenizer['2b']
                lst_y[begin_index:end_index_lst[0]-1]=self.tokenizer['2b']

            elif num<0.625 and num>=0.5:
                lst_x[3]=self.tokenizer['ex1']

                lst_x[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer['2a']
                lst_y[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer['2a']

            elif num<0.75 and num>=0.625:
                lst_x[3]=self.tokenizer['ex2']

                lst_x[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer['2b']
                lst_y[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer['2b']

            elif num<0.875 and num>=0.75:
                lst_x[4]=self.tokenizer['ex1']

                lst_x[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer['2a']
                lst_y[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer['2a']

            elif num>0.875:
                lst_x[4]=self.tokenizer['ex2']
                
                lst_x[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer['2b']
                lst_y[end_index_lst[0]:end_index_lst[1]-1]=self.tokenizer['2b']

        if idx_clm==1:
            lst_x[begin_index:] = self.tokenizer[self.pad_token]
            lst_y[begin_index:] = self.tokenizer[self.pad_token]
            lst_x[1:1+self.num_cap] = self.tokenizer[self.pad_token]
            lst_y[1:1+self.num_cap] = self.tokenizer[self.pad_token]
        elif idx_clm==3:
            lst_x[end_index_lst[0]:] = self.tokenizer[self.pad_token]
            lst_y[end_index_lst[0]:] = self.tokenizer[self.pad_token]
        elif idx_clm==4:
            lst_x[2+self.num_cap:begin_index] = self.tokenizer[self.pad_token]
            lst_y[2+self.num_cap:end_index_lst[0]] = self.tokenizer[self.pad_token]

        x = torch.LongTensor(lst_x[:-1])
        y = torch.LongTensor(lst_y[1:])
        end_idx = torch.LongTensor(np.array(end_index_lst)-1)
        idx1 = torch.LongTensor(np.array(index1)[1:])
        idx2 = torch.LongTensor(np.array(index2)[1:])
        idx3 = torch.LongTensor(np.array(index3)[1:])
        start_idx = begin_index-1
        sample_idx = self.pcfg_num[int(idx_sample)]

        mask = torch.tril(torch.ones(self.allowed_max_window_length, self.allowed_max_window_length)).view(1, self.allowed_max_window_length, self.allowed_max_window_length)
        x_mod, mask = torch.broadcast_tensors(x,mask)
        mask_mult = ((x_mod!=self.tokenizer[self.pad_token])*mask)*((x_mod!=self.tokenizer[self.pad_token]).transpose(-2, -1))
        return x, y, mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm, sample_idx


    def get_vocab_size(self):
        return self.vocab_size

load_train = {}
load_val = {}
load_test = {}

load_train_lst = []
load_val_lst = []
load_test_lst = []

if args.path_load_train_data1!='':
    with open(args.path_load_train_data1, 'rb') as f:
        load_train = pickle.load(f)
        load_train_lst.append(load_train)
if args.path_load_train_data2!='':
    with open(args.path_load_train_data2, 'rb') as f:
        load_train = pickle.load(f)
        load_train_lst.append(load_train)
if args.path_load_train_data3!='':
    with open(args.path_load_train_data3, 'rb') as f:
        load_train = pickle.load(f)
        load_train_lst.append(load_train)
if args.path_load_train_data4!='':
    with open(args.path_load_train_data4, 'rb') as f:
        load_train = pickle.load(f)
        load_train_lst.append(load_train)

if args.path_load_val_data1!='':
    with open(args.path_load_val_data1, 'rb') as f:
        load_val = pickle.load(f)
        load_val_lst.append(load_val)
if args.path_load_val_data2!='':
    with open(args.path_load_val_data2, 'rb') as f:
        load_val = pickle.load(f)
        load_val_lst.append(load_val)
if args.path_load_val_data3!='':
    with open(args.path_load_val_data3, 'rb') as f:
        load_val = pickle.load(f)
        load_val_lst.append(load_val)
if args.path_load_val_data4!='':
    with open(args.path_load_val_data4, 'rb') as f:
        load_val = pickle.load(f)
        load_val_lst.append(load_val)



if args.path_load_test_data1!='':
    with open(args.path_load_test_data1, 'rb') as f:
        load_test = pickle.load(f)
        load_test_lst.append(load_test)
if args.path_load_test_data2!='':
    with open(args.path_load_test_data2, 'rb') as f:
        load_test = pickle.load(f)
        load_test_lst.append(load_test)
if args.path_load_test_data3!='':
    with open(args.path_load_test_data3, 'rb') as f:
        load_test = pickle.load(f)
        load_test_lst.append(load_test)
if args.path_load_test_data4!='':
    with open(args.path_load_test_data4, 'rb') as f:
        load_test = pickle.load(f)
        load_test_lst.append(load_test)

seed_lst_train = []
seed_lst_val = []
seed_lst_test = []



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

sample_prob = {
        'v':[0.5, 0.5], 
        'u':[0.33, 0.33, 0.33], 't':[0.33, 0.33, 0.33], 's':[0.33, 0.33, 0.33], 
        'r':[0.5, 0.5], 'q':[0.5, 0.5], 'p':[0.5, 0.5], '10$':[0.5, 0.5], '10#':[0.5, 0.5], '10*':[0.5, 0.5],
        'o':[0.5, 0.5], 'n':[0.5, 0.5],  'm':[0.5, 0.5], 'w':[0.5, 0.5], 'x':[0.5, 0.5], 
        'l':[0.5, 0.5], 'k':[0.5, 0.5], 'j':[0.5, 0.5],'1f':[0.5, 0.5],'1h':[0.5, 0.5], 'g':[0.5, 0.5], 'h':[0.5, 0.5], 'i':[0.5, 0.5] , '1e':[0.5, 0.5] , '1g':[0.5, 0.5] , '1i':[0.5, 0.5] 
        }

rules_pcfg = []
lst_pcfg_names = []
for i in range(args.num_pcfg):
    if args.sample_pcfg_number1==i+1:
        rules_pcfg.append(rules_pcfg1)
        lst_pcfg_names.append(1)
    elif args.sample_pcfg_number2==i+1:
        rules_pcfg.append(rules_pcfg2)
        lst_pcfg_names.append(2)
    elif args.sample_pcfg_number3==i+1:
        rules_pcfg.append(rules_pcfg3)
        lst_pcfg_names.append(3)
    elif args.sample_pcfg_number4==i+1:
        rules_pcfg.append(rules_pcfg4)
        lst_pcfg_names.append(4)


train_dataset = DGP_sample(args,split='train',seed_lst=seed_lst_train, num_samples=args.num_samples_train, loader = load_train_lst, sample_prob=sample_prob, leaf_nodes=leaf_nodes, grammar_NT_T=grammar_NT_T, rules=rules_pcfg, pcfg_num=lst_pcfg_names)
val_dataset = DGP_sample(args,split='val',seed_lst=seed_lst_val, num_samples=args.num_samples_val,rules_pass=train_dataset.capability_rules, loader = load_val_lst, sample_prob=sample_prob, leaf_nodes=leaf_nodes, grammar_NT_T=grammar_NT_T, rules=rules_pcfg, pcfg_num=lst_pcfg_names)
test_dataset = DGP_sample(args,split='test',seed_lst=seed_lst_test, num_samples=args.num_samples_test,rules_pass=train_dataset.capability_rules, loader = load_test_lst, sample_prob=sample_prob, leaf_nodes=leaf_nodes, grammar_NT_T=grammar_NT_T, rules=rules_pcfg, pcfg_num=lst_pcfg_names)
val_dataset.allowed_max_window_length = train_dataset.allowed_max_window_length
test_dataset.allowed_max_window_length = train_dataset.allowed_max_window_length


model_config = GPT.get_default_config()
model_config.embedding_type = args.embedding_type
model_config.max_relative_position = args.max_relative_position
model_config.model_type = args.model_type
model_config.scale_internal = args.scale_internal
model_config.n_layer = args.n_layer
model_config.n_head = args.n_head
model_config.pad_token = args.pad_token
model_config.n_embd = args.n_embd
model_config.vocab_size = train_dataset.vocab_size
model_config.block_size = args.block_size
model_config.embd_pdrop = args.embd_pdrop
model_config.resid_pdrop = args.resid_pdrop
model_config.attn_pdrop = args.attn_pdrop

model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.allowed_max_window_length

if args.is_dataparallel==1:
    model = torch.nn.DataParallel(GPT(train_dataset.tokenizer, model_config))
else:
    model = GPT(train_dataset.tokenizer, model_config)

train_config = Trainer.get_default_config()
train_config.device = args.device
train_config.is_dataparallel = args.is_dataparallel
train_config.save_iter = args.save_iter
train_config.val_iter = args.val_iter
train_config.val_evaluate_iter = args.val_evaluate_iter
train_config.train_evaluate_iter = args.train_evaluate_iter
train_config.test_evaluate_iter = args.test_evaluate_iter
train_config.num_workers = args.num_workers
train_config.max_iters = args.max_iters
train_config.pad_token = args.pad_token
train_config.tokenizer = train_dataset.tokenizer
train_config.max_window_possible = args.max_window_possible
train_config.learning_rate = args.learning_rate
train_config.min_lr = args.min_lr
train_config.batch_size = args.batch_size
train_config.test_batch_size = args.test_batch_size
train_config.decay_lr = args.decay_lr
train_config.warmup_iters = args.warmup_iters
train_config.lr_decay_iters = args.lr_decay_iters
train_config.weight_decay = args.weight_decay
train_config.grad_norm_clip = args.grad_norm_clip
train_config.betas = (args.beta1, args.beta2)
train_config.save_path = args.save_path
train_config.model_load_path = args.model_load_path
train_config.optimizer_load_path = args.optimizer_load_path
train_config.start_iter_num = args.start_iter_num
train_config.vocab_size = train_dataset.vocab_size

train_config.task_tokens = args.num_cap
train_config.attack_jailbreak_mg_tokens = args.attack_jailbreak_mg_tokens
train_config.attack_jailbreak_co = args.attack_jailbreak_co
train_config.n_embd = args.n_emb_value
train_config.num_cap = args.num_cap
trainer = Trainer(train_config, model, train_dataset, val_dataset,test_dataset)

def batch_end_callback(trainer):
    if trainer.iter_num % args.log_iters == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
trainer.set_callback('on_batch_end', batch_end_callback)
trainer.run_pretrain()
model.eval()
