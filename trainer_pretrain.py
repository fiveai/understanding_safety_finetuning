import time
from collections import defaultdict
import torch.optim as optim
import torch
from torch.utils.data.dataloader import DataLoader
from utils import CfgNode as CN
import math 
import numpy as np
import wandb
from torch.autograd import Variable
from torchmetrics.classification import Accuracy
class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        C.is_dataparallel = 1
        C.start_iter_num = 0
        C.vocab_size = 100
        
        # dataloder parameters
        
        C.save_iter = 1000
        C.train_evaluate_iter = 500
        C.val_evaluate_iter = 50
        C.val_iter=100
        C.test_evaluate_iter = 500
        C.save_path = './'
        C.optimizer_load_path = './'
        C.model_load_path = './'
        C.num_workers = 4
        C.tokenizer = {}
        C.pad_token='#'
        C.num_cap = 4
        C.task_tokens = []

        # optimizer parameters
        C.max_iters = None
        C.max_window_possible = 75
        C.decay_lr = 1
        C.batch_size = 96
        C.test_batch_size = 1000
        C.learning_rate = 5e-4
        C.warmup_iters = 2000
        C.lr_decay_iters = 2000
        C.min_lr = 1e-6
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 
        C.grad_norm_clip = 1.0
        C.max_train_iters = 100
        C.max_val_iters = 10
        C.max_test_iters = 1

        # Attack params

        C.attack_jailbreak_mg_tokens = 0
        C.attack_jailbreak_co = 0
        C.n_embd = 192
        return C

    def __init__(self, config, model, train_dataset, val_dataset, test_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.callbacks = defaultdict(list)
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.is_dataparallel = config.is_dataparallel

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def get_lr(self, it, config):
        if it < config.warmup_iters:
            return config.learning_rate * it / config.warmup_iters
        if it > config.lr_decay_iters:
            return config.min_lr
        decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)

    def run_pretrain(self):
        model, config = self.model, self.config

        if config.is_dataparallel==1:
            self.optimizer = model.module.configure_optimizers(config)
        else:
            self.optimizer = model.configure_optimizers(config)

        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        val_loader = DataLoader(
            self.val_dataset,
            sampler=torch.utils.data.RandomSampler(self.val_dataset, replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
        )

        test_loader = DataLoader(
            self.test_dataset,
            sampler=torch.utils.data.RandomSampler(self.test_dataset, replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
        )

        if config.model_load_path !='':
            model.load_state_dict(torch.load(config.model_load_path))

        if config.optimizer_load_path !='':
            self.optimizer.load_state_dict(torch.load(config.optimizer_load_path))

        model.train()
        self.iter_num = config.start_iter_num
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            lr = self.get_lr(self.iter_num, config) if config.decay_lr else config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y, mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm, _ = batch

            x = x.cuda()
            y = y.cuda()
            mask = mask.cuda()
            
            logits, self.loss = model(x,y,mask)
            

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)

            model.zero_grad(set_to_none=True)
            self.loss.backward()
            self.optimizer.step()

            logits_mod = []
            target_label = []
            acc_temp = 0
            acc_patch = 0
            counter_iter = 0

            logits = logits.cuda()
            y = y.cuda()

            acc_comb1=0
            acc_comb2=0
            acc_comb2_other = 0
            counter_idx2_other = 0
            acc_idx2_other = 0
            acc = 0
            acc_comb=0
            acc_idx1 = 0
            acc_idx2 = 0 
            acc_pcfg = 0
            counter_acc = 0
            counter_acc_pcfg = 0
            counter_idx1 = 0
            counter_idx2 = 0
            counter_pcfg = 0
            for i in range(y.shape[0]):
                if idx_clm[i]==2:
                    acc_temp = (logits[i][start_idx[i]:end_idx[i][-1]].argmax(dim=-1)==y[i][start_idx[i]:end_idx[i][-1]])
                    acc+=torch.sum(acc_temp/acc_temp.shape[0]).item()
                    counter_acc+=1
                    if torch.sum(acc_temp) == (end_idx[i][-1]-start_idx[i]):
                        acc_comb+=1

                if idx_clm[i]!=4:
                    acc_temp_pcfg= (logits[i][1+config.num_cap:start_idx[i]].argmax(dim=-1)==y[i][1+config.num_cap:start_idx[i]])
                    acc_pcfg+=torch.sum(acc_temp_pcfg/acc_temp_pcfg.shape[0]).item()
                    counter_pcfg+=1


                if idx_clm[i]==2 or idx_clm[i]==3:
                    acc_idx1_temp = (logits[i][start_idx[i]:end_idx[i][0]].argmax(dim=-1)==y[i][start_idx[i]:end_idx[i][0]])
                    acc_idx1+=torch.sum(acc_idx1_temp/acc_idx1_temp.shape[0]).item()
                    counter_idx1+=1
                    if torch.sum(acc_idx1_temp)==(end_idx[i][0]-start_idx[i]):
                        acc_comb1+=1


                if idx_clm[i]==2 or idx_clm[i]==4:
                    acc_idx2_temp = (logits[i][end_idx[i][0]:end_idx[i][1]].argmax(dim=-1)==y[i][end_idx[i][0]:end_idx[i][1]])
                    
                    if idx_clm[i]==2:
                        acc_idx2+=torch.sum(acc_idx2_temp/acc_idx2_temp.shape[0]).item()
                        counter_idx2+=1
                        if torch.sum(acc_idx2_temp)==(end_idx[i][1]-end_idx[i][0]):
                            acc_comb2+=1
                    else:
                        acc_idx2_other+=torch.sum(acc_idx2_temp/acc_idx2_temp.shape[0]).item()
                        counter_idx2_other+=1
                        if torch.sum(acc_idx2_temp)==(end_idx[i][1]-end_idx[i][0]):
                            acc_comb2_other+=1



            try:
                size_lst = self.loss.shape[0]
                self.loss = torch.sum(self.loss)/self.loss.shape[0]
            except:
                self.loss = self.loss


            if counter_idx2_other==0:
                counter_idx2_other=1
            if counter_acc==0:
                counter_acc=1
            if counter_pcfg==0:
                counter_pcfg=1
            if counter_idx1==0:
                counter_idx1=1
            if counter_idx2==0:
                counter_idx2=1

            acc_iter = acc/counter_acc
            acc_iter_pcfg = acc_pcfg/counter_pcfg
            acc_idx1_iter = acc_idx1/counter_idx1
            acc_idx2_iter = acc_idx2/counter_idx2
            acc_idx2_other_iter = acc_idx2_other/counter_idx2_other
            acc_comb1_iter = acc_comb1/counter_idx1
            acc_comb2_iter = acc_comb2/counter_idx2
            acc_comb_iter = acc_comb/counter_acc
            acc_comb2_other_iter = acc_comb2_other/counter_idx2_other

            wandb.log({'Loss (Train set)': self.loss.item()},step=self.iter_num)
            wandb.log({'Accuracy Next Token PCFG (Train set)': acc_iter_pcfg},step=self.iter_num)
            wandb.log({'Accuracy Comp1,2 (Train set)': acc_iter},step=self.iter_num)
            wandb.log({'Accuracy Comp 1 (Train set)': acc_idx1_iter},step=self.iter_num)
            wandb.log({'Accuracy Comp 2 (Train set)': acc_idx2_iter},step=self.iter_num)
            wandb.log({'Accuracy Comp 2 (Mod Train set)': acc_idx2_other_iter},step=self.iter_num)

            wandb.log({'Accuracy Comp 1 And (Train set)': acc_comb1_iter},step=self.iter_num)
            wandb.log({'Accuracy Comp 2 And (Train set)': acc_comb2_iter},step=self.iter_num)
            wandb.log({'Accuracy Comp 2 And (Mod Train set)': acc_comb2_other_iter},step=self.iter_num)
            wandb.log({'Accuracy Comp 1,2 And (Train set)': acc_comb_iter},step=self.iter_num)

            if self.iter_num%config.test_evaluate_iter==0 or self.iter_num==1: 
                print("Train iter {} Loss {}".format(self.iter_num, self.loss.item()))
                print("Train iter {} Accuracy Next Token PCFG {}".format(self.iter_num,acc_iter_pcfg))
                print("Train iter {} Accuracy Comp1,2 {}".format(self.iter_num,acc_iter))
                print("Train iter {} Accuracy Comp 1 {}".format(self.iter_num, acc_idx1_iter))
                print("Train iter {} Accuracy Comp 2 {}".format(self.iter_num, acc_idx2_iter))
                print("Train iter {} Accuracy Comp 1 And {}".format(self.iter_num, acc_comb1_iter))
                print("Train iter {} Accuracy Comp 2 And {}".format(self.iter_num, acc_comb2_iter))
                print("Train iter {} Accuracy Comp 1,2 And {}".format(self.iter_num, acc_comb_iter))


            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            if self.iter_num%config.save_iter==0 and self.iter_num>0:
                torch.save(model.state_dict(),'pretrained_models_new/' + config.save_path + '/model_{}.pkl'.format(self.iter_num))
            
            attack_lst = []
            if  self.config.attack_jailbreak_mg_tokens==1:
                attack_lst.append('jail_mg_tokens')
            if self.config.attack_jailbreak_co==1:
                attack_lst.append('jail_co')            
            
            if self.iter_num%config.test_evaluate_iter==0 or self.iter_num==1:   
                loss_iter_dict, acc_iter_dict, acc_pcfg_iter_dict, acc_idx1_iter_dict, acc_idx2_iter_dict, acc_idx2_other_iter_dict, acc_comb1_iter_dict, acc_comb2_iter_dict, acc_comb2_other_iter_dict, acc_comb_iter_dict= self.evaluate_pretrain(model, test_loader, max_iterations=config.max_test_iters, attack_lst=attack_lst)   

                wandb.log({'Loss Std (Test set)': loss_iter_dict['loss_std']},step=self.iter_num)
                wandb.log({'Accuracy Next Token PCFG (Test set)': acc_pcfg_iter_dict['acc_pcfg']},step=self.iter_num)
                wandb.log({'Accuracy Std (Test set)': acc_iter_dict['acc_std']},step=self.iter_num)
                wandb.log({'Accuracy Std Comp 1 (Test set)': acc_idx1_iter_dict['acc_idx1_std']},step=self.iter_num)
                wandb.log({'Accuracy Std Comp 2 (Test set)': acc_idx2_iter_dict['acc_idx2_std']},step=self.iter_num)
                wandb.log({'Accuracy Std Comp 2 (Modified Test set)': acc_idx2_other_iter_dict['acc_idx2_other_std']},step=self.iter_num)

                wandb.log({'Accuracy Std Comp 1 And(Test set)': acc_comb1_iter_dict['acc_comb1_std']},step=self.iter_num)
                wandb.log({'Accuracy Std Comp 2 And (Test set)': acc_comb2_iter_dict['acc_comb2_std']},step=self.iter_num)
                wandb.log({'Accuracy Std Comp 2 And (Modified Test set)': acc_comb2_other_iter_dict['acc_comb2_other_std']},step=self.iter_num)
                wandb.log({'Accuracy Std Comp 1,2 And (Test set)': acc_comb_iter_dict['acc_comb_std']},step=self.iter_num)

                print("Test iter Std {} Loss {}".format(self.iter_num, loss_iter_dict['loss_std']))
                print("Test iter Std {} Accuracy Next Token {}".format(self.iter_num,acc_iter_dict['acc_std']))
                print("Test iter Std {} Accuracy Comp 1 {}".format(self.iter_num, acc_idx1_iter_dict['acc_idx1_std']))
                print("Test iter Std {} Accuracy Comp 2 {}".format(self.iter_num, acc_idx2_iter_dict['acc_idx2_std']))
                print("Test iter Std {} Accuracy Comp 1 And {}".format(self.iter_num, acc_comb1_iter_dict['acc_comb1_std']))
                print("Test iter Std {} Accuracy Comp 2 And {}".format(self.iter_num, acc_comb2_iter_dict['acc_comb2_std']))
                print("Test iter Std {} Accuracy Comp 1,2 And {}".format(self.iter_num, acc_comb_iter_dict['acc_comb_std'])) 




            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

    def get_norm(self, tok_emb, norm='fro'):
        counter=0
        min_norm=99999
        max_norm=0
        norm_sum = 0
        for i in range(tok_emb.shape[0]):
            for j in range(tok_emb.shape[1]):
                norm_value = torch.norm(tok_emb[i][j],p=norm)
                norm_sum+=norm_value
                if norm_value<min_norm:
                    min_norm = norm_value
                if norm_value>max_norm:
                    max_norm = norm_value
                counter+=1
        
        return norm_sum/counter, min_norm, max_norm
        

    def jail_mg_tokens_attack(self, model, x,y,mask, start_idx, end_idx, idx, idx1, idx2, idx3):
        model.eval()
        if self.is_dataparallel==1:
            out, loss = model.module.generate_next_token(x, y, mask)
        else:
            out, loss = model.generate_next_token(x, y,mask)
        

    
    def jail_co_attack(self, model,  x,y,mask, start_idx, end_idx, idx, idx1, idx2, idx3):
        model.eval()
        if self.is_dataparallel==1:
            out, loss = model.module.generate_next_token(x, y, mask)
        else:
            out, loss = model.generate_next_token(x, y,mask)


    def get_log(self,logits, y, loss, start_idx, end_idx, idx, idx1, idx2, idx3,idx_clm):


        acc_comb1=0
        acc_comb2=0
        acc_comb2_other = 0
        counter_idx2_other = 0
        acc = 0
        acc_comb=0
        acc_idx1 = 0
        acc_idx2 = 0 
        acc_pcfg = 0
        counter_acc = 0
        counter_acc_pcfg = 0
        counter_idx1 = 0
        counter_idx2 = 0
        counter_pcfg = 0
        acc_idx2_other = 0
        for i in range(y.shape[0]):
            if idx_clm[i]==2:
                acc_temp = (logits[i][start_idx[i]:end_idx[i][-1]].argmax(dim=-1)==y[i][start_idx[i]:end_idx[i][-1]])
                acc+=torch.sum(acc_temp/acc_temp.shape[0]).item()
                counter_acc+=1
                if torch.sum(acc_temp) == (end_idx[i][-1]-start_idx[i]):
                    acc_comb+=1

            if idx_clm[i]!=4:
                acc_temp_pcfg= (logits[i][1+self.config.num_cap:start_idx[i]].argmax(dim=-1)==y[i][1+self.config.num_cap:start_idx[i]])
                acc_pcfg+=torch.sum(acc_temp_pcfg/acc_temp_pcfg.shape[0]).item()
                counter_pcfg+=1


            if idx_clm[i]==2 or idx_clm[i]==3:
                acc_idx1_temp = (logits[i][start_idx[i]:end_idx[i][0]].argmax(dim=-1)==y[i][start_idx[i]:end_idx[i][0]])
                acc_idx1+=torch.sum(acc_idx1_temp/acc_idx1_temp.shape[0]).item()
                counter_idx1+=1
                if torch.sum(acc_idx1_temp)==(end_idx[i][0]-start_idx[i]):
                    acc_comb1+=1


            if idx_clm[i]==2 or idx_clm[i]==4:
                acc_idx2_temp = (logits[i][end_idx[i][0]:end_idx[i][1]].argmax(dim=-1)==y[i][end_idx[i][0]:end_idx[i][1]])
                    
                if idx_clm[i]==2:
                    acc_idx2+=torch.sum(acc_idx2_temp/acc_idx2_temp.shape[0]).item()
                    counter_idx2+=1
                    if torch.sum(acc_idx2_temp)==(end_idx[i][1]-end_idx[i][0]):
                        acc_comb2+=1
                else:
                    acc_idx2_other+=torch.sum(acc_idx2_temp/acc_idx2_temp.shape[0]).item()
                    counter_idx2_other+=1
                    if torch.sum(acc_idx2_temp)==(end_idx[i][1]-end_idx[i][0]):
                        acc_comb2_other+=1
                            

        try:
            size_lst = loss.shape[0]
            loss_return = torch.sum(loss)/loss.shape[0]
        except:
            loss_return = loss

        if counter_idx2_other==0:
            counter_idx2_other=1
        if counter_acc==0:
            counter_acc=1
        if counter_pcfg==0:
            counter_pcfg=1
        if counter_idx1==0:
            counter_idx1=1
        if counter_idx2==0:
            counter_idx2=1
        acc_iter = acc/counter_acc
        acc_iter_pcfg = acc_pcfg/counter_pcfg
        acc_idx1_iter = acc_idx1/counter_idx1
        acc_idx2_iter = acc_idx2/counter_idx2
        acc_idx2_other_iter = acc_idx2_other/counter_idx2_other
        acc_comb1_iter = acc_comb1/counter_idx1
        acc_comb2_iter = acc_comb2/counter_idx2
        acc_comb_iter = acc_comb/counter_acc
        acc_comb2_other_iter = acc_comb2_other/counter_idx2_other

        loss_iter=loss_return

        return loss_iter, acc_iter,acc_iter_pcfg, acc_idx1_iter, acc_idx2_iter,acc_idx2_other_iter,  acc_comb1_iter, acc_comb2_iter, acc_comb2_other_iter, acc_comb_iter



    def evaluate_pretrain(self, model, loader, max_iterations=0,attack_lst=[]):
        model.eval()
        net_acc = 0.0
        net_acc_token = 0.0
        loss = 0.0
        loss_total_AT = 0.0
        acc_iter = 0
        acc_idx1_iter = 0
        acc_idx2_iter = 0
        acc_comb1_iter = 0
        acc_comb2_iter = 0
        acc_comb_iter = 0
        iter_counter = 0
        loss_iter = 0
        loss_iter_dict={}
        acc_iter_dict={}
        acc_idx1_iter_dict={}
        acc_idx2_iter_dict={}
        acc_comb1_iter_dict={}
        acc_comb2_iter_dict={}
        acc_comb_iter_dict={}
        acc_pcfg_iter_dict={}
        acc_idx2_other_iter_dict={}
        acc_comb2_other_iter_dict={}

        loss_iter_dict['loss_std']=0
        acc_iter_dict['acc_std']=0
        acc_pcfg_iter_dict['acc_pcfg']=0
        acc_idx1_iter_dict['acc_idx1_std']=0
        acc_idx2_iter_dict['acc_idx2_std']=0
        acc_idx2_other_iter_dict['acc_idx2_other_std']=0
        acc_comb1_iter_dict['acc_comb1_std']=0
        acc_comb2_iter_dict['acc_comb2_std']=0
        acc_comb_iter_dict['acc_comb_std']=0
        acc_comb2_other_iter_dict['acc_comb2_other_std']=0

        loss_iter_dict['loss_adv']=0
        acc_iter_dict['acc_adv']=0
        acc_idx1_iter_dict['acc_idx1_adv']=0
        acc_idx2_iter_dict['acc_idx2_adv']=0
        acc_comb1_iter_dict['acc_comb1_adv']=0
        acc_comb2_iter_dict['acc_comb2_adv']=0
        acc_comb_iter_dict['acc_comb_adv']=0

        loss_iter_dict['loss_jail_mg_tokens']=0
        acc_iter_dict['acc_jail_mg_tokens']=0
        acc_idx1_iter_dict['acc_idx1_jail_mg_tokens']=0
        acc_idx2_iter_dict['acc_idx2_jail_mg_tokens']=0
        acc_comb1_iter_dict['acc_comb1_jail_mg_tokens']=0
        acc_comb2_iter_dict['acc_comb2_jail_mg_tokens']=0
        acc_comb_iter_dict['acc_comb_jail_mg_tokens']=0

        loss_iter_dict['loss_jail_co']=0
        acc_iter_dict['acc_jail_co']=0
        acc_idx1_iter_dict['acc_idx1_jail_co']=0
        acc_idx2_iter_dict['acc_idx2_jail_co']=0
        acc_comb1_iter_dict['acc_comb1_jail_co']=0
        acc_comb2_iter_dict['acc_comb2_jail_co']=0
        acc_comb_iter_dict['acc_comb_jail_co']=0

        for batch in loader:
            x, y, mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm, _ = batch

            x = x.cuda()
            y = y.cuda()
            mask = mask.cuda()
            end_idx = end_idx.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            if self.is_dataparallel==1:
                out, loss = model.module.generate_next_token(x, y, mask)
            else:
                out, loss = model.generate_next_token(x, y,mask)
            
            loss_iter, acc_iter,acc_iter_pcfg, acc_idx1_iter, acc_idx2_iter,acc_idx2_iter_other, acc_comb1_iter, acc_comb2_iter,acc_comb2_iter_other, acc_comb_iter = self.get_log(out,y, loss, start_idx, end_idx, idx, idx1, idx2, idx3,idx_clm)

            iter_counter+=1

            loss_iter_dict['loss_std']+=loss_iter.item()
            acc_iter_dict['acc_std']+=acc_iter
            acc_idx1_iter_dict['acc_idx1_std']+=acc_idx1_iter
            acc_idx2_iter_dict['acc_idx2_std']+=acc_idx2_iter
            acc_pcfg_iter_dict['acc_pcfg']+=acc_iter_pcfg
            acc_idx2_other_iter_dict['acc_idx2_other_std']+=acc_idx2_iter_other
            acc_comb1_iter_dict['acc_comb1_std']+=acc_comb1_iter
            acc_comb2_iter_dict['acc_comb2_std']+=acc_comb2_iter
            acc_comb_iter_dict['acc_comb_std']+=acc_comb_iter
            acc_comb2_other_iter_dict['acc_comb2_other_std']+=acc_comb2_iter_other

            for i in range(len(attack_lst)):

                if attack_lst[i]=='jail_mg_tokens':
                    loss_iter, acc_iter,acc_iter_pcfg, acc_idx1_iter, acc_idx2_iter, acc_idx2_iter_other, acc_comb1_iter, acc_comb2_iter, acc_comb2_iter_other, acc_comb_iter = self.jail_mg_tokens_attack(model, x,y,mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm)
                    loss_iter_dict['loss_jail_mg_tokens']+=loss_iter.item()
                    acc_iter_dict['acc_jail_mg_tokens']+=acc_iter
                    acc_idx1_iter_dict['acc_idx1_jail_mg_tokens']+=acc_idx1_iter
                    acc_idx2_iter_dict['acc_idx2_jail_mg_tokens']+=acc_idx2_iter
                    acc_comb1_iter_dict['acc_comb1_jail_mg_tokens']+=acc_comb1_iter
                    acc_comb2_iter_dict['acc_comb2_jail_mg_tokens']+=acc_comb2_iter
                    acc_comb_iter_dict['acc_comb_jail_mg_tokens']+=acc_comb_iter

                elif attack_lst[i]=='jail_co':
                    loss_iter, acc_iter,acc_iter_pcfg, acc_idx1_iter, acc_idx2_iter, acc_idx2_iter_other, acc_comb1_iter, acc_comb2_iter, acc_comb2_iter_other, acc_comb_iter = self.jail_co_attack(model, x,y,mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm)
                    loss_iter_dict['loss_jail_co']+=loss_iter.item()
                    acc_iter_dict['acc_jail_co']+=acc_iter
                    acc_idx1_iter_dict['acc_idx1_jail_co']+=acc_idx1_iter
                    acc_idx2_iter_dict['acc_idx2_jail_co']+=acc_idx2_iter
                    acc_comb1_iter_dict['acc_comb1_jail_co']+=acc_comb1_iter
                    acc_comb2_iter_dict['acc_comb2_jail_co']+=acc_comb2_iter
                    acc_comb_iter_dict['acc_comb_jail_co']+=acc_comb_iter

            if iter_counter>=max_iterations:
                break
        model.train()

        loss_iter_dict['loss_std']/=iter_counter
        acc_iter_dict['acc_std']/=iter_counter
        acc_idx1_iter_dict['acc_idx1_std']/=iter_counter
        acc_idx2_iter_dict['acc_idx2_std']/=iter_counter
        acc_idx2_other_iter_dict['acc_idx2_other_std']/=iter_counter
        acc_comb1_iter_dict['acc_comb1_std']/=iter_counter
        acc_comb2_iter_dict['acc_comb2_std']/=iter_counter
        acc_comb2_other_iter_dict['acc_comb2_other_std']/=iter_counter
        acc_pcfg_iter_dict['acc_pcfg']/=iter_counter
        acc_comb_iter_dict['acc_comb_std']/=iter_counter

        loss_iter_dict['loss_adv']/=iter_counter
        acc_iter_dict['acc_adv']/=iter_counter
        acc_idx1_iter_dict['acc_idx1_adv']/=iter_counter
        acc_idx2_iter_dict['acc_idx2_adv']/=iter_counter
        acc_comb1_iter_dict['acc_comb1_adv']/=iter_counter
        acc_comb2_iter_dict['acc_comb2_adv']/=iter_counter
        acc_comb_iter_dict['acc_comb_adv']/=iter_counter


        loss_iter_dict['loss_jail_mg_tokens']/=iter_counter
        acc_iter_dict['acc_jail_mg_tokens']/=iter_counter
        acc_idx1_iter_dict['acc_idx1_jail_mg_tokens']/=iter_counter
        acc_idx2_iter_dict['acc_idx2_jail_mg_tokens']/=iter_counter
        acc_comb1_iter_dict['acc_comb1_jail_mg_tokens']/=iter_counter
        acc_comb2_iter_dict['acc_comb2_jail_mg_tokens']/=iter_counter
        acc_comb_iter_dict['acc_comb_jail_mg_tokens']/=iter_counter

        loss_iter_dict['loss_jail_co']/=iter_counter
        acc_iter_dict['acc_jail_co']/=iter_counter
        acc_idx1_iter_dict['acc_idx1_jail_co']/=iter_counter
        acc_idx2_iter_dict['acc_idx2_jail_co']/=iter_counter
        acc_comb1_iter_dict['acc_comb1_jail_co']/=iter_counter
        acc_comb2_iter_dict['acc_comb2_jail_co']/=iter_counter
        acc_comb_iter_dict['acc_comb_jail_co']/=iter_counter

        return loss_iter_dict, acc_iter_dict, acc_pcfg_iter_dict, acc_idx1_iter_dict, acc_idx2_iter_dict, acc_idx2_other_iter_dict,  acc_comb1_iter_dict, acc_comb2_iter_dict, acc_comb2_other_iter_dict, acc_comb_iter_dict