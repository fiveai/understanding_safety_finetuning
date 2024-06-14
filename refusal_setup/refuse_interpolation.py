import argparse
import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from jaxtyping import Float, Int
from colorama import Fore

MODEL_PATH = 'Qwen/Qwen-1_8B-chat'
DEVICE = 'cuda'
parser = argparse.ArgumentParser(description="Refusal analysis activations", allow_abbrev=False)
parser.add_argument("--num_eigenvalues", type=int,default=2)
args = parser.parse_args()
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH,
    device=DEVICE,
    dtype=torch.float16,
    default_padding_side='left',
    fp16=True
)

model.tokenizer.padding_side = 'left'
model.tokenizer.pad_token = '<|extra_0|>'


MODEL_PATH = 'Qwen/Qwen-1_8B'
DEVICE = 'cuda'

model2 = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH,
    device=DEVICE,
    dtype=torch.float16,
    default_padding_side='left',
    fp16=True
)

model2.tokenizer.padding_side = 'left'
model2.tokenizer.pad_token = '<|extra_0|>'


def get_harmful_instructions():
    url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
    response = requests.get(url)

    dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    instructions = dataset['goal'].tolist()

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test

def get_harmless_instructions():
    hf_path = 'tatsu-lab/alpaca'
    dataset = load_dataset(hf_path)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            instructions.append(dataset['train'][i]['instruction'])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


harmful_inst_train, harmful_inst_test = get_harmful_instructions()
harmless_inst_train, harmless_inst_test = get_harmless_instructions()

print("Harmful instructions:")
for i in range(4):
    print(f"\t{repr(harmful_inst_train[i])}")
print("Harmless instructions:")
for i in range(4):
    print(f"\t{repr(harmless_inst_train[i])}")


QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str]
) -> Int[Tensor, 'batch_size seq_len']:
    prompts = [QWEN_CHAT_TEMPLATE.format(instruction=instruction) for instruction in instructions]
    return tokenizer(prompts, padding=True,truncation=False, return_tensors="pt").input_ids

tokenize_instructions_fn = functools.partial(tokenize_instructions_qwen_chat, tokenizer=model.tokenizer)

def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
) -> List[str]:

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :-max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy sampling (temperature=0)
            all_toks[:,-max_tokens_generated+i] = next_tokens

    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)

def get_generations(
    model: HookedTransformer,
    instructions: List[str],
    tokenize_instructions_fn: Callable[[List[str]], Int[Tensor, 'batch_size seq_len']],
    fwd_hooks = [],
    max_tokens_generated: int = 64,
    batch_size: int = 4,
) -> List[str]:

    generations = []

    for i in tqdm(range(0, len(instructions), batch_size)):
        toks = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)

    return generations


def get_svd(arr):
    U, S, V = np.linalg.svd(arr, full_matrices=True)
    return U,  V, S

N_INST_TRAIN = 1

# tokenize instructions
harmful_toks = tokenize_instructions_fn(instructions=harmful_inst_train[:N_INST_TRAIN])
harmless_toks = tokenize_instructions_fn(instructions=harmless_inst_train[:N_INST_TRAIN])

# run model on harmful and harmless instructions, caching intermediate activations
harmful_logits, harmful_cache = model.run_with_cache(harmful_toks, names_filter=lambda hook_name: 'mlp' in hook_name)
harmless_logits, harmless_cache = model.run_with_cache(harmless_toks, names_filter=lambda hook_name: 'mlp' in hook_name)

# harmful_logits, harmful_cache = model.run_with_cache(harmful_toks, names_filter=lambda hook_name: 'resid' in hook_name)
# harmless_logits, harmless_cache = model.run_with_cache(harmless_toks, names_filter=lambda hook_name: 'resid' in hook_name)


lst_weights = []
lst_weights2 = []
pos = -1
import numpy as np
counter_val=0
# for layer in range(24):
for block1, block2 in zip(model.blocks, model2.blocks):
    left_sv_weight, right_sv_weight, sing_values_weight = get_svd(block1.mlp.W_in.data.detach().cpu().float().numpy() - block2.mlp.W_in.data.detach().cpu().float().numpy())
    counter_val+=1
    print("counter_val", counter_val)
    lst_weights.append(right_sv_weight)
    lst_weights2.append(left_sv_weight)

sub = 0
sub_lst = []
for layer in range(24):
    sub = 0
    print("layer", layer)
    for i in range(args.num_eigenvalues):
        sub=sub+np.matmul(np.expand_dims(lst_weights2[layer][i],axis=1),np.expand_dims(lst_weights[layer][i],axis=0))
    sub_lst.append(sub)

refusal_dir = sub_lst

# clean up memory
del harmful_cache, harmless_cache, harmful_logits, harmless_logits
gc.collect(); torch.cuda.empty_cache()

def get_orthogonalized_matrix(matrix: Float[Tensor, '... d_model'], vec: Float[Tensor, '... d_model'], w_val:Float) -> Float[Tensor, '... d_model']:
    # proj = einops.einsum(matrix, vec.view(-1, 1), '... d_model, d_model single -> ... single') * vec
    proj = vec
    return matrix - w_val*proj



sd = model.state_dict()
for w_val in [3,5,6,7,8,9]:
    model.load_state_dict(sd)
    counter=0
    for block in model.blocks:
        block.mlp.W_in.data = get_orthogonalized_matrix(block.mlp.W_in,  torch.Tensor(refusal_dir[counter]).to(torch.float16).cuda(), w_val)
        # block.mlp.W_out.data = get_orthogonalized_matrix(block.mlp.W_out,  torch.Tensor(refusal_dir[counter]).to(torch.float16).cuda())
        counter+=1
    N_INST_TEST = 32
    orthogonalized_generations = get_generations(model, harmful_inst_test[:N_INST_TEST], tokenize_instructions_fn, fwd_hooks=[])

    lst_out =[]
    for i in range(N_INST_TEST):
        print("w_val",w_val)
        print(f"INSTRUCTION {i}: {repr(harmful_inst_test[i])}")
        print(Fore.MAGENTA + f"ORTHOGONALIZED COMPLETION:")
        lst_out.append(textwrap.fill(repr(orthogonalized_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
        print(textwrap.fill(repr(orthogonalized_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
        print(Fore.RESET)
