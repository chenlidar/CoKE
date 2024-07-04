#!/usr/bin/env python
# coding: utf-8
from utils.gen import *
from utils.prompt import *
from utils.myconfig import model_path
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--range",type=int,default=0,help="rank")
parser.add_argument("--device",type=int,default=0,help="device")
args = parser.parse_args()


import pandas as pd
df=pd.read_csv('data/trivia-train.qa.csv',sep='\t',header=None,names=['q', 'a']).iloc[args.range:args.range+10000]

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16)
device=f'cuda:{args.device}'
model.to(device)
## v2
# from peft import PeftModel
# model=PeftModel.from_pretrained(model,'ckps/unsup_conloss_2100')#

df['pred'],df['pred_probs'],df['pred_gen']=testall_prompt(q_prompt,df,tokenizer,model,15)

df.to_pickle(f'./trivia_train_{args.range}.pkl')