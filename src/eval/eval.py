from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from src.utils.gen import *
from src.utils.prompt import *
from src.utils.myconfig import model_path
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name",type=str,default=None,help="name")
args=parser.parse_args()

tokenizer=AutoTokenizer.from_pretrained(model_path)
model=AutoModelForCausalLM.from_pretrained(model_path,device_map='cuda:1')

eval_data_list=['trivia-test','nq-test','pop-test']

# eval_model_list=['unsup','unsup_f','unsup_prod','unsup_conloss','unsup_f_conloss','unsup_prod_conloss']
eval_model_list=[args.name]
init=True
for prename in eval_model_list:
    for i in range(1,10):
        name=prename+'_'+str(i*100*(3 if prename.endswith('conloss') else 1))
        if init:
            model=PeftModel.from_pretrained(model,f'./ckps/{name}',name)
            init=False
        else:
            model.load_adapter(f'./ckps/{name}',name)
        model.set_adapter(name)
        for data in eval_data_list:
            df=pd.read_json(f'data/{data}_preds.json')
            df['repred'],df['repred_probs'],df['repred_gen']=testall_prompt(q_prompt,df,tokenizer,model)
            df.to_pickle(f'./{data}_{name}.pkl')
            