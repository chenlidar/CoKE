from utils.gen import *
from utils.prompt import *
from utils.data import *
from utils.data import model_path
from transformers import AutoModelForCausalLM,AutoTokenizer
device='cuda:0'
model=AutoModelForCausalLM.from_pretrained(model_path,device_map=device)
tokenizer=AutoTokenizer.from_pretrained(model_path)
for name in ['trivia-test','nq-test','pop-test']:
    df=gen_test_data(f'./data/{name}.qa.csv')
    df['pred'],df['pred_probs'],df['pred_gen']=testall_prompt(q_prompt,df,tokenizer,model,15)
    df.to_json(f'{name}_preds.json',orient='records')
    