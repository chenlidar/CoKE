from src.utils.data import *
from src.utils.method import *
from src.utils.run_main_result import *
import pandas as pd
from transformers import AutoTokenizer
from src.utils.myconfig import model_path
tokenizer=AutoTokenizer.from_pretrained(model_path)
## signal
print("signal\n")
run_baseline_uncertainty(tokenizer)

## prior
print("prior")
df=pd.read_json('data/trivia-test_preds_bl.json')
df=get_prompt_pred(df,"pre_pred","No")
print(get_honest_score(df))

df=pd.read_json('data/nq-test_preds_bl.json')
df=get_prompt_pred(df,"pre_pred","No")
print(get_honest_score(df))

df=pd.read_json('data/pop-test_preds_bl.json')
df=get_prompt_pred(df,"pre_pred","No")
print(get_honest_score(df))


## post
print("post\n")
df=pd.read_json('data/trivia-test_preds_bl.json')
df=get_prompt_pred(df,"post_pred","Unsure")
print(get_honest_score(df))

df=pd.read_json('data/nq-test_preds_bl.json')
df=get_prompt_pred(df,"post_pred","Unsure")
print(get_honest_score(df))

df=pd.read_json('data/pop-test_preds_bl.json')
df=get_prompt_pred(df,"post_pred","Unsure")
print(get_honest_score(df))

## icl
print("icl")
df=pd.read_json('data/trivia-test_preds_bl.json')
df=get_prompt_pred(df,"icl_pred","Unknow")
print(get_honest_score(df))

df=pd.read_json('data/nq-test_preds_bl.json')
df=get_prompt_pred(df,"icl_pred","Unknow")
print(get_honest_score(df))

df=pd.read_json('data/pop-test_preds_bl.json')
df=get_prompt_pred(df,"icl_pred","Unknow")
print(get_honest_score(df))

## verb
print('verb')
def extract_fields(text):
    lines = text.split('\n')
    guess = ""
    probability = 0.0
    for line in lines:
        if line.startswith('Guess:'):
            guess = line.split('Guess:')[1].strip()
        elif line.startswith('Probability:'):
            try:
                probability = float(line.split('Probability:')[1].strip())
            except:
                pass
    return guess, probability

df=pd.read_json('data/trivia-test_preds_bl.json')
dft=pd.read_json('data/labeled-data_preds_bl.json')
df['pred_verbprobs']=df['verb_pred'].apply(lambda x:extract_fields(x)[1])
df['pred']=df['verb_pred'].apply(lambda x:extract_fields(x)[0])
dft['pred_verbprobs']=dft['verb_pred'].apply(lambda x:extract_fields(x)[1])
dft['pred']=dft['verb_pred'].apply(lambda x:extract_fields(x)[0])
_,t=get_threshod(dft,'pred_verbprobs',tokenizer.eos_token_id)
print(baseline(df,t,'pred_verbprobs'),tokenizer.eos_token_id)

df=pd.read_json('data/nq-test_preds_bl.json')
dft=pd.read_json('data/labeled-data_preds_bl.json')
df['pred_verbprobs']=df['verb_pred'].apply(lambda x:extract_fields(x)[1])
df['pred']=df['verb_pred'].apply(lambda x:extract_fields(x)[0])
dft['pred_verbprobs']=dft['verb_pred'].apply(lambda x:extract_fields(x)[1])
dft['pred']=dft['verb_pred'].apply(lambda x:extract_fields(x)[0])
_,t=get_threshod(dft,'pred_verbprobs',tokenizer.eos_token_id)
print(baseline(df,t,'pred_verbprobs',tokenizer.eos_token_id))

df=pd.read_json('data/pop-test_preds_bl.json')
dft=pd.read_json('data/labeled-data_preds_bl.json')
df['pred_verbprobs']=df['verb_pred'].apply(lambda x:extract_fields(x)[1])
df['pred']=df['verb_pred'].apply(lambda x:extract_fields(x)[0])
dft['pred_verbprobs']=dft['verb_pred'].apply(lambda x:extract_fields(x)[1])
dft['pred']=dft['verb_pred'].apply(lambda x:extract_fields(x)[0])
_,t=get_threshod(dft,'pred_verbprobs',tokenizer.eos_token_id)
print(baseline(df,t,'pred_verbprobs',tokenizer.eos_token_id))

## ft
print('ft')
def p_f(df_path):
    dfa=pd.read_pickle(df_path)
    print(get_honest_score(dfa))
p_f('./data/trivia-test_ft_400.pkl')
p_f('./data/nq-test_ft_400.pkl')
p_f('./data/pop-test_ft_400.pkl')

## ft-idk
print('ft-idk')
p_f('./data/trivia-test_ft_unk_400.pkl')
p_f('./data/nq-test_ft_unk_400.pkl')
p_f('./data/pop-test_ft_unk_400.pkl')
## CoKE
print('CoKE')
p_f('./data/trivia-test_unsup_conloss_2700.pkl')
p_f('./data/nq-test_unsup_conloss_2700.pkl')
p_f('./data/pop-test_unsup_conloss_2700.pkl')