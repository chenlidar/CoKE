from utils.method import *
from utils.prompt import *
from utils.gen import *
import pandas as pd
def run_baseline_uncertainty():
    ## data
    labeled_data=pd.read_json('./data/trivia-train_preds.json').sample(100,random_state=1)
    trivia_test=pd.read_json('./data/trivia-test_preds.json')
    nq_test=pd.read_json('./data/nq-test_preds.json')
    pop_test=pd.read_json('./data/pop-test_preds.json')
    ## Uncertainty-based method
    for signal in ["pred_minprobs","pred_fprobs","pred_prodprobs"]:
        _,best_t=get_threshod(labeled_data,signal)
        tq_R_k,tq_R_unk,tq_S_aware=baseline(trivia_test,best_t,signal)
        nq_R_k,nq_R_unk,nq_S_aware=baseline(nq_test,best_t,signal)
        pq_R_k,pq_R_unk,pq_S_aware=baseline(pop_test,best_t,signal)

        print(f"{signal}:\ttq_R_k:\t{tq_R_k:0.3f}\ttq_R_unk:\t{tq_R_unk:0.3f}\ttq_S_aware:\t{tq_S_aware:0.3f}\
            \tnq_R_k:\t{nq_R_k:0.3f}\tnq_R_unk:\t{nq_R_unk:0.3f}\tnq_S_aware:\t{nq_S_aware:0.3f}\
                \tpq_R_k:\t{pq_R_k:0.3f}\tpq_R_unk:\t{pq_R_unk:0.3f}\tpq_S_aware:\t{pq_S_aware:0.3f}")
        
def run_baseline_prompt(tokenizer,model):
    ## data
    labeled_data=pd.read_json('./data/trivia-train_preds.json').sample(100,random_state=1)
    trivia_test=pd.read_json('./data/trivia-test_preds.json')
    nq_test=pd.read_json('./data/nq-test_preds.json')
    pop_test=pd.read_json('./data/pop-test_preds.json')
    
    labeled_data['verb_pred'],labeled_data['verb_pred_probs'],labeled_data['verb_pred_gen']=testall_prompt(verbq_prompt,labeled_data,tokenizer,model,max_new_tokens=25)
    labeled_data.to_json('labeled-data_preds_bl.json')
    ## Prompt-based method
    trivia_test['pre_pred'],trivia_test['pre_pred_probs'],trivia_test['pre_pred_gen']=testall_prompt(preq_prompt,trivia_test,tokenizer,model,max_new_tokens=3)
    nq_test['pre_pred'],nq_test['pre_pred_probs'],nq_test['pre_pred_gen']=testall_prompt(preq_prompt,nq_test,tokenizer,model,max_new_tokens=3)
    pop_test['pre_pred'],pop_test['pre_pred_probs'],pop_test['pre_pred_gen']=testall_prompt(preq_prompt,pop_test,tokenizer,model,max_new_tokens=3)
    
    trivia_test['post_pred'],trivia_test['post_pred_probs'],trivia_test['post_pred_gen']=testall_prompt2(postq_prompt,trivia_test,tokenizer,model,max_new_tokens=3)
    nq_test['post_pred'],nq_test['post_pred_probs'],nq_test['post_pred_gen']=testall_prompt2(postq_prompt,nq_test,tokenizer,model,max_new_tokens=3)
    pop_test['post_pred'],pop_test['post_pred_probs'],pop_test['post_pred_gen']=testall_prompt2(postq_prompt,pop_test,tokenizer,model,max_new_tokens=3)
    
    trivia_test['icl_pred'],trivia_test['icl_pred_probs'],trivia_test['icl_pred_gen']=testall_prompt(iclq_prompt,trivia_test,tokenizer,model,max_new_tokens=15)
    nq_test['icl_pred'],nq_test['icl_pred_probs'],nq_test['icl_pred_gen']=testall_prompt(iclq_prompt,nq_test,tokenizer,model,max_new_tokens=15)
    pop_test['icl_pred'],pop_test['icl_pred_probs'],pop_test['icl_pred_gen']=testall_prompt(iclq_prompt,pop_test,tokenizer,model,max_new_tokens=15)
    
    trivia_test['verb_pred'],trivia_test['verb_pred_probs'],trivia_test['verb_pred_gen']=testall_prompt(verbq_prompt,trivia_test,tokenizer,model,max_new_tokens=25)
    nq_test['verb_pred'],nq_test['verb_pred_probs'],nq_test['verb_pred_gen']=testall_prompt(verbq_prompt,nq_test,tokenizer,model,max_new_tokens=25)
    pop_test['verb_pred'],pop_test['verb_pred_probs'],pop_test['verb_pred_gen']=testall_prompt(verbq_prompt,pop_test,tokenizer,model,max_new_tokens=25)
    
    trivia_test.to_json('trivia-test_preds_bl.json')
    nq_test.to_json('nq-test_preds_bl.json')
    pop_test.to_json('pop-test_preds_bl.json')
    
def run_test_consistance(tokenizer,model):
    trivia_test=pd.read_json('./data/trivia-test_preds.json')
    nq_test=pd.read_json('./data/nq-test_preds.json')
    pop_test=pd.read_json('./data/pop-test_preds.json')
    
    trivia_test['pre_pred'],trivia_test['pre_pred_probs'],trivia_test['pre_pred_gen']=testall_prompt(preq_prompt,trivia_test,tokenizer,model,max_new_tokens=3)
    nq_test['pre_pred'],nq_test['pre_pred_probs'],nq_test['pre_pred_gen']=testall_prompt(preq_prompt,nq_test,tokenizer,model,max_new_tokens=3)
    pop_test['pre_pred'],pop_test['pre_pred_probs'],pop_test['pre_pred_gen']=testall_prompt(preq_prompt,pop_test,tokenizer,model,max_new_tokens=3)
    
    trivia_test['post_pred'],trivia_test['post_pred_probs'],trivia_test['post_pred_gen']=testall_prompt2(postq_prompt,trivia_test,tokenizer,model,max_new_tokens=3)
    nq_test['post_pred'],nq_test['post_pred_probs'],nq_test['post_pred_gen']=testall_prompt2(postq_prompt,nq_test,tokenizer,model,max_new_tokens=3)
    pop_test['post_pred'],pop_test['post_pred_probs'],pop_test['post_pred_gen']=testall_prompt2(postq_prompt,pop_test,tokenizer,model,max_new_tokens=3)
    
    trivia_test.to_json('trivia-test_preds_nocons.json')
    nq_test.to_json('nq-test_preds_nocons.json')
    pop_test.to_json('pop-test_preds_nocons.json')