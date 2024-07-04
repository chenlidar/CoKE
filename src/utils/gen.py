
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

@torch.inference_mode()
def generate_with_prob(model,input_ids,attention_mask,max_new_tokens,eos_token_id):
    past_key_values = None
    bs,len0=attention_mask.shape[0],attention_mask.shape[1]
    gens=[]
    probs=[]
    for j in range(max_new_tokens):
        output=model(input_ids=input_ids,attention_mask=attention_mask,return_dict=True,use_cache=True,past_key_values=past_key_values,output_hidden_states=False)
        past_key_values=output.past_key_values
        logits=output.logits[:,-1]#[batchsize,D]
        logits=F.softmax(logits,dim=-1)
        pred=logits
        next_tokens_probs,next_tokens=torch.max(pred,dim=-1,keepdim=True)#[batchsize,1]
        
        input_ids=next_tokens
        attention_mask=torch.concat([attention_mask, torch.ones((bs,1), dtype=torch.long, device=attention_mask.device)], dim=-1)      
        
        gens.append(next_tokens)
        probs.append(next_tokens_probs)
    #后处理eos 注意input_ids
    gens=torch.concat(gens,dim=-1)
    probs=torch.concat(probs,dim=-1)
    mask = (gens == eos_token_id)
    mask=mask.cumsum(dim=-1).clamp(max=1)
    gens = gens * (1 - mask) + mask * eos_token_id
    
    return gens,probs

def testall_prompt(prompt,df,tokenizer,model,max_new_tokens=15):
    ans_list=[]
    probs_list=[]
    gen_list=[]
    for idx,test in tqdm(df.iterrows()):
        inputs=tokenizer(prompt.format(test['q'].strip()),add_special_tokens=True,return_tensors='pt')
        gen,probs=generate_with_prob(model,inputs.input_ids.to(model.device),inputs.attention_mask.to(model.device),max_new_tokens,2)
        ans=tokenizer.decode(gen.to('cpu')[0],skip_special_tokens=False).replace('</s>','').strip()
        ans_list.append(ans)
        probs_list.append(probs.float().to('cpu')[0].tolist())
        gen_list.append(gen.to('cpu')[0].tolist())
    return ans_list,probs_list,gen_list

def testall_prompt2(prompt,df,tokenizer,model,max_new_tokens=3):
    ans_list=[]
    probs_list=[]
    gen_list=[]
    for idx,test in tqdm(df.iterrows()):
        inputs=tokenizer(prompt.format(test['q'].strip(),test['pred'].strip()),add_special_tokens=True,return_tensors='pt')
        gen,probs=generate_with_prob(model,inputs.input_ids.to(model.device),inputs.attention_mask.to(model.device),max_new_tokens,2)
        ans=tokenizer.decode(gen.to('cpu')[0],skip_special_tokens=False).replace('</s>','').strip()
        ans_list.append(ans)
        probs_list.append(probs.float().to('cpu')[0].tolist())
        gen_list.append(gen.to('cpu')[0].tolist())
    return ans_list,probs_list,gen_list

@torch.inference_mode()
def getAnsProb(model,inputs,outputs):
    input_ids=torch.hstack([inputs.input_ids,outputs["input_ids"]])
    attention_mask=torch.hstack([inputs.attention_mask,outputs["attention_mask"]])
    last_tokens = outputs["input_ids"][0]
    outputs = model(input_ids=input_ids.to(model.device),attention_mask=attention_mask.to(model.device))
    predictions = outputs[0]
    probabilities = torch.nn.functional.softmax(predictions[0, -last_tokens.shape[0]-1:-1], dim=-1)
    
    return [probability[token].item() for token, probability in zip(last_tokens, probabilities)]