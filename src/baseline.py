from utils.run_main_result import *
from transformers import AutoTokenizer,AutoModelForCausalLM
from utils.myconfig import model_path
tokenizer=AutoTokenizer.from_pretrained(model_path)
model=AutoModelForCausalLM.from_pretrained(model_path,device_map='cuda:0')
# run_baseline_prompt(tokenizer,model)
from peft import PeftModel
model=PeftModel.from_pretrained(model,f'./ckps/unsup_700')
run_test_consistance(tokenizer,model)