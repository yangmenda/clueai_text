from transformers import T5Tokenizer, T5ForConditionalGeneration

# 使用
import torch
from transformers import AutoTokenizer

def get_answer(text,device):
    tokenizer = T5Tokenizer.from_pretrained("ClueAI/ChatYuan-large-v1",cache_dir="./pretrain")
    model = T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v1",cache_dir="./pretrain")
    device = torch.device(device)
    model.to(device)
    input_text = "用户：" + text + "\n小元："
    out_put=answer(input_text,tokenizer=tokenizer,model=model,device=device)
    return out_put



def preprocess(text):
  text = text.replace("\n", "\\n").replace("\t", "\\t")
  return text

def postprocess(text):
  return text.replace("\\n", "\n").replace("\\t", "\t")

def answer(text,tokenizer,model,device,sample=True, top_p=1, temperature=0.7):
 
  text = preprocess(text)
  encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device) 
  if not sample:
    out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, num_beams=1, length_penalty=0.6)
  else:
    out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
  out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
  return postprocess(out_text[0])



