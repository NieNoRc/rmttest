from tokenizers import Encoding
from transformers import PreTrainedTokenizer
from torch import tensor
class AddSuppInfo():
    def __init__(self, device: str):
        super().__init__()
        self.device=device       
    def methodattract(self,tokenizer:PreTrainedTokenizer, words:list,encoding:Encoding):
        method_infos=[]
        for i in range(0,len(words)):
            if(len(words[i])<2):
                continue
            supportsentense=''
            if words[i][-3:-1]=='()' and '_' in words[i]:
                supportsentense=words[i]+' is almost a method.'
            elif words[i][-3:-1]=='()':
                supportsentense=words[i]+' is probably a method.'
            if len(supportsentense) > 0:
                insert_pos=[]
                for j in range(0,len(encoding.word_ids)):
                    if encoding.word_ids[j]==i:
                        insert_pos.append(j)
                support_input=tokenizer(supportsentense)
                support_id=[]
                for j in range(0,len(support_input.encodings[0].word_ids)):
                    if support_input.encodings[0].word_ids[j]==0:
                        support_id.append(j)
                support_input={k: tensor(v).to(self.device) for k, v in support_input.items()}
                method_infos.append({'bert_input':support_input,'insert_pos':insert_pos,'support_wp_ids':support_id})
        return method_infos
    def gen_support_infos(self,tokenizer:PreTrainedTokenizer, words:list,encoding:Encoding):
        return self.methodattract(tokenizer, words,encoding)



