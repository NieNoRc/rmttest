from tokenizers import Encoding
from transformers import PreTrainedTokenizer
from torch import tensor
import re
class AddSuppInfo():
    def __init__(self):
        super().__init__()      
    def methodattract(self,tokenizer:PreTrainedTokenizer, words:list,encoding:Encoding):
        method_infos=[]
        for i in range(0,len(words)):
            if(len(words[i])<2):
                continue
            supportsentense=[]
            if words[i][-3:-1]=='()' and '_' in words[i]:
                supportsentense=[words[i],'is','almost', 'a', 'method','.']
            elif words[i][-3:-1]=='()':
                supportsentense=[words[i],'is', 'probably', 'a', 'method','.']
            if len(supportsentense) > 0:
                insert_pos=[]
                for j in range(0,len(encoding.word_ids)):
                    if encoding.word_ids[j]==i:
                        insert_pos.append(j)
                        break
                support_input=tokenizer(supportsentense,is_split_into_words=True)
                support_id=[]
                for j in range(0,len(support_input.encodings[0].word_ids)):
                    if support_input.encodings[0].word_ids[j]==0:
                        support_id.append(j)
                support_input={k: tensor(v) for k, v in support_input.items()}
                method_infos.append({'bert_input':support_input,'insert_pos':insert_pos,'support_wp_ids':support_id})
        return method_infos
    def methodattractnew(self,tokenizer:PreTrainedTokenizer, words:list,encoding:Encoding):
        method_infos=[]
        trust_level_str=['probably','likely','almost']
        for i in range(0,len(words)):
            trust_level=0
            if(len(words[i])<2):
                continue
            supportsentense=[]
            if words[i][-3:-1]=='()':
                trust_level+=1
            if '_' in words[i]:
                trust_level+=1
            if '::' in words[i]:
                trust_level+=1
            if len(words[i].split('.'))>2:
                trust_level+=1
            upper_split=re.split(pattern='[A-Z]',string=words[i])
            upper_split_cnt=0
            if len(upper_split)>1:
                for entry in upper_split:
                    if len(entry)>0:
                        upper_split_cnt+=1
            if upper_split_cnt>=2:
                trust_level+=1
            if trust_level>0:
                supportsentense=[words[i],'is']
                if trust_level<len(trust_level_str):
                    supportsentense.append(trust_level_str[trust_level-1])
                supportsentense+=['a', 'method','.']
            if len(supportsentense) > 0:
                insert_pos=[]
                for j in range(0,len(encoding.word_ids)):
                    if encoding.word_ids[j]==i:
                        insert_pos.append(j)
                        break
                support_input=tokenizer(supportsentense,is_split_into_words=True)
                support_id=[]
                for j in range(0,len(support_input.encodings[0].word_ids)):
                    if support_input.encodings[0].word_ids[j]==0:
                        support_id.append(j)
                support_input={k: tensor(v) for k, v in support_input.items()}
                method_infos.append({'bert_input':support_input,'insert_pos':insert_pos,'support_wp_ids':support_id})
        return method_infos
    def methodattractsent(self, words:list,encoding:Encoding):
        method_infos=[]
        
        trust_level_str=['probably','likely','almost']
        for i in range(0,len(words)):
            trust_level=0
            if(len(words[i])<2):
                continue
            supportsentense=[]
            if words[i][-3:-1]=='()':
                trust_level+=1
            if '_' in words[i]:
                trust_level+=1
            if '::' in words[i]:
                trust_level+=1
            if len(words[i].split('.'))>2:
                trust_level+=1
            upper_split=re.split(pattern='[A-Z]',string=words[i])
            upper_split_cnt=0
            if len(upper_split)>1:
                for entry in upper_split:
                    if len(entry)>0:
                        upper_split_cnt+=1
            if upper_split_cnt>=2:
                trust_level+=1
            if trust_level>0:
                supportsentense=[words[i],'is']
                if trust_level<len(trust_level_str):
                    supportsentense.append(trust_level_str[trust_level-1])
                supportsentense+=['a', 'method','.']
            if len(supportsentense) > 0:
                insert_pos=[]
                for j in range(0,len(encoding.word_ids)):
                    if encoding.word_ids[j]==i:
                        insert_pos.append(j)
                        break
                method_infos.append({'words':supportsentense,'insert_pos':insert_pos})
        return method_infos
    def gen_support_infos(self,tokenizer:PreTrainedTokenizer, words:list,encoding:Encoding):
        return self.tokenize_supportsentence(tokenizer=tokenizer,sentences=self.methodattractsent(words,encoding))
    def tokenize_supportsentence(self,tokenizer:PreTrainedTokenizer,sentences:list):
        support_infos=[]
        for entry in sentences:
            support_input=tokenizer(entry['words'],is_split_into_words=True)
            support_id=[]
            for j in range(0,len(support_input.encodings[0].word_ids)):
                if support_input.encodings[0].word_ids[j]==0:
                    support_id.append(j)
            support_input={k: tensor(v) for k, v in support_input.items()}
            support_infos.append({'bert_input':support_input,'insert_pos':entry['insert_pos'],'support_wp_ids':support_id})
        return support_infos


