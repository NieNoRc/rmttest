from tokenizers import Encoding
from transformers import PreTrainedTokenizer
from torch import tensor
import json
import re
class AddSuppInfo():
    def __init__(self,index_fname:str,wiki_info_fname:str):
        super().__init__()   
        self.support_word_index,self.wiki_support_info=self.init_wiki(index_fname,wiki_info_fname)
    def init_wiki(self,index_fname:str,wiki_info_fname:str):
        with open(index_fname,'r') as i_f:
            index_json=json.load(fp=i_f)
        with open(wiki_info_fname,'r') as wiki_f:
            wiki_json=json.load(fp=wiki_f)
        return index_json,wiki_json

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
        ipre='[0-9]+\.[0-9]+\.[0-9]+\.[0-9]'
        ip_patt=re.compile(pattern=ipre)
        trust_level_str=['probably','likely','almost']
        for i in range(0,len(words)):
            if ip_patt.fullmatch(words[i]) is not None:
                supportsentense=[words[i],'is','an','IP','address']
            else:
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
                method_infos.append({'words':supportsentense,'insert_pos':insert_pos,'support_wdid':[0,0]})
        return method_infos
    def gen_support_infos(self,tokenizer:PreTrainedTokenizer, words:list,encoding:Encoding):
        support_info=self.methodattractsent(words,encoding)
        support_info+=self.insert_wiki(words,encoding)
        return self.tokenize_supportsentence(tokenizer=tokenizer,sentences=support_info)
    def tokenize_supportsentence(self,tokenizer:PreTrainedTokenizer,sentences:list):
        support_infos=[]
        for entry in sentences:
            support_input=tokenizer(entry['words'],is_split_into_words=True)
            support_id=[]
            if len(support_input['input_ids'])>512:
                continue
            for j in range(0,len(support_input.encodings[0].word_ids)):
                current_word_id=support_input.encodings[0].word_ids[j]
                if current_word_id is not None and current_word_id>=entry['support_wdid'][0] and current_word_id<=entry['support_wdid'][1]:
                    support_id.append(j)
                elif len(support_id)>0:
                    break
            support_input={k: tensor(v) for k, v in support_input.items()}
            support_infos.append({'bert_input':support_input,'insert_pos':entry['insert_pos'],'support_wp_ids':support_id})
        return support_infos
    def insert_wiki(self, words:list,encoding:Encoding):
        wiki_infos=[]
        for i in range(0,len(words)):
            keywords=''
            temp_index=self.support_word_index
            match_flag=False
            for j in range(i,len(words)):
                if words[j] in temp_index:
                    keywords+=words[j]+' '
                    temp_index=temp_index[words[j]]
                else:
                    if len(keywords)>0:
                        keywords=keywords.strip()
                        if keywords in self.wiki_support_info and self.wiki_support_info[keywords]['flag']==1:
                            insert_pos=[]
                            insert_words=re.split(pattern=r'[ .,()]',string=self.wiki_support_info[keywords]['summary'])
                            for k in range(0,len(encoding.word_ids)):
                                if encoding.word_ids[k]==i:
                                    insert_pos.append(k)
                                    break
                            wiki_infos.append({'words':insert_words,'insert_pos':insert_pos,'support_wdid':self.wiki_support_info[keywords]['index']})
                            i=j
                            break
                    else:
                        break
        return wiki_infos
            

