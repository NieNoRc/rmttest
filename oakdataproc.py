import json
from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer
from gensupportinfo import AddSuppInfo
label_map_set = {'I-vendor', 'B-hardware', 'B-language', 'B-parameter', 'I-relevant_term', 'I-os', 'I-hardware', 'B-cve id', 'B-method', 'B-programming', 'B-function', 'I-update', 'B-version',
                 'I-version', 'B-cve', 'I-application', 'B-update', 'O', 'B-file', 'B-relevant_term', 'B-programming language', 'B-application', 'B-os', 'B-edition', 'I-edition', 'B-vendor'}
label_map_ext = {'PAD': -100, 'O': 0, 'B-application': 1, 'I-application': 2, 'B-cve': 3, 'B-cve id': 3, 'I-cve': 4, 'I-cve id': 4, 'B-edition': 5, 'I-edition': 6, 'B-file': 7, 'I-file': 8, 'B-function': 9, 'I-function': 10, 'B-hardware': 11, 'I-hardware': 12, 'B-language': 13, 'I-language': 14, 'B-method': 15,
                 'I-method': 16, 'B-os': 17, 'I-os': 18, 'B-parameter': 19, 'I-parameter': 20, 'B-programming': 21, 'B-programming language': 21, 'I-programming': 22, 'I-programming language': 22, 'B-relevant_term': 23, 'I-relevant_term': 24, 'B-update': 25, 'I-update': 26, 'B-vendor': 27, 'I-vendor': 28, 'B-version': 29, 'I-version': 30}
label_list = ['I-vendor', 'B-hardware', 'B-language', 'B-parameter', 'I-relevant_term', 'I-os', 'I-hardware', 'B-cve id', 'B-method', 'B-programming', 'B-function', 'I-update',
              'B-version', 'I-version', 'B-cve', 'I-application', 'B-update', 'B-file', 'B-relevant_term', 'B-programming language', 'B-application', 'B-os', 'B-edition', 'I-edition', 'B-vendor']
labels_map_inverse=['O', 'B-application', 'I-application', 'B-cve', 'I-cve', 'B-edition', 'I-edition', 'B-file', 'I-file', 'B-function', 'I-function', 'B-hardware', 'I-hardware', 'B-language', 'I-language', 'B-method', 'I-method', 'B-os', 'I-os', 'B-parameter', 'I-parameter', 'B-programming', 'I-programming', 'B-relevant_term', 'I-relevant_term', 'B-update', 'I-update', 'B-vendor', 'I-vendor', 'B-version', 'I-version']

class OakDataSet(Dataset):
    def __init__(self, datadir: str, modelpath: str,supp_info_producer:AddSuppInfo):
        super().__init__()
        self.data = []        
        self.traindata = []
        self.testdata = []
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.supp_info_producer=supp_info_producer
        self.readjson(datadir)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def readtxt(self, fname):
        with open(file=fname, mode='r') as f:
            labels = []
            words = []
            for line in f:
                tokens: [] = line.strip().split(' ')
                if len(tokens) > 1:
                    labels.append(tokens[1])
                    words.append(tokens[0])
                else:
                    if len(words)>0:
                        entry = self.tokenizer(words, is_split_into_words=True)
                        new_label = self.label_proc(
                            labels, entry.encodings[0].word_ids)
                        self.data.append({'data': entry, 'labels': new_label})
                    labels = []
                    words = []

    def readjson(self, fname):
        with open(file=fname, mode='r') as f:
            datas = json.load(f)
            
            for topkey in datas.keys():
                datanum=0
                second = datas.get(topkey)
                for seckey in second.keys():
                    third = second.get(seckey)
                    sentence = ''
                    words = []
                    labels = []
                    for token in third:
                        labels.append(token[1])
                        sentence += token[0]+' '
                        words.append(token[0])
                    entry = self.tokenizer(words, is_split_into_words=True)
                    new_label = self.label_proc(
                        labels, entry.encodings[0].word_ids)
                    support_infos=self.supp_info_producer.gen_support_infos(tokenizer=self.tokenizer,words=words,encoding=entry.encodings[0])
                    #self.data.append({'data': entry, 'labels': new_label})
                    if len(entry['input_ids'])>512:
                        continue
                    if datanum%10<7:
                        self.traindata.append({'data': entry, 'labels': new_label,'support_infos':support_infos})
                    else:
                        self.testdata.append({'data': entry, 'labels': new_label,'support_infos':support_infos})
                    datanum+=1

    def shift_label(self, origin_label: str) -> int:
        if 'O' == origin_label:
            return label_map_ext[origin_label]
        label_piece: list = origin_label.split('-')
        if len(label_piece) > 1:
            return label_map_ext['I-'+label_piece[1]]
        else:
            return label_map_ext[origin_label]

    def label_proc(self, origin_label, word_ids):
        new_label = []
        current_word = None
        for word_id in word_ids:
            if word_id is None:
                new_label.append(-100)
            elif word_id != current_word:
                current_word = word_id
                new_label.append(label_map_ext[origin_label[word_id]])
            else:
                new_label.append(self.shift_label(origin_label[word_id]))
        return new_label
