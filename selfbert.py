from torch import nn,cat,zeros_like
from transformers import BertModel
class PartialTrainableBERT(nn.Module):
    def __init__(self,modelpath:str,flag_layer_num:int=11,num_labels:int=2,dropout_rate:float=0.1): 
        super().__init__()
        self.hugginBERT=BertModel.from_pretrained(modelpath)
        self.hugginBERT.requires_grad_(False)
        for name,parm in self.hugginBERT.encoder.named_parameters():
            if(int(name.split('.')[1])>flag_layer_num):
                parm.requires_grad_(True)
        self.classifier = nn.Linear((self.hugginBERT.config.hidden_size)*2, num_labels)
        self.dropout_layer=nn.Dropout(p=dropout_rate)
    def forward(self,bertinput,supportinput:list,insert_support_flag:bool=True):
        bertoutput=self.hugginBERT(**bertinput)
        support_info=self.insert_support_info(bertoutdup=zeros_like(bertoutput[0]),supportinput=supportinput,insert_support_flag=insert_support_flag)
        dropout=self.dropout_layer(cat([bertoutput[0],support_info],dim=2))
        output=self.classifier(dropout)
        return output
    def insert_support_info(self,bertoutdup,supportinput:list,insert_support_flag:bool=True):
        if insert_support_flag is not True:
            return bertoutdup
        for entry in supportinput:
            entryoutput=self.hugginBERT(**(entry['bert_input']))
            for i in entry['insert_pos'][0].tolist():
                cnt=0
                for j in entry['support_wp_ids'][0].tolist():
                    bertoutdup[0][i+cnt]=entryoutput[0][0][j]
                    cnt+=1
        return bertoutdup

