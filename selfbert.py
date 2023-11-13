from torch import nn,cat,zeros_like,Tensor
from transformers import BertModel
class PartialTrainableBERTold(nn.Module):
    def __init__(self,modelpath:str,flag_layer_num:int=11,num_labels:int=2,dropout_rate:float=0.1,feature_layer:int=2): 
        super().__init__()
        self.hugginBERT=BertModel.from_pretrained(modelpath)
        self.hugginBERT.requires_grad_(False)
        for name,parm in self.hugginBERT.encoder.named_parameters():
            if(int(name.split('.')[1])>flag_layer_num):
                parm.requires_grad_(True)
        self.classifier = nn.Linear((self.hugginBERT.config.hidden_size)*2, num_labels)
        self.dropout_layer=nn.Dropout(p=dropout_rate)
    def forward(self,bertinput,supportinput:list,insert_support_flag:bool=True):
        bertinput['output_hidden_states']=True
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

class PartialTrainableBERT(nn.Module):
    def __init__(self,modelpath:str,flag_layer_num:int=11,num_labels:int=2,dropout_rate:float=0.1,feature_layer:int=2): 
        super().__init__()
        self.hugginBERT=BertModel.from_pretrained(modelpath)
        self.hugginBERT.requires_grad_(False)
        for name,parm in self.hugginBERT.encoder.named_parameters():
            if(int(name.split('.')[1])>flag_layer_num):
                parm.requires_grad_(True)
        self.classifier = nn.Linear((self.hugginBERT.config.hidden_size)*2*feature_layer, num_labels)
        self.dropout_layer=nn.Dropout(p=dropout_rate)
        self.feature_layer=feature_layer if feature_layer<=13 else 13
    def forward(self,bertinput,supportinput:list,insert_support_flag:bool=True):
        bertinput['output_hidden_states']=True
        bertoutput=self.hugginBERT(**bertinput)
        features=[bertoutput.hidden_states[12-i] for i in range(0,self.feature_layer)]
        support_info=self.insert_support_info(bertoutdup=[zeros_like(bertoutput[0]) for i in range(0,self.feature_layer)],supportinput=supportinput,insert_support_flag=insert_support_flag)
        features+=support_info
        dropout=self.dropout_layer(cat(features,dim=2))
        output=self.classifier(dropout)
        return output
    def insert_support_info(self,bertoutdup:list[Tensor],supportinput:list,insert_support_flag:bool=True)->list[Tensor]:
        if insert_support_flag is not True:
            return bertoutdup
        for entry in supportinput:
            entry['bert_input']['output_hidden_states']=True
            entryoutput=self.hugginBERT(**(entry['bert_input']))
            for i in entry['insert_pos'][0].tolist():
                for j in range(0,self.feature_layer):
                    cnt=0
                    for k in entry['support_wp_ids'][0].tolist():
                        bertoutdup[j][0][i+cnt]=entryoutput.hidden_states[12-j][0][k]
                        cnt+=1
        return bertoutdup
    
