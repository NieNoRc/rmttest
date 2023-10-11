from torch import nn
from transformers import BertModel
class PartialTrainableBERT(nn.Module):
    def __init__(self,modelpath:str,flag_layer_num:int=11,num_labels:int=2,dropout_rate:float=0.5): 
        super().__init__()
        self.hugginBERT=BertModel.from_pretrained(modelpath)
        self.hugginBERT.requires_grad_(False)
        for name,parm in self.hugginBERT.encoder.named_parameters():
            if(int(name.split('.')[1])>flag_layer_num):
                parm.requires_grad_(True)
        self.classifier = nn.Linear(self.hugginBERT.config.hidden_size, num_labels)
        self.dropout_layer=nn.Dropout(p=dropout_rate)
    def forward(self,bertinput):
        bertoutput=self.hugginBERT(**bertinput)
        dropout=self.dropout_layer(bertoutput[0])
        output=self.classifier(dropout)
        return output

