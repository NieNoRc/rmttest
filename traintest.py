from transformers import pipeline, BertForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
from torch.nn import Module,utils,CrossEntropyLoss
from torch import no_grad,optim,argmax,tensor,zeros,Tensor
from selfbert import PartialTrainableBERT
from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
from oakdataproc import OakDataSet
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from accelerate import Accelerator
class ResultStat():
    def __init__(self):
        self.pred = []
        self.truth = []

  

    def accum(self, predict:Tensor, truth:Tensor,labelsmap_inverse:list):
        #similar with accum(). BUT the input is different due to the output is different between LSTM and CRF
        batch_size, max_seq_len, label_num = predict.shape
        for i in range(0,batch_size):
            temp_seq_len = max_seq_len
            temp_pred = []
            temp_truth = []
            for j in range(0,temp_seq_len):
                temp_pred.append(labelsmap_inverse[argmax(predict[i][j]).item()])
                if truth[i][j].item()>=0:
                    temp_truth.append(labelsmap_inverse[truth[i][j].item()])
                else:
                    temp_truth.append('O')
            self.pred.append(temp_pred)
            self.truth.append(temp_truth)

    def get_result(self):
        return classification_report(self.truth, self.pred, mode='strict', scheme=IOB2,digits=4)
        #return classification_report(self.truth, self.pred, scheme=IOB2,digits=4)
def train_loop(dataloader:DataLoader, model:Module, loss_fn:CrossEntropyLoss, optimizer:optim.AdamW, device,insert_support_flag:bool,accelor:Accelerator=None):
    size = len(dataloader.dataset)
    model = model.train()
    num_batches = len(dataloader)
    total_loss = 0
    loss=0
    batchnum=1
    for batch in dataloader:
        bertinput=batch['data'] 
        support_infos=batch['support_infos']
        truth=batch['labels']
        if accelor is None:
            bertinput = {k: v.to(device) for k, v in bertinput.items()}
            truth=truth.to(device)
            for i in range(0,len(support_infos)):
                support_infos[i]['bert_input']={k: v.to(device) for k, v in support_infos[i]['bert_input'].items()} 
        if bertinput['input_ids'].size()[1]>512:
            continue
        bertinput = {k: v.to(device) for k, v in bertinput.items()}
        predict=model(bertinput=bertinput,supportinput=support_infos,insert_support_flag=insert_support_flag)
        loss=loss_fn(input=predict.permute(0,2,1),target=truth)
        total_loss += loss
        optimizer.zero_grad()
        if accelor is not None:
            accelor.backward(loss)
            #accelor.clip_grad_value_(model.parameters(),5)
        else:
            loss.backward()
            #utils.clip_grad_value_(model.parameters(),5) #gradiant clipping
        optimizer.step()

        if batchnum % 1024 == 0:
            loss, current = loss.item(), batchnum
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        batchnum+=1
    avg_loss = total_loss / num_batches
    print(f"Train avg loss: {avg_loss:>7f}")

def test_loop(dataloader:DataLoader, model:Module, loss_fn:CrossEntropyLoss, device:str,labels_map_inverse:list,insert_support_flag:bool,accelor:Accelerator=None):
    model = model.eval()
    RS = ResultStat()
    num_batches = len(dataloader)
    test_loss = 0
    with no_grad():
        for batch in dataloader:
            #print('ss')
            bertinput=batch['data'] 
            support_infos=batch['support_infos']
            truth=batch['labels']
            if accelor is None:
                bertinput = {k: v.to(device) for k, v in bertinput.items()}
                truth=truth.to(device)
                for i in range(0,len(support_infos)):
                    support_infos[i]['bert_input']={k: v.to(device) for k, v in support_infos[i]['bert_input'].items()} 
            if bertinput['input_ids'].size()[1]>512:
                continue
            else:
                predict=model(bertinput=bertinput,supportinput=support_infos,insert_support_flag=insert_support_flag)
                test_loss +=loss_fn(input=predict.permute(0,2,1),target=truth).item()
            RS.accum(predict=predict,truth=truth,labelsmap_inverse=labels_map_inverse)
    test_loss /= num_batches
    print(f"Test avg loss: {test_loss:>8f} \n")
    #accelor.wait_for_everyone()
    RS.pred=accelor.gather_for_metrics((RS.pred))
    RS.truth=accelor.gather_for_metrics((RS.truth))
    #accelor.wait_for_everyone()
    if accelor.is_main_process:
        print(RS.get_result())