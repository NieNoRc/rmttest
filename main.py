import torch
from selfbert import PartialTrainableBERT
from oakdataproc import label_map_ext,OakDataSet,labels_map_inverse
from timeit import default_timer
from traintest import train_loop,test_loop
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from gensupportinfo import AddSuppInfo
import os
def main():
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    modelpath='models/bertbasecased'
    datapath='datas/oakcorpus/full_corpus.json'
    learn_rate=2e-4
    epochs=90
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model=PartialTrainableBERT(modelpath=modelpath,flag_layer_num=11,num_labels=31).to(device=device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=label_map_ext['PAD'])
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    supp_info_proc=AddSuppInfo(device=device)
    oakdataset = OakDataSet(
        datadir=datapath, modelpath=modelpath,supp_info_producer=supp_info_proc)
    train_dataset=Dataset.from_list(oakdataset.traindata)
    train_dataset.set_format('torch')
    train_dataloader=DataLoader(dataset=train_dataset,shuffle=True)
    test_dataset=Dataset.from_list(oakdataset.testdata)
    test_dataset.set_format('torch')
    test_dataloader=DataLoader(dataset=test_dataset,shuffle=True)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        tic = default_timer()
        train_loop(train_dataloader, model, loss_fn, optimizer=optimizer,device=device)
        toc = default_timer()
        print('Time: ' + str(toc - tic))

        test_loop(test_dataloader, model, loss_fn,device,labels_map_inverse)
    print("Done!")
if __name__ == "__main__":
    main()