from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import json
import torch

def DataLoadeing(train_X , train_y, validation_X, validation_y, batch_size=64,drop_last=False):
    
    train_X=torch.from_numpy(train_X).to(torch.float)
    train_y=torch.from_numpy(train_y).to(torch.float)
    
    validation_X=torch.from_numpy(validation_X).to(torch.float)
    validation_y=torch.from_numpy(validation_y).to(torch.float)
    
    #test_X=torch.from_numpy(test_X).to(torch.float)
    #test_y=torch.from_numpy(test_y).to(torch.float)
    
    train_data_loader=DataLoader(BatchGenerator(train_X , train_y), 
                                 shuffle=False, batch_size=batch_size, drop_last=drop_last)
    validation_data_loader=DataLoader(BatchGenerator(validation_X , validation_y), 
                                 shuffle=False, batch_size=batch_size, drop_last=drop_last)
    #test_data_loader=DataLoader(BatchGenerator(test_X , test_y), 
                                 #shuffle=False, batch_size=batch_size, drop_last=drop_last)
    return train_data_loader, validation_data_loader#, test_data_loader
       

class SaveBestModel:

    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion,path):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nSaving best model for epoch: {epoch}\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer,
                'loss': criterion,}, path+'best_model.pth')

class PlotLossCurves:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.epoch=[]
        self.data={}
        self.count=0
    
    def __call__(self, train_loss,val_loss,epoch,path,load_model_epoch=None):
                   
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.epoch.append(epoch)
        if load_model_epoch is None:
            self.data = {"Epoch": self.epoch, "TrainLoss": self.train_loss, "ValLoss":self.val_loss}
        else:
            for i in range(len(self.epoch)-load_model_epoch):
                self.train_loss.pop()
                self.val_loss.pop()
                self.epoch.pop()
            self.data = {"Epoch": self.epoch, "TrainLoss": self.train_loss, "ValLoss":self.val_loss}
        load_model_epoch=None   
        with open(path+"//loss.json", "w") as file:
            json.dump(self.data, file)

        self.count=self.count+1
        if self.count>=2:
            plt.figure(figsize=(10, 7))
            plt.plot(self.train_loss, color='orange', linestyle='-', label='train loss')
            plt.plot(self.val_loss, color='red', linestyle='-', label='validataion loss')
            plt.grid(True)
            plt.style.use("ggplot")
#             plt.autoscale(axis='x',tight=True)
            plt.xticks(self.epoch)
            plt.title('Model Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(path+'//loss.png')
            plt.close()


class LoadModel:

    def __call__(self, model_path, lr,train_X , train_y, validation_X, validation_y, Batch_Size=True):
        best_model_cp= torch.load(model_path)
        model=best_model_cp['model_state_dict']
        opt=best_model_cp['optimizer']
        best_epoch=best_model_cp['epoch']+1
        opt.param_groups[0]['lr']=lr
        print('New lr =',opt.param_groups[0]['lr'])
        if type(Batch_Size) is bool:
            return model, opt, best_epoch
        else:
            train_data_loader, validation_data_loader = DataLoadeing(train_X , train_y, validation_X, validation_y, batch_size=Batch_Size)
            return model, opt, best_epoch, train_data_loader, validation_data_loader

class TestModel():
    def __call__(self, test_X, model, load_model_path):
        best_model_cp= torch.load(load_model_path)
        model__=best_model_cp['model_state_dict']
        model.load_state_dict(model__)
        testt_X=torch.from_numpy(test_X).to(torch.float)
        with torch.no_grad():
                pred= model(testt_X)
                prediction=torch.Tensor.numpy(pred)
        return prediction
    
    

def train(start_epoch,num_epochs,
          best_model_path,fig_path,model,
          train_data_loader,validation_data_loader,criterion,optimizer,
          save_best_model,Plot_Loss,load_model_epoch=None):
    
    Tsteps=len(train_data_loader)
    Vsteps=len(validation_data_loader)
    for epoch in range(start_epoch,num_epochs+1):
        val_loss_list,train_loss_list=list(),list()

        for i, batchT in enumerate(train_data_loader,1):
            model.train()

            inputsT, labelsT = batchT
            outputsT = model(inputsT)
            train_loss = criterion(outputsT, labelsT)

            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_loss_list.append(train_loss.item())

            if i == len(train_data_loader):
                epoch_train_loss = sum(train_loss_list) / len(train_loss_list)
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{Tsteps}], Training Loss: {epoch_train_loss:.4f}')
            else:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{Tsteps}], Training Loss: {train_loss.item():.4f}',  end='\r')

        for i, batchV in enumerate(validation_data_loader,1):
            model.eval()
            with torch.no_grad():
                inputsV, labelsV = batchV
                outputsV = model(inputsV)
                val_loss = criterion(outputsV, labelsV)
                val_loss_list.append(val_loss.item())

            if i == len(validation_data_loader):
                epoch_val_loss = sum(val_loss_list) / len(val_loss_list)
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{Vsteps}], Val Loss: {epoch_val_loss:.4f}')
            else:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{Vsteps}], Val Loss: {val_loss.item():.4f}',  end='\r')

        path=best_model_path+str(epoch)
        save_best_model(epoch_val_loss, epoch, model, optimizer, criterion,path)
        if epoch==load_model_epoch:
            Plot_Loss(epoch_train_loss, epoch_val_loss,epoch,fig_path, load_model_epoch)
        else:
            Plot_Loss(epoch_train_loss, epoch_val_loss,epoch,fig_path)

        

class BatchGenerator(Dataset):
    '''
    BatchGenerator(train_X , train_y) # train_X , train_y is numpy array
    To check the detail of Batches:
        dataset.BatchGenerators(loader) # datastet is BatchGenerator object and loader is the object of DataLoader
    To check the number of samples:
        len(BatchGenerator Object)
    While len(DataLoader Object) gives total number of Batches
        
    '''
    def __init__(self,X , Y):
        self.data = list(zip(X , Y))
        self.l=None
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx][0]
        target = self.data[idx][1]
        return features, target

    @staticmethod
    def batches(an_loader):
        l=list(an_loader)
        
        print(f"Total Number Of Batches = {len(l)}")
        print(f"Each Batch have a shape of {l[0][0].shape} # batch_size, time-step, num_features") #,list(train)[0][1].shape
        
        if an_loader.drop_last!=True:
            print(f"Last Batch have {l[-1][0].shape} tensors")
        