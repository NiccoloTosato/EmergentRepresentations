import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data import *

def get_model(modelname, **kwargs):
    if modelname=='ff':
        return NetFF(backprop=False, **kwargs)
    elif modelname=='bp_ff':
        return NetFF(backprop=True, **kwargs)
    else:
        return FullyConnected(**kwargs)
        
class ForwardLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, custom_nlinear = torch.nn.Sigmoid(), goodness_function = lambda x : torch.norm(x, dim=1, p=torch.inf), epochs=1000, lr=1e-4,
                 dropout=0,normalize_function=lambda x : torch.nn.functional.normalize(x, p=2)):
        super().__init__()
        self.threshold = 3
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)
        self.custom_nlinear = custom_nlinear
        self.goodness_function = goodness_function 
        self.dropout=torch.nn.Dropout(p=dropout, inplace=False)
        self.num_epochs=epochs
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.normalize_function=normalize_function
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1.0)

    def normalize(self,x):
        return self.normalize_function(x)

    def forward(self, x):
        x_direction = self.normalize(x)
        x_direction = self.custom_nlinear(self.linear(x_direction))
        x_direction = self.dropout(x_direction)
        return x_direction
    
    def train_layer(self, x_pos, x_neg, path, device, batch_size=60000):
        dataset=torch.utils.data.TensorDataset(x_pos, x_neg)
        dataloader=torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        pos=[]
        neg=[]
        for i in range(self.num_epochs):
            if (i % 100) == 0:
                print("Epoch: ", i,flush=True)
            avg_pos=0
            avg_neg=0
            for xpos, xneg in iter(dataloader):
                xpos=xpos.to(device)
                xneg=xneg.to(device)
                x_pos_output=self.forward(xpos)
                x_neg_output=self.forward(xneg)
                positive_goodness = self.goodness_function(x_pos_output)
                negative_goodness = self.goodness_function(x_neg_output) 
                
                avg_pos+=positive_goodness.mean().item()
                avg_neg+=negative_goodness.mean().item()
                l = torch.log(1 + torch.exp(torch.cat([
                -positive_goodness + self.threshold,
                negative_goodness - self.threshold]))).mean()
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            pos.append(avg_pos/len(dataloader))
            neg.append(avg_neg/len(dataloader))
            if i%10==0:
                plt.plot(pos, label='positive')
                plt.plot(neg, label='negative')
                plt.legend()
                plt.savefig(path+'_goodness.png')
                plt.close()
        self.eval()
        return self.forward(x_pos.to(device)).detach(), self.forward(x_neg.to(device)).detach()
            
        
    

class NetFF(torch.nn.Module):
    def __init__(self,backprop=False, input_dim=784, hidden_dim=1000, n_layers=4, custom_nlinear = torch.nn.ReLU(), goodness_function = lambda x : torch.norm(x,dim=1,p=torch.inf), num_epochs=1000, lr=1e-4, dropout=0, batch_size=512,normalize_function=lambda x : torch.nn.functional.normalize(x,p=2) ):
        super().__init__()
        self.threshold=3
        self.backprop=backprop
        self.num_epochs=num_epochs
        self.hidden_dim=hidden_dim
        self.layers = torch.nn.ModuleList([ForwardLayer(input_dim if i==0 else hidden_dim, hidden_dim, custom_nlinear, goodness_function, num_epochs, lr, dropout) for i in range(n_layers)])
        self.goodness_function=goodness_function
        self.batch_size=batch_size
        self.lr=lr
        return 
 
    def predict(self, x,exclude_first=False):
        device=x.get_device()
        goodness_per_label = torch.empty((x.shape[0],10,len(self.layers))).to(device)

        for label in range(10):            
            x_lab = label_images(x, label, rgb=(len(x.shape)>2)).reshape(x.shape[0], -1)
            goodness = []
            for i, layer in enumerate(self.layers):
                x_lab = layer(x_lab)
                if not (i==0 and exclude_first==True):
                    goodness_per_label[:,label,i]=self.goodness_function(x_lab)
        goodness_per_label=torch.nn.functional.softmax(goodness_per_label,dim=1)
        goodness_per_label=torch.sum(goodness_per_label,dim=2)
        return torch.argmax(goodness_per_label, dim=1)

    def predict_old(self, x,exclude_first=False):
        goodness_per_label = []
        for label in range(10):            
            x_lab = label_images(x, label, rgb=(len(x.shape)>2)).reshape(x.shape[0], -1)
            goodness = []
            for i, layer in enumerate(self.layers):
                x_lab = layer(x_lab)
                if not (i==0 and exclude_first==True):
                    goodness.append(self.goodness_function(x_lab))
            goodness_per_label.append(sum(goodness).unsqueeze(1))
            
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return torch.argmax((goodness_per_label), dim=1)
       
    def extract_and_store_representations(self, testloader, device):
        x_te,y_te=next(iter(testloader))
        x=label_images(x_te, y_te, rgb=(len(x_te.shape)>2)).to(device)
        x=x.reshape(x.shape[0],-1)
        allrepresentations=np.zeros((x_te.shape[0], len(self.layers), self.hidden_dim))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            allrepresentations[:,i,:]=x.detach().cpu().numpy()
        return allrepresentations, y_te.detach().cpu().numpy()
    
    def extract_saliency_map(self, testloader, device):
        x_te,y_te=next(iter(testloader))
        x_te=x_te.to(device)
        x_te.requires_grad_()
        #x=label_images(x_te, y_te).to(device)
        acc=0
        for i, layer in enumerate(self.layers):
            if i==0:
                x = layer(x_te)
            else:
                x = layer(x)
            acc+=torch.norm(x,dim=1,p=torch.inf)
        acc.mean().backward()
        return x_te.grad
    

    def forward(self,x):
        goodness = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            goodness.append(self.goodness_function(x))
        return sum(goodness).unsqueeze(1)
    
    def train_net_bp(self, x_pos, x_neg, testloader, path, device):
        dataset=torch.utils.data.TensorDataset(x_pos, x_neg)
        dataloader=torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        pos=[]
        neg=[]
        for epoch in range(self.num_epochs):
            self.train()
            avg_pos=0
            avg_neg=0
            if (epoch % 100) ==0:
                print("Epoch: ", epoch,flush=True)
            for xpos, xneg in dataloader:
                xpos=xpos.to(device)
                xneg=xneg.to(device)
                pos_goodness=self.forward(xpos)
                neg_goodness=self.forward(xneg)
                l = torch.log(1 + torch.exp(torch.cat([
                -pos_goodness + self.threshold,
                neg_goodness - self.threshold]))).mean()
                avg_pos+=pos_goodness.mean().item()
                avg_neg+=neg_goodness.mean().item()
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            pos.append(avg_pos/len(dataloader))
            neg.append(avg_neg/len(dataloader))
            if epoch%50==0:
                plt.figure()
                plt.plot(pos, label='pos')
                plt.plot(neg, label='neg')
                plt.legend()
                plt.savefig(path+f'goodness_{epoch}.png')
                plt.close()
                pself.eval()
                self.evaluate(testloader, device)
            
    def evaluate(self,testloader,device):
        x_te,y_te=next(iter(testloader))
        x_te=x_te.to(device)
        y_te=y_te.to(device)
        print(device,x_te.get_device(),y_te.get_device(),flush=True)
        print('Test accuracy:', self.predict(x_te).eq(y_te).float().mean().item(),flush=True)
        
    
    def train_(self, data,path,device):
        
        trainloader,testloader=data
        x,y=next(iter(trainloader))
        x,y=multilabel_batch(x, y)
        y=y.reshape(-1)
        x_pos=x[y==1].to(device)
        x_neg=x[y==0].to(device)
        rnd=torch.randperm(x_neg.size(0))# comment these
        x_neg=x_neg[rnd][:x_pos.size(0)]# two lines to use all data (batchsize=600000)
        print(x_pos.size)
        print("Training")
        if self.backprop:
            self.train_net_bp(x_pos,x_neg, testloader, path, device)
        else:
            for l,layer in enumerate(self.layers):
                #layer.to(device)
                self.train()
                layer_path=path+f'layer{l}_'
                x_pos, x_neg = layer.train_layer(x_pos, x_neg, layer_path, device, self.batch_size)
                self.eval()
                self.evaluate(testloader,device)


class FullyConnected(torch.nn.Module):
    
    def __init__(self, input_dim=784, hidden_dim=1000, n_layers=4, custom_nlinear = torch.nn.ReLU(), num_epochs=20, lr=1e-4, dropout=0):
        super().__init__()
        self.num_epochs=num_epochs
        self.lr=lr
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_features=input_dim if i==0 else hidden_dim, out_features=hidden_dim, bias=True)for i in range(n_layers)])
        self.head = torch.nn.Linear(in_features=hidden_dim, out_features=10)
        self.dropout=torch.nn.Dropout(p=dropout, inplace=False)
        self.activation = custom_nlinear
        self.hidden_dim=hidden_dim
        
    def get_accuracy(self, dataloader):
        device=next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            correct=0
            for x, y in dataloader:
                x=x.reshape(x.shape[0], -1)
                out=self.forward(x.to(device))
                correct+=(torch.argmax(out, axis=1)==y.to(device)).sum()
        return correct/len(dataloader.dataset)
    
    def extract_and_store_representations(self, testloader, device):

        x,y=next(iter(testloader))
        #x=label_images(x_te, y_te, rgb=(len(x_te.shape)>2)).to(device)
        x=x.reshape(x.shape[0],-1).to(device)
        allrepresentations=np.zeros((x.shape[0], len(self.layers), self.hidden_dim))
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
            allrepresentations[:,i,:]=x.detach().cpu().numpy()
        return allrepresentations, y.detach().cpu().numpy()
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
            x = self.dropout(x)
        return self.head(x)
        
    def forward2(self, x):
        activation=list()
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
            x = self.dropout(x)
            activation.append(x)
        activation=torch.cat(activation,dim=0)
        return self.head(x),activation
    
    def train_(self, data, path, device):
        lam=os.environ.get('REGULARIZATION')
        if lam is not None:
            lam=float(os.environ.get('REGULARIZATION'))
            print(f"reg lambda: {lam}")
            
        trainloader,testloader=data
        loss=torch.nn.CrossEntropyLoss()
        optimizer=torch.optim.Adam(self.parameters(), lr=self.lr)
        losses=list()
        l1_norm=list()
        for epoch in range(self.num_epochs):
            if (epoch % 5) == 0:
                print("Epoch: ", epoch,flush=True)
            self.train()
            l_epoch=0
            for x,y in trainloader:
                x=x.to(device).reshape(x.shape[0],-1)
                y=y.to(device)
                out,act=self.forward2(x)
                #l1=sum(p.abs().sum() for p in self.parameters())
                l1=torch.norm(act,p=1)
                if lam is not None:
                    l=loss(out, y)+lam*l1
                else:
                    l=loss(out, y)
                optimizer.zero_grad()
                l.backward()
                l_epoch+=l.item()
                l1_norm.append(l1.detach().cpu().numpy())
                optimizer.step()
            losses.append(l_epoch/len(trainloader))
            if epoch%2==0:
                print(f"L1 norm: {l1}")
                plt.figure()
                plt.plot(losses)
                plt.savefig(path+f'loss.png')
                plt.close()

                plt.figure()
                plt.plot(l1_norm)
                plt.savefig(path+f'norm.png')
                plt.close()

            self.eval()
            print("Test acc: ", self.get_accuracy(testloader),flush=True)
            
    def predict(self,x):
        x=x.reshape(x.shape[0], -1)
        return torch.argmax(self(x), -1)

