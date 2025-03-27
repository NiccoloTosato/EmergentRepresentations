import os
import numpy as np
import torch
import time
from data import get_data
from models import get_model

def expandExperiments(configuration):
    experiments_queue=list()
    configuration_experiments = configuration["Experiments"]
    for experiment in configuration_experiments:
        config_dictionary=experiment["experiment"]
        for single_lr in config_dictionary["learningRate"]:
            for single_epochs in  config_dictionary["epochs"]:
                for count in range(0,config_dictionary["repetitionCount"]):
                    tmp=dict(config_dictionary)
                    tmp['learningRate']=single_lr
                    tmp['epochs']=single_epochs
                    tmp['runId']=count
                    del tmp['repetitionCount']
                    experiments_queue.append(tmp)
    return experiments_queue

class ExperimentConfigurator:
    def __init__(self, config,base_path):

        self.model=config["model"]
        self.learning_rate=config["learningRate"]
        self.epochs=config["epochs"]
        self.n_layers=config["nLayers"]
        self.dataset=config["dataSet"].lower()
        self.dropout=config["dropout"]
        self.batch_size=config["batch_size"]
        self.base_path=base_path+f"/{self.model}/{self.dataset}/{config['nonLinearity']}_l{config['norm']}_{self.epochs}_{self.batch_size}_{self.learning_rate}/"
        self.nonLinearity=self.set_activation(config["nonLinearity"])
        self.norm,self.normalize_function=self.set_norm(config["norm"])

        self.runId=config["runId"]
        print(self.base_path,flush=True)
        
    def __repr__(self):
        s=f'Model: {self.model}\n'
        s+=f'\tLearningRate: {self.learning_rate}\n'
        s+=f'\tEpochs: {self.epochs}\n'
        s+=f'\tActivation: {self.nonLinearity}\n'
        s+=f'\tGoodness: {self.norm}\n'
        s+=f'\t Dataset: {self.dataset}\n'
        s+=f'\t Layer count: {self.n_layers}\n'
        s+=f'\t Dropout: {self.dropout}\n'
        s+=f'\t Coding: {self.base_path}\n'
        return s

    def set_norm(self,norm):
        if str(norm).lower()=="1":
            return lambda x : torch.norm(x, dim=1, p=1),lambda x :torch.nn.functional.normalize(x,p=1)
        if str(norm).lower()=="2":
            return lambda x : torch.norm(x, dim=1, p=2),lambda x :torch.nn.functional.normalize(x,p=2)
        if str(norm).lower()=="inf":
            return lambda x : torch.norm(x, dim=1, p=torch.inf),lambda x : torch.nn.functional.normalize(x,p=torch.inf)
    
    def set_activation(self,function):
        if function.lower()=="sigmoid":
            return torch.nn.Sigmoid()
        if function.lower()=="relu":
            return torch.nn.ReLU()
    
    def run(self):
        # check if exist the folder
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.dataset=='svhn':
            input_dim=3072
            hidden_dim=1024*3
        elif self.dataset=='cifar':
            input_dim=32*32*3
            hidden_dim=1024*3
        if self.model=='ff' or self.model=='bp_ff':
            model=get_model(self.model, custom_nlinear = self.nonLinearity, lr=self.learning_rate,
                        n_layers=self.n_layers,
                        dropout=self.dropout,
                        num_epochs=self.epochs,
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        batch_size=self.batch_size,
                            goodness_function = self.norm,
                            normalize_function=self.normalize_function).to(device)
        else:
            model=get_model(self.model, custom_nlinear = self.nonLinearity, lr=self.learning_rate,
                            n_layers=self.n_layers, dropout=self.dropout, num_epochs=self.epochs,
                            input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        
        data=get_data(self.dataset, self.model)
        model.train_(data, self.base_path, device)
        torch.save(model.state_dict(), self.base_path+f'/weights_{self.runId}.pt')
        model.eval()
        rep,lbl=model.extract_and_store_representations(data[1], device)
        x_test,_=next(iter(data[1]))
        prediction=model.predict(x_test.to(device))
        
        np.save(self.base_path+f"/prediction_{self.runId}.npy", prediction.detach().cpu())
        np.save(self.base_path+f"/reps_{self.runId}.npy", rep)
        np.save(self.base_path+f"/lbls_{self.runId}.npy", lbl)
        
