import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys, traceback

def get_experiments_list(base_path="/home/path/change_me",model="ff",dataset="mnist"):
    path=f"{base_path}/{model}/{dataset}/"
    experiments_list=os.listdir(path)
    path_list=list()
    for exp in experiments_list:
        exp=f"{path}/{exp}"
        path_list.append(exp)
    return path_list

def load_data(experiment,only_correct=True):
    '''
    Load testset representation and labels
    '''
    runId=0
    rep=np.load(experiment+f'/reps_{runId}.npy')
    lbl=np.load(experiment+f'/lbls_{runId}.npy')
    prediction=np.load(experiment+f'/prediction_{runId}.npy')
    if only_correct:
        if '/bp/' in experiment :
#            prediction=np.argmax(prediction,axis=1)
            return rep[lbl == prediction], lbl[lbl == prediction],np.sum(lbl==prediction)/len(prediction)
        else: 
            return rep[lbl == prediction], lbl[lbl == prediction],np.sum(lbl==prediction)/len(prediction)
    return rep, lbl

def wall_plot(experiment,rep,lbl):
    idx = np.argsort(lbl)
    plt.figure(figsize=(20,60))
    for layer in range(np.shape(rep)[1]):
        plt.subplot(np.shape(rep)[1],1,1+layer)
        #print(f"Layer {layer} max value {np.max(rep[idx,layer,:])}")
        plt.imshow(rep[idx,layer,:][::5],aspect=0.2,vmin=0.0,cmap='Blues',interpolation='bicubic')
        plt.xticks(labels=[],ticks=[])
        plt.yticks(labels=range(0,10),ticks=range(100,2100,200),fontsize=17)
        plt.ylabel("Classes",fontsize=20)
        plt.xlabel("Neurons",fontsize=20)
        plt.title(f"Layer {layer+1}")
        #plt.colorbar()
        plt.savefig(experiment+"/wall_0.png")
        # print( f"Layer {layer} sparseness {sparseness(torch.from_numpy(rep[idx,layer,:]))}")
        
        
def sparseness(x):
    #Calculate the row sparseness, average on columns
    rateo=torch.norm(x,dim=1,p=1)/torch.norm(x,dim=1,p=2)
    s=(np.sqrt(x.shape[1])-(rateo))/(np.sqrt(x.shape[1])-1)
    print(x.shape)
    return s.nanmean().item()
    
def save_stat(experiment,rep,lbl,accuracy):
    '''
    Generate a heatmap to visualize the activations across all layers.    
    '''
    with open(f"{experiment}/stats.txt", mode="w") as file:
        file.write(f"Accuracy {accuracy}\n")
        file.write("--------------Sparseness--------------\n")
        for layer in range(np.shape(rep)[1]):
            file.write(f"Layer {layer} sparseness {sparseness(torch.from_numpy(rep[:,layer,:]))}\n")
        file.write("--------------Max value--------------\n")
        for layer in range(np.shape(rep)[1]):
            file.write(f"Layer {layer} max value {np.max(rep[:,layer,:])}\n")

    #must change the number of layer dinamically


def analyze(experiment):
    rep,lbl,accuracy=load_data(experiment)
    wall_plot(experiment,rep,lbl)
    save_stat(experiment,rep,lbl,accuracy)

if __name__ == "__main__":
    for model in ["bp","ff","bp_ff"]:
        for dataset in ["mnist","fashionmnist"]:
            experiments=get_experiments_list(model=model,dataset=dataset)
            for experiment in experiments:
                try:
                    analyze(experiment)
                    print(f"Done {experiment}")
                except Exception as error:
                    print(f"Failed analyze {experiment}")
                    print(error)
                    traceback.print_exc(file=sys.stdout)
    
