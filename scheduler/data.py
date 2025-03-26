import torch
import torchvision

def get_data(dataset, modelname, batch_size_bp=2048):
    if dataset=='svhn':
        max_train=73257
        max_test=26032
    else:
        max_train=60000
        max_test=10000
    if modelname=='ff':
        dataloaders=get_dataloaders(dataset, max_train, max_test)
    elif modelname=='bp_ff':
        dataloaders=get_dataloaders(dataset, max_train, max_test)
    elif modelname=='bp':
        dataloaders=get_dataloaders(dataset, batch_size_bp, max_test)
    return dataloaders

def label_images(images, labels, rgb=False):
    labeled_images = images.clone()
    if rgb:
        labeled_images[:,:,0,:10]=0
        labeled_images[range(len(images)),:,0, labels]=images.max()
    else:
        labeled_images[:,:10]=0
        labeled_images[range(len(images)), labels]=images.max()
    
    return labeled_images

def get_dataloaders(dataset='mnist', train_batch_size=128, test_batch_size=128):
    mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,)),torchvision.transforms.Lambda(torch.flatten)])
    svhn_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))])
    cifar_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset=='mnist':
        trainset = torchvision.datasets.MNIST(f'./data/', transform=mnist_transform,  train=True, download=True)
        testset = torchvision.datasets.MNIST(f'./data/', transform=mnist_transform, train=False, download=True)
    elif dataset=='fashionmnist':
        trainset = torchvision.datasets.FashionMNIST(f'./data/', transform=mnist_transform,  train=True, download=True)
        testset = torchvision.datasets.FashionMNIST(f'./data/', transform=mnist_transform, train=False, download=True)
    elif dataset=='svhn':
        trainset = torchvision.datasets.SVHN(f'./data/', transform=svhn_transform,  split='train', download=True)
        testset = torchvision.datasets.SVHN(f'./data/', transform=svhn_transform, split='test', download=True)
    elif dataset=='cifar':
        trainset = torchvision.datasets.CIFAR10(f'./data/', transform=cifar_transform,  train=True, download=True)
        testset = torchvision.datasets.CIFAR10(f'./data/', transform=cifar_transform, train=False, download=True)
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)  
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)
    return trainloader, testloader

def multilabel_batch(data, labels):
    images=torch.empty((0,)+data.shape[1:])
    newlabels=torch.empty(0, dtype=torch.float32)
    for i in range(10):
        images=torch.cat((images,label_images(data, torch.ones_like(labels)*i,
                                              rgb=(len(data.shape)>2))))
        newlabels=torch.cat((newlabels, torch.eq(labels,i)))
        newlabels.type(torch.FloatTensor)
    return images.reshape(images.shape[0], -1), newlabels.reshape(-1,1)

        
