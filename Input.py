import torch
from torchvision import datasets, transforms

def load_data(source):
    #Path Variables
    data_dir = source
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Defining Transformation Parameters for Tensors.
    train_data_transforms = transforms.Compose([transforms.RandomRotation(45),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
    valid_data_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
    test_data_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
    #Loading Image Datasets
    train_image_datasets = datasets.ImageFolder(train_dir, transform= train_data_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform= valid_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform= test_data_transforms)
    #Initialize the Dataloader with Image Dataset and Transformation
    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=64)
    return train_dataloaders, valid_dataloaders, test_dataloaders, train_image_datasets



