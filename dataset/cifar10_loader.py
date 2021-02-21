import pickle as plk 
import numpy as np
import torch
import torchvision.transforms as transforms
import os
from PIL import Image

train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
test_list = ['test_batch']

def unpickle(filename):
    with open(filename, 'rb') as f:
        datadict = plk.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) # W,H,F
        Y = np.array(Y).reshape(-1, 1)
    
    return X, Y

def load_cifar10(root, dataset_type):
    X = []
    Y = []
    if dataset_type == 'train':
        for i in range(5):
            f = os.path.join(root, train_list[i])
            x_batch, y_batch = unpickle(f)
            X.append(x_batch)
            Y.append(y_batch)
        X = np.concatenate(X)
        Y = np.concatenate(Y)
    elif dataset_type == 'test':
        X, Y = unpickle(os.path.join(root, test_list[0]))
    else:
        print('No target data files found!')
    
    return X, Y

class TransformLoader:
    def __init__(self, image_size,
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.25), int(self.image_size*1.25)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()


    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class Dataset:
    def __init__(self, root, transform, dataset_type='train'):       
        self.transform = transform
        self.imgs, self.labels = load_cifar10(root, dataset_type)
    
    def __getitem__(self, key):
        img = Image.fromarray(self.imgs[key])
        img = self.transform(img)
        labels = torch.from_numpy(self.labels).long()
        # labels_onehot = torch.zeros(labels.shape[0], CIFAR10_Loader.num_classes).scatter_(1,labels,1)
        # label = labels_onehot[key]
        return img, labels[key].squeeze()
    
    def __len__(self):
        return len(self.labels)

class CIFAR10_Loader(object):
    inputchannel = 3
    num_classes = 10
    # root='F:\GITHUB_CODE\data\cifar10\cifar-10-batches-py'

    def __init__(self, image_size = (32,32), root='F:\GITHUB_CODE\data\cifar10\cifar-10-batches-py'):
        super(CIFAR10_Loader, self).__init__()
        self.trans_loader = TransformLoader(image_size)
        self.root = root
        

    def get_dataset(self, dataset_type='train', aug=False):     
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = Dataset(self.root, transform, dataset_type)
        return dataset

    # def get_data_loader(self, dataset_type='train', aug=False, **data_loader_params):
    #     dataset = get_dataset(dataset_type=dataset_type, aug=aug)
    #     # data_loader_params = dict(batch_size = self.batch_size, shuffle = True)       
    #     data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    #     return data_loader
    


if __name__ == '__main__':
    cifar10 = CIFAR10_Loader()
    cifar10_dataset = cifar10.get_dataset()
    len = cifar10_dataset.__len__()
    print(len)




        

