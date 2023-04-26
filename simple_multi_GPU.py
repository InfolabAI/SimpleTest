import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import time
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from pdb import set_trace

SEED=1223

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

gpus = '0,1'
os.environ["CUDA_VISIBLE_DEVICES"]=gpus
num_gpus = len(gpus.split(','))
batch_size = 1024

def get_CIFAR10():
    data_tr = datasets.CIFAR10('/mnt/share_nfs/dataset/cifar10', train=True, download=True)
    data_te = datasets.CIFAR10('/mnt/share_nfs/dataset/cifar10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te

class get_DataHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X)

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = 10
        self.stem_conv = conv_block(3,128)
        self.module_list = nn.ModuleList()
        for i in range(blocks):
            self.module_list += [conv_block(128,128)]

        self.module_list2 = nn.ModuleList()
        for i in range(blocks):
            self.module_list2 += [conv_block(128,128)]

        self.fc1 = nn.Linear(128, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        e = self.stem_conv(x)
        for md in self.module_list:
           e = md(e) 

        e = F.max_pool2d(e, 2)
        for md in self.module_list2:
           e = md(e) 

        e = F.max_pool2d(e, 2)
        e = F.adaptive_avg_pool2d(e, 1)
        e = e.view(-1, 128)
        e = F.relu(self.fc1(e))
        out = self.fc2(e)
        return out

device = torch.device("cuda")
X_tr, Y_tr, X_te, Y_te = get_CIFAR10()
model = ConvNet().to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model).to(device)
else:
    model = model.to(device)

op = torch.optim.SGD(model.parameters(), lr=0.01)
transform = transforms.Compose([transforms.ToTensor(), 
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])
tr_loader = torch.utils.data.DataLoader(get_DataHandler(X_tr, Y_tr, transform), shuffle=True, batch_size=batch_size, num_workers=1)
te_loader = torch.utils.data.DataLoader(get_DataHandler(X_te, Y_te, transform), shuffle=False, batch_size=batch_size, num_workers=1)
ce_loss = nn.CrossEntropyLoss()

#Evaluation like Discriminator
###
start = time.time()
for epoch in range(10):
    model.train()
    loss_t = 0
    for x, y in tr_loader:
        op.zero_grad()
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = ce_loss(out, y)
        loss.backward()
        op.step()

        loss_t += float(loss.detach())

    print('mean_loss', loss_t/len(tr_loader))

    model.eval()
    pred_list = []
    with torch.no_grad():
        for x, y in te_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = torch.topk(out, 1)[1]
            pred_list += [pred.view(-1).cpu()] 

        pred_te = torch.cat(pred_list, 0)
        acc = (1.0*(pred_te == Y_te)).mean()
        print("acc", float(acc.detach()))

print("elapsed time",  time.time() - start)
