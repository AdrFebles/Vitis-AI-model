import argparse
import torch
from pathlib import Path # convenient way to deal w/ paths
import numpy as np # standard for data processing
import pandas as pd # standard for data processing
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import StandardScaler # Useful module to standarize the values
from pytorch_nndct.apis import torch_quantizer
from torch.utils.data import dataset, dataloader
import random

parser = argparse.ArgumentParser(description='CNN pytorch quantizer test')

parser.add_argument('--quant_mode',
                    type=str, 
                    default='calib', 
                    help='CNN pytorch quantization mode, calib for calibration of quantization, test for evaluation of quantized model')
parser.add_argument('--subset_len',
                    type=int,
                    default=None,
                    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--batch_size',
    default=3,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')
parser.add_argument('--inspect', 
    dest='inspect',
    action='store_true',
    help='inspect model')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is: ', device)
#print('GPUs existentes: ', torch.cuda.device_count())
deploy = args.deploy
batch_size = args.batch_size
subset_len = args.subset_len
inspect = args.inspect
quant_mode = args.quant_mode

train = pd.read_csv('vitis_data/rds_cpu_utilization_cc0c53.csv')
#print(train)
valid = pd.read_csv('vitis_data/calib_data.csv')

def parse_and_standardize(df: pd.DataFrame, scaler: StandardScaler = None):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['stand_value'] = df['value']
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(df['stand_value'].values.reshape(-1, 1))
    df['stand_value'] = scaler.transform(df['stand_value'].values.reshape(-1, 1))
    return scaler

data_scaler = parse_and_standardize(train)
parse_and_standardize(valid, data_scaler)

#test_data= dataset.TensorDataset(torch.FloatTensor(test_data))
#test_loader= dataloader.DataLoader(test_data, batch_size=1, shuffle=False)

class CPUDataset(Dataset):
    def __init__(self, data: pd.DataFrame, size: int,
                 step: int = 1):
        self.chunks = torch.FloatTensor(data['stand_value']).unfold(0, size + 1, step)
        self.chunks = self.chunks.view(-1, 1, size + 1)

    def __len__(self):
        return self.chunks.size(0)

    def __getitem__(self, i):
        x = self.chunks[i, :, :-1]
        print('x size:', x)
        y = self.chunks[i, :, -1:].squeeze(1)
        return x, y

n_factors =10
train_ds = CPUDataset(train, n_factors)
valid_ds1 = CPUDataset(valid, n_factors)
#print('Valid unfolding:', valid_ds.size(0))
#print(valid_ds)
valid_dataset=dataset.TensorDataset(torch.FloatTensor(valid['stand_value']).unfold(0, 11, 1))
#print('Test data:',valid_dataset)
valid_ds= dataloader.DataLoader(valid_dataset, batch_size=1, shuffle=False)
#print(valid_ds)

#--------------Definición del modelo----------------------

def conv_layer(in_feat, out_feat, kernel_size=3, stride=1,
               padding=1, relu=True):
    res = [
        nn.Conv1d(in_feat, out_feat, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm1d(out_feat),
    ]
    if relu:
        res.append(nn.ReLU())
    return nn.Sequential(*res)


class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.in_feat, self.out_feat = in_feat, out_feat
        self.conv1 = conv_layer(in_feat, out_feat)
        self.conv2 = conv_layer(out_feat, out_feat, relu=False)
        if self.apply_shortcut:
            self.shortcut = conv_layer(in_feat, out_feat,
                                       kernel_size=1, padding=0,
                                       relu=False)

    def forward(self, x):
        out = self.conv1(x)
        if self.apply_shortcut:
            x = self.shortcut(x)
        return x + self.conv2(out)

    @property
    def apply_shortcut(self):
        return self.in_feat != self.out_feat


class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool1d(1)
        self.mp = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)
class CNN(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.base = nn.Sequential(
            ResBlock(1, 8),  # shape = batch, 8, n_factors
            ResBlock(8, 8),
            ResBlock(8, 16),  # shape = batch, 16, n_factors
            ResBlock(16, 16),
            ResBlock(16, 32),  # shape = batch, 32, n_factors
            ResBlock(32, 32),
            ResBlock(32, 64),  # shape = batch, 64, n_factors
            ResBlock(64, 64),
        )
        self.head = nn.Sequential(
            AdaptiveConcatPool1d(),  # shape = batch, 128, 1
            nn.Flatten(),
            nn.Linear(128, out_size)
        )

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out

def calculate_prediction_errors(model, dataset: CPUDataset,
    device: torch.device):
    #model=model.eval()
    model=model.to(device)
    criterion = nn.MSELoss().to(device)
    with torch.no_grad():
        errors = []
        for x, y in tqdm(dataset):
            #x = x.to(device)[None]
            #y = y.to(device)[None]
            x = x.to(device)
            y = y.to(device)
            print('La x es:',x)
            print('La y es:',y)
            predicted = model(x)
            print(predicted)
            prediction_error = criterion(predicted, y)
            errors.append(prediction_error)
        return errors/float(len(errors))

#
#------------ MODEL --------------------------------------
checkpoint= torch.load('./pretrained.pth')
#print('Checkpoint',checkpoint)

cnn_model = CNN(out_size=1)
model=cnn_model.to(device)
model.load_state_dict(checkpoint)
print(model)
model.eval()

#-------- NNDCT QUANTIZATION -------------------------
if args.quant_mode != 'test' and deploy:
    deploy = False
    print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
if deploy and (batch_size != 1 or subset_len != 1):
    print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
    batch_size = 1
    subset_len = 1

A = [ ]
for i in range (11):
    A.append(random.randint(0, 100))
print('A es:',A)
class InpDataset(Dataset):
    def __init__(self, data: pd.DataFrame, size: int,
                 step: int = 1):
        self.chunks = torch.FloatTensor(data).unfold(0, size + 1, step)
        self.chunks = self.chunks.view(-1, 1, size + 1)

    def __len__(self):
        return self.chunks.size(0)

    def __getitem__(self, i):
        x = self.chunks[i, :, :-1]
        print('Cadena inputs',x)
        y = self.chunks[i, :, -1:].squeeze(1)
        print(y)
        return x,y
inputs=InpDataset(A,n_factors)

if args.quant_mode == 'test':
   quant_model = model
   import sys
   from pytorch_nndct.apis import Inspector
     # create inspector
     # inspector = Inspector("0x603000b16013831") # by fingerprint
   inspector = Inspector("DPUCAHX8L_ISA0_SP")  # by name
     # start to inspect
   inspector.inspect(quant_model, inputs[:][0])
   sys.exit()
else:
  ## new api
  ####################################################################################
  quantizer = torch_quantizer(
       quant_mode, model, inputs[:][0],device=device)
  model = quantizer.quant_model                           

#--------- NNDCT QUANTIZATION FORWARDING ---------------
valid_pred_errors = calculate_prediction_errors(model, inputs, device)

#------ HANDLE QUANTIZATION RESULTS ---------------------
if args.quant_mode == 'calib':
    quantizer.export_quant_config()
if deploy:
    quantizer.export_xmodel()
    quantizer.export_onnx_model()
