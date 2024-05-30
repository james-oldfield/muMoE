import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import pickle
from PIL import Image
from matplotlib import pyplot as plt
from fvcore.nn import FlopCountAnalysis
import math
import argparse
import timm
from tqdm import tqdm
import logging
import os
import tensorly as tl
import numpy as np

import sys
sys.path.append("../")
from MuMoE import MuMoE, CPMuMoE, TRMuMoE
from utils import cosine_lr

tl.set_backend('pytorch')
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("device", type=int, help="GPU id")
parser.add_argument('--load_weights', action='store_true', help='Flag to load model weights')
parser.add_argument('--weights_path', type=str, default='./checkpoints/model-0.pth', help='Path to model weights file')

# add argparse for float learning rate
parser.add_argument('--learning_rate', type=float, default=3e-05, help='Learning rate')
parser.add_argument('--reg', type=float, default=0.1, help='Weight reg term')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--decay_type', type=str, choices=['cosine', 'linear', 'exponential'], help='', default='linear')
parser.add_argument('--rank', type=int, default=128, help='Rank of layer')
parser.add_argument('--batch_size', type=int, default=4096, help='batch_size')
parser.add_argument('--use_bias', type=int, default=1, help='use bias?')
parser.add_argument('--num_experts', type=int, help='number of experts to use')
parser.add_argument('--hierarchy', type=int, default=1)
parser.add_argument('--layer', type=str, choices=['MLP', 'CPMuMoE', 'TRMuMoE'], help='Choose layer type', required=True)
parser.add_argument('--activation', type=str, choices=['entmax'], default='entmax')
parser.add_argument('--custom_name', type=str, help='Custom name for model', default='')
parser.add_argument('--dataset', type=str, choices=['ImageNET1k'], help='dataset', default='ImageNET1k')
parser.add_argument('--generate_grids', action='store_true', help='')
parser.add_argument('--model', type=str, default='vit_base_patch32_clip_224.openai')
parser.add_argument('--tqdm', action='store_true', help='')

args = parser.parse_args()

DEVICE_ID = str(args.device)
print(f'Using device: {DEVICE_ID}')

def Model(basemodel, num_classes, input_dim, hidden_dim=512, act=torch.sigmoid):
    if args.layer == 'MLP':
        layer = nn.Linear(input_dim, hidden_dim)

    elif args.layer == 'TRMuMoE':
        r1 = r2 = 4

        # parameter match linear layers
        I = config['input_dim']
        O = config['hidden_dim']
        N = config['num_experts']
        e1 = config['num_experts']
        
        linear_params = 769_000 # to match
        exp_params = N*I
        r3 = math.ceil((linear_params - (4*N*4 + exp_params)) / ((I+1)*4 + 4*O))
        config['rank'] = r3
        layer = MuMoE(TRMuMoE, input_dim, hidden_dim, act=act, expert_dims=[e1], ranks=[[r1, e1, r2], [r2, input_dim, r3], [r3, hidden_dim, r1]], use_bias=config['use_bias'], hierarchy=1)

    elif args.layer == 'CPMuMoE':
        # parameter match linear layers
        I = config['input_dim']
        O = config['hidden_dim']
        N = config['num_experts']
        
        linear_params = 769_000 # to match
        exp_params = N*I
        
        rank = math.ceil((linear_params - exp_params) / (I+1+O+N))
        config['rank'] = rank

        e1 = config['num_experts']
        layer = MuMoE(CPMuMoE, input_dim, hidden_dim, act=act, expert_dims=[e1], ranks=rank, use_bias=config['use_bias'], hierarchy=1)

    return nn.Sequential(
        basemodel,
        layer,
    )

config = {
    'bs': args.batch_size,
    'num_epochs': 10,
    'num_experts': int(args.num_experts),
    'use_bias': bool(args.use_bias),
    'rank': None,
    'custom_name': args.custom_name,
    'lr': args.learning_rate,
    'reg': args.reg,
    'dropout': args.dropout,
    'layer': args.layer,
    'train_base': True,
    'decay_type': args.decay_type,
    'input_dim': 768,
}

config['moe_activ'] = args.activation
config['pre_model'] = args.model

dataset_hiddendims = {
    'ImageNET1k': 1000,
}

try:
    config['dataset'] = args.dataset
    config['hidden_dim'] = dataset_hiddendims[args.dataset]
except:
    print('Failed to get dataset configs')
    raise

from entmax import entmax15
act = lambda x: entmax15(x, dim=-1)

n_seeds = 1

g_accs = []
for seed in range(n_seeds):
    torch.manual_seed(seed)

    basemodel = timm.create_model(config['pre_model'], pretrained=True, num_classes=0)
    data_cfg = timm.data.resolve_data_config(basemodel.pretrained_cfg)
    transform_tr = timm.data.create_transform(**data_cfg)
    transform_te = timm.data.create_transform(**data_cfg, is_training=False)
    
    # use random-resize crop
    transform_tr = transforms.Compose([transforms.RandomResizedCrop(224, antialias=True, interpolation=Image.BICUBIC), transforms.ToTensor(), transform_tr.transforms[-1]])
    
    for param in basemodel.parameters():
        param.requires_grad = True

    if config['dataset'] == 'ImageNET1k':
        from utils import get_IN_classes

        trainset = datasets.ImageFolder('/data/ImageNet-2012/ILSVRC2012/train', transform=transform_tr)
        testset = datasets.ImageFolder('/data/ImageNet-2012/ILSVRC2012/val', transform=transform_te)
        class_names = trainset.classes = get_IN_classes()
        n_classes = 1000
    else:
        print('Dataset not supported')
        raise

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['bs'], shuffle=True, num_workers=8, drop_last=config['dataset'] == 'Caltech101')
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['bs'], shuffle=False, num_workers=8)

    num_batches = len(trainloader)

    model = Model(basemodel, input_dim=config['input_dim'], hidden_dim=config['hidden_dim'], num_classes=n_classes, act=act)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['reg'])

    # Define the loss function and oputimizer
    criterion = nn.CrossEntropyLoss() if config['dataset'] != 'CelebA' else nn.BCEWithLogitsLoss()
    scheduler = cosine_lr(optimizer, [pg['lr'] for pg in optimizer.param_groups], 500, config['num_epochs']*num_batches)
    
    scaler = GradScaler()

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if seed == 0:
        model_name = f"{config['custom_name']}-{config['layer']}-{config['dataset']}-{config['moe_activ']}-lr_{config['lr']}-trainbase_{config['train_base']}-numseeds_{n_seeds}-{config['pre_model']}-experts_{config['num_experts']}-bs_{config['bs']}-rank_{config['rank']}-numepochs_{config['num_epochs']}"

        model[1].eval()
        flops = FlopCountAnalysis(model[1].cuda(), torch.randn(1, config['input_dim']).cuda())
        print(f'# Flops: {format(int(flops.total()), ",")}, # Params: {format(sum([p.numel() for p in model[1].parameters() if p.requires_grad]), ",")}')
        layer_params = sum([p.numel() for p in model[1].parameters() if p.requires_grad])
        layer_flops = int(flops.total())
        print('---')
        model[1].train()

    accs1 = [0]
    correct = 0; total = -1

    for epoch in range(config['num_epochs']):
        running_loss = []
        model.train()

        tr_enum = enumerate(tqdm(trainloader)) if args.tqdm else enumerate(trainloader)
        model.train()

        for i, data in tr_enum:
            optimizer.zero_grad()

            step = i + epoch * num_batches
            scheduler(step)

            if type(data) == dict: inputs = data['image'] ; labels = data['label']
            else:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

            with autocast(device_type='cuda'):
                outputs = model(inputs).squeeze(dim=1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            # unscale for grad clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += [loss.item()]

        # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0; total = 0
            c1 = 0; c2=0; c3=0

            for data in testloader:
                if type(data) == dict:
                    images = data['image'] ; labels = data['label']
                else:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)

                with autocast(device_type='cuda'):
                    outputs = model(images).squeeze(dim=1)

                _, predicted = torch.topk(outputs.data, k=2, dim=1)
                correct += (predicted[:, 0] == labels).sum().item()

                total += images.size(0)

            print(f'Epoch {epoch}: Total accuracy:', correct/total*100)

    if config['dataset'] == 'CelebA':
        g_accs += [torch.mean(correct/total*100)]  # average accuracy over all 40 attributes
    else:
        g_accs += [correct/total*100]

mean_acc = np.mean(np.array(g_accs), 0)
std_acc = np.std(np.array(g_accs), 0)

results = {'Atts': {}}
results['Acc'] = [mean_acc, std_acc]
print('Mean acc:', mean_acc, 'std:', std_acc)
results['LayerParams'] = layer_params
results['LayerFLOPS'] = layer_flops