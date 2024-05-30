import torch
from functools import partial
import evaluate
import time
import torch.nn as nn
import torch.optim as optim
from timm.layers import DropPath
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import v2
from fvcore.nn import FlopCountAnalysis
from PIL import Image
import logging
from einops.layers.torch import Rearrange
import argparse
from tqdm import tqdm
import wandb

import tensorly as tl
import numpy as np
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
pair = lambda x: x if isinstance(x, tuple) else (x, x)

import sys
sys.path.append("../")
from MuMoE import CPMuMoE, TRMuMoE, MuMoE
from utils import linear_lr

from accelerate import Accelerator
from accelerate.utils import set_seed

def save_model_weights(model, model_name):
    torch.save(model.state_dict(), f'./checkpoints/model-{model_name}.pth')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_weights', action='store_true', help='Flag to load model weights')
    parser.add_argument('--weights_path', type=str, default='', help='Path to model weights file')

    # add argparse for float learning rate
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--reg', type=float, default=1e-4, help='Weight reg term')
    parser.add_argument('--dropout', type=float, default=0.0, help='Weight reg term')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch_size')
    parser.add_argument('--warmup', type=int, default=10_000, help='batch_size')
    parser.add_argument('--layer', type=str, choices=['MLP', 'CPMuMoE', 'TRMuMoE'], help='Choose layer type')
    parser.add_argument('--dataset', type=str, choices=['ImageNET1k'], help='dataset', default='ImageNET1k')
    parser.add_argument('--gradient_acc_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--custom_name', type=str, help='Custom name for model', default='')
    parser.add_argument('--tqdm', action='store_true', help='')

    parser.add_argument('--stochastic_depth', action='store_true', help='Use stochastic depth?')
    parser.add_argument('--mixup', type=float, default=0.5, help='Mixup strength')
    parser.add_argument('--randaugment', type=int, default=15, help='Randaugment magnitude')

    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.gradient_acc_steps)
    device = accelerator.device

    tl.set_backend('pytorch')
    torch.manual_seed(0)
    set_seed(0)

    # Codebase builds off: https://github.com/lucidrains/mlp-mixer-pytorch
    def pair(t):
        return t if isinstance(t, tuple) else (t, t)

    class PreNormResidual(nn.Module):
        def __init__(self, dim, fn, dp):
            super().__init__()
            self.fn = fn
            self.norm = nn.LayerNorm(dim)
            self.dp = dp

        def forward(self, x):
            return self.dp(self.fn(self.norm(x))) + x

    def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear, type='linear', depth=0):
        inner_dim = expansion_factor
        
        return dense(input_dim=dim, output_dim=inner_dim, second_dim=dim) if 'MuMoE' in args.layer else \
        nn.Sequential(
            dense(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            dense(inner_dim, dim),
            nn.Dropout(dropout)
        )
 
    def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 2048, expansion_factor_token = 256, act=torch.sigmoid, dropout = 0., stochastic_depth=False):
        image_h, image_w = pair(image_size)
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_h // patch_size) * (image_w // patch_size)

        # default to linear layers all round
        layer_c = layer_t = layer_i = layer_o = nn.Linear

        n_experts = 64
        t_dim = 196
        if args.layer == 'CPMuMoE':
            ########################### CP factorisation
            multiplier = 0.75
            rankt = int(t_dim*multiplier); rankc = int(dim*multiplier)

            layer_t = partial(MuMoE, MuMoE_layer=CPMuMoE, act=act, expert_dims=[n_experts], ranks=rankt, use_bias=True, hierarchy=1)
            layer_c = partial(MuMoE, MuMoE_layer=CPMuMoE, act=act, expert_dims=[n_experts], ranks=rankc, use_bias=True, hierarchy=1)

        elif args.layer == 'TRMuMoE':
            multiplier = 0.19
            r1 = r2 = 4 ; rankt = int(t_dim*multiplier); rankc = int(dim*multiplier)

            layer_t = partial(MuMoE, MuMoE_layer=TRMuMoE, act=act, expert_dims=[n_experts], ranks=[[r1, None, r2], [r2, None, rankt], [rankt, None, r1]], use_bias=True, hierarchy=1)
            layer_c = partial(MuMoE, MuMoE_layer=TRMuMoE, act=act, expert_dims=[n_experts], ranks=[[r1, None, r2], [r2, None, rankc], [rankc, None, r1]], use_bias=True, hierarchy=1)

        print(f'Using layer {layer_t} and {layer_c}')

        class MLP_Mixer(nn.Module):
            def __init__(self, channels, patch_size, dim, num_classes, depth, expansion_factor_token, expansion_factor, dropout, stochastic_depth):
                super(MLP_Mixer, self).__init__()
                self.stochastic_depth = stochastic_depth
                self.depth = depth
                
                num_patches = (224 // patch_size) ** 2  # assuming image size is 224x224

                self.stem = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                    layer_i((patch_size ** 2) * channels, dim),
                )

                self.mixer_blocks = nn.ModuleList([])
                for i in range(depth):
                    drop_prob = 0.1 * (i / (depth-1))
                    dp = DropPath(drop_prob) if i > 0 and stochastic_depth else nn.Identity()
                    self.mixer_blocks.append(nn.Sequential(
                        PreNormResidual(dim,
                            nn.Sequential(
                                Rearrange('b t c -> b c t'),
                                FeedForward(num_patches, expansion_factor_token, dropout, layer_t, type='token_mix', depth=i),
                                Rearrange('b c t -> b t c'),
                            ), dp=dp),  # TOKEN mix mlp
                        PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, layer_c, type='channel_mix', depth=i), dp=dp),  # CHANNEL mix mlp
                    ))

                self.head = nn.Sequential(
                    nn.LayerNorm(dim),
                    Reduce('b n c -> b c', 'mean'),
                    layer_o(dim, num_classes),
                )

            def forward(self, x):
                x = self.stem(x)
                for mixer_block in self.mixer_blocks:
                    x = mixer_block(x)
                x = self.head(x)
                return x
        
        return MLP_Mixer(channels, patch_size, dim, num_classes, depth, expansion_factor_token, expansion_factor, dropout, stochastic_depth)

    config = {
        'bs': args.batch_size // accelerator.num_processes,
        'lr': args.learning_rate,
        'reg': args.reg,
        'layer': args.layer,
        'custom_name': args.custom_name,
        'num_experts': None,
        'rank': None,
        'moe_activ': 'entmax',
        'dropout': args.dropout,
        'mixup': args.mixup,
        'randaugment': args.randaugment,
        'stochastic_depth': args.stochastic_depth,
        'gradient_acc_steps': args.gradient_acc_steps,
        'warmup': args.warmup,
        'patch_size': 16,
        'img_size': 224,
    }
    try:
        config['dataset'] = args.dataset
    except:
        print('Failed to get dataset configs')
        raise

    from entmax import entmax15
    act = lambda x: entmax15(x, dim=-1)

    num_epochs = 300

    # s16 model config
    config['depth'] = 8
    config['dim'] = 512
    config['expansion_factor'] = 2048
    config['expansion_factor_token'] = 256
    config['type'] = 's16'
    
    if config['randaugment']:
        transform_tr = transforms.Compose([transforms.RandomResizedCrop(config['img_size'], antialias=True, interpolation=Image.BICUBIC), torchvision.transforms.RandomHorizontalFlip(p=0.5), transforms.RandAugment(2, magnitude=config['randaugment']), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])
    else:
        transform_tr = transforms.Compose([transforms.RandomResizedCrop(config['img_size'], antialias=True, interpolation=Image.BICUBIC), torchvision.transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])

    if accelerator.is_main_process:
        wandb.init(project="mlp_mixer", config=config, name=args.custom_name if args.custom_name else None)

    transform_te = transforms.Compose([transforms.Resize(256, antialias=True), transforms.CenterCrop(config['img_size']), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if config['dataset'] == 'ImageNET1k':
            from utils import get_IN_classes

            # modify last element of transform_tr to use ImageNet stats
            transform_tr.transforms[-1] = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transform_te.transforms[-1] = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

            trainset = datasets.ImageFolder('/mydatapathhere/ImageNet-2012/ILSVRC2012/train', transform=transform_tr)
            testset = datasets.ImageFolder('/mydatapathhere/ImageNet-2012/ILSVRC2012/val', transform=transform_te)

            n_classes = 1000
    else:
        print('Dataset not supported')
        raise

    n_seeds = 1

    results = {}
    g_accs = []

    for seed in range(n_seeds):
        print(f'using seed: {seed}')
        torch.manual_seed(seed)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['bs'], shuffle=True, num_workers=8)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config['bs'], shuffle=False, num_workers=8)

        model = MLPMixer(
            image_size=(config['img_size'], config['img_size']), channels=3, patch_size=config['patch_size'], dim=config['dim'],
            expansion_factor=config['expansion_factor'], expansion_factor_token=config['expansion_factor_token'], depth=config['depth'], num_classes=n_classes, act=act,
            dropout=config['dropout'], stochastic_depth=config['stochastic_depth']
        )

        model_name = f"mixer-{config['custom_name']}-{config['type']}-{config['layer']}-{config['dataset']}-{config['moe_activ']}-experts_{config['num_experts']}-rank_{config['rank']}-numepochs_{num_epochs}-numseeds_{n_seeds}-RA_{config['randaugment']}-SD_{config['stochastic_depth']}-mu_{config['mixup']}-warmup_{config['warmup']}"

        log_level    = logging.INFO
        log_format   = '  %(message)s'
        log_handlers = [logging.FileHandler(f'./results/{model_name}.log'), logging.StreamHandler()]
        logging.basicConfig(level = log_level, format = log_format, handlers = log_handlers)
        logger = logging.getLogger("mixer")

        if accelerator.is_main_process:
            logger.info('---------------')
            logger.info('Config:')
            logger.info(config)
            logger.info('---------------')

        num_batches = len(trainloader)

        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        criterion = nn.CrossEntropyLoss()
        scheduler = linear_lr(optimizer, config['lr'], config['warmup'], num_epochs*num_batches)

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
        model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
            model, optimizer, trainloader, testloader, scheduler
        )

        if config['mixup']: mixup = v2.MixUp(alpha=config['mixup'], num_classes=n_classes)

        if accelerator.is_main_process:
            model.eval()
            flops = FlopCountAnalysis(model.cuda(), torch.randn(1, 3, config['img_size'], config['img_size']).cuda())
            layer_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
            layer_flops = flops.total()
            c2 = {'model_params': layer_params, 'model_flops': layer_flops}
            wandb.config.update(c2, allow_val_change=True)
            logger.info(f'# Flops: {format(int(flops.total()), ",")}, # Params: {format(sum([p.numel() for p in model.parameters() if p.requires_grad]), ",")}')
            model.train()

        metric = evaluate.load("accuracy", experiment_id=model_name)

        accs1 = [0]
        for epoch in range(num_epochs):
            running_loss = []
            model.train()
            t0 = time.time()

            tr_enum = enumerate(tqdm(trainloader, file=sys.stdout)) if args.tqdm else enumerate(trainloader)
            for i, data in tr_enum:
                with accelerator.accumulate(model):
                    if type(data) == dict: inputs = data['image'] ; labels = data['label']
                    else: inputs, labels = data
                    if config['mixup']: inputs, labels = mixup(inputs, labels)

                    step = i + epoch * num_batches
                    scheduler(step)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()

                    running_loss += [loss.item()]

            # Test the model
            model.eval()
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    outputs = model(images)
                    predicted = outputs.argmax(dim=-1)
                    predicted, labels = accelerator.gather_for_metrics((predicted, labels))
                    metric.add_batch(predictions = predicted, references = labels)

                accuracy = metric.compute()['accuracy']
                accs1 += [accuracy]

                if accelerator.is_main_process:
                    current_lr = optimizer.param_groups[-1]['lr']
                    logger.info(f'Epoch {epoch} -- Test acc: {accuracy} -- Train loss: {np.mean(running_loss)} -- (max: {np.max(np.array(accs1))} at: {np.argmax(np.array(accs1))-1}), LR: {current_lr}, epoch time: {time.time() - t0}')
                    wandb.log({"val_acc": accuracy, "train_loss": np.mean(running_loss), "learing_rate": current_lr})
                    
                    if epoch % 10 == 0:
                        save_model_weights(model, model_name)

        g_accs += [accs1]
    mean_acc = np.mean(np.array(g_accs), 0)
    std_acc = np.std(np.array(g_accs), 0)

    results['LayerParams'] = layer_params
    results['LayerFLOPS'] = layer_flops
    results['Acc'] = [mean_acc, std_acc]
    results['config'] = config

    if accelerator.is_main_process:
        save_model_weights(model, model_name)
        print('...model saved.')

        import pickle
        with open(f'./results/{model_name}.json', 'wb') as fp:
            pickle.dump(results, fp)
        print('...saved results')

if __name__ == "__main__":
    main()