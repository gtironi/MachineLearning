# FixMatch - Implementação Original
# Este arquivo implementa o FixMatch seguindo exatamente a implementação do repositório FixMatch-pytorch

import math
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

DEVICE = get_device()
print("Usando dispositivo:", DEVICE)
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memória GPU disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("Rodando na CPU")

# Configurações do experimento
SEED = 5
NUM_LABELED = 4000  # Total de dados rotulados (400 por classe)
BATCH_SIZE = 64
MU = 7  # Multiplicador para batch não supervisionado
TOTAL_STEPS = 2**20  # 1,048,576 steps
EVAL_STEP = 1024
THRESHOLD = 0.95  # tau
LAMBDA_U = 1.0
LR = 0.03
WEIGHT_DECAY = 5e-4
NESTEROV = True
USE_EMA = True
EMA_DECAY = 0.999
WARMUP = 0
T = 1  # Temperature para pseudo-labels

# Constantes CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
NUM_CLASSES = 10

set_seed(SEED)

# RandAugment implementation (adaptado do repositório original)
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw

PARAMETER_MAX = 10

def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX

def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)

def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)

def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)

def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)

def Identity(img, **kwarg):
    return img

def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)

def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)

def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)

def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

def augment_list():
    l = [
        (AutoContrast, None, None),
        (Brightness, 0.9, 0.05),
        (Color, 0.9, 0.05),
        (Contrast, 0.9, 0.05),
        (Equalize, None, None),
        (Identity, None, None),
        (Posterize, 4, 4),
        (Rotate, 30, 0),
        (Sharpness, 0.9, 0.05),
        (ShearX, 0.3, 0),
        (ShearY, 0.3, 0),
        (Solarize, 256, 0),
        (TranslateX, 0.3, 0),
        (TranslateY, 0.3, 0)
    ]
    return l

class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img

# Transformações (exatamente como no repositório original)
class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

# Datasets SSL
class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def x_u_split(num_labeled, num_classes, labels):
    """Split dataset into labeled and unlabeled"""
    label_per_class = num_labeled // num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = np.array(range(len(labels)))

    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)

    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == num_labeled

    # Expand labels se necessário
    if num_labeled < BATCH_SIZE:
        num_expand_x = math.ceil(BATCH_SIZE * EVAL_STEP / num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])

    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

# Wide ResNet (implementação simplificada para CIFAR-10)
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, drop_rate=0.0, num_classes=10):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def build_wideresnet(depth=28, widen_factor=2, dropout=0, num_classes=10):
    return WideResNet(depth=depth, widen_factor=widen_factor, drop_rate=dropout, num_classes=num_classes)

# EMA Model
class ModelEMA(object):
    def __init__(self, model, decay):
        self.ema = self.deepcopy_model(model)
        self.decay = decay

    def deepcopy_model(self, model):
        ema_model = build_wideresnet(depth=28, widen_factor=2, dropout=0, num_classes=NUM_CLASSES)
        ema_model.load_state_dict(model.state_dict())
        for param in ema_model.parameters():
            param.detach_()
        return ema_model

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

# Scheduler com warmup
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=7./16., last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

# Interleave functions para MixMatch
def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

# Accuracy function
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Average Meter
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Preparar datasets
def get_cifar10(root='./data'):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        NUM_LABELED, NUM_CLASSES, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True, transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=CIFAR10_MEAN, std=CIFAR10_STD))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

if __name__ == "__main__":
    # Criar datasets
    labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10()

    print(f"Dados rotulados: {len(labeled_dataset)}")
    print(f"Dados não rotulados: {len(unlabeled_dataset)}")
    print(f"Dados de teste: {len(test_dataset)}")

    # DataLoaders
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=BATCH_SIZE,
        num_workers=4,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=RandomSampler(unlabeled_dataset),
        batch_size=BATCH_SIZE * MU,
        num_workers=4,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=BATCH_SIZE,
        num_workers=4)

    # Criar modelo e otimizador
    model = build_wideresnet(depth=28, widen_factor=2, dropout=0, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    print(f"Total de parâmetros: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Otimizador com weight decay diferenciado
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = optim.SGD(grouped_parameters, lr=LR, momentum=0.9, nesterov=NESTEROV)

    # Scheduler
    epochs = math.ceil(TOTAL_STEPS / EVAL_STEP)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP, TOTAL_STEPS)

    # EMA
    if USE_EMA:
        ema_model = ModelEMA(model, EMA_DECAY)

    print(f"Épocas: {epochs}")
    print(f"Steps por época: {EVAL_STEP}")
    print(f"Total steps: {TOTAL_STEPS}")

    # Função de teste
    def test(test_loader, model, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        test_loader_tqdm = tqdm(test_loader, desc=f"Test Epoch {epoch+1}")

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader_tqdm):
                model.eval()

                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)

                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.shape[0])
                top1.update(prec1.item(), inputs.shape[0])
                top5.update(prec5.item(), inputs.shape[0])
                batch_time.update(time.time() - end)
                end = time.time()

                test_loader_tqdm.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Top1': f'{top1.avg:.2f}%',
                    'Top5': f'{top5.avg:.2f}%'
                })

        print(f"Test - Loss: {losses.avg:.4f}, Top-1: {top1.avg:.2f}%, Top-5: {top5.avg:.2f}%")
        return losses.avg, top1.avg

    # Função de treinamento principal
    def train(labeled_trainloader, unlabeled_trainloader, test_loader, model, optimizer, ema_model, scheduler):
        best_acc = 0
        test_accs = []
        end = time.time()

        labeled_iter = iter(labeled_trainloader)
        unlabeled_iter = iter(unlabeled_trainloader)

        model.train()
        for epoch in range(epochs):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            losses_x = AverageMeter()
            losses_u = AverageMeter()
            mask_probs = AverageMeter()

            p_bar = tqdm(range(EVAL_STEP), desc=f"Train Epoch {epoch+1}/{epochs}")

            for batch_idx in range(EVAL_STEP):
                # Pegar dados rotulados
                try:
                    inputs_x, targets_x = next(labeled_iter)
                except:
                    labeled_iter = iter(labeled_trainloader)
                    inputs_x, targets_x = next(labeled_iter)

                # Pegar dados não rotulados
                try:
                    (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
                except:
                    unlabeled_iter = iter(unlabeled_trainloader)
                    (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

                data_time.update(time.time() - end)
                batch_size = inputs_x.shape[0]

                # Interleave para processamento eficiente
                inputs = interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*MU+1).to(DEVICE)
                targets_x = targets_x.to(DEVICE)

                # Forward pass
                logits = model(inputs)
                logits = de_interleave(logits, 2*MU+1)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits

                # Loss supervisionada
                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                # Pseudo-labels
                pseudo_label = torch.softmax(logits_u_w.detach()/T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(THRESHOLD).float()

                # Loss não supervisionada
                Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

                # Loss total
                loss = Lx + LAMBDA_U * Lu

                # Backward
                loss.backward()

                # Update
                losses.update(loss.item())
                losses_x.update(Lx.item())
                losses_u.update(Lu.item())
                optimizer.step()
                scheduler.step()

                if USE_EMA:
                    ema_model.update(model)

                model.zero_grad()

                batch_time.update(time.time() - end)
                end = time.time()
                mask_probs.update(mask.mean().item())

                p_bar.set_postfix({
                    'LR': f'{scheduler.get_last_lr()[0]:.4f}',
                    'Loss': f'{losses.avg:.4f}',
                    'Loss_x': f'{losses_x.avg:.4f}',
                    'Loss_u': f'{losses_u.avg:.4f}',
                    'Mask': f'{mask_probs.avg:.2f}'
                })
                p_bar.update()

            p_bar.close()

            # Teste
            if USE_EMA:
                test_model = ema_model.ema
            else:
                test_model = model

            test_loss, test_acc = test(test_loader, test_model, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            test_accs.append(test_acc)
            print(f'Melhor Top-1 acc: {best_acc:.2f}%')
            print(f'Média Top-1 acc (últimas 20): {np.mean(test_accs[-20:]):.2f}%\n')

        return test_accs

    # Executar treinamento
    print("***** Iniciando Treinamento *****")
    print(f"Dataset: CIFAR-10 com {NUM_LABELED} rótulos")
    print(f"Épocas: {epochs}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total steps: {TOTAL_STEPS}")
    print(f"Threshold (tau): {THRESHOLD}")
    print(f"Lambda_u: {LAMBDA_U}")
    print(f"EMA: {USE_EMA}")
    print()

    model.zero_grad()
    test_accs = train(labeled_trainloader, unlabeled_trainloader, test_loader,
                      model, optimizer, ema_model if USE_EMA else None, scheduler)

    # Plotar resultados
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(test_accs)
    plt.title('Acurácia de Teste por Época')
    plt.xlabel('Época')
    plt.ylabel('Acurácia (%)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    # Média móvel das últimas 20 épocas
    moving_avg = []
    for i in range(len(test_accs)):
        start_idx = max(0, i-19)
        moving_avg.append(np.mean(test_accs[start_idx:i+1]))

    plt.plot(moving_avg)
    plt.title('Média Móvel da Acurácia (20 épocas)')
    plt.xlabel('Época')
    plt.ylabel('Acurácia (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"\nResultados Finais:")
    print(f"Melhor acurácia: {max(test_accs):.2f}%")
    print(f"Acurácia final: {test_accs[-1]:.2f}%")
    print(f"Média das últimas 20 épocas: {np.mean(test_accs[-20:]):.2f}%")
