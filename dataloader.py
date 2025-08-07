import sys
sys.path.append('./AutoAugment/')

import torch
import torchvision
import torchvision.transforms as transforms
from autoaugment import CIFAR10Policy


def get_dataloaders(args):
    train_transform, test_transform = get_transform(args)

    if args.dataset == "c10":
        train_ds = torchvision.datasets.CIFAR10('./datasets', train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10('./datasets', train=False, transform=test_transform, download=True)
        args.num_classes = 10
    else:
        raise ValueError(f"No such dataset:{args.dataset}")

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_dl, test_dl

def get_transform(args):
    if args.dataset in ["c10"]:
        args.padding=4
        args.size = 32
        if args.dataset=="c10":
            args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    else:
        args.padding=28
        args.size = 224
        args.mean, args.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_transform_list = [transforms.RandomCrop(size=(args.size,args.size), padding=args.padding)]
    if args.dataset!="svhn":
        train_transform_list.append(transforms.RandomCrop(size=(args.size,args.size), padding=args.padding))

    if args.autoaugment:
        if args.dataset == 'c10' or args.dataset=='c100':
            train_transform_list.append(CIFAR10Policy())
        else:
            print(f"No AutoAugment for {args.dataset}")   

    train_transform = transforms.Compose(   #裁剪 → AutoAugment（可選）→ 轉成 tensor → 正規化
        train_transform_list+[
            transforms.ToTensor(),
            transforms.Normalize(
                mean=args.mean,
                std = args.std
            )
        ]
    )
    test_transform = transforms.Compose([   #測試資料只做 tensor 化與正規化，不做增強。
        transforms.ToTensor(),
        transforms.Normalize(
            mean=args.mean,
            std = args.std
        )
    ])

    return train_transform, test_transform