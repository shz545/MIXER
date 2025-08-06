import argparse  # 處理命令列參數
import torch     # PyTorch 主框架
import wandb     # 用來追蹤與視覺化訓練過程
wandb.login()    # 登入 wandb 帳號，初始化追蹤功能


from dataloader import get_dataloaders
from utils import get_model
from train import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['c10'])
parser.add_argument('--model', required=True, choices=['mlp_mixer'])
parser.add_argument('--batch-size', type=int, default=128)           # 訓練批次大小
parser.add_argument('--eval-batch-size', type=int, default=1024)     # 評估批次大小
parser.add_argument('--num-workers', type=int, default=4)            # dataloader 的子進程數量
parser.add_argument('--seed', type=int, default=3407)                # 隨機種子（確保可重現）
parser.add_argument('--epochs', type=int, default=300)               # 總訓練 epoch 數
# parser.add_argument('--precision', type=int, default=16)

parser.add_argument('--patch-size', type=int, default=4) # patch 大小
parser.add_argument('--hidden-size', type=int, default=128) # 隱藏層大小
parser.add_argument('--hidden-c', type=int, default=512) #channel-mixing 隱藏層大小
parser.add_argument('--hidden-s', type=int, default=256) #token-mixing 隱藏層大小
parser.add_argument('--num-layers', type=int, default=6) # Mixer 層數
parser.add_argument('--drop-p', type=int, default=0.) # dropout 機率
parser.add_argument('--off-act', action='store_true', help='Disable activation function') # 是否關閉激活函數
parser.add_argument('--is-cls-token', action='store_true', help='Introduce a class token.') # 是否使用 cls token

parser.add_argument('--lr', type=float, default=3e-3)                # 初始學習率
parser.add_argument('--min-lr', type=float, default=1e-5)            # 最小學習率（給 scheduler 用）
parser.add_argument('--momentum', type=float, default=0.9)           # Momentum（給 SGD 用）
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])  # 使用的 optimizer
parser.add_argument('--scheduler', default='cosine', choices=['step', 'cosine'])  # 學習率排程器
parser.add_argument('--beta1', type=float, default=0.9)              # Adam β₁ 參數
parser.add_argument('--beta2', type=float, default=0.99)             # Adam β₂ 參數
parser.add_argument('--weight-decay', type=float, default=5e-5)      # 權重衰減（L2 regularization）
parser.add_argument('--off-nesterov', action='store_true')          # 是否關閉 Nesterov（SGD 專用）
parser.add_argument('--label-smoothing', type=float, default=0.1)   # label smoothing 系數
parser.add_argument('--gamma', type=float, default=0.1)             # learning rate step decay 的 gamma
parser.add_argument('--warmup-epoch', type=int, default=5)          # warmup epoch 數
parser.add_argument('--autoaugment', action='store_true')           # 是否使用 AutoAugment
parser.add_argument('--clip-grad', type=float, default=0, help="0 means disabling clip-grad") # 梯度裁剪
parser.add_argument('--cutmix-beta', type=float, default=1.0)       # CutMix 的 beta 值
parser.add_argument('--cutmix-prob', type=float, default=0.)        # 使用 CutMix 的機率

args = parser.parse_args()
args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #使用gpu,cpu
args.nesterov = not args.off_nesterov
torch.random.manual_seed(args.seed)

#取名
experiment_name = f"{args.model}_{args.dataset}_{args.optimizer}_{args.scheduler}"
if args.autoaugment:
    experiment_name += "_aa"
if args.clip_grad:
    experiment_name += f"_cg{args.clip_grad}"
if args.off_act:
    experiment_name += f"_noact"
if args.cutmix_prob>0.:
    experiment_name += f'_cm'
if args.is_cls_token:
    experiment_name += f"_cls"

if __name__=='__main__':
    with wandb.init(project='mlp_mixer', config=args, name=experiment_name):
        train_dl, test_dl = get_dataloaders(args)
        model = get_model(args)
        trainer = Trainer(model, args)
        trainer.fit(train_dl, test_dl)