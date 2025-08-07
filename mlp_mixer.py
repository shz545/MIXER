import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange  # 用於張量重排
import torchsummary  # 顯示模型摘要

# 主模型：MLP-Mixer
class MLPMixer(nn.Module):
    def __init__(self, in_channels=3, img_size=32, patch_size=4, hidden_size=512, hidden_s=256, hidden_c=2048, num_layers=8, num_classes=10, drop_p=0., off_act=False, is_cls_token=False):
        super(MLPMixer, self).__init__()

        # 計算 patch 數量
        num_patches = (img_size // patch_size) ** 2
        self.is_cls_token = is_cls_token

        # 將影像切成 patch 並嵌入為向量
        self.patch_emb = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size),  # 每個 patch 轉成 hidden_size 維度
            Rearrange('b d h w -> b (h w) d')  # 重排為 (batch, num_patches, hidden_size)
        )

        # 是否使用 cls token（類似 ViT）
        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))  # 可訓練的分類 token
            num_patches += 1  # 增加一個 token

        # 疊加多層 MixerLayer
        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer(num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act) 
                for _ in range(num_layers)
            ]
        )

        self.ln = nn.LayerNorm(hidden_size)  # 最後 LayerNorm
        self.clf = nn.Linear(hidden_size, num_classes)  # 分類器

    def forward(self, x):
        out = self.patch_emb(x)  # Patch 嵌入
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)  # 加上 cls token
        out = self.mixer_layers(out)  # 通過所有 Mixer 層
        out = self.ln(out)  # LayerNorm
        out = out[:, 0] if self.is_cls_token else out.mean(dim=1)  # 取 cls token 或平均
        out = self.clf(out)  # 分類
        return out

# 單層 Mixer：包含 token-mixing 和 channel-mixing
class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act):
        super(MixerLayer, self).__init__()
        self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, off_act)  # token-mixing
        self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, off_act)  # channel-mixing

    def forward(self, x):
        out = self.mlp1(x)
        out = self.mlp2(out)
        return out

# Token-mixing MLP：在 patch 維度上混合資訊
class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, off_act):
        super(MLP1, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1)  # Conv1d 模擬 Linear
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x: x  # GELU 或關閉激活

    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out + x  # 殘差連接

# Channel-mixing MLP：在通道維度上混合資訊
class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, off_act):
        super(MLP2, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_c)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x: x

    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out + x  # 殘差連接

# 測試模型結構
if __name__ == '__main__':
    net = MLPMixer(
        in_channels=3,  # 輸入通道數(RGB)
        img_size=32,    # 圖像大小
        patch_size=4,   # patch 大小
        hidden_size=128,    # 隱藏層大小
        hidden_s=512,   # token-mixing 隱藏層大小
        hidden_c=64,    # 通道混合隱藏層大小
        num_layers=8,  # 層數
        num_classes=10,     # 分類數量
        drop_p=0.,  # dropout 機率
        off_act=False, # 不關閉激活函數
        is_cls_token=True # 使用 cls token
    )
    torchsummary.summary(net, (3, 32, 32))  # 顯示模型摘要
