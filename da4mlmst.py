import sys
import pickle
import numpy as np

sys.path.insert(0, "/home/shz545/da4ml/src")
from da4ml.cmvm.util.mat_decompose import kernel_decompose

# 讀取 Flax 權重
with open("mlp_mixer_params.pkl", "rb") as f:
    params = pickle.load(f)

# 遍歷所有 kernel 權重
kernel_layers = []

def search_kernels(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == "kernel":
                kernel_layers.append((prefix, v))
            else:
                search_kernels(v, f"{prefix}/{k}" if prefix else k)

search_kernels(params)

# 顯示所有可選 kernel
print("可選擇的 kernel 權重：")
for idx, (name, _) in enumerate(kernel_layers):
    print(f"{idx}: {name.removeprefix('params/')}")

# 讓使用者選擇
idx = int(input("請輸入要分解的 kernel 編號："))
name, kernel = kernel_layers[idx]
print(f"選擇：{name}")

# 量化（如需 Q8.8，可調整）
W = np.round(np.array(kernel) * 256).astype(np.int16)

# 執行 kernel_decompose
result = kernel_decompose(W, dc=-2)

# result 可能是 tuple (M1, M2)
if isinstance(result, tuple) and len(result) == 2:
    M1, M2 = result
    # 儲存分解後的 M1, M2 為 .mem
    for arr, tag in zip([M1, M2], ["M1", "M2"]):
        mem_name = f"mst_{name.replace('/', '_')}_kernel_{tag}.mem"
        with open(mem_name, "w") as f:
            for v in arr.flatten():
                v_int = int(v)
                f.write(f"{(v_int & 0xFFFF):04X}\n")
        print(f"已儲存分解後 {tag} 為 {mem_name}")

    # 驗證分解正確性
    recon = M1 @ M2
    print("\n=== 分解驗證 ===")
    print("\nW (原量化):")
    print(W)
    print("\nM1 @ M2 (重建):")
    print(recon)
    print("\nM1:")
    print(M1)
    print("M2:")
    print(M2)
    print("完全相等？", np.array_equal(W, recon))
    print("最大誤差：", np.max(np.abs(W - recon)))
else:
    print("kernel_decompose 回傳格式非 (M1, M2)，請檢查 API 文件。")