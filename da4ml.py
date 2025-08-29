import sys
import os
import pickle
import numpy as np

sys.path.insert(0, "/home/shz545/da4ml/src")
from da4ml.cmvm.util.mat_decompose import kernel_decompose
from da4ml.cmvm.core import cmvm, to_solution
'''
合併da4ml的所有階段
但沒有cse的函式
所以用cmvm()當替代方案
目前發現沒有把加法器使用率降下來
'''
def save_mem(arr, path):
    with open(path, "w") as f:
        for v in arr.flatten():
            v_int = int(v)
            f.write(f"{(v_int & 0xFFFF):04X}\n")

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

# 輸入 shape
shape_str = input("請輸入矩陣 shape，預設為 64,64（直接按 Enter 使用預設）：").strip()
if shape_str:
    shape = tuple(map(int, shape_str.replace('(', '').replace(')', '').split(',')))
else:
    shape = (64, 64)

# 執行 kernel_decompose
print("\n=== 執行 kernel_decompose（第一階段分解） ===")
M1, M2 = kernel_decompose(W, dc=-2)

# 儲存分解後的 M1, M2
mst_dir = "mst_kernel"
os.makedirs(mst_dir, exist_ok=True)
m1_name = os.path.join(mst_dir, f"mst_{name.replace('/', '_')}_M1.mem")
m2_name = os.path.join(mst_dir, f"mst_{name.replace('/', '_')}_M2.mem")
save_mem(M1, m1_name)
save_mem(M2, m2_name)
print(f"已儲存分解後 M1：{m1_name}")
print(f"已儲存分解後 M2：{m2_name}")

# 第二階段：對 M1, M2 各自做 cmvm 分解（method='wmc'）
cse_dir = "cse_kernel"
os.makedirs(cse_dir, exist_ok=True)

solutions = []
for mat, tag in zip([M1, M2], ["M1", "M2"]):
    print(f"\n=== 對 {name}_{tag} 做 cmvm 分解（method='wmc'） ===")
    dense_adders = shape[0] * (shape[1] - 1)
    state = cmvm(mat, method='wmc')
    solution = to_solution(state, adder_size=-1, carry_size=-1)
    solutions.append(solution)
    cost = solution.cost
    ratio = 100 * (1 - cost / dense_adders)
    print(f"原始 dense 加法器數量（理論值）：{dense_adders}")
    print(f"分解後 cost（視為加法器數量）：{cost}")
    print(f"節省比例：{ratio:.2f}%")
    print(f"分解後 latency：{solution.latency}")
    print("shape:", solution.shape)
    print("ops 數量:", len(solution.ops))
    out_name = os.path.join(cse_dir, f"cse_{name.replace('/', '_')}_{tag}_wmc.mem")
    save_mem(solution.kernel, out_name)
    print(f"已儲存分解後 kernel 為 {out_name}")
    if ratio > 0:
        print(f"\n*** {name}_{tag} 用 wmc 分解有節省硬體資源！節省比例：{ratio:.2f}% ***")

# 合併運算圖與資源分析
solution1, solution2 = solutions
total_cost = solution1.cost + solution2.cost
# latency 合併方式依實際 pipeline 結構而定，這裡直接加總
if isinstance(solution1.latency, tuple) and isinstance(solution2.latency, tuple):
    total_latency = tuple(np.add(solution1.latency, solution2.latency))
else:
    total_latency = (solution1.latency, solution2.latency)
total_ops = len(solution1.ops) + len(solution2.ops)

print("\n=== 合併運算圖後的總資源消耗 ===")
print(f"總加法器數量（cost）：{total_cost}")
print(f"總延遲（latency）：{total_latency}")
print(f"總運算步驟數量（ops）：{total_ops}")