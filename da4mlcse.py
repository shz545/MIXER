import sys
import os
import numpy as np

sys.path.insert(0, "/home/shz545/da4ml/src")
from da4ml.cmvm.core import cmvm, to_solution
'''
進行da4ml的第二階段
但沒有cse的函式
所以用cmvm()當替代方案
目前發現沒有把加法器使用率降下來
'''
def hex_to_int16(s):
    v = int(s, 16)
    return v if v < 0x8000 else v - 0x10000

def load_mem(path, shape=(64, 64)):
    with open(path) as f:
        data = [hex_to_int16(line.strip()) for line in f if line.strip()]
    return np.array(data, dtype=np.int16).reshape(shape)

# 只列出 mst_kernel 資料夾下的 .mem 檔
mst_dir = "mst_kernel"
mem_files = [f for f in os.listdir(mst_dir) if f.endswith('.mem') and os.path.isfile(os.path.join(mst_dir, f))]
if not mem_files:
    print("mst_kernel 資料夾中找不到 .mem 檔案！")
    exit(1)

# 讓使用者選擇檔案類型（例如 MixerBlock_0_token_mixing_Dense_0_kernel）
# 自動找出對應的 M1、M2
# 先找出所有可選的 kernel 名稱（不含 _M1/_M2）
kernel_types = set()
for fname in mem_files:
    if "_M1" in fname:
        kernel_types.add(fname.replace("_M1.mem", ""))
    elif "_M2" in fname:
        kernel_types.add(fname.replace("_M2.mem", ""))

kernel_types = sorted(list(kernel_types))
if not kernel_types:
    print("找不到可用的 kernel 類型！")
    exit(1)

print("可選擇的 kernel 類型：")
for idx, ktype in enumerate(kernel_types):
    print(f"{idx}: {ktype}")

sel_idx = int(input("請輸入要分解的 kernel 類型編號："))
if sel_idx < 0 or sel_idx >= len(kernel_types):
    print("類型編號錯誤！")
    exit(1)
kernel_base = kernel_types[sel_idx]

# 自動選擇對應的 M1, M2 檔案
m1_file = f"{kernel_base}_M1.mem"
m2_file = f"{kernel_base}_M2.mem"
m1_path = os.path.join(mst_dir, m1_file)
m2_path = os.path.join(mst_dir, m2_file)

if not (os.path.exists(m1_path) and os.path.exists(m2_path)):
    print(f"找不到 {m1_file} 或 {m2_file}！")
    exit(1)

# 輸入 shape，預設 64x64
shape_str = input("請輸入矩陣 shape，預設為 64,64（直接按 Enter 使用預設）：").strip()
if shape_str:
    shape = tuple(map(int, shape_str.replace('(', '').replace(')', '').split(',')))
else:
    shape = (64, 64)

methods = ['wmc-dc', 'wmc', 'mc-dc', 'mc']

# 確保 cse_kernel 資料夾存在
out_dir = "cse_kernel"
os.makedirs(out_dir, exist_ok=True)

# 分別對 M1, M2 做多種 cmvm 分解
for mat_path, tag in zip([m1_path, m2_path], ["M1", "M2"]):
    base = f"{kernel_base}_kernel_{tag}"
    mat = load_mem(mat_path, shape)
    print(f"\n=== 對 {base} 做 cmvm 分解（多種方法） ===")
    dense_adders = shape[0] * (shape[1] - 1)
    for method in methods:
        print(f"\n--- method: {method} ---")
        state = cmvm(mat, method=method)
        solution = to_solution(state, adder_size=-1, carry_size=-1)
        print("CMVM 分解完成！")
        print(f"原始 dense 加法器數量（理論值）：{dense_adders}")
        print(f"分解後 cost（視為加法器數量）：{solution.cost}")
        print(f"節省比例：{100 * (1 - solution.cost / dense_adders):.2f}%")
        print(f"分解後 latency：{solution.latency}")
        print("shape:", solution.shape)
        print("ops 數量:", len(solution.ops))
        out_name = os.path.join(out_dir, f"cse_{base}_{method}.mem")
        with open(out_name, "w") as f:
            for v in solution.kernel.flatten():
                v_int = int(v)
                f.write(f"{(v_int & 0xFFFF):04X}\n")
        print(f"已儲存分解後 kernel 為 {out_name}")