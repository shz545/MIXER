import sys
import os
import numpy as np

sys.path.insert(0, "/home/shz545/da4ml/src")
from da4ml.cmvm.core import cmvm, to_solution

def hex_to_int16(s):
    v = int(s, 16)
    return v if v < 0x8000 else v - 0x10000

def load_mem(path, shape=(64, 64)):
    with open(path) as f:
        data = [hex_to_int16(line.strip()) for line in f if line.strip()]
    return np.array(data, dtype=np.int16).reshape(shape)

# 只列出同一資料夾下的 .mem 檔（不允許路徑跳脫）
cwd = os.getcwd()
mem_files = [f for f in os.listdir(cwd) if f.endswith('.mem') and os.path.isfile(os.path.join(cwd, f))]
if not mem_files:
    print("找不到任何 .mem 檔案！")
    exit(1)

print("可選擇的 .mem 檔案（僅限本資料夾）：")
for idx, fname in enumerate(mem_files):
    print(f"{idx}: {fname}")

# 只允許選單中的檔案
file_idx = int(input("請輸入要分解的檔案編號："))
if file_idx < 0 or file_idx >= len(mem_files):
    print("檔案編號錯誤！")
    exit(1)
mem_path = mem_files[file_idx]
base_name = os.path.splitext(mem_path)[0]

# 預設 shape 為 64x64，可自訂
shape_str = input("請輸入矩陣 shape，預設為 64,64（直接按 Enter 使用預設）：").strip()
if shape_str:
    shape = tuple(map(int, shape_str.replace('(', '').replace(')', '').split(',')))
else:
    shape = (64, 64)

W = load_mem(mem_path, shape=shape)

# 執行 cmvm，method 設為 'wmc-dc'
state = cmvm(W, method='wmc-dc')
solution = to_solution(state, adder_size=-1, carry_size=-1)

# 輸出分解摘要
print("CMVM (wmc-dc) 分解完成！")
print("adder_size:", solution.adder_size)
print("carry_size:", solution.carry_size)
print("cost:", solution.cost)
print("latency:", solution.latency)
print("shape:", solution.shape)
print("ops 數量:", len(solution.ops))

# 儲存分解後 kernel，檔名為 cse{原檔名}.mem
out_name = f"cse_{base_name}.mem"
with open(out_name, "w") as f:
    for v in solution.kernel.flatten():
        v_int = int(v)
        f.write(f"{(v_int & 0xFFFF):04X}\n")
print(f"已儲存分解後 kernel 為 {out_name}")