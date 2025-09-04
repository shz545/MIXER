import numpy as np
from collections import Counter
import sys

# da4ml 路徑（請依實際安裝路徑調整）
sys.path.insert(0, "/home/shz545/da4ml/src")
from da4ml.cmvm.util.mat_decompose import kernel_decompose
from da4ml.cmvm.api import solve

def to_binary(val, bits=8):
    return [(val >> i) & 1 for i in range(bits)]

def hcmvm_expand(matrix, bits=8):
    expr_counter = Counter()
    for row in matrix:
        for val in row:
            rep = to_binary(int(val), bits)
            for i, bit in enumerate(rep):
                if bit != 0:
                    expr_counter[f"shift_{i}"] += 1
    adder_count = sum(expr_counter.values())
    # latency: 最大 shift 層（critical path）
    latency = max([int(k.split('_')[1]) for k in expr_counter.keys()]) + 1 if expr_counter else 0
    # ops: 所有 shift-and-add 的總數
    ops = adder_count
    return expr_counter, adder_count, latency, ops

# 測試用隨機矩陣
np.random.seed(0)
matrix = np.random.randint(-128, 128, size=(8, 8))

# HCMVM (binary)
hcmvm_counter, hcmvm_adder, hcmvm_latency, hcmvm_ops = hcmvm_expand(matrix, bits=8)
print("\n=== Hcmvm ===")
print(f"加法器總數（Adder count）：{hcmvm_adder}")
print(f"最大 shift 層（Step/Latency）：{hcmvm_latency}")
print(f"總運算步驟數量（ops）：{hcmvm_ops}")

# DA4ML (kernel_decompose + solve)
print("\n=== DA4ML (kernel_decompose + solve) ===")
M1, M2 = kernel_decompose(matrix, dc=-2)
solution1 = solve(M1, method0='wmc')
solution2 = solve(M2, method0='wmc')
total_cost = solution1.cost + solution2.cost
# latency 合併方式依實際 pipeline 結構而定，這裡直接加總
if isinstance(solution1.latency, tuple) and isinstance(solution2.latency, tuple):
    total_latency = tuple(np.add(solution1.latency, solution2.latency))
else:
    total_latency = (solution1.latency, solution2.latency)
def get_ops_count(solution):
    if hasattr(solution, "solutions"):
        return sum(get_ops_count(s) for s in solution.solutions)
    if hasattr(solution, "ops"):
        return len(solution.ops)
    return 0
total_ops = get_ops_count(solution1) + get_ops_count(solution2)

print(f"總加法器數量（cost）：{total_cost}")
print(f"總延遲（latency）：{total_latency}")
print(f"總運算步驟數量（ops）：{total_ops}")
print(matrix)