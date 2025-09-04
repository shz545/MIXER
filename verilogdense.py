import os
import pickle
import numpy as np
from da4ml.cmvm.api import solve
from da4ml.trace import FixedVariableArrayInput, comb_trace
from da4ml.trace.ops import quantize
from da4ml.codegen import VerilogModel

# 設定輸出資料夾
output_dir = "verilogdense0_output"
os.makedirs(output_dir, exist_ok=True)

with open("mlp_mixer_params.pkl", "rb") as f:
    params = pickle.load(f)["params"]

W = np.array(params["MixerBlock_0"]["token_mixing"]["Dense_0"]["kernel"])

solution = solve(W)


input_shape = (64,)

def operation(inp):
    inp_q = quantize(inp, 1, 7, 0)
    # 加入 dense 運算
    W = np.array(params["MixerBlock_0"]["token_mixing"]["Dense_0"]["kernel"])
    return inp_q @ W  # 或 np.dot(inp_q, W)

inp = FixedVariableArrayInput(input_shape)
out = operation(inp)
comb_logic = comb_trace(inp, out)

# latency_cutoff=4 代表 pipeline 最多 4 級，會自動做資源共用
verilog_model = VerilogModel(comb_logic, 'dense0', output_dir, latency_cutoff=8)
verilog_model.write()
print(f"✅ 已產生 pipeline/resource sharing 的 dense0 層 Verilog 檔案至 {output_dir} 資料夾")