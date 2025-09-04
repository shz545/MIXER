import numpy as np
from da4ml.trace import FixedVariableArrayInput, comb_trace
from da4ml.trace.ops import relu, quantize
from da4ml.codegen import VerilogModel

# 假設你要一個 32 個元素的 ReLU 層，每個元素 16 bits
input_shape = (32,)

def operation(inp):
    inp_q = quantize(inp, 1, 7, 0)  # 先量化
    return relu(inp_q)

# 建立符號輸入
inp = FixedVariableArrayInput(input_shape)  # 不加 bit_width
out = operation(inp)

# 產生組合邏輯運算圖
comb_logic = comb_trace(inp, out)

# 產生 Verilog
verilog_model = VerilogModel(comb_logic, 'relu32', './verilogrelu_output')
verilog_model.write()
print("✅ 已產生 verilogrelu_output/relu32.v")