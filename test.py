def q8_8(val):
    return int(round(val * 256))

def simulate_dense_q8_8(x, weights, bias):
    out = []
    for i in range(len(weights)):
        acc = sum([x[j] * weights[i][j] for j in range(len(x))])
        acc += bias[i] * 256
        out.append(acc >> 8)
    return out

# 測試資料
x = [q8_8(0.25), q8_8(-0.5), q8_8(0.125), q8_8(0.0)]
weights = [
    [q8_8(1.0), q8_8(0.5), q8_8(-1.0), q8_8(0.0)],
    [q8_8(-0.5), q8_8(0.25), q8_8(0.5), q8_8(1.0)]
]
bias = [q8_8(0.0), q8_8(0.25)]

# 執行
output = simulate_dense_q8_8(x, weights, bias)
print("輸出結果：", output)
