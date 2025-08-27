import numpy as np

def csd_nonzero_digits(n: int) -> int:
    """
    回傳整數 n 的 Canonical Signed Digit（CSD）表示中，非零位元的數量。
    這個數字可用來衡量硬體實現時的加法/減法資源消耗。
    參考演算法：對 n 取尾數 mod4 來決定 ±1 消元，直至 n 變 0。
    """
    count = 0
    x = n
    while x != 0:
        if x & 1 == 0:
            x //= 2
        else:
            # u = 2 - (x mod 4) 可取 +1 或 -1
            m4 = x & 3
            u = 1 if m4 == 1 else -1  # m4 == 3 時走 -1
            x -= u
            count += 1
            x //= 2
    return count

def csd_vector(vec: np.ndarray) -> int:
    """
    對向量中每個元素應用 csd_nonzero_digits，並加總。
    這個總和就是該向量在 CSD 表示下的總非零位元數，
    可作為圖論分解時的邊權重（越小越省硬體資源）。
    """
    return int(sum(csd_nonzero_digits(int(v)) for v in vec))

def graph_decompose(M: np.ndarray):
    """
    Stage1: 圖基分解，M.shape = (din, dout)
    以 Prim 演算法建立最小生成樹（MST），
    每條邊的權重就是差分向量的 CSD 非零位元數（越小越省硬體資源）。
    回傳 M1 (din x k) 與 M2 (k x dout)，使得 M == M1 @ M2。
    M1 只包含 parent 是 root 的基底，其餘 column 由 M1 線性組合重建。
    """
    din, dout = M.shape
    # 節點 0: zero vector；1..dout: M 的各列
    nodes = [np.zeros(din, dtype=int)] + [M[:, i].copy() for i in range(dout)]
    N = len(nodes)

    in_mst = [False]*N
    in_mst[0] = True  # root 節點已在 MST

    # Prim 初始距離：從 root(0) 到其他節點的 csd weight
    dist = [float('inf')]*N
    edge_info = [None]*N  # (parent_idx, diff_vector, sign)
    for i in range(1, N):
        # 初始邊權重就是該向量的 CSD 非零位元數
        dist[i] = csd_vector(nodes[i])
        # diff = nodes[i] - nodes[0] = nodes[i]
        edge_info[i] = (0, nodes[i].copy(), 1)

    edges = []  # list of (p, c, diff, sign)
    for _ in range(N-1):
        # 找目前還沒進入 MST 的節點
        cand = [i for i in range(N) if not in_mst[i]]
        if not cand: break
        # 先嘗試用所有已在 MST 的 parent 更新所有候選 child 的 dist
        print("[DEBUG] 各 parent 到每個候選 child 的 CSD 非零位元數:")
        for child in cand:
            for parent in range(N):
                if in_mst[parent]:
                    plus_vec = nodes[parent] + nodes[child]
                    minus_vec = nodes[child] - nodes[parent]
                    w_plus = csd_vector(plus_vec)
                    w_minus = csd_vector(minus_vec)
                    print(f"  parent={parent}, child={child}, +: {w_plus}, -: {w_minus}")
                    # 嘗試用 parent 更新 child 的 dist
                    if w_plus <= w_minus:
                        w, dv, ss = w_plus, plus_vec, -1
                    else:
                        w, dv, ss = w_minus, minus_vec, 1
                    if w < dist[child]:
                        print(f"[DEBUG] 更新dist: parent={parent}, child={child}, sign={ss}, 新CSD非零位元數={w}, 向量={dv}")
                        dist[child] = w
                        edge_info[child] = (parent, dv.copy(), ss)
        # 選 dist 最小的 child
        j = min(cand, key=lambda i: dist[i])
        p, diff, sgn = edge_info[j]
        print(f"[DEBUG] 選邊: parent={p}, child={j}, sign={sgn}, CSD非零位元數={csd_vector(diff)}, 向量={diff}")
        edges.append((p, j, diff, sgn))
        in_mst[j] = True

    # 所有 MST 的邊都作為基底
    k = len(edges)
    M1 = np.column_stack([e[2] for e in edges])  # shape (din, k)
    M2 = np.zeros((k, dout), dtype=int)          # shape (k, dout)
    # 建立 child->edge index 對應表
    edge_idx_map = {c: idx for idx, (p, c, _, _) in enumerate(edges)}
    # 對每個 column，回溯 MST 路徑，將經過的邊係數累加到 M2
    for col in range(1, N):
        cur = col
        coeffs = np.zeros(k, dtype=int)
        debug_path = []
        while cur != 0:
            for idx, (p, c, _, s) in enumerate(edges):
                if c == cur:
                    coeffs[idx] += s
                    debug_path.append((p, c, s))
                    cur = p
                    break
        print(f"[DEBUG] col={col-1} 回溯路徑: {debug_path}，係數={coeffs}")
        M2[:, col-1] = coeffs
    return M1, M2

if __name__ == "__main__":
    # 範例：3×3 常數矩陣
    M = np.array([
        [0, 1, 3],
        [1, 4, 4],
        [3, 6, 5]
    ])

    M1, M2 = graph_decompose(M)
    print("M1 (din×k):")
    print(M1)
    print("\nM2 (k×dout):")
    print(M2)
    print("\nReconstruct M1 @ M2:")
    print(M1 @ M2)
    print("\n原矩陣 M:")
    print(M)