import jax.numpy as jnp

def evaluate(model, params, batch_stats, test_data):
    accs = []
    for imgs, labels in test_data:
        # 組合 Flax 需要的 variables 結構
        variables = {
            'params': params,
            'batch_stats': batch_stats
        }

        # Forward pass（不更新 batch_stats）
        logits = model.apply(variables, imgs, train=False, mutable=False)

        # 預測與計算準確率
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        accs.append(acc)

    # 回傳整體平均準確率
    return float(jnp.mean(jnp.array(accs)))
