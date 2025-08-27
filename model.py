import einops
import flax.linen as nn
import jax.numpy as jnp

class MlpBlock(nn.Module):
    mlp_dim: int
    dropout_rate: float
    use_bn: bool = False

    @nn.compact
    def __call__(self, x, train: bool):
        y = nn.Dense(self.mlp_dim)(x)
        if self.use_bn:
            y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        y = nn.Dense(x.shape[-1])(y)
        if self.use_bn:
            y = nn.BatchNorm(use_running_average=not train)(y)
        return y

class MixerBlock(nn.Module):
    tokens_mlp_dim: int
    channels_mlp_dim: int
    dropout_rate: float
    use_bn: bool = False

    @nn.compact
    def __call__(self, x, train: bool):
        y = x
        if not self.use_bn:
            y = nn.LayerNorm()(y)
        y = jnp.swapaxes(y, 1, 2)
        y = MlpBlock(self.tokens_mlp_dim, self.dropout_rate, self.use_bn, name='token_mixing')(y, train=train)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y

        y = x
        if not self.use_bn:
            y = nn.LayerNorm()(y)
        y = MlpBlock(self.channels_mlp_dim, self.dropout_rate, self.use_bn, name='channel_mixing')(y, train=train)
        return x + y

class MlpMixer(nn.Module):
    num_classes: int
    num_blocks: int
    patch_size: int
    hidden_dim: int
    tokens_mlp_dim: int
    channels_mlp_dim: int
    dropout_rate: float
    use_bn: bool = False

    @nn.compact
    def __call__(self, x, train: bool):
        s = self.patch_size
        x = nn.Conv(self.hidden_dim, (s, s), strides=(s, s), name='stem')(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not train)(x)
        x = einops.rearrange(x, 'n h w c -> n (h w) c')

        for _ in range(self.num_blocks):
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim, self.dropout_rate, self.use_bn)(x, train=train)

        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not train, name='pre_head_bn')(x)
        else:
            x = nn.LayerNorm(name='pre_head_layer_norm')(x)

        x = jnp.mean(x, axis=1)
        return nn.Dense(self.num_classes, name='head', kernel_init=nn.initializers.zeros)(x)