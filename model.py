import einops
import flax.linen as nn
import jax.numpy as jnp

class MlpBlock ( nn . Module ) :
    mlp_dim : int
    @nn . compact
    def __call__ ( self , x ) :
        y = nn . Dense ( self . mlp_dim ) (x ) #第一層：隱藏層
        y = nn . gelu ( y ) #激發函數：gelu
        return nn . Dense ( x . shape [ -1]) ( y ) #第二層：輸出層

class MixerBlock ( nn . Module ) :
    tokens_mlp_dim : int
    channels_mlp_dim : int
    @nn . compact
    def __call__ ( self , x ) :
        y = nn . LayerNorm () ( x ) #對輸入進行layernorm
        y = jnp . swapaxes (y , 1 , 2) #進行軸交換
        y = MlpBlock ( self . tokens_mlp_dim , name = 'token_mixing' ) ( y ) #進行token_mixing
        y = jnp . swapaxes (y , 1 , 2)
        x = x +y #殘差連接 
        y = nn . LayerNorm () ( x ) #再次進行layernorm
        return x + MlpBlock ( self . channels_mlp_dim , name =' channel_mixing ') ( y ) #進行channel_mixing並加回輸入值 再進行輸出

class MlpMixer ( nn . Module ) :
    num_classes : int
    num_blocks : int
    patch_size : int
    hidden_dim : int
    tokens_mlp_dim : int
    channels_mlp_dim : int
    @nn . compact
    def __call__ ( self , x ) :
        s = self . patch_size 
        x = nn . Conv ( self . hidden_dim , (s , s ) , strides =( s , s ) , name ='stem' ) ( x ) #用卷積切割圖像
        x = einops . rearrange (x , 'n h w c -> n (h w) c') #把patch展平成序列
        for _ in range ( self . num_blocks ) : #由num_blocks決定要幾層
            x = MixerBlock ( self . tokens_mlp_dim , self . channels_mlp_dim ) ( x ) 
        x = nn . LayerNorm ( name ='pre_head_layer_norm ') ( x ) #對patch序列進行layernorm
        x = jnp . mean (x , axis =1) #對所有patch特徵做平均
        return nn . Dense ( self . num_classes , name ='head ', #連接分類層
                            kernel_init = nn . initializers . zeros ) (x) #初始化權重為零