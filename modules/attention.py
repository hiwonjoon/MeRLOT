import gin
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Attention

class MultiHeadedAttention(Layer):
    def __init__(self,num_heads,dim):
        super().__init__()

        self.num_heads = num_heads
        self.dim = self.d_w = dim

        self.w_q = Dense(num_heads*dim,use_bias=False)
        self.w_k = Dense(num_heads*dim,use_bias=False)
        self.w_v = Dense(num_heads*dim,use_bias=False)
        self.w_o = Dense(num_heads*dim,use_bias=False)

        self.attn = [Attention() for _ in range(num_heads)] # you just need a single instance, but...

    #@tf.function
    def call(self, inputs, training=None):
        q,k,v = inputs

        q = self.w_q(q) / (self.dim)**.5 # scaling to prevent too large dot-product when self.dim set high.
        k = self.w_k(k)
        v = self.w_v(v)

        o = []
        for _attn, _q,_k,_v in zip(
            self.attn,
            tf.split(q,self.num_heads,axis=-1),
            tf.split(k,self.num_heads,axis=-1),
            tf.split(v,self.num_heads,axis=-1)):

            _o = _attn([_q,_v,_k])
            o.append(_o)

        o = self.w_o(tf.concat(o,axis=-1))
        return o

@gin.configurable(module=__name__)
class Encoder(Layer):
    def __init__(self,num_heads,d_model,d_ff=None):
        d_ff = 2*d_model if d_ff is None else d_ff

        super().__init__()

        self.mha = MultiHeadedAttention(num_heads,d_model//num_heads)
        self.ffn = [
            Dense(d_ff,activation='relu'),
            Dense(d_model),
        ]
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    #@tf.function
    def call(self, inputs, training=None):
        x = inputs[0]

        agg = self.mha([x, x, x])
        o1 = self.ln1(x + agg)  # (batch_size, input_seq_len, d_model)

        o2 = o1
        for l in self.ffn:
            o2 = l(o2)

        o = self.ln2(o1 + o2)  # (batch_size, input_seq_len, d_model)
        return o

@gin.configurable(module=__name__)
class Decoder(Layer):
    def __init__(self,num_heads,d_model,d_ff=None):
        d_ff = 2*d_model if d_ff is None else d_ff

        super().__init__()

        self.mha = MultiHeadedAttention(num_heads,d_model//num_heads)
        self.ffn = [
            Dense(d_ff,activation='relu'),
            Dense(d_model),
        ]
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    #@tf.function
    def call(self, inputs, training=None):
        e,q = inputs

        agg = self.mha([q, e, e])
        o1 = self.ln1(q + agg)  # (batch_size, input_seq_len, d_model)

        o2 = o1
        for l in self.ffn:
            o2 = l(o2)

        o = self.ln2(o1 + o2)  # (batch_size, input_seq_len, d_model)

        return o

def dot_product_attention(K,Q,scale=False):
    """
    input:
        K: [Meta Batch size, key_len, key_dims]
        Q: [Meta Batch size, query_len, value_dims]
    output:
        logits: [Meta Batch Size, query_len, key_len]
        att_map: [Meta Batch Size, query_len, key_len]
    """
    scores = logits = tf.matmul(Q, K, transpose_b=True) #[-1,query_len,key_len]
    if scale:
        key_dims = tf.shape(K)[2]
        scores /= tf.cast(key_dims,tf.float32)**0.5

    return logits, tf.nn.softmax(scores,axis=-1)
