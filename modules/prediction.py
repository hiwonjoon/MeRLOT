import gin
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Attention
import tensorflow_probability as tfp
tfd = tfp.distributions

from . import MultiInputDense

@gin.configurable(module=f'{__name__}')
class Prediction(Layer):
    def __init__(self,num_layers,dim,y_dim):
        super().__init__()
        self.pred = MultiInputDense(num_layers,dim,y_dim)

    def call(self, inputs, training=None):
        """
        input:
            q: [Meta Batch Size, q_len, x_dim]
            f_embed: [Meta Batch Size, context_len, embed_dim]
            att_logits: [Meta Batch Size, q_len, context_len]
        output:
            y_hat: prediction made with x
                [Meta Batch Size, q_len, y_dim]
        """
        q,f_embed,att_logits = inputs

        context_len = tf.shape(att_logits)[2]

        qq = tf.tile(q[:,:,None,:],[1,1,context_len,1])

        if len(f_embed.get_shape()) == 3:
            q_len = tf.shape(att_logits)[1]
            ff_embed = tf.tile(f_embed[:,None,:,:],[1,q_len,1,1])
        else:
            ff_embed = f_embed

        yy_hat = self.pred((qq,ff_embed))

        att_map = tf.nn.softmax(att_logits,axis=-1)
        y_hat = tf.reduce_sum(att_map[:,:,:,None] * yy_hat,axis=2)
        dist = None

        return y_hat, dist

@gin.configurable(module=f'{__name__}')
class ProbabilisticPrediction(Layer):
    def __init__(self,num_layers,dim,y_dim,anp_style_sigma=False):
        super().__init__()
        self.pred = MultiInputDense(num_layers,dim,y_dim*2)
        self.anp_style_sigma = anp_style_sigma

    def call(self, inputs, training=None):
        """
        input:
            q: [Meta Batch Size, q_len, x_dim]
            f_embed: [Meta Batch Size, context_len, embed_dim]
            att_logits: [Meta Batch Size, q_len, context_len]
        output:
            y_hat: prediction made with x
                [Meta Batch Size, q_len, y_dim]
        """
        q,f_embed,att_logits = inputs

        context_len = tf.shape(att_logits)[2]

        qq = tf.tile(q[:,:,None,:],[1,1,context_len,1])

        if len(f_embed.get_shape()) == 3:
            q_len = tf.shape(att_logits)[1]
            ff_embed = tf.tile(f_embed[:,None,:,:],[1,q_len,1,1])
        else:
            ff_embed = f_embed

        mu,sigma = tf.split(self.pred((qq,ff_embed)),2,axis=-1) #[M,Q,C,y_dim]
        if self.anp_style_sigma:
            sigma = 0.1 + 0.9 * tf.nn.softplus(sigma)
        else:
            sigma = tf.nn.softplus(tf.maximum(-15.,sigma))

        component_dist = tfd.MultivariateNormalDiag(mu,scale_diag=sigma,validate_args=False,allow_nan_stats=False) # Batch dim [M,Q,C]

        dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=att_logits),
            components_distribution=component_dist)
        y_hat = dist.sample()

        return y_hat, dist
