import gin
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Attention

from . import MultiInputDense

@gin.configurable(module=__name__)
class MetaFunUpdater(Layer):
    def __init__(self,num_layers,dim,embed_dim,alpha):
        super().__init__()
        self.u = MultiInputDense(num_layers,dim,embed_dim)
        self.alpha = alpha
        self.embed_dim = embed_dim

    #@tf.function
    def call(self, inputs, training=None):
        """
        input:
            x: [Meta Batch Size, C, x_dim]
            y: [Meta Batch Size, C, y_dim]
            r_c: [Meta Batch Size, C, r_embed_dim]
            r_q: [Meta Batch Size, Q, r_embed_dim]
            c_att_map: [Meta Batch Size, C, C]
            q_att_map: [Meta Batch Size, Q, C]
        output:
            f_embed_updated
        """
        x,y,r_c,r_q,c_att_map,q_att_map = inputs

        U = self.u((x,y,r_c)) #[M,C,r_embed_dim]

        delta_r_c = tf.matmul(c_att_map,U) #[M,C,r_embed_dim]
        delta_r_q = tf.matmul(q_att_map,U) #[M,Q,r_embed_dim]

        next_r_c = r_c - self.alpha * delta_r_c
        next_r_q = r_q - self.alpha * delta_r_q

        return next_r_c, next_r_q

@gin.configurable(module=__name__)
class MetaFunUpdaterLocal(Layer):
    def __init__(self,num_layers,dim,embed_dim,alpha):
        super().__init__()
        self.u = MultiInputDense(num_layers,dim,embed_dim)
        self.alpha = alpha
        self.embed_dim = embed_dim

    #@tf.function
    def call(self, inputs, training=None):
        """
        input:
            x: [Meta Batch Size, C, x_dim]
            y: [Meta Batch Size, C, y_dim]
            r_ik_c: [Meta Batch Size, C, C, r_embed_dim]
            r_ik_q: [Meta Batch Size, C, Q, r_embed_dim]
            c_att_map: [Meta Batch Size, C, C]
            q_att_map: [Meta Batch Size, Q, C]
        output:
            f_embed_updated
        """
        x,y,r_c,r_q,c_att_map,q_att_map = inputs

        C = tf.shape(r_c)[1]

        xx = tf.tile(x[:,None,:,:],[1,C,1,1]) #[M,C,C,x_dim]
        yy = tf.tile(y[:,None,:,:],[1,C,1,1]) #[M,C,C,y_dim]

        U = self.u((xx,yy,r_c)) #[M,C,C,r_embed_dim]

        delta_r_c = tf.matmul(c_att_map[:,None,:,:],U) #[M,C,C,r_embed_dim]
        delta_r_q = tf.matmul(q_att_map[:,None,:,:],U) #[M,C,Q,r_embed_dim]

        next_r_c = r_c - self.alpha * delta_r_c
        next_r_q = r_q - self.alpha * delta_r_q

        return next_r_c, next_r_q
