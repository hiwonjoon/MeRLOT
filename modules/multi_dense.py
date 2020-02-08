import gin
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

@gin.configurable(module=__name__)
class MultiInputDense(Layer):
    def __init__(self,num_layers,dim,out_dim,activation='relu'):
        super().__init__()
        self.layers = [
            Dense(dim,activation=activation)
            for _ in range(num_layers)
        ] + [Dense(out_dim)]

    def call(self, inputs, training=None):
        o = tf.concat(inputs,axis=-1)
        for l in self.layers:
            o = l(o,training=training)
        return o
