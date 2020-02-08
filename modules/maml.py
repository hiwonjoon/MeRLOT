import gin
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
tfd = tfp.distributions

class MetaDense(layers.Dense):
    def build(self,*args,**kwargs):
        assert self.built == False

        super().build(*args,**kwargs)
        self.input_spec = [self.input_spec,
                           layers.InputSpec(self.kernel.dtype,self.kernel.get_shape()),
                           layers.InputSpec(self.bias.dtype,self.bias.get_shape())]

    #@tf.function
    def call(self,inputs):
        assert self.built == True

        if isinstance(inputs,tuple):
            x,w,b = inputs

            tf.assert_equal(tf.shape(w),tf.shape(self.kernel))
            tf.assert_equal(tf.shape(b),tf.shape(self.bias))

            outputs = tf.matmul(x, w)
            if self.use_bias:
                outputs = tf.nn.bias_add(outputs, b)
            if self.activation is not None:
                return self.activation(outputs)  # pylint: disable=not-callable
            return outputs
        else:
            return super().call(inputs)


@gin.configurable(module=__name__)
class DenseNet(layers.Layer):
    def __init__(self,in_dim,out_dim,num_layers,dim,activation='relu'):
        super().__init__()
        self.layers = [
            MetaDense(dim,activation=activation)
            for _ in range(num_layers)
        ] + [MetaDense(out_dim)]

        #build network
        self.layers[0].build(in_dim)
        for l in self.layers[1:]: l.build(dim)

    @property
    def params_list(self):
        return [l.weights for l in self.layers]

    def call(self, inputs):
        assert False, "You don't need to call this function"

    #@tf.function
    def call_with_params(self, x, params_list, raw=False):
        assert len(params_list) == len(self.layers), f'{len(params_list)} != {len(self.layers)}'
        o = x
        for l,params in zip(self.layers,params_list):
            o = l((o,*params))
        return o

@gin.configurable(module=__name__)
class DenseNetProbPred(layers.Layer):
    def __init__(self,in_dim,out_dim,num_layers,dim,activation='relu',anp_style_sigma=False):
        super().__init__()
        self.layers = [
            MetaDense(dim,activation=activation)
            for _ in range(num_layers)
        ] + [MetaDense(out_dim*2)]

        #build network
        self.layers[0].build(in_dim)
        for l in self.layers[1:]: l.build(dim)

        self.anp_style_sigma = anp_style_sigma

    @property
    def params_list(self):
        return [l.weights for l in self.layers]

    def call(self, inputs):
        assert False, "You don't need to call this function"

    def call_with_params(self, x, params_list, raw=False):
        assert len(params_list) == len(self.layers), f'{len(params_list)} != {len(self.layers)}'
        o = x
        for l,params in zip(self.layers,params_list):
            o = l((o,*params))

        if raw:
            return o
        else:
            return self.make_dist(o)

    def make_dist(self, o):
        mu_y,sigma_y = tf.split(o,2,axis=-1) #[B,d_y],[B,d_y]

        if self.anp_style_sigma:
            sigma_y = 0.1 + 0.9 * tf.nn.softplus(sigma_y)
        else:
            sigma_y = tf.nn.softplus(tf.maximum(-15.,sigma_y))

        y_dist = tfd.MultivariateNormalDiag(mu_y,scale_diag=sigma_y)
        y_hat = y_dist.sample()

        return y_hat, y_dist
