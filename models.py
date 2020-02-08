from functools import partial
import gin
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import tensorflow_probability as tfp
tfd = tfp.distributions

import modules as M

@gin.configurable(module=__name__)
class MAML(Model):
    def __init__(
        self,
        x_dim,
        y_dim,
        net_fn,
        num_inner_updates,
        alpha,
        fo_approx,
        meta_batch_size,
        grad_clip = 10,
        clip_by_global_norm=False,
        finetune_grad_clip = 0,
    ):
        super().__init__()

        self.net = net_fn(in_dim=x_dim,out_dim=y_dim)

        self.alpha = alpha
        self.num_inner_updates = num_inner_updates
        self.fo_approx = fo_approx

        self.clip_by_global_norm = clip_by_global_norm
        self.finetune_grad_clip = finetune_grad_clip

        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        @tf.function(input_signature=(
            tf.TensorSpec(shape=[meta_batch_size,None,x_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[meta_batch_size,None,y_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[meta_batch_size,None,x_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[meta_batch_size,None,y_dim],dtype=tf.float32)))
        def update(x,y,q,a):
            with tf.GradientTape() as tape:
                loss = tf.map_fn(self._update,(x,y,q,a),dtype=tf.float32, parallel_iterations=meta_batch_size)
                #loss = tf.reduce_mean(loss,axis=0)
                loss = tf.reduce_mean(tf.boolean_mask(loss, tf.math.is_finite(loss)),axis=0)

            gradients = tape.gradient(loss, self.trainable_variables)

            if grad_clip > 0:
                if clip_by_global_norm:
                    clipped_gradients, grad_norm = tf.clip_by_global_norm(gradients, grad_clip)
                else:
                    clipped_gradients = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in gradients] # checkout cbfinn/MAML
            else:
                clipped_gradients = gradients

            self.optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables))

            self.train_loss(loss)

        self.update = update

    def call(self,inputs):
        bx,by,bq = inputs

        o = self.call_without_eager(bx,by,bq)

        if isinstance(self.net,M.maml.DenseNetProbPred):
            ba_hat, ba_dist = self.net.make_dist(o)
            return ba_hat, ba_dist
        else:
            return o

    @tf.function
    def call_without_eager(self,bx,by,bq):
        return tf.map_fn(partial(self._call,raw=True),(bx,by,bq),dtype=tf.float32, parallel_iterations=10)

    ###########################################################
    # the functions below this line does not assume meta_batch dimension
    ###########################################################
    def _update(self,inp):
        x,y,q,a = inp

        if isinstance(self.net,M.maml.DenseNetProbPred):
            a_hat, a_dist = self._call((x,y,q))
            loss = -1.* tf.reduce_mean(a_dist.log_prob(a),axis=0)
        else:
            a_hat = self._call((x,y,q))
            loss = tf.reduce_mean(tf.reduce_sum((a-a_hat)**2,axis=-1),axis=0)

        return loss

    def _finetune(self,x,y,params_list):
        with tf.GradientTape() as tape:
            tape.watch(params_list)

            if isinstance(self.net,M.maml.DenseNetProbPred):
                _, y_dist = self.net.call_with_params(x, params_list)
                loss = -1.*tf.reduce_mean(y_dist.log_prob(y),axis=0)
            else:
                y_hat = self.net.call_with_params(x, params_list)
                loss = tf.reduce_mean(tf.reduce_sum((y-y_hat)**2,axis=-1),axis=0)

        gs_list = tape.gradient(loss,params_list)
        assert len(gs_list) == len(params_list)

        # This part does not exist in the original paper.
        if self.finetune_grad_clip > 0:
            if self.clip_by_global_norm: #TODO: this is not global norm clipping
                gs_list = [tf.clip_by_global_norm(g, self.finetune_grad_clip)[0] for g in gs_list]
            else:
                gs_list = [tf.clip_by_value(g, -self.finetune_grad_clip, self.finetune_grad_clip) for g in gs_list]

        finetuned_params_list = []
        for params,gs in zip(params_list,gs_list):
            assert len(params) == len(gs)

            finetuned_params = [
                param - self.alpha * (tf.stop_gradient(g) if self.fo_approx else g)
                for param, g in zip(params,gs)
            ]

            finetuned_params_list.append(finetuned_params)

        return finetuned_params_list

    def _call(self,inputs, raw=False):
        x,y,q = inputs

        params_list = self.net.params_list

        for _ in range(self.num_inner_updates):
            params_list = self._finetune(x,y,params_list)

        return self.net.call_with_params(q, params_list, raw)

@gin.configurable(module=__name__)
class ANP(Model):
    def __init__(
        self,
        x_dim,
        y_dim,
        d_dim,
        num_heads,
        xy_embedder_fn_det,
        xy_embedder_fn_lat,
        x_embedder_fn,
        num_self_attention_det,
        num_self_attention_lat,
        z_dist_fn,
        decoder_fn,
        anp_style_sigma=True,
        uncertainty_prediction=True,
        max_grad_norm=0.,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.anp_style_sigma = anp_style_sigma

        self.xy_embedder_det = xy_embedder_fn_det()
        self.self_attention_det = [M.attention.MultiHeadedAttention(num_heads=num_heads,dim=d_dim//num_heads) for _ in range(num_self_attention_det)]
        self.x_embedder = x_embedder_fn()
        self.cross_attention = M.attention.MultiHeadedAttention(num_heads=num_heads,dim=d_dim//num_heads)

        self.xy_embedder_lat = xy_embedder_fn_lat()
        self.self_attention_lat= [M.attention.MultiHeadedAttention(num_heads=num_heads,dim=d_dim//num_heads) for _ in range(num_self_attention_lat)]
        self.z_dist = z_dist_fn(out_dim=2*d_dim)

        self.uncertainty_prediction = uncertainty_prediction
        if uncertainty_prediction:
            self.decoder = decoder_fn(out_dim=2*y_dim)
        else:
            self.decoder = decoder_fn(out_dim=y_dim)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.kl = tf.keras.metrics.Mean(name='kl')
        self.grad_norm = tf.keras.metrics.Mean(name='grad_norm')

        self.reports = {
            'loss': self.train_loss,
            'kl': self.kl,
            'grad_norm': self.grad_norm
        }

        @tf.function(input_signature=(
            tf.TensorSpec(shape=[None,None,x_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[None,None,y_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[None,None,x_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[None,None,y_dim],dtype=tf.float32)))
        def update(x,y,q,a):
            with tf.GradientTape() as tape:
                a_hat, dist, kl = self((x,y,q,a),training=True)
                if self.uncertainty_prediction:
                    loss = -1. * tf.reduce_mean(dist.log_prob(a)) + tf.reduce_mean(kl) / tf.cast(tf.shape(q)[1],tf.float32) # sum of log_p - single kl. Look Eq. (3) of the ANP closely.
                else:
                    loss = tf.reduce_mean(tf.reduce_sum(0.5 * (a - a_hat) ** 2,axis=-1)) + tf.reduce_mean(kl) / tf.cast(tf.shape(q)[1],tf.float32)

            gradients = tape.gradient(loss, self.trainable_variables)

            gradients_clipped, grad_norm = tf.clip_by_global_norm(gradients, max_grad_norm)
            self.grad_norm(grad_norm)

            if max_grad_norm > 0:
                gradients = gradients_clipped

            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            self.train_loss(loss)
            self.kl(tf.reduce_mean(kl) / tf.cast(tf.shape(q)[1],tf.float32))

        self.update = update

    def latent_path(self,x,y):
        s = self.xy_embedder_lat((x,y)) #[B,C,d]
        for sa in self.self_attention_lat:
            s = sa((s,s,s))

        s_c = tf.reduce_mean(s,axis=1)
        mu_z,sigma_z = tf.split(self.z_dist((s_c,)),2,axis=-1) #[B,d], [B,d]

        sigma_z = 0.1 + 0.9 * tf.sigmoid(sigma_z)

        z_dist = tfd.MultivariateNormalDiag(mu_z,scale_diag=sigma_z)
        return z_dist

    def det_path(self,x,y,q):
        # Deterministic Path
        r = self.xy_embedder_det((x,y)) #[B,C,d]
        for sa in self.self_attention_det:
            r = sa((r,r,r))

        key = self.x_embedder((x,)) #[B,C,d]
        query = self.x_embedder((q,)) #[B,Q,d]

        r_star = self.cross_attention((query,key,r)) #[B,Q,d]
        return r_star

    def call(self,inputs,training=None):
        if training:
            assert len(inputs) == 4
            x,y,q,a = inputs
        else:
            assert len(inputs) == 3
            x,y,q = inputs

        q_len = tf.shape(q)[1]

        # Det path
        r_star = self.det_path(x,y,q)

        # Latent path
        z_dist = self.latent_path(x,y)
        z = z_dist.sample() #[B,d]

        if training:
            z_post = self.latent_path(tf.concat([x,q],axis=1),tf.concat([y,a],axis=1))
            kl = z_post.kl_divergence(z_dist) #[B]

            z = z_dist.sample() #Don't use sample from prior

        # Decoder
        d_in = tf.concat([q,r_star,tf.tile(z[:,None,:],[1,q_len,1])],axis=-1)

        if self.uncertainty_prediction:
            mu_y,sigma_y = tf.split(self.decoder((d_in,)),2,axis=-1) #[B,d_y],[B,d_y]

            if self.anp_style_sigma:
                sigma_y = 0.1 + 0.9 * tf.nn.softplus(sigma_y)
            else:
                sigma_y = tf.nn.softplus(tf.maximum(-15.,sigma_y))


            y_dist = tfd.MultivariateNormalDiag(mu_y,scale_diag=sigma_y)
            y_hat = y_dist.sample()
        else:
            y_hat = self.decoder((d_in,))
            y_dist = None

        if training:
            return y_hat, y_dist, kl
        else:
            if self.uncertainty_prediction:
                return y_hat, y_dist
            else:
                return y_hat

@gin.configurable(module=__name__)
class MetaFun(Model):
    def __init__(
        self,
        x_dim,
        y_dim,
        updater_fn,
        x_embedder_fn,
        decoder_fn,
        L,
        max_grad_norm=0,
        anp_style_sigma=False,
        uncertainty_prediction=True,
        learning_rate=1e-3,
    ):
        super().__init__()

        self.L = L

        self.x_embedder = x_embedder_fn()
        self.updater = updater_fn()

        self.uncertainty_prediction = uncertainty_prediction
        if uncertainty_prediction:
            self.decoder = decoder_fn(out_dim=2*y_dim)
        else:
            self.decoder = decoder_fn(out_dim=y_dim)

        self.anp_style_sigma = anp_style_sigma

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.grad_norm = tf.keras.metrics.Mean(name='grad_norm')

        self.reports = {
            'loss': self.train_loss,
            'grad_norm': self.grad_norm
        }

        @tf.function(input_signature=(
            tf.TensorSpec(shape=[None,None,x_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[None,None,y_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[None,None,x_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[None,None,y_dim],dtype=tf.float32)))
        def update(x,y,q,a):
            with tf.GradientTape() as tape:
                if self.uncertainty_prediction:
                    a_hat, dist = self((x,y,q),training=True)
                    loss = tf.reduce_mean(-1. * dist.log_prob(a))
                else:
                    a_hat = self((x,y,q),training=True)
                    loss = tf.reduce_mean(tf.reduce_sum(0.5 * (a - a_hat) ** 2,axis=-1))

            gradients = tape.gradient(loss, self.trainable_variables)

            gradients_clipped, grad_norm = tf.clip_by_global_norm(gradients, max_grad_norm)
            self.grad_norm(grad_norm)

            if max_grad_norm > 0:
                gradients = gradients_clipped

            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            self.train_loss(loss)

        self.update = update

    def call(self,inputs,training=None):
        x,y,q = inputs

        meta_batch_size = tf.shape(x)[0]
        c_len = tf.shape(x)[1]
        q_len = tf.shape(q)[1]

        key = self.x_embedder((x,)) #[B,C,d]
        query = self.x_embedder((q,)) #[B,Q,d]

        _, c_att_map = M.dot_product_attention(key,key,scale=True) #[B,C,C]
        _, q_att_map = M.dot_product_attention(key,query,scale=True) #[B,Q,C]

        init_r_c = tf.zeros((meta_batch_size,c_len,self.updater.embed_dim),tf.float32)
        init_r_q = tf.zeros((meta_batch_size,q_len,self.updater.embed_dim),tf.float32)

        rs = [(init_r_c,init_r_q)]
        for _ in range(self.L):
            rs.append(self.updater((x,y,rs[-1][0],rs[-1][1],c_att_map,q_att_map)))

        r_c,r_q = rs[-1]

        if self.uncertainty_prediction:
            mu_sigma = self.decoder((q,r_q,)) #[B,Q,y_dim*2]
            mu,sigma = tf.split(mu_sigma,2,axis=-1) #[M,Q,C,y_dim]

            if self.anp_style_sigma:
                sigma = 0.1 + 0.9 * tf.nn.softplus(sigma)
            else:
                sigma = tf.nn.softplus(tf.maximum(-15.,sigma))

            y_dist = tfd.MultivariateNormalDiag(mu,scale_diag=sigma)
            y_hat = y_dist.sample() #[B,Q,y_dim]

            return y_hat, y_dist
        else:
            y_hat = self.decoder((q,r_q,))
            return y_hat

@gin.configurable(module=__name__)
class MetaFunWithANP(Model):
    def __init__(
        self,
        x_dim,
        y_dim,
        updater_fn,
        predictor_fn,
        xy_embedder_fn,
        x_embedder_fn,
        L,
        num_encoder_decoder_stack=0,
        i_dim_reduction=0,
        o_dim_reduction=0,
        dot_product_attention_scale=False,
        max_grad_norm=0,
        learning_rate=1e-3,
        anp_style_sigma=False,
    ):
        super().__init__()

        self.L = L
        self.anp_style_sigma = anp_style_sigma

        self.predictor = predictor_fn(out_dim=2*y_dim)

        self.xy_embedder = xy_embedder_fn()
        self.xq_embedder = x_embedder_fn()
        self.encoder = [M.attention.Encoder() for _ in range(num_encoder_decoder_stack)]
        self.decoder = [M.attention.Decoder() for _ in range(num_encoder_decoder_stack)]
        self.encoder_decoder = [(e,d) for e,d in zip(self.encoder,self.decoder)]

        self.i_transform = Dense(i_dim_reduction,use_bias=False) if i_dim_reduction > 0 else None
        self.o_transform = Dense(o_dim_reduction,use_bias=False) if o_dim_reduction > 0 else None
        self.dot_product_attention_scale = dot_product_attention_scale

        self.updater = updater_fn()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.grad_norm = tf.keras.metrics.Mean(name='grad_norm')

        self.reports = {
            'loss': self.train_loss,
            'grad_norm': self.grad_norm
        }

        @tf.function(input_signature=(
            tf.TensorSpec(shape=[None,None,x_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[None,None,y_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[None,None,x_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[None,None,y_dim],dtype=tf.float32)))
        def update(x,y,q,a):
            with tf.GradientTape() as tape:
                a_hat, dist = self((x,y,q),training=True)
                loss = tf.reduce_mean(-1. * dist.log_prob(a))
            gradients = tape.gradient(loss, self.trainable_variables)

            gradients_clipped, grad_norm = tf.clip_by_global_norm(gradients, max_grad_norm)
            self.grad_norm(grad_norm)

            if max_grad_norm > 0:
                gradients = gradients_clipped

            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            self.train_loss(loss)

        self.update = update

    def _get_att_maps(self,x,y,q):
        xy_embed = self.xy_embedder((x,y))
        x_embed = self.xq_embedder((x,))
        q_embed = self.xq_embedder((q,))

        i,o1,o2 = xy_embed, x_embed, q_embed
        for e,d in self.encoder_decoder:
            i = e((i,))
            o1 = d((i,o1))
            o2 = d((i,o2))

        if self.i_transform: i = self.i_transform(i)
        if self.o_transform:
            o1 = self.o_transform(o1)
            o2 = self.o_transform(o2)

        _, c_att_map = M.dot_product_attention(o1,o1,scale=self.dot_product_attention_scale)
        _, q_att_map = M.dot_product_attention(o1,o2,scale=self.dot_product_attention_scale)

        return c_att_map, q_att_map

    def call(self,inputs,training=None):
        x,y,q = inputs

        meta_batch_size = tf.shape(x)[0]
        c_len = tf.shape(x)[1]
        q_len = tf.shape(q)[1]

        c_att_map, q_att_map = self._get_att_maps(x,y,q)

        init_r_c = tf.zeros((meta_batch_size,c_len,self.updater.embed_dim),tf.float32)
        init_r_q = tf.zeros((meta_batch_size,q_len,self.updater.embed_dim),tf.float32)

        rs = [(init_r_c,init_r_q)]
        for _ in range(self.L):
            rs.append(self.updater((x,y,rs[-1][0],rs[-1][1],c_att_map,q_att_map)))

        r_c,r_q = rs[-1]

        mu_sigma = self.predictor((q,r_q,)) #[B,Q,y_dim*2]
        mu,sigma = tf.split(mu_sigma,2,axis=-1) #[M,Q,C,y_dim]
        if self.anp_style_sigma:
            sigma = 0.1 + 0.9 * tf.nn.softplus(sigma)
        else:
            sigma = tf.nn.softplus(tf.maximum(-15.,sigma))

        y_dist = tfd.MultivariateNormalDiag(mu,scale_diag=sigma)
        y_hat = y_dist.sample() #[B,Q,y_dim]

        return y_hat, y_dist

@gin.configurable(module=__name__)
class Merlot(Model):
    def __init__(
        self,
        x_dim,
        y_dim,
        f_i_embedder_fn,
        r_ik_embedder_fn,
        predictor_fn,
        updater_fn,
        xy_embedder_fn,
        x_embedder_fn,
        L,
        num_encoder_decoder_stack,
        i_dim_reduction=0,
        o_dim_reduction=0,
        dot_product_attention_scale=False,
        max_grad_norm=0,
        entropy_lambda=0,
        learning_rate=1e-3,
        alter_psi=False,
    ):
        Model.__init__(self)

        self.L = L
        self.alter_psi = alter_psi

        self.f_i_embedder = f_i_embedder_fn()
        self.r_ik_embedder = r_ik_embedder_fn()
        self.predictor = predictor_fn(y_dim=y_dim)

        self.xy_embedder = xy_embedder_fn()
        self.xq_embedder = x_embedder_fn()
        self.encoder = [M.attention.Encoder() for _ in range(num_encoder_decoder_stack)]
        self.decoder = [M.attention.Decoder() for _ in range(num_encoder_decoder_stack)]
        self.encoder_decoder = [(e,d) for e,d in zip(self.encoder,self.decoder)]

        self.i_transform = Dense(i_dim_reduction,use_bias=False) if i_dim_reduction > 0 else None
        self.o_transform = Dense(o_dim_reduction,use_bias=False) if o_dim_reduction > 0 else None
        self.dot_product_attention_scale = dot_product_attention_scale

        self.updater = updater_fn()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.grad_norm = tf.keras.metrics.Mean(name='grad_norm')
        self.entropy = tf.keras.metrics.Mean(name='entropy')

        self.reports = {
            'loss': self.train_loss,
            'grad_norm': self.grad_norm,
            'entropy': self.entropy,
        }

        @tf.function(input_signature=(
            tf.TensorSpec(shape=[None,None,x_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[None,None,y_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[None,None,x_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[None,None,y_dim],dtype=tf.float32)))
        def update(x,y,q,a):
            with tf.GradientTape() as tape:
                if isinstance(self.predictor,M.ProbabilisticPrediction):
                    a_hat, dist = self((x,y,q),training=True)
                    loss = tf.reduce_mean(-1. * dist.log_prob(a))

                    entropy = tf.reduce_mean(dist.mixture_distribution.entropy())
                    self.entropy(entropy)

                    if entropy_lambda > 0:
                        loss += entropy_lambda * entropy

                else:
                    a_hat, dist = self((x,y,q),training=True)
                    loss = tf.reduce_mean(tf.reduce_sum(0.5 * (a - a_hat) ** 2,axis=-1))

                    if dist is not None:
                        entropy = tf.reduce_mean(dist.mixture_distribution.entropy())
                        self.entropy(entropy)

            gradients = tape.gradient(loss, self.trainable_variables)

            gradients_clipped, grad_norm = tf.clip_by_global_norm(gradients, max_grad_norm)
            self.grad_norm(grad_norm)

            if max_grad_norm > 0:
                gradients = gradients_clipped

            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            self.train_loss(loss)
        self.update = update

    def _get_att_maps(self,x,y,q):
        xy_embed = self.xy_embedder((x,y))
        x_embed = self.xq_embedder((x,))
        q_embed = self.xq_embedder((q,))

        i,o1,o2 = xy_embed, x_embed, q_embed
        for e,d in self.encoder_decoder:
            i = e((i,))
            o1 = d((i,o1))
            o2 = d((i,o2))

        if self.i_transform: i = self.i_transform(i)
        if self.o_transform:
            o1 = self.o_transform(o1)
            o2 = self.o_transform(o2)

        c_att_logits, c_att_map = M.dot_product_attention(o1,o1,scale=self.dot_product_attention_scale)
        q_att_logits, q_att_map = M.dot_product_attention(o1,o2,scale=self.dot_product_attention_scale)

        return c_att_logits, c_att_map, q_att_logits, q_att_map

    def _get_r_ik_0(self,x,y,q):
        """
        out:
            r_ik_c: [M,C,C,r_dim]
            r_ik_q: [M,C,Q,r_dim]
        """

        C = tf.shape(x)[1]
        Q = tf.shape(q)[1]

        if self.alter_psi: # For ablation study
            r_i = self.f_i_embedder((x,)) #[M,C,f_embed_dim]
        else:
            r_i = self.f_i_embedder((x,y)) #[M,C,f_embed_dim]

        _in = tf.concat([tf.tile(r_i[:,:,None,:],[1,1,C,1]),tf.tile(x[:,None,:,:],[1,C,1,1])],axis=-1) # _in[:,i,k,:] is concatenation of r_i and x_k
        r_ik_c = self.r_ik_embedder((_in,))

        _in = tf.concat([tf.tile(r_i[:,:,None,:],[1,1,Q,1]),tf.tile(q[:,None,:,:],[1,C,1,1])],axis=-1) # _in[:,i,k,:] is concatenation of r_i and x_k
        r_ik_q = self.r_ik_embedder((_in,))

        return r_ik_c, r_ik_q

    def enc(self,x,y,q):
        c_att_logits, c_att_map, q_att_logits, q_att_map = self._get_att_maps(x,y,q)

        r_ik_c_0, r_ik_q_0 = self._get_r_ik_0(x,y,q)

        rs = [(r_ik_c_0,r_ik_q_0)]
        for _ in range(self.L):
            rs.append(self.updater((x,y,rs[-1][0],rs[-1][1],c_att_map,q_att_map)))

        return (c_att_logits, c_att_map, q_att_logits, q_att_map), rs

    def dec(self,q,r_ik_q,q_att_logits):
        return self.predictor((q,tf.transpose(r_ik_q,[0,2,1,3]),q_att_logits))

    def call(self,inputs,training=None):
        x,y,q = inputs

        (c_att_logits, c_att_map, q_att_logits, q_att_map), rs = self.enc(x,y,q)

        r_ik_c, r_ik_q = rs[-1]
        return self.predictor((q,tf.transpose(r_ik_q,[0,2,1,3]),q_att_logits))
