import argparse
import logging
import gin
import os
import numpy as np
import matplotlib
matplotlib.use('module://imgcat')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tqdm import tqdm

def _predict(model,cx,cy,x,num_samples=100):
    _, dist = model((cx,cy,x))

    if num_samples <= 0: # use mode instead of sample mean
        if isinstance(dist,tfd.MultivariateNormalDiag):
            y_hat = dist.mean()
        elif isinstance(dist,tfd.MixtureSameFamily):
            mode_mixture_idx = dist.mixture_distribution.mode()
            y_hat = tf.gather_nd(
                dist.components_distribution.mean(),
                mode_mixture_idx[:,:,None],
                batch_dims=2)
        y_hat = y_hat.numpy()
    else:
        y_hat = np.mean(dist.sample(num_samples),axis=0)

    return dist, y_hat

def _eval_on(model,dataset,batch_size,reset_seed=True):
    if reset_seed:
        dataset.reset_seed()

    cx,cy,qx,qy = next(dataset.batch(batch_size))
    dist, y_hat = _predict(model,cx,cy,qx,num_samples=100)

    nlpd = -1. * np.mean(dist.log_prob(qy))
    rmse = np.mean((qy - y_hat)**2)**0.5

    return nlpd, rmse

def _draw(ax,x,y,cx,cy,dist,y_hat,num_samples=20):
    samples = []
    for _ in range(num_samples):
        y_hat_sample = dist.sample()
        ax.plot(x,y_hat_sample[0,:,0],'*',color='blue',alpha=0.03)

        samples.append(y_hat_sample[0,:,0].numpy())

    ax.plot(x,y_hat[0,:,0],'--',color='blue')
    ax.plot(x,np.mean(np.stack(samples),axis=0),'--',color='green')
    ax.plot(x,y,'-',color='red')
    ax.plot(cx,cy,'+',color='black')

    return x,y,cx,cy,y_hat[0,:,0],samples

def _qualitative_test(log_dir,model,dataset,num,prefix='',num_shots=None):
    dataset.reset_seed()
    for i in range(num):
        fig,ax = plt.subplots()

        (fn,x,y,_,_),cx,cy,_,_ = dataset.gen_task()
        if num_shots is not None:
            cx, cy = cx[:num_shots], cy[:num_shots]

        dist, y_hat = _predict(model,cx[None,:,None],cy[None,:,None],x[None,:,None],num_samples=0)

        info = _draw(ax,x,y,cx,cy,dist,y_hat)

        import pickle
        with open(os.path.join(log_dir,f'eval_{i}_{prefix}.pkl'),'wb') as f:
            pickle.dump(info,f)

        fig.savefig(os.path.join(log_dir,f'eval_{i}_{prefix}.pdf'),bbox_inches='tight')
        plt.close(fig)

def _scale_test(log_dir,model,dataset,batch_size=1000,save=False):
    mb_size = 50
    assert batch_size % mb_size == 0

    dataset.reset_seed()

    nlpds,rmses = [], []

    cx,cy,qx,qy = next(dataset.batch(batch_size))
    for k in tqdm(range(5,100+1,5)):
        nlpd = []
        rmse = []
        for b in range(batch_size//mb_size):
            sli = slice(b*mb_size,(b+1)*mb_size)

            dist, y_hat = _predict(model,cx[sli,:k],cy[sli,:k],qx[sli],num_samples=100)
            nlpd_ = -1. * dist.log_prob(qy[sli])
            rmse_ = (qy[sli] - y_hat)**2

            nlpd.append(nlpd_)
            rmse.append(rmse_)

        nlpd = np.mean(np.concatenate(nlpd,axis=0))
        rmse = np.mean(np.concatenate(rmse,axis=0)) ** 0.5

        nlpds.append(nlpd)
        rmses.append(rmse)

    if save:
        np.savetxt(os.path.join(log_dir,'nlpds.txt'),nlpds)
        np.savetxt(os.path.join(log_dir,'rmses.txt'),rmses)

    return nlpds, rmses

@gin.configurable(module=__name__)
def eval_(log_dir, dataset, model, it):
    K = num_shots = 50

    summary_writer = logging.getLogger('summary_writer')
    logger = logging.getLogger('stdout')

    from tasks.one_d_regression import Dataset

    valid_ds = Dataset(dataset.seed,'training',dataset.c_interval_size,dataset.c_len_range,dataset.q_len_range)
    int_ds = Dataset(dataset.seed,'int',dataset.c_interval_size,dataset.c_len_range,dataset.q_len_range)
    ext_ds = Dataset(dataset.seed,'ext',dataset.c_interval_size,dataset.c_len_range,dataset.q_len_range)
    scale_100_ds = Dataset(0,'training',0,(100,100),(100,100))

    nlpd, rmse = _eval_on(model,valid_ds,100)
    summary_writer.info('raw','nlpd/train',nlpd,it)
    summary_writer.info('raw','rmse/train',rmse,it)
    logger.info(f'[{it}] train: {nlpd} {rmse}')

    nlpd, rmse = _eval_on(model,int_ds,100)
    summary_writer.info('raw','nlpd/int',nlpd,it)
    summary_writer.info('raw','rmse/int',rmse,it)
    logger.info(f'[{it}] int: {nlpd} {rmse}')

    nlpd, rmse = _eval_on(model,ext_ds,100)
    summary_writer.info('raw','nlpd/ext',nlpd,it)
    summary_writer.info('raw','rmse/ext',rmse,it)
    logger.info(f'[{it}] ext: {nlpd} {rmse}')

    _qualitative_test(log_dir,model,scale_100_ds,1,f'scale_10_it_{it}',10)
    _qualitative_test(log_dir,model,scale_100_ds,1,f'scale_25_it_{it}',25)
    _qualitative_test(log_dir,model,scale_100_ds,1,f'scale_50_it_{it}',50)
    _qualitative_test(log_dir,model,scale_100_ds,1,f'scale_100_it_{it}',100)

    nlpds, rmses = _scale_test(log_dir,model,scale_100_ds,batch_size=100,save=False)
    summary_writer.info('raw','nlpd/scale_100',nlpds[-1],it)
    summary_writer.info('raw','rmse/scale_100',rmses[-1],it)
    logger.info(f'[{it}] scale_100: {nlpds[-1]} {rmses[-1]}')


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    parser = argparse.ArgumentParser(description=None)
    ########### expr setting
    parser.add_argument('--log_dir',required=True)
    parser.add_argument('--extra_config_params', nargs='*')
    parser.add_argument('--show_fig',action='store_true')
    args = parser.parse_args()

    import sys
    sys.path.append('.') #assuming the code is excuted from the top directory

    from run import load_model
    dataset, model = load_model(args.log_dir,args.extra_config_params)

    from tasks.one_d_regression import Dataset

    int_ds = Dataset(dataset.seed,'int',dataset.c_interval_size,dataset.c_len_range,dataset.q_len_range)
    ext_ds = Dataset(dataset.seed,'ext',dataset.c_interval_size,dataset.c_len_range,dataset.q_len_range)
    scale_100_ds = Dataset(dataset.seed,'training',0,(100,100),(100,100))

    result = _eval_on(model,int_ds,1000)
    print(f'int {result}')

    result = _eval_on(model,ext_ds,1000)
    print(f'ext {result}')

    _qualitative_test(args.log_dir,model,scale_100_ds,10,'scale_10',10)
    _qualitative_test(args.log_dir,model,scale_100_ds,10,'scale_25',25)
    _qualitative_test(args.log_dir,model,scale_100_ds,10,'scale_50',50)
    _qualitative_test(args.log_dir,model,scale_100_ds,10,'scale_100',100)

    nlpds, rmses = _scale_test(args.log_dir,model,scale_100_ds,save=True)
    print(nlpds)
    print(rmses)
