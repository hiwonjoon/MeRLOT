import argparse
import gin
import os
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import schedule

from utils import setup_logger

@gin.configurable()
def train(
    args,
    log_dir,
    seed,
    Dataset,
    Model,
    train_iter,
    batch_size,
    log_every_it=10,
    eval_fn = lambda *args: None,
    final_eval_fn = lambda *args: None,
    eval_every_minutes=1,
    save_every_minutes=30,
    #train_chkpt=None,
):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    np.random.seed(seed)
    tf.random.set_seed(seed)

    dataset = Dataset(task_random_seed=seed)
    model = Model(dataset.x_dim,dataset.y_dim)

    #if train_chkpt:
    #    model.load_weights(train_chkpt)

    # Define Logger
    setup_logger(log_dir,args)
    summary_writer = logging.getLogger('summary_writer')
    logger = logging.getLogger('stdout')

    # Write down gin config
    with open(os.path.join(args.log_dir,'config.gin'),'w') as f:
        f.write(gin.operative_config_str())
    summary_writer.info('text','gin/config',gin.operative_config_str(),0)

    eval_fn(log_dir, dataset, model, 0)

    schedule.every(eval_every_minutes).minutes.do(
        lambda: eval_fn(log_dir, dataset, model, it))
    schedule.every(save_every_minutes).minutes.do(
        lambda: model.save_weights(os.path.join(log_dir,f'model-{it}.tf')))

    try:
        get_batch = dataset.batch(batch_size)
        for it in tqdm(range(train_iter),desc=log_dir,dynamic_ncols=True):
            cx,cy,qx,qy = next(get_batch)

            model.update(cx,cy,qx,qy)

            if it % log_every_it == 0:
                try:
                    for name,item in model.reports.items():
                        val = item.result().numpy()
                        summary_writer.info('raw',f'loss/{name}',val,it)
                        logger.info(f'[{it}] {name}: {val}')
                except AttributeError:
                    loss = model.train_loss.result().numpy()

                    summary_writer.info('raw','loss/loss',loss,it)
                    logger.info(f'[{it}] loss: {loss}')

            schedule.run_pending()

    except KeyboardInterrupt:
        pass

    model.save_weights(os.path.join(log_dir,'model.tf'))
    final_eval_fn(log_dir, dataset, model)

def load_model(
    log_dir,
    extra_config_params,
    model_file='model.tf',
):
    # Allows overriding configuration with config_params; change environment or something
    gin.parse_config_files_and_bindings([os.path.join(log_dir,'config.gin')], extra_config_params)

    Dataset = gin.query_parameter('train.Dataset').scoped_configurable_fn
    Model = gin.query_parameter('train.Model').scoped_configurable_fn

    dataset = Dataset()
    model = Model(dataset.x_dim,dataset.y_dim)

    model.load_weights(os.path.join(log_dir,model_file))
    return dataset, model

if __name__ == "__main__":
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    parser = argparse.ArgumentParser(description=None)
    ########### expr setting
    parser.add_argument('--log_dir',required=True)
    parser.add_argument('--config_file',required=True, nargs='+')
    parser.add_argument('--config_params', nargs='*')
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.config_file, args.config_params)

    train(args,args.log_dir,args.seed)
