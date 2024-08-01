import os
import glob
import socket
import tensorflow as tf
import time
import datetime
# import neuralgym as ng

from model import HinpaintModel
from utils import load_yml
from trainer import HiTrainer, d_graph_deploy, get_batch, g_graph_deploy, get_input_queue
import dataModule.ng as ng

# tf.compat.v1.disable_eager_execution()


if __name__ == "__main__":
    # load config.yml file
    config = load_yml('config.yml')
    if config.GPU_ID != -1:
        gpu_ids = config.GPU_ID
    else:
        gpu_ids = [0]
    
    
    print('building networks and losses...')
    enq_ops = []

    '''    # load training data
    with open(config.TRAIN_LIST) as f:
        fnames = f.read().splitlines()
        endnd = (len(fnames) // config.BATCH_SIZE) * config.BATCH_SIZE
        fnames = fnames[:endnd]
        
    data = ng.DataFromFNames(fnames, config.IMG_SHAPE, random_crop=config.RANDOM_CROP,
                             enqueue_size=32, queue_size=256, nthreads=config.N_THREADS)


    images =  data.data_pipeline(config.BATCH_SIZE) #from neuralgym
    '''

    # load training data
    data_dir = config.TRAIN_LIST
    data_dir = './data/examples/train/'
    train_ds =  tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        validation_split=0.2,
        subset="training",
        seed=123,
        shuffle=True,
        image_size=(512, 512),
        batch_size=config.BATCH_SIZE,
        color_mode="rgb"
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        validation_split=0.2,
        subset="validation",
        seed=123,
        shuffle=True,
        image_size=(512, 512),
        batch_size=config.BATCH_SIZE,
        color_mode="rgb"
    )

    # image shape = (4, 512, 512, 3)
    # print("-----------images:",images.shape)
    # images = data.batch(config.BATCH_SIZE) #from tensorflow

    # images = ng.data_pipeline(fnames, config.IMG_SHAPE, config.BATCH_SIZE, random_crop=config.RANDOM_CROP,
    #                       enqueue_size=32, queue_size=256, nthreads=config.N_THREADS)
    model = HinpaintModel()

    

    g_vars, d_vars, losses = model.build_graph_with_losses(train_ds, config=config)

    # validation graphs
    print('bbuilding validation graph...')
  
    test_results = model.build_static_graph(val_images, config, name='test/')


    # training settings
    lr = tf.compat.v1.get_variable('lr', shape=[], trainable=False, dtype=tf.float32,
        initializer=tf.constant_initializer(1e-4))
    d_optimizer = tf.compat.v1.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
    g_optimizer = tf.compat.v1.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
    # log dir
    date_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    log_prefix = 'model_logs/' + '_'.join([date_str, config.LOG_DIR])
    dis_kwargs = {
            'var_list':d_vars, 
            'graph_def':d_graph_deploy, 
            'graph_def_kwargs':{'model': model, 'data': data, 'config': config}, 
            'optimizer':d_optimizer,
            'max_iters': 3,
            'gpu_ids': gpu_ids}

    gen_kwargs = {
            'var_list':g_vars, 
            'graph_def':g_graph_deploy, 
            'graph_def_kwargs':{'model': model, 'data': data, 'config': config}, 
            'optimizer':g_optimizer,
            'max_iters':config.MAX_ITERS,
            'gpu_ids': gpu_ids,
            'SPE': config.TRAIN_SPE }

    restore_dir = 'model_logs/'+config.MODEL_RESTORE if config.MODEL_RESTORE != '' else None
    saver_kwargs = {
            'period': config.TRAIN_SPE, 
            'dir': log_prefix + '/ckpt', 
            'restore_dir': restore_dir}

    summary_kwargs = {
            'period': config.VAL_PSTEPS, 
            'dir': log_prefix}
    trainer = HiTrainer(gen_kwargs, dis_kwargs, saver_kwargs, summary_kwargs, enq_ops)

    # start training
    print('training...')
    trainer.train()
