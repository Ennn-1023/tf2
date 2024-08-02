# def trainer
class Trainer:
    def __init__(self, gen_kwargs, dis_kwargs, saver_kwargs, summary_kwargs):
        self.gen_kwargs = gen_kwargs
        self.dis_kwargs = dis_kwargs
        self.saver_kwargs = saver_kwargs
        self.summary_kwargs = summary_kwargs
    
    @tf.function
    def train_step(self, dataset):
        gen_kwargs = {
            'var_list':g_vars, 
            'graph_def':g_graph_deploy, 
            'graph_def_kwargs':{'model': model, 'data': data, 'config': config}, 
            'optimizer':g_optimizer,
            'max_iters':config.MAX_ITERS,
            'gpu_ids': gpu_ids,
            'SPE': config.TRAIN_SPE }
        
        for img_batch in dataset:
            fake_img = model.generator(img_batch)

    def train(self):
        start_time = time.time()
        for step in range(self.gen_kwargs['max_iters']):
            #dloss = self.run_d_optimizer()

            #gloss = self.run_g_optimizer()
            #self.run_summary_writer(step)
            #self.run_saver(step)
            pass
            