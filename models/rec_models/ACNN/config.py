# coding:utf8
import warnings
import torch as t
import numpy as np

class DefaultConfig(object):
    env = 'mri_34'  # visdom environment
    vis_port = 8098 # visdom port num
    model = 'AcnnModel'  # model used whose name should consist with the name in 'models/__init__.py'

    train_data_root = '/data2/dutia/data1/python_mri/train/'
    val_data_root = '/data2/dutia/data1/python_mri/val/'
    load_model_path = None
    test_model_path = './checkpoints/mri_34_350.pth'  # path of pretrain model

    undersample_mask = "./data/mask_smp.mat"

    batch_size = 1 # batch size
    val_batch_size = 1
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch
    slice_num = 3  
    test_num = 400

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 351
    lr = 0.00001  # initial learning rate
    lr_list = np.logspace(-4, -5, 300)
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 

    def _parse(self, kwargs):
        """
        update parameter in config file
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        opt.device =t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
