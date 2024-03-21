import sys
import torch
sys.path.insert(0, '..')


class MemoryMatrixBlockConfig():
    num_memory = 4    # square of num_patches
    num_slots = 500
    slot_dim = 2048
    shrink_thres = 0.0005
    mask_ratio = 0.95

class BaseConfig():
    def __init__(self):

        #---------------------
        # Training Parameters
        #---------------------
        self.data_root = '/media/administrator/1305D8BDB8D46DEE/jhu'
        self.print_freq = 10
        self.device = 'cuda:0'
        self.epochs = 400
        self.lr = 1e-4#1e-3 # learning rate
        self.batch_size = 16
        self.test_batch_size = 2
        self.opt = torch.optim.Adam
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR
        self.scheduler_args = dict(milestones=[200, 300], gamma=0.2)
        self.analyze_memory = False
        self.val_freq = 1

        # GAN
        self.discriminator_type = 'basic'
        self.enbale_gan = 0 #100
        self.lambda_gp = 10
        self.size = 4
        self.n_critic = 1
        self.sample_interval = 1000
        self.scheduler_d = torch.optim.lr_scheduler.MultiStepLR
        self.scheduler_args_d = dict(milestones=[200-self.enbale_gan, 300-self.enbale_gan], gamma=0.2)

        # model
        self.num_in_ch = 1
        self.num_patch = 2 #4
        self.level = 4 #
        self.shrink_thres = 0.0005
        self.initial_combine = 2
        self.drop = 0.
        self.dist = True
        self.num_slots = 1000
        self.mem_num_slots = 500
        self.memory_channel = 2048
        self.img_size = 128
        self.mask_ratio = 0.95
        self.ops = ['concat', 'concat', 'none', 'none']
        self.decoder_memory = [None,
                               None,
                               dict(type='MemoryMatrixBlock', multiplier=64, num_memory=self.num_patch**2),
                               dict(type='MemoryMatrixBlock', multiplier=16, num_memory=self.num_patch**2)]

        # loss weight
        self.t_w = 0.5
        self.recon_w = 1.
        self.dist_w = 0.1
        self.g_w = 0.0005
        self.d_w = 1.

        # misc
        self.disable_tqdm = False
        self.dataset_name = 'zhang'
        self.early_stop = 200
        self.limit = None

        self.use_memory_inpaint_block = True
        self.teacher_stop_gradient = True

        # alert
        self.alert = None#Alert(lambda1=1., lambda2=1.)
    
    memory_config = MemoryMatrixBlockConfig()
