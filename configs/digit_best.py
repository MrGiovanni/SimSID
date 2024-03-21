import torch
from dataloader.dataloader_digit import DigitAnatomy, DigitAnatomyTest
from configs.base import BaseConfig

class MemoryMatrixBlockConfig():
    memory_layer_type = 'default'    # ['default', 'dim_reduce']
    num_memory = 9    # square of num_patches
    num_slots = 200
    slot_dim = 256    # used for memory_layer_type = dim_reduce
    shrink_thres = 5
    mask_ratio = 1.0

class InpaintBlockConfig():
    use_memory_queue = False
    use_inpaint = True
    num_slots = 200
    memory_channel = 64 * 4 * 4
    shrink_thres = 5
    drop = 0.    # used in the mlp in the transformer layer
    mask_ratio = 1.0


class Config(BaseConfig):

    memory_config = MemoryMatrixBlockConfig()
    inpaint_config = InpaintBlockConfig()

    def __init__(self):
        super(Config, self).__init__()

        #---------------------
        # Training Parameters
        #---------------------
        self.print_freq = 10
        self.device = 'cuda:0'
        self.epochs = 200
        self.lr = 1e-4 # learning rate
        self.batch_size = 16
        self.test_batch_size = 2
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        self.scheduler_args = dict(T_max=300, eta_min=self.lr*0.01)

        # GAN
        self.gan_lr = 1e-4
        self.discriminator_type = 'basic'
        self.enbale_gan = 0#100
        self.lambda_gp = 10.
        self.num_layers = 4
        self.size = 3
        self.n_critic = 1
        self.sample_interval = 1000
        self.scheduler_d = torch.optim.lr_scheduler.MultiStepLR
        self.scheduler_args_d = dict(milestones=[150-self.enbale_gan, 250-self.enbale_gan], gamma=0.2)

        # model
        self.img_size = 96
        self.normalize_tanh = False
        self.num_patch = 3 #4
        self.level = 3 #
        self.initial_combine = 1 # from top to bottom
        self.dist = True
        self.ops = ['none', 'none', 'none']
        self.decoder_memory = ['none', 'none', 'none']
        # self.ops = ['concat', 'concat', 'none']
        # self.decoder_memory = ['V3', 'V3', 'none']
                            # dict(type='MemoryMatrixBlockV3', multiplier=256, num_memory=self.num_patch**2),
                            # dict(type='MemoryMatrixBlockV3', multiplier=64, num_memory=self.num_patch**2)]


        # loss weight
        self.t_w = 0.01
        self.recon_w = 1.
        self.dist_w = 0.001
        self.g_w = 0.005
        self.d_w = 0.005

        # misc
        self.disable_tqdm = False
        self.dataset_name = 'digitanatomy'
        self.early_stop = 100
        self.data_type = 'none'

        self.data_root = '/mnt/data0/yixiao'
        self.train_dataset = DigitAnatomy(self.data_root+'/digitanatomy/MNIST/',  length=32 * 50)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.val_dataset = DigitAnatomyTest(self.data_root+'/digitanatomy/test/')
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0)

        self.test_dataset = DigitAnatomyTest(self.data_root+'/digitanatomy/test/')
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0) 