from cv2 import normalize
import torch
from dataloader.dataloader_chexpert import CheXpert
from configs.base import BaseConfig


class MemoryMatrixBlockConfig():
    memory_layer_type = 'default'    # ['default', 'dim_reduce']
    num_memory = 4    # square of num_patches
    num_slots = 200
    slot_dim = 256    # used for memory_layer_type = dim_reduce
    shrink_thres = 5
    mask_ratio = 1.0

class InpaintBlockConfig():
    use_memory_queue = False
    use_inpaint = True
    num_slots = 200
    memory_channel = 128 * 4 * 4
    shrink_thres = 5
    drop = 0.    # used in the mlp in the transformer layer
    mask_ratio = 0.9

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
        self.epochs = 600
        self.lr = 1e-4 # learning rate
        self.batch_size = 16
        self.test_batch_size = 2
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        self.scheduler_args = dict(T_max=300, eta_min=self.lr*0.5)

        # GAN
        self.gan_lr = 1e-4
        self.discriminator_type = 'basic'
        self.enbale_gan = 0 #100
        self.lambda_gp = 10.
        self.size = 4
        self.num_layers = 4
        self.n_critic = 2
        self.sample_interval = 1000
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR
        self.scheduler_args_d = dict(T_max=200, eta_min=self.lr*0.2)

        # model
        self.img_size = 128
        self.normalize_tanh = True
        self.num_patch = 2
        self.level = 4
        self.initial_combine = 2
        self.dist = True
        self.ops = ['concat', 'concat', 'none', 'none']
        self.decoder_memory = ['V1', 'V1', 'none', 'none']

        # loss weight
        self.t_w = 0.01
        self.recon_w = 10.
        self.dist_w = 0.001
        self.g_w = 0.005
        self.d_w = 0.005

        self.positive_ratio = 0.9

        # misc
        self.disable_tqdm = False
        self.dataset_name = 'chexpert'
        self.early_stop = 200    # used in alert.collect()
        # self.limit = 84    # number of iterations per epoch
        self.data_type = 'pa'
        self.test_disease_type = 'all'    # {'all', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 
                                                   # 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                                                   # 'Pneumothorax'}

        self.data_root = '/mnt/data0/yixiao/chexpert'
        self.train_dataset = CheXpert(
            self.data_root+'/train_256_'+self.data_type, 
            train=True, 
            img_size=(self.img_size, self.img_size), 
            normalize_tanh=self.normalize_tanh,
            data_type=self.data_type,
            positive_ratio=self.positive_ratio,
        )
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=False)
        self.val_dataset = CheXpert(
            self.data_root+'/val_256_'+self.data_type, 
            train=False, 
            img_size=(self.img_size, self.img_size),
            normalize_tanh=self.normalize_tanh,
            full=True,  
            data_type=self.data_type,
            positive_ratio=self.positive_ratio,
        )
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
        self.test_dataset = CheXpert(
            self.data_root+'/our_test_256_'+self.data_type, 
            train=False, 
            img_size=(self.img_size, self.img_size),
            normalize_tanh=self.normalize_tanh,
            full=True,  
            data_type=self.data_type,
            test_disease_type=self.test_disease_type,
            positive_ratio=self.positive_ratio,
        )
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)