import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

import warnings
warnings.filterwarnings('ignore')

psnr_sum = 0
ssim_sum = 0
i = 0
class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=54, clamp=True, increase_log_steps=True, # max_images改变log图片数量
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=6)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

        source_img_path = "./image_log/train/reconstruction" + "_gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
        target_img_path = "./image_log/train/samples_cfg_scale_9.00" + "_gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
        source_img = cv2.imread(source_img_path)
        target_img = cv2.imread(target_img_path)

        # if source_img.shape[0] != target_img.shape[0] or source_img.shape[1] != target_img.shape[1]:
        #     pil_img = Image.fromarray(target_img)
        #     pil_img = pil_img.resize((source_img.shape[1], source_img.shape[0]))  # 和clear_img的宽和高保持一致
        #     target_img = np.array(pil_img)
        # PSNR = peak_signal_noise_ratio(source_img, target_img)
        # print('PSNR: ', PSNR)
        # SSIM = structural_similarity(source_img, target_img, multichannel = True, win_size = 3)
        # print('SSIM: ', SSIM)

        if source_img.shape[0] != target_img.shape[0] or source_img.shape[1] != target_img.shape[1]:
            pil_img = Image.fromarray(target_img)
            pil_img = pil_img.resize((source_img.shape[1], source_img.shape[0]))  # 和clear_img的宽和高保持一致
            target_img = np.array(pil_img)
        PSNR = peak_signal_noise_ratio(source_img, target_img)
        global psnr_sum
        global ssim_sum
        global i
        # if (i % 700 == 0):
        if (i % 35000 == 0):
            psnr_sum = 0
            ssim_sum = 0
            i = 0
        i += 1
        psnr_sum += PSNR
        psnr_avg = psnr_sum / i
        print('PSNR_AVG: ', psnr_avg)
        SSIM = structural_similarity(source_img, target_img, multichannel = True, win_size = 3)
        ssim_sum +=SSIM
        ssim_avg = ssim_sum / i
        print('SSIM_AVG: ', ssim_avg)
        print('PSNR:', PSNR)
        print('SSIM:', SSIM)



    def log_img(self, pl_module, batch, batch_idx, split="train"):
        # loss = pl_module.get_clip_loss(batch, split=split, **self.log_images_kwargs)
        # loss.backward()
        # optimizer = torch.optim.Adam(pl_module.parameters(), lr=0.001)
        # optimizer.step()
        # optimizer.zero_grad()
        # print("clip_loss:" + str(loss.item()))
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N] #通过索引操作将 images[k] 截取到前 N 个图像，并将结果重新赋值给 images[k]。确保 images[k] 中仅包含 N 个图像，而不是原始的全部图像数据。
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
