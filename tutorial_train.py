from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset, MyTestDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict



# Configs
resume_path = './models/control_sd15_ini.ckpt'
# resume_path = './lightning_logs/200epoch_ckpt/checkpoints/epoch=199-step=209799.ckpt'
# resume_path = './lightning_logs/version_48/checkpoints/epoch=34-step=25654.ckpt'
# resume_path = './lightning_logs/new/checkpoints/epoch=15-step=9823.ckpt'

batch_size = 36
logger_freq = 2000
learning_rate = 1e-5 #可选2e-6，但效果还没有验证
sd_locked = False
only_mid_control = False



# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# train
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger], max_epochs=35) #改成半精度16，9.23
# model.eval()
trainer.fit(model, dataloader)

# test
test_batch_size = 1
test_logger_freq = 1
test_logger = ImageLogger(batch_frequency=test_logger_freq)
test_dataset = MyTestDataset()
test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=test_batch_size)
trainer = pl.Trainer(gpus=1, precision=16, callbacks=[test_logger], max_epochs=1)
model.eval()
trainer.fit(model, test_dataloader)