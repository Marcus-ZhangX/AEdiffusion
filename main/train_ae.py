import logging
import os
import sys

import hydra
import pytorch_lightning as pl
# print(pl.__version__)
import torchvision.transforms as T
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
# from models.vae import VAE
print(os.getcwd())  # 显示当前路径
sys.path.append(r"E:\Coding_path\DiffuseVAE\main")
from models.vae import VAE
from util import configure_device, get_dataset

logger = logging.getLogger(__name__)
import torch
print(torch.__version__)

import os
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"  # 修改环境变量 防止报错

@hydra.main(version_base=None,config_path="configs")  # 这里需要添加version_base=None，否则会报错
def train(config):
    # Get config and setup

    config = config.dataset.vae  # config.dataset.vae 赋值给变量 config，
    logger.info(OmegaConf.to_yaml(config))  # 然后将其转换为 YAML 格式并打印出来。OmegaConf.to_yaml() 是将 OmegaConf 对象转换为 YAML 格式的函数，而 logger.info() 则是将信息记录到日志中

    # Set seed
    seed_everything(config.training.seed, workers=True)
       # seed_everything(config.training.seed)  因为先前pl版本问题，先前我删除了worker

    # Dataset
    root = config.data.root
    d_type = config.data.name  # d_type就是数据的名字
    image_size = config.data.image_size
    dataset = get_dataset(d_type, root, image_size, norm=False, flip=config.data.hflip)
    N = len(dataset)
    batch_size = config.training.batch_size
    batch_size = min(N, batch_size)  # 但是，如果数据集的大小小于指定的批次大小，则 batch_size 被设置为数据集的大小。这确保了批次大小不会大于数据集的大小，否则会在训练过程中出现错误。

    # Model
    vae = VAE(
        input_res=image_size,
        enc_block_str=config.model.enc_block_config,
        dec_block_str=config.model.dec_block_config,
        enc_channel_str=config.model.enc_channel_config,
        dec_channel_str=config.model.dec_channel_config,
        lr=config.training.lr,
        alpha=config.training.alpha,
    )

    # Trainer
    train_kwargs = {}
    restore_path = config.training.restore_path
    if restore_path is not None:
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path   # "resume_from_checkpoint" 用于从之前保存的检查点（checkpoint）处恢复训练。
        # 在该代码中，如果 "restore_path" 参数不为空，则将其值作为 "resume_from_checkpoint" 参数传递给 "train_kwargs" 字典，
        # 以便在训练过程中调用。这意味着程序将从之前保存的检查点处继续训练，而不是重新开始。
        # 需要注意的是restore_path应该是一个checkpoint文件，而不是一个路径

    results_dir = config.training.results_dir
    chkpt_prefix = "cmhq128_alpha=1.0"   # 这个chkpt_prefix是保存的vae模型的前缀
    chkpt_callback = ModelCheckpoint(   #   关于该函数的介绍https://blog.csdn.net/zengNLP/article/details/94589469
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"vae-{chkpt_prefix}"
        + "-{epoch:02d}-{train_loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,   # 间隔多少个epoch保存一次模型
        # save_weights_only=False,     #  若设置为False，占用内存大（包括了模型结构和配置信息），下次调用可以直接载入，不需要再次搭建神经网络结构  .但是后面本来就是要先搭建网络结构的呀
        save_on_train_epoch_end=True,  # 不知道这句是用来干嘛的，先注释掉了  （可能用来表示在最后一个epoch也要保存模型）
        save_top_k=-1,  # 当此参数被设置为-1时，所有的checkpoints都会被保存，如果不设置这个参数，后面的checkpoints会将前面的覆盖掉
    )

    train_kwargs["default_root_dir"] = results_dir   # 设置保存训练log以及checkpoint的目录
    train_kwargs["max_epochs"] = config.training.epochs  # 设置训练最大/最小的epoch
    train_kwargs["log_every_n_steps"] = config.training.log_step  # 日志登记的间隔
    train_kwargs["callbacks"] = [chkpt_callback]

    device = config.training.device  # 选择训练的硬件类型
    loader_kws = {}
    if device.startswith("gpu"):  # 如果设备名称以 "gpu" 开头，就调用 configure_device 函数配置设备，并将返回结果中的设备列表（devs）赋值给 train_kwargs 字典中的 "gpus" 键。
        _, devs = configure_device(device)  # 换句话说，它是用于指定训练过程中使用的 GPU 设备的代码。
        train_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        from pytorch_lightning.plugins import DDPPlugin

        train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
        loader_kws["persistent_workers"] = True  # 原参数是true 默认是false，但是我设置了num worker为0 ，如果设置了大于0，那么就必须是true
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8

    # Half precision training
    if config.training.fp16:
        train_kwargs["precision"] = 16  # 训练精度选择（占用内存和计算速度）

    # Loader
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config.training.workers,  # num_workers = 2  2个worker的加载速度更快，可以这样理解
        pin_memory=True,    # 按照官方的建议[3]是你默认设置pin_memory为True就对了
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)   # 这里pytorch_lightning 这个包把训练的函数整合了，直接调用trainer就行，可以点开查看含内部函数
    trainer.fit(vae, train_dataloader=loader)


if __name__ == "__main__":
    train()
