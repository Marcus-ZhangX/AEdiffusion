import copy
import logging
import os
import sys

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader

print(os.getcwd())  # 显示当前路径
sys.path.append(r"E:\Coding_path\DiffuseVAE\main")

from models.callbacks import EMAWeightUpdate
from models.diffusion import DDPM, DDPMv2, DDPMWrapper, SuperResModel, UNetModel
from models.vae import VAE
from util import configure_device, get_dataset

logger = logging.getLogger(__name__)
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"  # 修改环境变量 防止报错

def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]
# 它接受一个字符串s作为参数。该函数首先使用逗号作为分隔符将输入字符串s分割成一个由多个子字符串组成的列表。
# 然后，函数遍历这个列表，并尝试将每个子字符串转换成整数类型。如果子字符串不是空字符串或者None，
# 那么就返回包含所有转换后的整数的列表。

@hydra.main(version_base=None,config_path="configs")
def train(config):
    # Get config and setup
    config = config.dataset.ddpm   # 这里的dataset就是指celebamaskhq128/train
    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    seed_everything(config.training.seed, workers=True)  # 这里设置seed为0

    # Dataset
    root = config.data.root   # dataset路径
    d_type = config.data.name  # d_type就是数据的名字
    image_size = config.data.image_size  # 128
    dataset = get_dataset(
        d_type, root, image_size, norm=config.data.norm, flip=config.data.hflip
    )
    N = len(dataset)
    batch_size = config.training.batch_size
    batch_size = min(N, batch_size)  # 但是，如果数据集的大小小于指定的批次大小，则 batch_size 被设置为数据集的大小。这确保了批次大小不会大于数据集的大小，否则会在训练过程中出现错误。

    # Model
    lr = config.training.lr  # 2e-5
    attn_resolutions = __parse_str(config.model.attn_resolutions)  # 在哪个地方添加attention
    dim_mults = __parse_str(config.model.dim_mults)
    ddpm_type = config.training.type

    # Use the superres model for conditional training
    decoder_cls = UNetModel if ddpm_type == "uncond" else SuperResModel  # if ddpm_type == "uncond" else SuperResModel   # 如果ddpm_type为字符串"uncond"，则decoder_cls指向UNetModel类；否则，decoder_cls指向SuperResModel类。
    decoder = decoder_cls(
        in_channels=config.data.n_channels,  # 为1
        model_channels=config.model.dim,  # Unet的起始通道数
        out_channels=1,  # 原为3
        num_res_blocks=config.model.n_residual,  # res block 的数量
        attention_resolutions=attn_resolutions,  # 在哪个地方添加attention
        channel_mult=dim_mults,   # 网络结构参数
        use_checkpoint=False,  # 没用到的参数
        dropout=config.model.dropout,  # drop out 取得是0.1,按照原始论文来的
        num_heads=config.model.n_heads,   # 这里的取值为1和open ai中的取值相同  the number of attention heads in each attention layer
        z_dim=config.training.z_dim,  # 100
        use_scale_shift_norm=config.training.z_cond,  # z_cond: False
        use_z=config.training.z_cond,   # # z_cond: True
    )

    # EMA parameters are non-trainable
    ema_decoder = copy.deepcopy(decoder)  # 这段代码的作用是创建decoder对象的深度副本并将其赋值给ema_decoder变量。深度副本意味着复制的是整个对象，包括其属性和子对象（如果有）。这种做法是为了防止修改decoder时影响到ema_decoder。
    for p in ema_decoder.parameters():
        p.requires_grad = False  # # 这段代码的意思是将 ema_decoder 中所有的参数设置为不需要梯度计算。

    ddpm_cls = DDPMv2 if ddpm_type == "form2" else DDPM  # 如果ddpm_type的值等于字符串"form2"，那么ddpm_cls将存储DDPMv2类，否则它将存储DDPM类。
# 这段代码是根据ddpm_type变量的值选择要使用哪个DDPM实现。如果ddpm_type是"form2"，那么会使用更新的DDPMv2实现，否则将使用原始的DDPM实现。
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )
    vae = VAE.load_from_checkpoint(
        config.training.vae_chkpt_path,  # 已经训练好的vae的checkpoint路径
        input_res=image_size,  # 128
    )
    vae.eval()  # 转化模式

    for p in vae.parameters():
        p.requires_grad = False   # 这段代码的意思是将 vae 中所有的参数设置为不需要梯度计算。为了加快模型的训练，防止权重更新，防止过拟合：

    assert isinstance(online_ddpm, ddpm_cls)
    assert isinstance(target_ddpm, ddpm_cls)
    logger.info(f"Using DDPM with type: {ddpm_cls} and data norm: {config.data.norm}")

    ddpm_wrapper = DDPMWrapper(   # 将DDPM的各种函数进行包装，可以点进DDPMwrapper中查看
        online_ddpm,
        target_ddpm,
        vae,
        lr=lr,
        cfd_rate=config.training.cfd_rate,  # 这个参数原始设置为0.5。这个参数只有在self condiitonal 参数设置为True的前提下才会起作用，详见DDPMwrapper函数中的具体代码
        n_anneal_steps=config.training.n_anneal_steps,  # 这里设置n_anneal_steps=0，即为学习率始终不变，设置为5000，lr_lambda函数会将当前步数step除以n_anneal_steps得到一个比例值，该值限制在[0,1]范围内，然后用该比例值作为学习率的缩放因子，从而实现学习率的逐渐降低
        loss=config.training.loss,  # 这里选择的是l2 loss
        conditional=False if ddpm_type == "uncond" else True,
        grad_clip_val=config.training.grad_clip,  # 对梯度进行裁剪，防止梯度爆炸
        z_cond=config.training.z_cond,  # z_cond: True
    )

    # Trainer
    train_kwargs = {}
    restore_path = config.training.restore_path
    if restore_path != "":
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    # Setup callbacks
    results_dir = config.training.results_dir   # 输出结果保存处
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"ddpm_{config.training.chkpt_prefix}" + "-{epoch:02d}-{loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,
        save_on_train_epoch_end=True,
        save_top_k=-1,  # 当此参数被设置为-1时，所有的checkpoints都会被保存，如果不设置这个参数，后面的checkpoints会将前面的覆盖掉
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.training.epochs
    train_kwargs["log_every_n_steps"] = config.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback]

    if config.training.use_ema:  # 如果use_ema为True的话
        ema_callback = EMAWeightUpdate(tau=config.training.ema_decay)  # ema_decay: 0.999
        train_kwargs["callbacks"].append(ema_callback)

    device = config.training.device  # 选择设备
    loader_kws = {}
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        train_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        from pytorch_lightning.plugins import DDPPlugin, DDPSpawnPlugin

        train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
        loader_kws["persistent_workers"] = True  # 原参数是true 默认是false，但是我设置了num worker为0 ，如果设置了大于0，那么就必须是true
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8

    # Half precision training
    if config.training.fp16:
        train_kwargs["precision"] = 16  # # 训练精度选择（占用内存和计算速度）

    # Loader
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    # Gradient Clipping by global norm (0 value indicates no clipping) (as in Ho et al.)
    # train_kwargs["gradient_clip_val"] = config.training.grad_clip

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(ddpm_wrapper, train_dataloader=loader)


if __name__ == "__main__":
    train()
