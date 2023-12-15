# Helper script to generate reconstructions from a conditional DDPM model
# Add project directory to sys.path
import os
import sys

print(os.getcwd())  # 显示当前路径
sys.path.append(r"E:\Coding_path\DiffuseVAE\main")

import copy
import hydra
import pytorch_lightning as pl
import torch
from models.callbacks import ImageWriter
from models.diffusion import DDPM, DDPMv2, DDPMWrapper, SuperResModel
from models.vae import VAE
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from util import configure_device, get_dataset


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(version_base=None, config_path=r"E:\Coding_path\DiffuseVAE\main\configs")
def generate_recons(config):
    config_ddpm = config.dataset.ddpm
    config_vae = config.dataset.vae
    seed_everything(config_ddpm.evaluation.seed, workers=True)  # 默认 0

    batch_size = config_ddpm.evaluation.batch_size  # 8
    n_steps = config_ddpm.evaluation.n_steps   # 50  这是我要进行实验的变量对象啦
    n_samples = config_ddpm.evaluation.n_samples  # 这个数量我暂时定一个500吧
    image_size = config_ddpm.data.image_size  # 128
    ddpm_latent_path = config_ddpm.data.ddpm_latent_path  # 这里的ddpm_latent_path应当为 “ ”，为空的话后面才能使得share_ddpm_latent=True
    ddpm_latents = torch.load(ddpm_latent_path) if ddpm_latent_path != "" else None

    # Load pretrained VAE
    vae = VAE.load_from_checkpoint(
        config_vae.evaluation.chkpt_path,
        input_res=image_size,
    )
    vae.eval()

    # Load pretrained wrapper
    attn_resolutions = __parse_str(config_ddpm.model.attn_resolutions)  # "32,16,8"
    dim_mults = __parse_str(config_ddpm.model.dim_mults)  # "1,2,2,3,4"
    decoder = SuperResModel(
        in_channels=config_ddpm.data.n_channels,   # 1
        model_channels=config_ddpm.model.dim,  # 128
        out_channels=1,
        num_res_blocks=config_ddpm.model.n_residual,  # 2
        attention_resolutions=attn_resolutions,  # "32,16,8"
        channel_mult=dim_mults,  # "1,2,2,3,4"
        use_checkpoint=False,
        dropout=config_ddpm.model.dropout,  # 0.0
        num_heads=config_ddpm.model.n_heads,   # 1
        z_dim=config_ddpm.evaluation.z_dim,  # 100
        use_scale_shift_norm=config_ddpm.evaluation.z_cond,  # 理论上来讲是True
        use_z=config_ddpm.evaluation.z_cond,  # 理论上来讲是True
    )

    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    ddpm_cls = DDPMv2 if config_ddpm.evaluation.type == "form2" else DDPM
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )

    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        config_ddpm.evaluation.chkpt_path,  # 模型保存路径
        online_network=online_ddpm,
        target_network=target_ddpm,
        vae=vae,                            # 在sample方法里，这里是none
        conditional=True,
        pred_steps=n_steps,   # 这里的n_steps是可以变化的，对应推理过程中的步长，可以与训练过程中的步长不相等
        eval_mode="recons",   # 这里选择的是reconstruction，那么对应在推理的时候操作会有不同
        resample_strategy=config_ddpm.evaluation.resample_strategy,  # "spaced"
        skip_strategy=config_ddpm.evaluation.skip_strategy,  # "uniform"
        sample_method=config_ddpm.evaluation.sample_method,  # "ddpm"
        sample_from=config_ddpm.evaluation.sample_from,  # "target"
        data_norm=config_ddpm.data.norm,  # False
        temp=config_ddpm.evaluation.temp,   # 1.0 看过代码了，这个参数应该影响不大
        guidance_weight=config_ddpm.evaluation.guidance_weight,   # 0.0  这个参数需要调参
        z_cond=config_ddpm.evaluation.z_cond,  # True
        ddpm_latents=ddpm_latents,  # 传入DDPMWrapper，传入spaced diffussion,在spaced_diff中的192行
        strict=True  # 原 True /这个语句就是指忽略掉模型和参数文件中不匹配的参数
    )

    # Dataset
    root = config_ddpm.data.root
    d_type = config_ddpm.data.name  # 这里选择"celebamaskhq"，实际上这个数据的名称的选择代表了一种处理数据集的方式，不同的名称是不同的处理方式
    image_size = config_ddpm.data.image_size  # 128
    dataset = get_dataset(
        d_type,
        root,
        image_size,
        norm=config_ddpm.data.norm,  # 和sample_cond 中一样，这个参数无所谓的
        flip=config_ddpm.data.hflip,  # 和sample_cond 中一样，这个参数无所谓的
        subsample_size=16,  # 这里直接改成我需要重建的图片的数量，取的是前16个参数
    )

    # import matplotlib.pyplot as plt
    # import numpy as np
    # root = "E:/Coding_path/DiffuseVAE/converted_TI_20000"
    # dataset = get_dataset(
    #     "celebamaskhq",
    #     root=root,
    #     image_size=128,
    #     norm=True,  # 和sample_cond 中一样，这个参数无所谓的
    #     flip=True,  # 和sample_cond 中一样，这个参数无所谓的
    #     subsample_size=2000 ,  # 这里直接改成我需要重建的图片的数量，取的是前16个参数
    # )
    # print(dataset[1].shape)  # torch.Size([1, 128, 128])
    # a = np.squeeze(dataset[100],0)
    # print(a.shape)
    # plt.imshow(a , cmap='gray')
    # plt.axis('off')
    # plt.show()

    # Setup devices
    test_kwargs = {}
    loader_kws = {}
    device = config_ddpm.evaluation.device
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        test_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        loader_kws["persistent_workers"] = True  # 默认是false，如果设置了大于0，那么就必须是true
    elif device == "tpu":
        test_kwargs["tpu_cores"] = 8

    # Predict loader
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,  # 8个，刚好一个batch
        drop_last=False,   # 不要drop last
        pin_memory=True,
        shuffle=False,   # 不打乱数据
        num_workers=config_ddpm.evaluation.workers,
        **loader_kws,
    )
    save_path = r"E:\Coding_path\DiffuseVAE\scripts\reconstruction_samples\reconstructed\Diffusevae_reconstruction"
    # Predict trainer
    write_callback = ImageWriter(
        save_path,
        "batch",
        n_steps=n_steps,
        eval_mode="recons",  # 这里和sample_cond函数当中的设置是不一样的
        conditional=True,
        sample_prefix=config_ddpm.evaluation.sample_prefix,
        # save_mode=config_ddpm.evaluation.save_mode,
        save_vae=config_ddpm.evaluation.save_vae,
        is_norm=config_ddpm.data.norm,  # # 按照save as np方式生成的图片是否采用normalize操作。选择True
    )

    test_kwargs["callbacks"] = [write_callback]
    test_kwargs["default_root_dir"] = config_ddpm.evaluation.save_path
    trainer = pl.Trainer(**test_kwargs)
    trainer.predict(ddpm_wrapper, val_loader)


if __name__ == "__main__":
    generate_recons()
