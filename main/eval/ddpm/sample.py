# Helper script to sample from an unconditional DDPM model
# Add project directory to sys.path
import os
import sys

# p = os.path.join(os.path.abspath("."), "main")
# sys.path.insert(1, p)

import copy
sys.path.append(r"E:\Coding_path\DiffuseVAE\main")
import hydra
import pytorch_lightning as pl
from datasets.latent import UncondLatentDataset
from models.callbacks import ImageWriter
from models.diffusion import DDPM, DDPMv2, DDPMWrapper, UNetModel
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from util import configure_device


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(version_base=None, config_path=r"E:\Coding_path\DiffuseVAE\main\configs")
def sample(config):
    # Seed and setup
    config = config.dataset.ddpm
    seed_everything(config.evaluation.seed, workers=True)

    # Ensure unconditional DDPM mode
    assert config.evaluation.type == "uncond"    # 这里根据训练时采用的模型改参数

    batch_size = config.evaluation.batch_size  # 8
    n_steps = config.evaluation.n_steps  # 1000
    n_samples = config.evaluation.n_samples  # 10
    image_size = config.data.image_size  # 128

    # Load pretrained wrapper
    attn_resolutions = __parse_str(config.model.attn_resolutions)  # "32,16,8"
    dim_mults = __parse_str(config.model.dim_mults)  # 一致
    decoder = UNetModel(           # 将unet模型赋给decoder
        in_channels=config.data.n_channels,  # 1
        model_channels=config.model.dim,  # 128
        out_channels=1,
        num_res_blocks=config.model.n_residual,  # 2
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=config.model.dropout,  # 0.0
        num_heads=config.model.n_heads,  # 1
    )

    ema_decoder = copy.deepcopy(decoder)  #  让的decoder变为ema_decoder。使用copy模块进行深拷贝操作。它将decoder对象进行深拷贝，并将结果赋值给变量ema_decoder。ema_decoder被创建为decoder的副本，这样两个对象之间的修改就可以独立进行，而不会相互干扰。这在需要对decoder进行修改但又需要保留原始状态的情况下非常有用。
    decoder.eval()  # 转换为推理模式
    ema_decoder.eval() # 转换为推理模式

    online_ddpm = DDPM(    # 如果选择的form2 就变为 DDPMv2
        decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
        var_type=config.evaluation.variance,
    )
    target_ddpm = DDPM(  # 如果选择的form2 就变为 DDPMv2
        ema_decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
        var_type=config.evaluation.variance,
    )

    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        config.evaluation.chkpt_path,  # 模型保存路径
        online_network=online_ddpm,
        target_network=target_ddpm,
        vae=None,
        conditional=False,
        pred_steps=n_steps,
        eval_mode="sample",  # sample模式
        resample_strategy=config.evaluation.resample_strategy,  # "spaced"
        skip_strategy=config.evaluation.skip_strategy,  # "uniform"
        sample_method=config.evaluation.sample_method,  # "ddim"
        sample_from=config.evaluation.sample_from,  # "target"
        data_norm=config.data.norm,  # False
        strict=False,  # 这个语句就是指忽略掉模型和参数文件中不匹配的参数
    )

    # Create predict dataset of latents
    z_dataset = UncondLatentDataset(
        (n_samples, 1, image_size, image_size),
    )

    # Setup devices
    test_kwargs = {}
    loader_kws = {}
    device = config.evaluation.device
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        test_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        loader_kws["persistent_workers"] = False
    elif device == "tpu":
        test_kwargs["tpu_cores"] = 8

    # Predict loader
    val_loader = DataLoader(
        z_dataset,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        num_workers=config.evaluation.workers,
        **loader_kws,
    )

    # Predict trainer
    write_callback = ImageWriter(
        config.evaluation.save_path,  # 生成的图片的保存路径
        "batch",
        n_steps=n_steps,
        eval_mode="sample",
        conditional=False,
        sample_prefix=config.evaluation.sample_prefix,
        save_mode=config.evaluation.save_mode,  # save_as_np
    )

    test_kwargs["callbacks"] = [write_callback]
    test_kwargs["default_root_dir"] = config.evaluation.save_path
    trainer = pl.Trainer(**test_kwargs)
    trainer.predict(ddpm_wrapper, val_loader)  # 这一步是在做推理，相当于ddpm_wrapper是模型，而val_loader是输入的参数


if __name__ == "__main__":
    sample()
    # 开始在这里写matlab相关的操作
    # z = torch.randn(1, 200)
    # output = G.forward(z.cuda())
    # img = output.detach().permute(0, 2, 3, 1).view(128, 128).cpu().numpy()
    # plt.contourf(img)
    # plt.colorbar()

