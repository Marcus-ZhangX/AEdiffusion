import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.diffusion.spaced_diff import SpacedDiffusion
from models.diffusion.spaced_diff_form2 import SpacedDiffusionForm2
from models.diffusion.ddpm_form2 import DDPMv2
from util import space_timesteps


class DDPMWrapper(pl.LightningModule):
    def __init__(
        self,
        online_network,
        target_network,
        vae,
        lr=2e-5,  # 一致
        cfd_rate=0.0,  # 一致
        n_anneal_steps=0,  # 一致
        loss="l2",  # 这里和配置中给的参数不一致，配置中给的参数是l2
        grad_clip_val=1.0,  # 对梯度进行裁剪，防止梯度爆炸
        sample_from="target",  # 如果self.sample_from等于字符串"target"，则选择self.target_network；否则选择self.online_network。
        resample_strategy="spaced",
        skip_strategy="uniform",
        sample_method="ddim",  # 原ddpm  # 采样策略有两种ddim，ddpm
        conditional=False,  # 原 True
        eval_mode="sample",  # 当转化为评估模式时，是sample还是reconstruction重建？这里选择sample
        pred_steps=None,
        pred_checkpoints=[],
        temp=1.0,
        guidance_weight=0.0,
        z_cond=False,
        ddpm_latents=None,
    ):
        super().__init__()
        assert loss in ["l1", "l2"]
        assert eval_mode in ["sample", "recons"]
        assert resample_strategy in ["truncated", "spaced"]
        assert sample_method in ["ddpm", "ddim"]
        assert skip_strategy in ["uniform", "quad"]

        self.z_cond = z_cond
        self.online_network = online_network
        self.target_network = target_network
        self.vae = vae
        self.cfd_rate = cfd_rate

        # Training arguments
        self.criterion = nn.MSELoss(reduction="mean") if loss == "l2" else nn.L1Loss()  # 如果选择l2 loss 那么就是mseloss，如果不是,那就是l1 loss
        self.lr = lr
        self.grad_clip_val = grad_clip_val
        self.n_anneal_steps = n_anneal_steps

        # Evaluation arguments
        self.sample_from = sample_from
        self.conditional = conditional
        self.sample_method = sample_method
        self.resample_strategy = resample_strategy
        self.skip_strategy = skip_strategy
        self.eval_mode = eval_mode
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps
        self.pred_checkpoints = pred_checkpoints
        self.temp = temp
        self.guidance_weight = guidance_weight
        self.ddpm_latents = ddpm_latents

        # Disable automatic optimization
        self.automatic_optimization = False

        # Spaced Diffusion (for spaced re-sampling)
        self.spaced_diffusion = None

    def forward(
        self,
        x,  # 给的x是一个128*128的噪声图像
        cond=None,
        z=None,   # z 对应了z_vae。 z也是DDPMWrapper函数外面传进来的
        n_steps=None,
        ddpm_latents=None,
        checkpoints=[],
    ):
        sample_nw = (  # 这个是用于非spaced diffussion 的计算
            self.target_network if self.sample_from == "target" else self.online_network  # 实际上这个参数选target或者online都没问题，两个网络结构是复制的
        )  # 如果self.sample_from等于字符串"target"，则选择self.target_network；否则选择self.online_network。

        spaced_nw = (  # 这个是用于spaced diffussion 的计算
            SpacedDiffusionForm2  # 如果选择了form2，那就spaced_nw =SpacedDiffusionForm2类，如果选择了form1，那就spaced_nw =SpacedDiffusion类
            if isinstance(self.online_network, DDPMv2)  # 检查self.online_network是否是DDPMv2类的实例
            else SpacedDiffusion
        )
        # For spaced resampling
        if self.resample_strategy == "spaced":
            num_steps = n_steps if n_steps is not None else self.online_network.T  # 这里的online_network.T 最初来自于 n_timesteps: 1000
            indices = space_timesteps(sample_nw.T, num_steps, type=self.skip_strategy)
            if self.spaced_diffusion is None:
                self.spaced_diffusion = spaced_nw(sample_nw, indices).to(x.device)

            if self.sample_method == "ddim":  # 如果选择ddim的话，那么就走这里
                return self.spaced_diffusion.ddim_sample(  # 在spaced_diffusion（也就是spaced_nw，也就是SpacedDiffusionForm2）里面有一个函数叫做ddim_sample
                    x,
                    cond=cond,
                    z_vae=z,
                    guidance_weight=self.guidance_weight,  # 0.0
                    checkpoints=checkpoints,
                )
            return self.spaced_diffusion(  # 如果没有选择ddim的话，那么就走这里
                x,
                cond=cond,
                z_vae=z,
                guidance_weight=self.guidance_weight,  # 0.0
                checkpoints=checkpoints,
                ddpm_latents=ddpm_latents,
            )

        # For truncated resampling
        if self.sample_method == "ddim":
            raise ValueError("DDIM is only supported for spaced sampling")  # 必须设置了spaced sampling，才能使用DDIM

        return sample_nw.sample(   # 如果没有选择spaced的话，非spaced diffussion 的计算,走这里，最终会传入到ddpm_form2文件中的sample函数中去
            x,
            cond=cond,
            z_vae=z,  # 将z传递给z_vae
            n_steps=n_steps,
            guidance_weight=self.guidance_weight,  # 0.0
            checkpoints=checkpoints,
            ddpm_latents=ddpm_latents,
        )
        # ddpm wrapper 的forward过程结束

    def training_step(self, batch, batch_idx):   # 这个函数用来训练的
        # Optimizers
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()

        cond = None
        z = None
        if self.conditional:  # 设置为了True
            x = batch
            with torch.no_grad():
                mu, logvar = self.vae.encode(x * 0.5 + 0.5)
                z = self.vae.reparameterize(mu, logvar)
                # print(z)
                cond = self.vae.decode(z)  # 通过decoder把z解码为 128*128  然后再将其作为条件cond
                cond = 2 * cond - 1

            # Set the conditioning signal based on clf-free guidance rate
            if torch.rand(1)[0] < self.cfd_rate:  # self.cfd_rate设置为了0.5
                cond = torch.zeros_like(x)
                z = torch.zeros_like(z)
        else:
            x = batch     # 如果不是条件生成，cond为false。那么就直接用的 batch传入的数据集

        # Sample timepoints
        t = torch.randint(   # 成一个大小为(x.size(0),)的随机整数张量t，其中的每个随机数的范围是从0到self.online_network.T(即为时间步长）
            0, self.online_network.T, size=(x.size(0),), device=self.device
        )

        # Sample noise
        eps = torch.randn_like(x)

        # Predict noise
        eps_pred = self.online_network(
            x, eps, t, low_res=cond, z=z.squeeze() if self.z_cond else None  # 如果 self.z_cond 为 True，则 z 会通过 z.squeeze() 进行压缩，将维度为1的维度去除，然后作为条件传递给 self.online_network 方法。否则，如果 self.z_cond 为 False，则不会使用 z 作为条件，而是将其设为 None。
        )   # low_res 是一个条件低分辨率图像（或特征）的张量，用于条件生成模型。


        # Compute loss
        loss = self.criterion(eps, eps_pred)

        # Clip gradients and Optimize
        optim.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.online_network.decoder.parameters(), self.grad_clip_val  # 对梯度进行裁剪，防止梯度爆炸
        )
        optim.step()

        # Scheduler step
        lr_sched.step()  # 用于调节学习率的，学习率会逐渐降低

        self.log("loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):  # 这个函数是用来生成的
        z = None
        if not self.conditional:  # 如果不是条件生成，即为无条件生成
            if self.guidance_weight != 0.0:  # 如果guidance_weight不等于0
                raise ValueError(
                    "Guidance weight cannot be non-zero when using unconditional DDPM"
                )
            x_t = batch
            return self(
                x_t,
                cond=None,
                z=None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=None,
            )

        if self.eval_mode == "sample":  # 用到了z变量

            x_t, z = batch
            recons = self.vae(z)
            recons = 2 * recons - 1

            # Initial temperature scaling
            x_t = x_t * self.temp

            # Formulation-2 initial latent
            if isinstance(self.online_network, DDPMv2):
                x_t = recons + self.temp * torch.randn_like(recons)
        else:  # 没用到z
            img = batch  # 如果是reconstruction函数，走这里
            recons = self.vae.forward_recons(img * 0.5 + 0.5)
            recons = 2 * recons - 1

            # DDPM encoder
            x_t = self.online_network.compute_noisy_input(
                img,
                torch.randn_like(img),
                torch.tensor(
                    [self.online_network.T - 1] * img.size(0), device=img.device
                ),
            )

            if isinstance(self.online_network, DDPMv2):
                x_t += recons
        # 这里我重新改写了，在不同的采样模式的情况下（主要就是sample模式和recons的模式）
        # 返回的值应该是不同的，sample情况下，返回值遵循原始代码
        # recons情况下，返回值中z直接命名为None,因为本来在recons情况下z也没用到，所以z在做squeeze操作时会报错“变量未声明”
        # 我直接写了一个if语句，使得在不同的情况下返回不同的值，防止报错
        if self.eval_mode == "sample":
            return (
                self(
                    x_t,
                    cond=recons,
                    z=z.squeeze() if self.z_cond else None,
                    n_steps=self.pred_steps,
                    checkpoints=self.pred_checkpoints,
                    ddpm_latents=self.ddpm_latents,
                ),
                recons,
            )
        else:
            return (
                self(
                    x_t,
                    cond=recons,
                    z=None,
                    n_steps=self.pred_steps,
                    checkpoints=self.pred_checkpoints,
                    ddpm_latents=self.ddpm_latents,
                ),
                recons,
            )


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.online_network.decoder.parameters(), lr=self.lr
        )

        # Define the LR scheduler (As in Ho et al.)
        if self.n_anneal_steps == 0:  # 如果n_anneal_steps等于0，则lr_lambda函数返回1.0，即学习率始终不变
            lr_lambda = lambda step: 1.0
        else:
            lr_lambda = lambda step: min(step / self.n_anneal_steps, 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
            },
        }
