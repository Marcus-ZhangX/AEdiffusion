import torch
import torch.nn as nn


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t).float()
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDPMv2(nn.Module):
    def __init__(
        self,
        decoder,
        beta_1=1e-4,
        beta_2=0.02,
        T=1000,
        var_type="fixedlarge",
    ):
        super().__init__()
        self.decoder = decoder
        self.T = T
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.var_type = var_type

        # Main constants
        self.register_buffer(
            "betas", torch.linspace(self.beta_1, self.beta_2, steps=self.T).double()
        )
        dev = self.betas.device
        alphas = 1.0 - self.betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_shifted = torch.cat([torch.tensor([1.0], device=dev), alpha_bar[:-1]])

        assert alpha_bar_shifted.shape == torch.Size(
            [
                self.T,
            ]
        )

        # Auxillary consts
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("minus_sqrt_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alpha_bar))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alpha_bar - 1)
        )

        # Posterior q(x_t-1|x_t,x_0,t) covariance of the forward process
        self.register_buffer(
            "post_variance", self.betas * (1.0 - alpha_bar_shifted) / (1.0 - alpha_bar)
        )
        # Clipping because post_variance is 0 before the chain starts
        self.register_buffer(
            "post_log_variance_clipped",
            torch.log(
                torch.cat(
                    [
                        torch.tensor([self.post_variance[1]], device=dev),
                        self.post_variance[1:],
                    ]
                )
            ),
        )

        # q(x_t-1 | x_t, x_0) mean coefficients
        self.register_buffer(
            "post_coeff_1",
            self.betas * torch.sqrt(alpha_bar_shifted) / (1.0 - alpha_bar),
        )
        self.register_buffer(
            "post_coeff_2",
            torch.sqrt(alphas) * (1 - alpha_bar_shifted) / (1 - alpha_bar),
        )
        self.register_buffer(
            "post_coeff_3",
            1 - self.post_coeff_2,
        )

    def _predict_xstart_from_eps(self, x_t, t, eps, cond=None):
        assert x_t.shape == eps.shape
        x_hat = 0 if cond is None else cond
        assert x_hat.shape == x_t.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_hat  # 这一行DDPMv2和DDPM不一样
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def get_posterior_mean_covariance(
        self, x_t, t, clip_denoised=True, cond=None, z_vae=None, guidance_weight=0.0
    ):
        B = x_t.size(0)
        t_ = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        assert t_.shape == torch.Size(
            [
                B,
            ]
        )
        x_hat = 0 if cond is None else cond

        # Compute updated score
        if guidance_weight == 0:  # 如果guidance_weight为零
            eps_score = self.decoder(x_t, t_, low_res=cond, z=z_vae)
        else:  # 如果guidance_weight不为零
            eps_score = (1 + guidance_weight) * self.decoder(  # 这是一种根据guidance_weight对两个解码结果进行加权的方法。
                x_t, t_, low_res=cond, z=z_vae
            ) - guidance_weight * self.decoder(
                x_t,
                t_,
                low_res=torch.zeros_like(cond),
                z=torch.zeros_like(z_vae) if z_vae is not None else None,
            )
        # Generate the reconstruction from x_t
        x_recons = self._predict_xstart_from_eps(x_t, t_, eps_score, cond=cond)

        # Clip
        if clip_denoised:
            x_recons.clamp_(-1.0, 1.0)

        # Compute posterior mean from the reconstruction
        post_mean = (
            extract(self.post_coeff_1, t_, x_t.shape) * x_recons
            + extract(self.post_coeff_2, t_, x_t.shape) * x_t  # 这一行DDPMv2和DDPM不一样
            + extract(self.post_coeff_3, t_, x_t.shape) * x_hat
        )

        # Extract posterior variance
        p_variance, p_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            "fixedlarge": (
                torch.cat(
                    [
                        torch.tensor([self.post_variance[1]], device=x_t.device),
                        self.betas[1:],
                    ]
                ),
                torch.log(
                    torch.cat(
                        [
                            torch.tensor([self.post_variance[1]], device=x_t.device),
                            self.betas[1:],
                        ]
                    )
                ),
            ),
            "fixedsmall": (
                self.post_variance,
                self.post_log_variance_clipped,
            ),
        }[self.var_type]
        post_variance = extract(p_variance, t_, x_t.shape)
        post_log_variance = extract(p_log_variance, t_, x_t.shape)
        return post_mean, post_variance, post_log_variance

    def sample(  # 非spaced diffussion 的计算，在条件推理的时候会传入这个函数
        self,
        x_t,
        cond=None,
        z_vae=None,
        n_steps=None,
        guidance_weight=0.0,
        checkpoints=[],   # 参数checkpoints是一个整数列表，表示在哪些步骤处保存采样结果，默认为空列表。
        ddpm_latents=None,
    ):
        # The sampling process goes here. This sampler also supports truncated sampling.
        # For spaced sampling (used in DDIM etc.) see SpacedDiffusion model in spaced_diff.py
        x = x_t
        B, *_ = x_t.shape
        sample_dict = {}

        if ddpm_latents is not None:  # 实际上是None，用不到
            ddpm_latents = ddpm_latents.to(x_t.device)

        num_steps = self.T if n_steps is None else n_steps
        checkpoints = [num_steps] if checkpoints == [] else checkpoints
        for idx, t in enumerate(reversed(range(0, num_steps))):  # 对于从0到num_steps的逆序范围的每个索引idx和值t，执行下面的代码块。
            z = (
                torch.randn_like(x_t)
                if ddpm_latents is None
                else torch.stack([ddpm_latents[idx]] * B)
            )
            (
                post_mean,
                post_variance,  # 计算后验均值（post_mean）、后验方差（post_variance）和后验对数方差（post_log_variance），用于生成采样。
                post_log_variance,
            ) = self.get_posterior_mean_covariance(
                x,
                t,
                cond=cond,
                z_vae=z_vae,
                guidance_weight=guidance_weight,
            )
            nonzero_mask = (  # 创建一个非零掩码张量nonzero_mask，其中元素为1表示t不等于0，元素为0表示t等于0。该掩码用于在t等于0时不添加噪声。
                torch.tensor(t != 0, device=x.device)
                .float()
                .view(-1, *([1] * (len(x_t.shape) - 1)))
            )  # no noise when t == 0

            # Langevin step!
            x = post_mean + nonzero_mask * torch.exp(0.5 * post_log_variance) * z

            if t == 0:   # 这一行DDPMv2和DDPM不一样
                # NOTE: In the final step we remove the vae reconstruction bias
                # added to the images as it degrades quality
                x -= cond  # 将条件张量cond从采样结果x中减去。这一步是为了在最后一步移除添加到图像的VAE重构偏差，以提高采样质量。

            # Add results
            if idx + 1 in checkpoints:
                sample_dict[str(idx + 1)] = x  # 将采样结果x添加到sample_dict字典中，以字符串形式使用idx + 1作为键。
        return sample_dict  # 返回采样结果

    def compute_noisy_input(self, x_start, eps, t, low_res=None):
        assert eps.shape == x_start.shape
        x_hat = 0 if low_res is None else low_res
        # Samples the noisy input x_t ~ N(x_t|x_0) in the forward process
        return (
            x_start * extract(self.sqrt_alpha_bar, t, x_start.shape)
            + x_hat
            + eps * extract(self.minus_sqrt_alpha_bar, t, x_start.shape)
        )

    def forward(self, x, eps, t, low_res=None, z=None):
        # Predict noise
        x_t = self.compute_noisy_input(x, eps, t, low_res=low_res)
        return self.decoder(x_t, t, low_res=low_res, z=z)
