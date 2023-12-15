import os
from typing import Sequence, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor
from torch.nn import Module
# from util import save_as_images, save_as_np, normalize, convert_to_np
from util import normalize, convert_to_np
import numpy as np

class EMAWeightUpdate(Callback):
    """EMA weight update
    Your model should have:
        - ``self.online_network``
        - ``self.target_network``
    Updates the target_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.
    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training step
    Example::
        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...
        trainer = Trainer(callbacks=[EMAWeightUpdate()])
    """

    def __init__(self, tau: float = 0.9999):
        """
        Args:
            tau: EMA decay rate
        """
        super().__init__()
        self.tau = tau

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # get networks
        online_net = pl_module.online_network.decoder
        target_net = pl_module.target_network.decoder

        # update weights
        self.update_weights(online_net, target_net)

    def update_weights(
        self, online_net: Union[Module, Tensor], target_net: Union[Module, Tensor]
    ) -> None:
        # apply MA weight update
        with torch.no_grad():
            for targ, src in zip(target_net.parameters(), online_net.parameters()):
                targ.mul_(self.tau).add_(src, alpha=1 - self.tau)


class ImageWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        write_interval,
        compare=False,
        n_steps=None,
        eval_mode="sample",
        conditional=True,
        sample_prefix="",
        save_vae=False,
        # save_mode="save_as_np",
        is_norm=True,
        # Ne=None,
    ):
        super().__init__(write_interval)
        assert eval_mode in ["sample", "recons"]
        self.output_dir = output_dir
        self.compare = compare
        self.n_steps = 1000 if n_steps is None else n_steps
        self.eval_mode = eval_mode
        self.conditional = conditional
        self.sample_prefix = sample_prefix
        self.save_vae = save_vae
        self.is_norm = is_norm
        # self.save_fn = save_as_images if save_mode == "image" else save_as_np
        # self.Ne = Ne  # ensemble 数量
    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        global obj  # 原来是没有这句的。我添加了全局语句
        rank = pl_module.global_rank  # 在分布式训练中，当使用多个设备或计算节点进行训练时，每个设备或节点都会被分配一个唯一的排名。这个排名可以用来区分不同的设备或节点，并根据需要执行特定的操作。
        if self.conditional:
            ddpm_samples_dict, vae_samples = prediction

            # if self.save_vae:  # 如果save_vae为true的话，会保存vae_samples
            #     vae_samples = vae_samples.cpu()
            #     vae_save_path = os.path.join(self.output_dir, "vae")
            #     os.makedirs(vae_save_path, exist_ok=True)
            #     self.save_fn(  # save_fn=save_as_np
            #         vae_samples,
            #         file_name=os.path.join(
            #             vae_save_path,
            #             f"output_vae_{self.sample_prefix}_{rank}_{batch_idx}",  # 这里是使用VAE ，sample输出的图像的命名规则
            #         ),
            #         denorm=self.is_norm,
            #     )
        else:
            ddpm_samples_dict = prediction
        #print(type(ddpm_samples_dict))   结果 <class 'dict'> 表示 ddpm_samples_dict 的类型是字典（dict）

        # Write output images
        # NOTE: We need to use gpu rank during saving to prevent
        # processes from overwriting images
        denorm = self.is_norm

        for k, ddpm_samples in ddpm_samples_dict.items():  # 将推理得到的ddpm_samples_dict，一个一个写进文件夹
            ddpm_samples = ddpm_samples.cpu()
            # print(ddpm_samples)
            # print(type(ddpm_samples))
            # Setup dirs
            base_save_path = os.path.join(self.output_dir, k)  # k=1000 将self.output_dir和k两个路径进行拼接，形成一个新的路径。
            img_save_path = os.path.join(base_save_path, "images")
            os.makedirs(img_save_path, exist_ok=True)
            file_name = os.path.join(
                img_save_path, f"output_{self.sample_prefix}_{rank}_{batch_idx}"  # 这里是使用ddpm ，sample输出的图像的命名规则
            ),
            # Save

            if denorm:
                obj = normalize(ddpm_samples)  # 这里是如果前面yaml文件中设置了norm: True，那么这里就需要将生成的图片归一化
            obj_list = convert_to_np(obj)

            file_name = str(file_name[0])  # 转化为字符串类型，不然file_name作为元组，会报错

            for i, out in enumerate(obj_list):
                current_file_name = file_name + "_%d.npy" % i
                np.save(current_file_name, out)


            # self.save_fn(  # save_fn=save_as_np
            #     ddpm_samples,
            #     file_name=os.path.join(
            #         img_save_path, f"output_{self.sample_prefix }_{rank}_{batch_idx}"   # 这里是使用ddpm ，sample输出的图像的命名规则
            #     ),
            #     denorm=self.is_norm,  # True
            #
            # )

        # FIXME: This is currently broken. Separate this from the core logic
        # into a new function. Uncomment when ready!
        # if self.compare:
        #     # Save comparisons
        #     (_, img_samples), _ = batch
        #     img_samples = normalize(img_samples).cpu()
        #     iter_ = vae_samples if self.eval_mode == "sample" else img_samples
        #     for idx, (ddpm_pred, pred) in enumerate(zip(ddpm_samples, iter_)):
        #         samples = {
        #             "VAE" if self.eval_mode == "sample" else "Original": pred,
        #             "DDPM": ddpm_pred,
        #         }
        #         compare_samples(
        #             samples,
        #             save_path=os.path.join(
        #                 self.comp_save_path, f"compare_form1_{rank}_{idx}.png"
        #             ),
        #         )
