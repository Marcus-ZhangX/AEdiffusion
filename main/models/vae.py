import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def parse_layer_string(s):  # 这个函数会传入一个block_config_str,可能是encoder的也可能是decoder的，然后会根据逗号分开，然后根据x,u,d,t分出对应的参数
    layers = []
    for ss in s.split(","):
        if "x" in ss:
            # Denotes a block repetition operation
            res, num = ss.split("x")  # 将其拆分成 res 和 num 两部分，并将 res 转换为整数。接着，函数会将 (int(res), None) 这样的元组重复添加 num 次到 layers 列表中。
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif "u" in ss:
            # Denotes a resolution upsampling operation
            res, mixin = [int(a) for a in ss.split("u")]  # 则它表示分辨率上采样操作，将其拆分成 res 和 mixin 两部分，并将二者都转换为整数。接着，函数会将 (res, mixin) 这样的元组添加到 layers 列表中。
            layers.append((res, mixin))
        elif "d" in ss:
            # Denotes a resolution downsampling operation
            res, down_rate = [int(a) for a in ss.split("d")]  # 则它表示分辨率下采样操作，将其拆分成 res 和 down_rate 两部分，并将二者都转换为整数。接着，函数会将 (res, down_rate) 这样的元组添加到 layers 列表中。
            layers.append((res, down_rate))
        elif "t" in ss:
            # Denotes a resolution transition operation
            res1, res2 = [int(a) for a in ss.split("t")] # 则它表示分辨率转换操作，将其拆分成 res1 和 res2 两部分，并将二者都转换为整数。接着，函数会将 ((res1, res2), None) 这样的元组添加到 layers 列表中。
            layers.append(((res1, res2), None))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def parse_channel_string(s):   # 将传入的channel_string中的信息应用到网络结构上
    channel_config = {}
    for ss in s.split(","):  # 代码首先创建了一个空字典 channel_config，随后通过对字符串 s 进行逗号分隔得到多个子串。对于每个子串，再用冒号分隔得到两部分，第一部分为输出特征图大小，即 res 值，第二部分为该层的输入通道数 in_channels。将这两部分转换为整数类型，并将其作为键值对加入到 channel_config 字典中。最后返回完整的 channel_config 字典。
        res, in_channels = ss.split(":")
        channel_config[int(res)] = int(in_channels)
    return channel_config
'''{
    128: 64,
    64: 64,
    32: 128,
    16: 128,
    8: 256,
    4: 512,
    1: 1024
}
'''
def get_conv( # 这个函数用于创建一个卷积层
    in_dim,  # 输入通道数
    out_dim,  # 输出通道数
    kernel_size,  # 卷积核大小
    stride,  # 步长
    padding,  # 填充
    zero_bias=True,  # 是否将偏置项初始化为0
    zero_weights=False,  # 是否将权重初始化为0
    groups=1,  # 卷积分组数（groups）
):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c

#  得到 一个3*3的卷积层
def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups)

#  得到 一个1*1的卷积层
def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups)

# def fc_layer(size_in, size_out):
#     layer = nn.Sequential(
#         nn.Linear(size_in, size_out),
#         nn.BatchNorm1d(size_out),
#         nn.ReLU()
#     )
#     return layer

class ResBlock(nn.Module):
    def __init__(
        self,
        in_width,
        middle_width,
        out_width,
        down_rate=None,
        residual=False,  # 这里的residual选的是False，我想后面如果可以的话，True会不会提高它的性能
        use_3x3=True,
        zero_last=False,
    ):
        super().__init__()  # super().__init__()的作用是让子类继承父类的构造方法，并在子类自己的构造方法中使用父类的构造方法进行初始化。这样可以避免重复编写构造方法，并且确保子类的初始化过程包含了父类的所有必要步骤。
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)  # 1*1卷积
        self.c2 = (
            get_3x3(middle_width, middle_width)  # 原则上是3*3卷积
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c3 = (
            get_3x3(middle_width, middle_width)  # 原则上是3*3卷积
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last) # 1*1卷积

    def forward(self, x):  # 前向过程总是包括两个部分：特征提取和输出计算
        # print(x)
        xhat = self.c1(F.gelu(x))  # 特征提取
        xhat = self.c2(F.gelu(xhat))  # 特征提取
        xhat = self.c3(F.gelu(xhat))  # 特征提取
        xhat = self.c4(F.gelu(xhat))  # 特征提取
        # 输出计算
        out = x + xhat if self.residual else xhat  # 将最终的卷积结果xhat与输入数据x相加（或直接返回xhat，根据residual参数的取值），得到最终的输出
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate) # 如果down_rate参数不为None，则将输出进行平均池化操作并返回
        return out


class Encoder(nn.Module):  # 定义encoder,继承自nn.Module父类
    def __init__(self, block_config_str, channel_config_str):
        super().__init__()
        self.in_conv = nn.Conv2d(1, 16, 3, stride=1, padding=1, bias=False)  # 创建了一个3×3的卷积层，输入通道数为原来是3，我改为了1，输出通道数为64，步长为1，填充为1.

        block_config = parse_layer_string(block_config_str)  # 解析块配置
        channel_config = parse_channel_string(channel_config_str)   # 解析通道数配置
        blocks = []  # 这里一个block空字典
        for _, (res, down_rate) in enumerate(block_config):  # 这里只处理了res, down_rate的配置 ，也就是字母d对应的数组
            if isinstance(res, tuple):  # 检查res是否是一个元组类型。如果 res 是元组类型，则条件为真，这个代码块中的语句将被执行
                # Denotes transition to another resolution
                res1, res2 = res
                blocks.append(
                    nn.Conv2d(channel_config[res1], channel_config[res2], 1, bias=False)  # 128-64
                )
                continue

            in_channel = channel_config[res]
            use_3x3 = res > 1  # 这句的意思是如果res大于1成立，那么use_3*3布尔值为true
            blocks.append(
                ResBlock(
                    in_channel,  # in_width
                    int(0.5 * in_channel),  # middle_width
                    in_channel,  # out_width
                    down_rate=down_rate,
                    residual=True,
                    use_3x3=use_3x3,  # 若res > 1，即为true
                )
            )
        # TODO: If the training is unstable try using scaling the weights
        self.block_mod = nn.Sequential(*blocks)  # 将前面累计的blocks连起来成为一个Sequential网络结构

        # Latents   这里两个卷积层分别计算潜变量z
        self.mu = nn.Conv2d(channel_config[1], channel_config[1], 1, bias=False)
        self.logvar = nn.Conv2d(channel_config[1], channel_config[1], 1, bias=False)

    def forward(self, input):
        x = self.in_conv(input)  # 先通过一个3*3的卷积层
        x = self.block_mod(x)  # 通过很多个block
        return self.mu(x), self.logvar(x)

class Decoder(nn.Module):
    def __init__(self, input_res, block_config_str, channel_config_str):
        super().__init__()
        # self.in_conv = nn.Conv2d(50, 64, 3, stride=1, padding=1, bias=False)
        block_config = parse_layer_string(block_config_str)  # 解析块配置
        channel_config = parse_channel_string(channel_config_str)  # 解析通道数配置
        blocks = []
        for _, (res, up_rate) in enumerate(block_config):
            if isinstance(res, tuple):  # # 检查res是否是一个元组类型。如果 res 是元组类型，则条件为真，这个代码块中的语句将被执行
                # Denotes transition to another resolution
                res1, res2 = res
                blocks.append(
                    nn.Conv2d(channel_config[res1], channel_config[res2], 1, bias=False)
                )
                continue

            if up_rate is not None:  # 这步在encoder没有
                blocks.append(nn.Upsample(scale_factor=up_rate, mode="nearest"))  # 将原始图像沿着水平和垂直方向分别放大两倍（scale_factor=2.0），并且使用最近邻插值算法（mode=nearest）来填充新的像素值。最近邻插值算法会在新图像中为每个像素分配与原图像中最近的像素值，从而生成一个放大后的图像。
                continue

            in_channel = channel_config[res]
            use_3x3 = res > 1
            blocks.append(
                ResBlock(
                    in_channel,
                    int(0.5 * in_channel),
                    in_channel,
                    down_rate=None,
                    residual=True,
                    use_3x3=use_3x3,
                )
            )
        # TODO: If the training is unstable try using scaling the weights
        self.block_mod = nn.Sequential(*blocks)  # 将前面累计的blocks连起来成为一个Sequential网络结构
        self.last_conv = nn.Conv2d(channel_config[input_res], 1, 3, stride=1, padding=1)  # 结束处有一个3*3的卷积，这里我把输出改成单通道了

    def forward(self, input):
        # x = self.in_conv(input)  # 将维度从100变为128
        x = self.block_mod(input)  # 通过很多个block
        x = self.last_conv(x)  # 一个3*3的卷积
        return torch.sigmoid(x)  # 通过一个sigmoid激活函数


# Implementation of the Resnet-VAE using a ResNet backbone as encoder
# and Upsampling blocks as the decoder
class VAE(pl.LightningModule):  # nn.Module
    def __init__(
        self,
        input_res,  #  input_res就是图片的大小
        enc_block_str,
        dec_block_str,
        enc_channel_str,
        dec_channel_str,
        alpha=1.0,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_res = input_res
        self.enc_block_str = enc_block_str
        self.dec_block_str = dec_block_str
        self.enc_channel_str = enc_channel_str
        self.dec_channel_str = dec_channel_str
        self.alpha = alpha  # 为1.0
        self.lr = lr  # 为0.0001

        # Encoder architecture
        self.enc = Encoder(self.enc_block_str, self.enc_channel_str)

        # Decoder Architecture
        self.dec = Decoder(self.input_res, self.dec_block_str, self.dec_channel_str)

    def encode(self, x):
        mu, logvar = self.enc(x)
        return mu, logvar

    def decode(self, z):
        return self.dec(z)

    def reparameterize(self, mu, logvar):  # 参数重整化
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, z):  # 这个函数只用来sample
        # Only sample during inference
        decoder_out = self.decode(z)
        return decoder_out

    def forward_recons(self, x):  # 这个函数只用来重建reconstruction
        # For generating reconstructions during inference
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoder_out = self.decode(z)
        return decoder_out

    def training_step(self, batch, batch_idx):
        x = batch

        # Encoder
        mu, logvar = self.encode(x)

        # Reparameterization Trick  # 参数重整化
        z = self.reparameterize(mu, logvar)

        # Decoder
        decoder_out = self.decode(z)

        # Compute loss   # 关于loss的计算，有的是采用的BCE loss,有的采用的是MSE loss,有的采用的是L1 loss,到底应该采用哪一个，我想，各有各的说法
        # 关于loss类型的采用，可以看看这篇 https://zhuanlan.zhihu.com/p/345360992
        mse_loss = nn.MSELoss(reduction="sum")  # 计算mse_loss
        recons_loss = mse_loss(decoder_out, x)  # 通过mse_loss计算recons_loss
        kl_loss = self.compute_kl(mu, logvar)   # 计算kl_loss
        self.log("Recons Loss", recons_loss, prog_bar=True)
        self.log("Kl Loss", kl_loss, prog_bar=True)

        total_loss = recons_loss + self.alpha * kl_loss  # 总的loss。 需要明确vae论文中使用的loss到底是什么
        self.log("Total Loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    # def __call__(self, x):  # 这里我重写call函数，为的是想让VAE网络返回的前向函数是默认的forward（）。而是forward_recons（）
    #     return self.forward_recons(x)  # 这是为了能够使用tensorboard 的 Summary writer写出完整的网络结构。这段代码是可以注释的


if __name__ == "__main__":

    enc_block_config_str = "128x1,128d2,128t64,64x3,64d2,64t32,32x3,32d2,32t16,16x7,16d2,16t8,8x3,8d2,8t4,4x3,4d4,4t1,1x2"
    enc_channel_config_str = "128:16,64:16,32:32,16:32,8:64,4:128,1:256"

    dec_block_config_str = "1x1,1u4,1t4,4x2,4u2,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
    dec_channel_config_str = "128:16,64:16,32:32,16:32,8:64,4:128,1:256"

    vae = VAE(
    input_res=128,
    enc_block_str=enc_block_config_str,
    dec_block_str=dec_block_config_str,
    enc_channel_str=enc_channel_config_str,
    dec_channel_str=dec_channel_config_str,
    )

    # 打印网络结构
    # sample = torch.randn(1, 1, 128, 128)
    # print(vae)
    # print(list(vae.children()))  # 效果和print(vae)相同
    # print(list(vae.named_children()))  # 给children()的输出加上了编号
    # for name, parameters in vae.named_parameters():  # 类似于展示结构
    #     print(name, ':', parameters.size())
    # print("Num params: ", sum(p.numel() for p in vae.parameters()))  # 计算出参数数量   当前：21098275

        # 想要在tensorboard上画出网络结构，但是失败
        # from tensorboardX import SummaryWriter
        # writer = SummaryWriter(r'E:\Coding_path\DiffuseVAE\scripts\results_dir\lightning_logs\architecture_vae')
        #
        # z = torch.randn(1024)
        # image = torch.stack([torch.tensor(3), z, torch.tensor(1), torch.tensor(1)])
        # writer.add_graph(vae, input_to_model=image)
        # writer.close()
