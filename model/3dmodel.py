import torch
import torch.nn as nn
from torch.nn import functional as F


# unet encoder + skip connection + channel attention & spatial attention + transformer encoder

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dhw=64):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels),
            # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            # nn.LayerNorm([dhw, dhw, dhw]),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SkipConnect(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipConnect, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        return self.residual(x)


# Attention Mechanism
class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool3d = nn.AdaptiveAvgPool3d(1)  # B*C*1*1*1
        self.max_pool3d = nn.AdaptiveMaxPool3d(1)  # B*C*1*1*1

        self.shared_mlp = nn.Sequential(
            nn.Conv3d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // ratio, channel, 1, bias=False)
        )
        # B*C*1*1*1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.avg_pool3d(x)
        avgout = self.shared_mlp(avgout)
        maxout = self.max_pool3d(x)
        maxout = self.shared_mlp(maxout)
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, channel, kernel_size=7, stride=1, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv3d = nn.Conv3d(2, 1, kernel_size, stride, padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv3d(x))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channel)
        self.spatial_att = SpatialAttention(channel)

    def forward(self, x):
        out = self.channel_att(x) * x
        out = self.spatial_att(out) * out
        return out


# if __name__ == '__main__':
#     m = CBAM(32)
#     x = torch.randn((2,32,64,64,64))
#     y = m(x)
#     print(y.shape)
#     print(x.shape)
#     print((x+y).shape)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False):
        super(Up, self).__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[4] - x1.size()[4]
        diffY = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# transformer encoder
# def drop_path(x, drop_prob=0, training=False):
#     if not training or drop_prob != 0:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()
#     x = x.div(keep_prob) * random_tensor
#
#     return x
#
#
# class DropPath(nn.Module):
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob)
#
#
# class PatchEmbed(nn.Module):
#     """
#     input_size: B*256*16*16*16
#     """
#     def __init__(self, embdd_dim=16*16*16, norm_layer=None):
#         super(PatchEmbed, self).__init__()
#         self.norm = norm_layer(embdd_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         x = x.flatten(2).transpose(1, 2)  # (B, 4096, 256)
#         x = self.norm(x)
#         return x   # torch.Size([10, 4096, 256])

# if __name__ == '__main__':
#     m = PatchEmbed()
#     x = torch.randn(10,256, 16, 16, 16)
#     y = m(x)
#     print(y.shape)


# class Attention(nn.Module):
#     def __init__(self,
#                  dim,   # 输入token的dim
#                  num_heads=8,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  attn_drop_ratio=0.,
#                  proj_drop_ratio=0.):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop_ratio)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop_ratio)
#
#     def forward(self, x):
#         # [batch_size, num_patches + 1, total_embed_dim]
#         B, N, C = x.shape
#
#         # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
#         # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
#         # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head] 调整顺序
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
#         # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
#         attn = (q @ k.transpose(-2, -1)) * self.scale # q dot-product k的转置，只对最后两个维度进行操作
#         attn = attn.softmax(dim=-1)  # 对每一行进行softmax
#         attn = self.attn_drop(attn)
#
#         # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
#         # reshape: -> [batch_size, num_patches + 1, total_embed_dim] 将多头的结果拼接在一起
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#
# class Mlp(nn.Module):
#     """
#     MLP as used in Vision Transformer, MLP-Mixer and related networks
#     """
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# class Block(nn.Module):
#     def __init__(self,
#                  dim,
#                  num_heads=8,
#                  mlp_ratio=4., # 第一个全连接层节点个数是输入的四倍
#                  qkv_bias=False,
#                  qk_scale=None,
#                  drop_ratio=0.,
#                  attn_drop_ratio=0.,
#                  drop_path_ratio=0.,
#                  act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm):
#         super(Block, self).__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                               attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
#
#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
#
#
# class TransEncoder(nn.Module):
#     def __init__(self, token_dim=256, num_tokens=4096, drop_ratio=0, num_classes=2):
#         super(TransEncoder, self).__init__()
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens+1, token_dim))
#         self.patch_embed = PatchEmbed()
#         self.pos_drop = nn.Dropout(p=drop_ratio)
#         self.block = Block(token_dim, 8)
#
#         self.norm = nn.LayerNorm(token_dim)
#         self.pre_logits = nn.Identity()
#         self.head = nn.Linear(token_dim, num_classes) if num_classes > 0 else nn.Identity()
#
#     def forward(self, x):
#         x = self.patch_embed(x)  # B 4096 256
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_token, x), dim=1)  # B 4097 256
#
#         x = self.pos_drop(x + self.pos_embed)
#         x = self.block(x)
#         x = self.norm(x)
#
#         x = self.pre_logits(x[:, 0])
#         x = self.head(x)
#
#         return x


class TransNet3d(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, n_channels=32):
        super(TransNet3d, self).__init__()
        self.in_channels = in_channels  # 1
        self.n_classes = n_classes  # 2
        # n_channels -> out_channels
        self.n_channels = n_channels  # 32

        self.conv = DoubleConv(in_channels, n_channels)
        self.skipconnect1 = nn.Sequential(
            nn.Conv3d(in_channels, n_channels, kernel_size=1, stride=1),
            nn.BatchNorm3d(n_channels)
        )
        self.cbam1 = CBAM(n_channels)

        self.enc1 = Down(n_channels, 2 * n_channels)
        self.skipconnect2 = SkipConnect(n_channels, 2 * n_channels)
        self.cbam2 = CBAM(2 * n_channels)

        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.skipconnect3 = SkipConnect(2 * n_channels, 4 * n_channels)
        self.cbam3 = CBAM(4 * n_channels)

        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.skipconnect4 = SkipConnect(4 * n_channels, 8 * n_channels)
        self.cbam4 = CBAM(8 * n_channels)

        self.enc4 = Down(8 * n_channels, 16 * n_channels)
        self.skipconnect5 = SkipConnect(8 * n_channels, 16 * n_channels)
        self.cbam5 =CBAM(16 * n_channels)

        self.enc5 = Down(16 * n_channels, 32 * n_channels)
        self.skipconnect6 = SkipConnect(16 * n_channels, 32 * n_channels)
        self.cbam6 = CBAM(32 * n_channels)

        # self.out = TransEncoder()
        self.out = nn.Sequential(
            nn.Linear(1024*4*4*4, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 2),
            nn.Dropout()
        )

    def forward(self, x):
        x1 = self.conv(x)
        x1 = x1 + self.skipconnect1(x)
        x1 = self.cbam1(x1) + x1
        print("x1", x1.shape)

        x2 = self.enc1(x1)
        print("x2", x2.shape)
        x2 = x2 + self.skipconnect2(x1)
        print("x3", x2.shape)
        x2 = self.cbam2(x2) + x2
        print("x2", x2.shape)

        x3 = self.enc2(x2)
        x3 = x3 + self.skipconnect3(x2)
        x3 = self.cbam3(x3) + x3
        print("x3", x3.shape)

        x4 = self.enc3(x3)
        x4 = x4 + self.skipconnect4(x3)
        x4 = self.cbam4(x4) + x4
        print("x4", x4.shape)

        x5 = self.enc4(x4)
        x5 = x5 + self.skipconnect5(x4)
        print("x5", x5.shape)

        x6 = self.enc5(x5)
        x6 = x6 + self.skipconnect6(x5)

        out = self.out(x6)
        return out


if __name__ == '__main__':
    # x = torch.randn((1, 1, 256, 256, 256))
    model = TransNet3d()
    # x = x.cuda()
    model = model.cuda()
    # model = nn.DataParallel(model, device_ids=None, output_device=None)
    # y = model(x)
    # print(torch.softmax(y, dim=1))
