import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


##### Feature Pyramid Network Component #####
class FPN_Block(nn.Module):
    """
    Feature Pyramid Network (FPN) block for top-down feature fusion.

    This block takes a higher-level feature map, upsamples it, and perform element-wise addition with the output of the lower HFF stage

    Args:
        in_channels (int): Number of input channels from the higher-level feature map.
        out_channels (int): Number of output channels after applying 1x1 convolution.
    """

    def __init__(self, hff_in_channels: int, out_channels: int):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.pwconv = nn.Conv2d(
            in_channels=hff_in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        # self.norm = nn.BatchNorm2d(out_channels) # An's Note: Try training without normalization first, then try BatchNorm and GroupNorm to see if there is any improvement
        # self.act = nn.ReLU(inplace=True)

    def forward(self, p, f):
        f = self.pwconv(f)
        p = self.upsample(p)
        p = p + f
        # p = self.norm(p)
        # p = self.act(p)
        return p


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.avg_pooled = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=in_channels // 4,
                        kernel_size=1,
                        bias=False,
                    ),
                    # nn.GroupNorm(32, out_channels),
                    # nn.ReLU(inplace=True),
                )
                for pool_size in pool_sizes
            ]
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels * 2,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            # nn.GroupNorm(32, out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        pooled_layers = [
            F.interpolate(layer(x), size=(h, w), mode="bilinear", align_corners=False)
            for layer in self.avg_pooled
        ]

        x = torch.cat([x] + pooled_layers, dim=1)
        return self.final_conv(x)


def drop_path_f(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class TriFuse(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        depths=(2, 2, 2, 2),
        num_heads=(4, 8, 16, 32),
        window_size=7,
        qkv_bias=True,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        HFF_dp=0.0,
        conv_depths=(2, 2, 2, 2),
        conv_drop_path_rate=0.0,
        conv_head_init_scale: float = 1.0,
        fpn_dim=256,  # Add fpn_dim argument
        **kwargs,
    ):
        super().__init__()
        conv_dims = (int(fpn_dim / 8), int(fpn_dim / 4), int(fpn_dim / 2), int(fpn_dim))
        embed_dim = int(fpn_dim / 8)
        self._out_feature_channels = [*conv_dims]
        ###### Local Branch Setting #######

        self.downsample_layers = nn.ModuleList()  # stem + 3 stage downsample
        stem = nn.Sequential(
            nn.Conv2d(in_chans, conv_dims[0], kernel_size=4, stride=4),
            LayerNorm(conv_dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        # stage2-4 downsample
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(conv_dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(conv_dims[i], conv_dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [
            x.item() for x in torch.linspace(0, conv_drop_path_rate, sum(conv_depths))
        ]
        cur = 0

        # Build stacks of blocks in each stage
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Local_block(dim=conv_dims[i], drop_rate=dp_rates[cur + j])
                    for j in range(conv_depths[i])
                ]
            )
            self.stages.append((stage))
            cur += conv_depths[i]

        ###### Global Branch Setting ######

        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        # The channels of stage4 output feature matrix
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_c=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        i_layer = 0
        self.layers1 = BasicLayer(
            dim=int(embed_dim * 2**i_layer),
            depth=depths[i_layer],
            num_heads=num_heads[i_layer],
            window_size=window_size,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
            norm_layer=norm_layer,
            downsample=PatchMerging if (i_layer > 0) else None,
            use_checkpoint=use_checkpoint,
        )

        i_layer = 1
        self.layers2 = BasicLayer(
            dim=int(embed_dim * 2**i_layer),
            depth=depths[i_layer],
            num_heads=num_heads[i_layer],
            window_size=window_size,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
            norm_layer=norm_layer,
            downsample=PatchMerging if (i_layer > 0) else None,
            use_checkpoint=use_checkpoint,
        )

        i_layer = 2
        self.layers3 = BasicLayer(
            dim=int(embed_dim * 2**i_layer),
            depth=depths[i_layer],
            num_heads=num_heads[i_layer],
            window_size=window_size,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
            norm_layer=norm_layer,
            downsample=PatchMerging if (i_layer > 0) else None,
            use_checkpoint=use_checkpoint,
        )

        i_layer = 3
        self.layers4 = BasicLayer(
            dim=int(embed_dim * 2**i_layer),
            depth=depths[i_layer],
            num_heads=num_heads[i_layer],
            window_size=window_size,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
            norm_layer=norm_layer,
            downsample=PatchMerging if (i_layer > 0) else None,
            use_checkpoint=use_checkpoint,
        )

        ###### Hierachical Feature Fusion Block Setting #######

        self.fu1 = HFF_block(
            ch_1=conv_dims[0],
            ch_2=conv_dims[0],
            r_2=16,
            ch_int=conv_dims[0],
            ch_out=conv_dims[0],
            drop_rate=HFF_dp,
        )
        self.fu2 = HFF_block(
            ch_1=conv_dims[1],
            ch_2=conv_dims[1],
            r_2=16,
            ch_int=conv_dims[1],
            ch_out=conv_dims[1],
            drop_rate=HFF_dp,
        )
        self.fu3 = HFF_block(
            ch_1=conv_dims[2],
            ch_2=conv_dims[2],
            r_2=16,
            ch_int=conv_dims[2],
            ch_out=conv_dims[2],
            drop_rate=HFF_dp,
        )
        self.fu4 = HFF_block(
            ch_1=conv_dims[3],
            ch_2=conv_dims[3],
            r_2=16,
            ch_int=conv_dims[3],
            ch_out=conv_dims[3],
            drop_rate=HFF_dp,
        )

        ###### Feature Pyramid Network Setting ######
        # Use fpn_dim for output channels
        # self.topconv = nn.Conv2d(
        #     in_channels=conv_dims[3], out_channels=fpn_dim, kernel_size=1, stride=1
        # )
        # self.p4 = FPN_Block(hff_in_channels=conv_dims[2], out_channels=fpn_dim)
        # self.p3 = FPN_Block(hff_in_channels=conv_dims[1], out_channels=fpn_dim)
        # self.p2 = FPN_Block(hff_in_channels=conv_dims[0], out_channels=fpn_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward(self, imgs, targets=None):
        # images (224, 224, 3)

        ######  Global Branch ######
        # H_embed = W_embed = 224
        x_s, H_embed, W_embed = self.patch_embed(imgs)
        x_s = self.pos_drop(x_s)
        x_s_1, H, W = self.layers1(x_s, H_embed, W_embed)
        x_s_2, H, W = self.layers2(x_s_1, H, W)
        x_s_3, H, W = self.layers3(x_s_2, H, W)
        x_s_4, H, W = self.layers4(x_s_3, H, W)

        # [B,L,C] ---> [B,C,H,W]

        x_s_1 = torch.transpose(x_s_1, 1, 2)
        x_s_1 = x_s_1.view(
            x_s_1.shape[0], -1, H_embed, W_embed
        )  # 56 because patch_size = 4
        x_s_2 = torch.transpose(x_s_2, 1, 2)
        x_s_2 = x_s_2.view(x_s_2.shape[0], -1, H_embed // 2, W_embed // 2)
        x_s_3 = torch.transpose(x_s_3, 1, 2)
        x_s_3 = x_s_3.view(x_s_3.shape[0], -1, H_embed // 4, W_embed // 4)
        x_s_4 = torch.transpose(x_s_4, 1, 2)
        x_s_4 = x_s_4.view(x_s_4.shape[0], -1, H_embed // 8, W_embed // 8)

        # print("Global Branch")
        # print(x_s_1.shape)
        # print(x_s_2.shape)
        # print(x_s_3.shape)
        # print(x_s_4.shape)

        ######  Local Branch ######
        x_c = self.downsample_layers[0](imgs)
        x_c_1 = self.stages[0](x_c)
        x_c = self.downsample_layers[1](x_c_1)
        x_c_2 = self.stages[1](x_c)
        x_c = self.downsample_layers[2](x_c_2)
        x_c_3 = self.stages[2](x_c)
        x_c = self.downsample_layers[3](x_c_3)
        x_c_4 = self.stages[3](x_c)

        # print("Local Branch")
        # print(x_c_1.shape)
        # print(x_c_2.shape)
        # print(x_c_3.shape)
        # print(x_c_4.shape)

        ###### Hierachical Feature Fusion ######
        x_f_1 = self.fu1(x_c_1, x_s_1, None)
        x_f_2 = self.fu2(x_c_2, x_s_2, x_f_1)
        x_f_3 = self.fu3(x_c_3, x_s_3, x_f_2)
        x_f_4 = self.fu4(x_c_4, x_s_4, x_f_3)

        # print("Hierachical Feature Fusion")
        # print(x_f_1.shape)
        # print(x_f_2.shape)
        # print(x_f_3.shape)
        # print(x_f_4.shape)

        ###### Feature Pyramid Network ######
        # [1, 96, 56, 56]
        # [1, 192, 28, 28]
        # [1, 384, 14, 14]
        # [1, 768, 7, 7]

        # x_p_4 = self.ppm(x_f_4)
        # x_p_4 = self.topconv(x_f_4)
        # x_p_3 = self.p4(x_p_4, x_f_3)
        # x_p_2 = self.p3(x_p_3, x_f_2)
        # x_p_1 = self.p2(x_p_2, x_f_1)

        # print("Feature Pyramid Network")
        # print(x_p_1.shape)
        # print(x_p_2.shape)
        # print(x_p_3.shape)
        # print(x_p_4.shape)

        ###### Feature Fusion ######
        x_4 = F.interpolate(x_f_4, scale_factor=8, mode="bilinear", align_corners=False)
        x_3 = F.interpolate(x_f_3, scale_factor=4, mode="bilinear", align_corners=False)
        x_2 = F.interpolate(x_f_2, scale_factor=2, mode="bilinear", align_corners=False)
        x_1 = x_f_1

        x_fu = [x_4, x_3, x_2, x_1]

        return x_fu
        # return self.head(x_f)


##### Local Feature Block Component #####


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Local_block(nn.Module):
    r"""Local Feature Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_rate=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv = nn.Linear(
            dim, dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        x = shortcut + self.drop_path(x)
        return x


# Hierachical Feature Fusion Block
class HFF_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.0):
        super(HFF_block, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // r_2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_2 // r_2, ch_2, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.Updim = Conv(
            ch_int // 2, ch_int, 1, bn=True, relu=True
        )  # 1x1 Conv - according to paper
        self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(
            ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first"
        )
        self.W3 = Conv(ch_int * 3, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)

        self.gelu = nn.GELU()

        self.residual = IRMLP(ch_1 + ch_2 + ch_int, ch_out)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward(self, l, g, f):
        W_local = self.W_l(
            l
        )  # local feature from Local Feature Block (NOT MENTIONED IN PAPER)
        W_global = self.W_g(
            g
        )  # global feature from Global Feature Block (NOT MENTIONED IN PAPER)
        if f is not None:
            W_f = self.Updim(f)
            W_f = self.Avg(W_f)
            shortcut = W_f
            X_f = torch.cat([W_f, W_local, W_global], 1)
            X_f = self.norm1(X_f)
            X_f = self.W3(X_f)
            X_f = self.gelu(X_f)
        else:
            shortcut = 0
            X_f = torch.cat([W_local, W_global], 1)
            X_f = self.norm2(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)

        # spatial attention for ConvNeXt branch
        l_jump = l
        max_result, _ = torch.max(l, dim=1, keepdim=True)
        avg_result = torch.mean(l, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        l = self.spatial(result)
        l = self.sigmoid(l) * l_jump

        # channel attetion for transformer branch
        g_jump = g
        max_result = self.maxpool(g)
        avg_result = self.avgpool(g)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        g = self.sigmoid(max_out + avg_out) * g_jump

        fuse = torch.cat([g, l, X_f], 1)
        fuse = self.norm3(fuse)
        fuse = self.residual(fuse)
        fuse = shortcut + self.drop_path(fuse)
        return fuse


class Conv(nn.Module):
    def __init__(
        self,
        inp_dim,
        out_dim,
        kernel_size=3,
        stride=1,
        bn=False,
        relu=True,
        bias=True,
        group=1,
    ):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(
            inp_dim,
            out_dim,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


#### Inverted Residual MLP
class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out


####### Shift Window MSA #############


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.dim = dim  # embed_dim * 2 ** (i_layer)
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # [2, Mh, Mw]
        # coords = [
        #     [
        #         [0, 1, 2, 3, 4, 5, 6],  # Row indices
        #         [0, 1, 2, 3, 4, 5, 6],
        #         ...
        #         [0, 1, 2, 3, 4, 5, 6]
        #     ],
        #     [
        #         [0, 0, 0, 0, 0, 0, 0],  # Column indices
        #         [1, 1, 1, 1, 1, 1, 1],
        #         ...
        #         [6, 6, 6, 6, 6, 6, 6]
        #     ]
        # ]

        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # coords_flatten = [
        #     [0, 1, 2, ..., 6, 0, 1, 2, ..., 6, ..., 0, 1, ..., 6],  # Row indices
        #     [0, 0, 0, ..., 0, 1, 1, 1, ..., 1, ..., 6, 6, ..., 6]   # Column indices
        # ]  # Shape: [2, 49]

        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        # Subtraction with extra dimension is to create two matrix representing Row indices and Column indices in 2D dimension (images).
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # [Mh*Mw, Mh*Mw, 2]

        # Adjust the column indices such that its > 0
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        # Adjust the row indices such that its > 0
        relative_coords[:, :, 1] += self.window_size[1] - 1

        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # ???????
        # Relative Positional Embeddings
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # print(x.shape)  # (64, 49, 32)
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


### Global Feature Block
class Global_block(nn.Module):
    r"""Global Feature Block from modified Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = act_layer()

    def forward(self, x, attn_mask):
        # patch size
        H, W = self.H, self.W
        # (1, 3136, 32)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # x = [
        #     patch1, patch2,...
        #     patch...
        # ] each patch has C which is embed_dim

        # padding for partition windows
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        # windows size (7, 7)
        # Divide (1, 56, 56, 32) into (64, 7, 7, 32)
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        # (64, 49, 32)
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C
        )  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(
            attn_windows, self.window_size, Hp, Wp
        )  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.fc1(x)
        x = self.act(x)
        x = shortcut + self.drop_path(x)

        return x


class BasicLayer(nn.Module):
    """
    Downsampling and Global Feature Block for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList(
            [
                Global_block(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else self.shift_size,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(
            -1, self.window_size * self.window_size
        )  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
            2
        )  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        return attn_mask

    def forward(self, x, H, W):
        # print(x.shape, H, W)
        if self.downsample is not None:
            x = self.downsample(
                x, H, W
            )  # patch merging stage2 in [6,3136,96] out [6,784,192]
            H, W = (H + 1) // 2, (W + 1) // 2

        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:  # global block
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        return x, H, W


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)

    Example:
        Divide (1, 14, 14, 3) -> (1, 2, 2, 7, 7, 3) -> (1 * 2 * 2, 7, 7, 3)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, patch_size=4, in_c=3, embed_dim=32, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top, H_bottom, C_front, C_back)
            x = F.pad(
                x,
                (
                    0,
                    self.patch_size[1] - W % self.patch_size[1],
                    0,
                    self.patch_size[0] - H % self.patch_size[0],
                    0,
                    0,
                ),
            )

        # downsample patch_size times
        x = self.proj(x)
        _, _, H, W = x.shape
        # H, W at this stage is the number of tokens (image_H, image_W // patch_size)
        # C is embed_dim
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        dim = dim // 2
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


def TriFuse_Tiny(fpn_dim: int = 256):  # Add fpn_dim
    model = TriFuse(
        depths=(2, 2, 2, 2),
        conv_depths=(2, 2, 2, 2),
        fpn_dim=fpn_dim,  # Pass fpn_dim
    )
    return model


def TriFuse_Small(fpn_dim: int = 256):  # Add fpn_dim
    model = TriFuse(
        depths=(2, 2, 6, 2),
        conv_depths=(2, 2, 6, 2),
        fpn_dim=fpn_dim,  # Pass fpn_dim
    )
    return model


def TriFuse_Base(fpn_dim: int = 256):  # Add fpn_dim
    model = TriFuse(
        depths=(2, 2, 18, 2),
        conv_depths=(2, 2, 18, 2),
        fpn_dim=fpn_dim,  # Pass fpn_dim
    )
    return model
