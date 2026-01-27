from typing import Type, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from segment_anything.modeling.image_encoder import ImageEncoderViT, Attention, window_partition, window_unpartition
from segment_anything.modeling.common import MLPBlock, LayerNorm2d
from networks.mlora import MultiLoraLinear 

nonlinearity = partial(F.relu, inplace=False)

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            scale: float = 0.5,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.scale = scale
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        xn = self.norm2(x)
        x = x + self.mlp(xn)
        return x

class Resize(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=True
        )

class AntiAlias(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.2):
        super().__init__()
        assert kernel_size % 2 == 1
        k = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-(k**2) / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d[:, None] @ gauss_1d[None, :]
        self.register_buffer("kernel2d", gauss_2d[None, None, :, :])
        self.ks = kernel_size

    def forward(self, x):
        B, C, H, W = x.shape
        weight = self.kernel2d.expand(C, 1, self.ks, self.ks).contiguous()
        return F.conv2d(x, weight, padding=self.ks // 2, groups=C)

class BFEM(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.reduced_channels = max(4, in_channels // reduction)
        self.bottom_up_convs = nn.ModuleList([
            nn.Conv2d(1 if i == 0 else self.reduced_channels, self.reduced_channels, kernel_size=3, padding=1, bias=False) for i in range(3)
        ])
        self.pools = nn.ModuleList([nn.MaxPool2d(2, stride=2) for _ in range(2)])
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=1, bias=False) for _ in range(3)
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=3, padding=1, bias=False) for _ in range(2)
        ])
        self.attn_proj = nn.Conv2d(self.reduced_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(self.reduced_channels)

        self.feat_reduce = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1, bias=False)
        self.feat_norm = nn.BatchNorm2d(self.reduced_channels)
        self.feat_proj = nn.Conv2d(self.reduced_channels, 1, kernel_size=1, bias=False)

        self.to_gray = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_kx", kx)
        self.register_buffer("sobel_ky", ky)
        self.eps = 1e-6

    def _edge(self, m):
        gx = F.conv2d(m, self.sobel_kx, padding=1)
        gy = F.conv2d(m, self.sobel_ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        mag = mag - mag.amin(dim=(2, 3), keepdim=True)
        mag = mag / (mag.amax(dim=(2, 3), keepdim=True) + self.eps)
        return mag

    def _cos_sim(self, a, b):
        B = a.shape[0]
        av = a.view(B, -1)
        bv = b.view(B, -1)
        num = (av * bv).sum(dim=1)
        den = torch.sqrt((av * av).sum(dim=1) * (bv * bv).sum(dim=1) + self.eps)
        s = num / (den + self.eps)
        return s.clamp(-1.0, 1.0).view(B, 1, 1, 1)

    def forward(self, x, mask):
        B, C, H, W = x.shape
        mask_resized = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)

        edge_m = self._edge(mask_resized)
        gray_f = self.to_gray(x)
        edge_f = self._edge(gray_f)

        sim = self._cos_sim(edge_m, edge_f)
        strength = edge_m.mean(dim=(2, 3), keepdim=True)
        w_mask = torch.sigmoid(6.0 * (0.5 * (sim + 1.0) + 0.5 * strength - 0.55))
        w_mask = w_mask.clamp(0.1, 0.9)
        w_feat = 1.0 - w_mask

        p = self.bottom_up_convs[0](edge_m)
        p1 = p
        p = self.pools[0](p); p = self.bottom_up_convs[1](p); p2 = p
        p = self.pools[1](p); p = self.bottom_up_convs[2](p); p3 = p

        t = self.lateral_convs[2](p3)
        t = F.interpolate(t, size=p2.shape[2:], mode='bilinear', align_corners=False)
        t = self.norm(self.smooth_convs[1](t + self.lateral_convs[1](p2)))
        t = F.interpolate(t, size=p1.shape[2:], mode='bilinear', align_corners=False)
        t = self.norm(self.smooth_convs[0](t + self.lateral_convs[0](p1)))

        attn_mask_logits = self.attn_proj(t)

        f = self.feat_reduce(x)
        f = self.feat_norm(f)
        f = F.relu(f, inplace=False)
        attn_feat_logits = self.feat_proj(f)

        logits = w_mask * attn_mask_logits + w_feat * attn_feat_logits
        attention = self.sigmoid(logits)

        return x * (1 + attention.expand_as(x))

class AdaImageEncoderViT(ImageEncoderViT):
    def __init__(
            self,
            img_size: int = 512,
            patch_size: int = 16,
            in_chans:int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            out_indices=(2, 5, 8, 11),
    ) -> None:
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_chans=out_chans,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_abs_pos=use_abs_pos,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes
        )
        self.out_indices = out_indices
        new_blocks = []
        for blk in self.blocks:
            new_blocks.append(
                Block(
                    dim=blk.dim,
                    num_heads=blk.num_heads,
                    mlp_ratio=blk.mlp_ratio,
                    qkv_bias=blk.qkv_bias,
                    norm_layer=blk.norm_layer,
                    act_layer=blk.act_layer,
                    use_rel_pos=blk.use_rel_pos,
                    rel_pos_zero_init=blk.rel_pos_zero_init,
                    window_size=blk.window_size,
                    input_size=blk.input_size,
                )
            )
        del self.blocks
        self.blocks = nn.ModuleList(new_blocks)
        self.pyramid_Adapter = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
                LayerNorm2d(out_chans),
                Resize(scale, mode='bilinear', align_corners=True),
            ) for scale in [4, 2, 1]
        ])
        self.last_resizer = Resize(0.5, mode='bilinear', align_corners=True)
        self.mask_attention = BFEM(embed_dim)
        self.anti_alias = AntiAlias(kernel_size=5, sigma=1.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.anti_alias(x)
        mask = x[:, 0:1, :, :]
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        out_feats = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                x_attn = self.mask_attention(x.permute(0, 3, 1, 2), mask)
                out_feats.append(x_attn)

        last_feat = out_feats.pop()
        pyramid_feats = []
        for out_feat, ada in zip(out_feats, self.pyramid_Adapter):
            pyramid_feats.append(ada(out_feat))
        pyramid_feats.append(self.last_resizer(self.neck(last_feat)))
        return pyramid_feats

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity
        self.se = SELayer(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.se(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BoundaryHead(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.ca = ChannelAttention(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.sa = SpatialAttention()
        self.conv3 = nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.final_conv = nn.Conv2d(mid_channels // 2, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.res_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)

    def forward(self, x):
        identity = self.res_conv(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.ca(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x + identity
        x = x * self.sa(x).expand_as(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

class ContextRefiner(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw1 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.dw2 = nn.Conv2d(channels, channels, kernel_size=5, padding=6, dilation=3, groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.se = SELayer(channels, reduction=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.dw1(x)))
        y = self.relu(self.bn2(self.dw2(y)))
        y = self.relu(self.bn3(self.pw(y)))
        y = self.se(y)
        return x + y

class LinkNetDecoder(nn.Module):
    def __init__(self, filters, num_classes=1):
        super().__init__()
        self.decoder4= DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.fusion4 = nn.Conv2d(filters[2] + 1, filters[2], 1)
        self.fusion3 = nn.Conv2d(filters[1] + 1, filters[1], 1)
        self.fusion2 = nn.Conv2d(filters[0] + 1, filters[0], 1)
        self.fusion1 = nn.Conv2d(filters[0] + 1, filters[0], 1)
        self.refine4 = ContextRefiner(filters[2])
        self.refine3 = ContextRefiner(filters[1])
        self.refine2 = ContextRefiner(filters[0])
        self.refine1 = ContextRefiner(filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)
        self.boundary_head = BoundaryHead(filters[0], filters[0] // 2)
        self.bnd_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, feats, mask):
        e1, e2, e3, e4 = feats
        d4 = self.decoder4(e4) + e3
        mask_d4 = F.interpolate(mask, size=d4.shape[2:], mode='bilinear', align_corners=False)
        d4_fused = self.fusion4(torch.cat([d4, mask_d4], dim=1))
        d4 = self.refine4(d4_fused)

        d3 = self.decoder3(d4) + e2
        mask_d3 = F.interpolate(mask, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3_fused = self.fusion3(torch.cat([d3, mask_d3], dim=1))
        d3 = self.refine3(d3_fused)

        d2 = self.decoder2(d3) + e1
        mask_d2 = F.interpolate(mask, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2_fused = self.fusion2(torch.cat([d2, mask_d2], dim=1))
        d2 = self.refine2(d2_fused)

        bnd = self.boundary_head(d2)

        d1 = self.decoder1(d2)
        mask_d1 = F.interpolate(mask, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1_fused = self.fusion1(torch.cat([d1, mask_d1], dim=1))
        d1 = self.refine1(d1_fused)

        bnd_up = F.interpolate(bnd, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = d1 * (1.0 + self.bnd_alpha * bnd_up)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        seg = torch.sigmoid(out)
        return seg, bnd

class MultiTaskEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder

    def forward(self, x):
        mask = x[:, 0:1, :, :]
        feats = self.enc(x)
        seg, bnd = self.dec(feats, mask)
        return seg, bnd

def resize_pretrained_pos(pos_embed: torch.Tensor, size):
    if pos_embed.ndim == 4:
        pos_embed = F.interpolate(pos_embed.permute(0, 3, 1, 2), size, mode='bicubic', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)
    elif pos_embed.ndim == 2:
        pos_embed = F.interpolate(pos_embed.unsqueeze(0).permute(0, 2, 1), size, mode='linear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 1).squeeze(0)
    return pos_embed

def resize_model_pos_embed(model, img_size, encoder_global_attn_indexes):
    supported_sizes = [128, 256, 512]
    if img_size not in supported_sizes:
        raise ValueError(f"图像尺寸 {img_size} 不受支持，仅支持 {supported_sizes}")

    pos_embed = F.interpolate(model.pos_embed.permute(0, 3, 1, 2), img_size // 16, mode='bicubic', align_corners=False)
    model.pos_embed.data = pos_embed.permute(0, 2, 3, 1)
    for i in encoder_global_attn_indexes:
        blk = model.blocks[i]
        new_size = 2 * img_size // 16 - 1
        new_rel_pos_h = F.interpolate(blk.attn.rel_pos_h.unsqueeze(0).permute(0, 2, 1), new_size, mode='linear',
                                      align_corners=False).permute(0, 2, 1).squeeze(0)
        blk.attn.rel_pos_h.data = new_rel_pos_h
        new_rel_pos_w = F.interpolate(blk.attn.rel_pos_w.unsqueeze(0).permute(0, 2, 1), new_size, mode='linear',
                                      align_corners=False).permute(0, 2, 1).squeeze(0)
        blk.attn.rel_pos_w.data = new_rel_pos_w
    return model

def setup_fine_tuning(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

    name_keywords = ('Adapter', 'mask_attention', 'lora_')
    for name, p in module.named_parameters():
        if any(k in name for k in name_keywords):
            p.requires_grad = True

    for name, p in module.named_parameters():
        if name.startswith('dec.'):
            p.requires_grad = True

    norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, LayerNorm2d)
    for m in module.modules():
        if isinstance(m, norm_types):
            for p in m.parameters():
                p.requires_grad = True

    for name, p in module.named_parameters():
        if name.endswith('.bias'):
            p.requires_grad = True

    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    trainable_param_names = [n for n, p in module.named_parameters() if p.requires_grad]

    print("Model configured for fine-tuning.")
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")
    print("Some trainable parameter names:")
    for name in trainable_param_names[:5]:
        print(f"  - {name}")
    if len(trainable_param_names) > 5:
        print(f"  ... and {len(trainable_param_names) - 5} more.")

class BoundaryHead(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.ca = ChannelAttention(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.sa = SpatialAttention()
        self.conv3 = nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.final_conv = nn.Conv2d(mid_channels // 2, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.res_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)

    def forward(self, x):
        identity = self.res_conv(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.ca(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x + identity
        x = x * self.sa(x).expand_as(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

class LinkNetDecoder(nn.Module):
    def __init__(self, filters, num_classes=1):
        super().__init__()
        self.decoder4= DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.fusion4 = nn.Conv2d(filters[2] + 1, filters[2], 1)
        self.fusion3 = nn.Conv2d(filters[1] + 1, filters[1], 1)
        self.fusion2 = nn.Conv2d(filters[0] + 1, filters[0], 1)
        self.fusion1 = nn.Conv2d(filters[0] + 1, filters[0], 1)
        self.refine4 = ContextRefiner(filters[2])
        self.refine3 = ContextRefiner(filters[1])
        self.refine2 = ContextRefiner(filters[0])
        self.refine1 = ContextRefiner(filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)
        self.boundary_head = BoundaryHead(filters[0], filters[0] // 2)
        self.bnd_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, feats, mask):
        e1, e2, e3, e4 = feats
        d4 = self.decoder4(e4) + e3
        mask_d4 = F.interpolate(mask, size=d4.shape[2:], mode='bilinear', align_corners=False)
        d4_fused = self.fusion4(torch.cat([d4, mask_d4], dim=1))
        d4 = self.refine4(d4_fused)

        d3 = self.decoder3(d4) + e2
        mask_d3 = F.interpolate(mask, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3_fused = self.fusion3(torch.cat([d3, mask_d3], dim=1))
        d3 = self.refine3(d3_fused)

        d2 = self.decoder2(d3) + e1
        mask_d2 = F.interpolate(mask, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2_fused = self.fusion2(torch.cat([d2, mask_d2], dim=1))
        d2 = self.refine2(d2_fused)

        bnd = self.boundary_head(d2)

        d1 = self.decoder1(d2)
        mask_d1 = F.interpolate(mask, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1_fused = self.fusion1(torch.cat([d1, mask_d1], dim=1))
        d1 = self.refine1(d1_fused)

        bnd_up = F.interpolate(bnd, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = d1 * (1.0 + self.bnd_alpha * bnd_up)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        seg = torch.sigmoid(out)
        return seg, bnd

def build_sam_adapter_linknet(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        checkpoint,
        image_size=512
):
    supported_sizes = [128, 256, 512]
    if image_size not in supported_sizes:
        raise ValueError(f"图像尺寸 {image_size} 不受支持，仅支持 {supported_sizes}")

    prompt_embed_dim = 256
    vit_patch_size = 16

    encoder = AdaImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
        out_indices=encoder_global_attn_indexes,
    )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        state_dict = {k.replace('image_encoder.', ''): v for k, v in state_dict.items() if 'image_encoder' in k}
        if image_size != 1024:
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'pos' in k:
                    if k == 'pos_embed':
                        v = resize_pretrained_pos(v, (image_size // 16, image_size // 16))
                    else:
                        blk_idx = int(k.split('.')[1])
                        if blk_idx in encoder_global_attn_indexes:
                            v = resize_pretrained_pos(v, 2 * image_size // 16 - 1)
                        else:
                            v = resize_pretrained_pos(v, 2 * 14 - 1)
                new_state_dict[k] = v
            state_dict = new_state_dict
        keys = encoder.load_state_dict(state_dict, strict=False)
        print(f'Loaded parameters from pretrained checkpoint, missing keys: {keys.missing_keys}')

    encoder = MultiLoraLinear.convert_lora_linear(
        encoder, 
        r=16,  
        num_lora=6,  
        lora_alpha=8, 
        lora_dropout=0.1, 
        merge_weights=False
    )

    decoder = LinkNetDecoder([prompt_embed_dim] * 4, num_classes=1)

    model = MultiTaskEncoderDecoder(encoder, decoder)

    setup_fine_tuning(model)

    return model

def build_sam_vit_b_adapter_linknet_multi_lora(checkpoint=None, image_size=512):
    return build_sam_adapter_linknet(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        image_size=image_size
    ), [2, 5, 8, 11]
