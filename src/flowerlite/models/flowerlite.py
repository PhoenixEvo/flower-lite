import torch
from torch import nn


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class PetalMixBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, drop_path_rate=0.0):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.bn_u = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU(inplace=True)
        
        self.c_a = in_channels // 2
        self.c_b = in_channels - self.c_a
        
        self.dwconv3 = nn.Conv2d(self.c_a, self.c_a, kernel_size=3, stride=stride, padding=1, groups=self.c_a, bias=False)
        self.dwconv5 = nn.Conv2d(self.c_b, self.c_b, kernel_size=5, stride=stride, padding=2, groups=self.c_b, bias=False)
        
        self.conv_e = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=1, bias=False)
        self.bn_e = nn.BatchNorm2d(2 * out_channels)
        
        se_hidden = max(16, (2 * out_channels) // 8)
        self.se_fc1 = nn.Linear(2 * out_channels, se_hidden)
        self.se_fc2 = nn.Linear(se_hidden, 2 * out_channels)
        
        self.conv_r = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1, bias=False)
        self.bn_r = nn.BatchNorm2d(out_channels)
        nn.init.constant_(self.bn_r.weight, 0.0)
        nn.init.constant_(self.bn_r.bias, 0.0)
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        h = x
        u = self.act(self.bn_u(h))
        u_a = u[:, :self.c_a]
        u_b = u[:, self.c_a:]
        
        v_a = self.dwconv3(u_a)
        v_b = self.dwconv5(u_b)
        v = torch.cat([v_a, v_b], dim=1)
        
        v_shuf = channel_shuffle(v, 2)
        e = self.act(self.bn_e(self.conv_e(v_shuf)))
        
        # SE
        se_pool = e.mean((2, 3))
        se = self.act(self.se_fc1(se_pool))
        se = torch.sigmoid(self.se_fc2(se))
        e_se = e * se.view(-1, 2 * self.out_channels, 1, 1)
        
        r = self.bn_r(self.conv_r(e_se))
        return self.shortcut(h) + self.drop_path(r)

class PollenAttentionBridge(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, drop_path_rate=0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * heads * head_dim, bias=True)
        self.proj = nn.Linear(heads * head_dim, dim, bias=True)
        
        # Relative position bias table for 8x8
        # The coordinates are from 0 to 7. Max relative distance is 7, min is -7.
        # Total relative positions: 15x15 = 225
        self.relative_position_bias_table = nn.Parameter(torch.zeros(225, heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        coords_h = torch.arange(8)
        coords_w = torch.arange(8)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) # 2, 8, 8
        coords_flatten = torch.flatten(coords, 1) # 2, 64
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # 2, 64, 64
        relative_coords[0, :, :] += 7 
        relative_coords[1, :, :] += 7
        relative_coords[0, :, :] *= 15
        relative_position_index = relative_coords.sum(0) # 64, 64
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        self.ln2 = nn.LayerNorm(dim)
        self.mlp_conv1 = nn.Conv2d(dim, mlp_dim, 1, bias=False)
        self.mlp_dwconv = nn.Conv2d(mlp_dim, mlp_dim, 3, padding=1, groups=mlp_dim, bias=False)
        self.mlp_bn = nn.BatchNorm2d(mlp_dim)
        self.mlp_act = nn.SiLU(inplace=True)
        self.mlp_conv2 = nn.Conv2d(mlp_dim, dim, 1, bias=False)

    def forward(self, x):
        # x is [B, 256, 8, 8]
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2) # [B, 64, 256]
        
        # Attention
        t_ln = self.ln1(t)
        qkv = self.qkv(t_ln).reshape(B, H*W, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(H*W, H*W, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        
        t_attn = (attn @ v).transpose(1, 2).reshape(B, H*W, self.heads * self.head_dim)
        t_attn = self.proj(t_attn)
        t_attn = t + self.drop_path(t_attn)
        
        # LocalFFN
        t_attn_ln = self.ln2(t_attn)
        t_attn_spatial = t_attn_ln.transpose(1, 2).reshape(B, C, H, W)
        
        mlp_out = self.mlp_conv1(t_attn_spatial)
        mlp_out = self.mlp_dwconv(mlp_out)
        mlp_out = self.mlp_bn(mlp_out)
        mlp_out = self.mlp_act(mlp_out)
        mlp_out = self.mlp_conv2(mlp_out)
        
        t_out = t_attn + self.drop_path(mlp_out.flatten(2).transpose(1, 2))
        return t_out.transpose(1, 2).reshape(B, C, H, W)

class FlowerLiteL(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        widths = [64, 128, 256, 384]
        depths = [3, 4, 6, 3]
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
        
        self.stages = nn.ModuleList()
        cur = 0
        in_c = 64
        for i in range(4):
            stage = nn.ModuleList()
            for j in range(depths[i]):
                stride = 2 if j == 0 and i > 0 else 1
                stage.append(PetalMixBlock(in_c, widths[i], stride, dpr[cur]))
                in_c = widths[i]
                cur += 1
            if i == 2:
                stage.append(PollenAttentionBridge(widths[i], 4, 64, 384, dpr[cur-1]))
            self.stages.append(stage)
            
        self.head_ln = nn.LayerNorm(384)
        self.head_fc1 = nn.Linear(384, 768)
        self.head_act = nn.SiLU(inplace=True)
        self.head_drop = nn.Dropout(0.15)
        self.head_fc2 = nn.Linear(768, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if m.weight.shape == (self.num_classes, 768) or m.weight.shape == (768, 384) or getattr(m, 'is_qkv_or_proj', False):
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        x = x.mean((2, 3)) # GAP
        x = self.head_ln(x)
        x = self.head_fc1(x)
        x = self.head_act(x)
        x = self.head_drop(x)
        x = self.head_fc2(x)
        return x
