import torch
import torch.nn as nn
from argparse import Namespace

class MLP(nn.Module):
    def __init__(self, tsne=False, hidden_dim=128, num_hidden_layers=7, output_dim=10):
        super().__init__()
        self.tsne = tsne
        layers = []
        in_dim = 2 if tsne else 784
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x_im, x_tsne=None):
        if self.tsne:
            return self.net(x_tsne)
        else:
            return self.net(x_im.view(x_im.size(0), -1))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.relu(out + identity)
        return out

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, base_channels=16, num_blocks=2):
        """
        Tiny UNet-like encoder for classification (MNIST-style images).

        Args:
            in_channels: input channels (1 for grayscale)
            num_classes: output classes
            base_channels: channels in the first conv layer
            num_blocks: number of residual blocks per resolution
        """
        super().__init__()

        # 28x28 -> 28x28
        self.enc_conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.enc_blocks1 = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])

        # 28 -> 14
        self.down1 = nn.AvgPool2d(2)
        self.enc_conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.enc_blocks2 = nn.Sequential(*[ResidualBlock(base_channels * 2) for _ in range(num_blocks)])

        # 14 -> 7
        self.down2 = nn.AvgPool2d(2)
        self.enc_conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.enc_blocks3 = nn.Sequential(*[ResidualBlock(base_channels * 4) for _ in range(num_blocks)])

        # classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x_im, x_tsne=None):
        x = self.enc_conv_in(x_im)
        x = self.enc_blocks1(x)

        x = self.down1(x)
        x = self.enc_conv2(x)
        x = self.enc_blocks2(x)

        x = self.down2(x)
        x = self.enc_conv3(x)
        x = self.enc_blocks3(x)

        x = self.avgpool(x)     # (B, C, 1, 1)
        x = torch.flatten(x, 1) # (B, C)
        out = self.fc(x)
        return out
    
def gn(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    # Make GroupNorm robust when channels < 32
    g = min(num_groups, num_channels)
    g = max(1, g)
    return nn.GroupNorm(g, num_channels)

class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

class PreActBottleneckResBlock(nn.Module):
    """
    Pre-activation bottleneck residual block:
      GN -> ReLU -> 1x1 (reduce)
      GN -> ReLU -> 3x3
      GN -> ReLU -> 1x1 (expand)
      (+ optional SE)
      Dropout (if > 0) applied before residual add
      Residual add (no activation after, as in pre-act ResNets)
    in_channels == out_channels == channels
    """
    def __init__(self, channels: int, bottleneck_ratio: float = 0.5,
                 use_se: bool = False, dropout: float = 0.0, activation: str = "silu"):
        super().__init__()
        mid = max(1, int(channels * bottleneck_ratio))

        self.n1 = gn(channels)
        self.c1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)

        self.n2 = gn(mid)
        self.c2 = nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False)

        self.n3 = gn(mid)
        self.c3 = nn.Conv2d(mid, channels, kernel_size=1, bias=False)

        self.use_se = use_se
        self.se = SqueezeExcite(channels) if use_se else nn.Identity()

        self.dropout = nn.Dropout2d(p=dropout) if dropout and dropout > 0 else nn.Identity()

        self.act = get_activation(activation)

    def forward(self, x):
        identity = x

        out = self.act(self.n1(x))
        out = self.c1(out)

        out = self.act(self.n2(out))
        out = self.c2(out)

        out = self.act(self.n3(out))
        out = self.c3(out)

        out = self.se(out)
        out = self.dropout(out)

        out = out + identity
        return out


class CNNDeluxe(nn.Module):
    """
    UNet-like encoder for classification with:
      - GroupNorm(<=32 groups, adaptive for low channel counts)
      - Pre-activation residual bottleneck blocks
      - Optional Squeeze-and-Excite and Dropout inside blocks
      - Configurable number of downsampling stages via `num_downsamples` (default 2)
        For 28x28 inputs, the hard maximum allowed is 4 (28->14->7->4->2->1).
        We raise if user tries more than that.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        base_channels: int = 16,
        num_blocks: int = 2,
        num_downsamples: int = 2,
        use_se: bool = True,
        dropout: float = 0.0,
        activation: str = "silu",
    ):
        super().__init__()

        if num_downsamples > 5:
            raise ValueError(
                "num_downsamples > 5 would downsample a 28x28 input below 1x1 (28→14→7→4→2→1)."
            )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.num_downsamples = num_downsamples
        self.use_se = use_se
        self.dropout = dropout
        self.activation = activation
        self.act = get_activation(activation)
        # Stem keeps it close to the original CNN
        self.enc_conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)

        # Stage 0 (no downsample)
        self.enc_blocks0 = nn.Sequential(
            *[PreActBottleneckResBlock(base_channels, use_se=use_se, dropout=dropout, activation=activation)
              for _ in range(num_blocks)]
        )

        # Build pyramid: each stage downsamples (AvgPool2d) then widens channels by 2x
        self.downs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.blocks = nn.ModuleList()

        curr_ch = base_channels
        for _ in range(num_downsamples):
            self.downs.append(nn.AvgPool2d(kernel_size=2, ceil_mode=True))
            self.convs.append(nn.Conv2d(curr_ch, curr_ch * 2, kernel_size=3, padding=1, bias=False))
            curr_ch *= 2
            self.blocks.append(nn.Sequential(
                *[PreActBottleneckResBlock(curr_ch, use_se=use_se, dropout=dropout, activation=activation)
                  for _ in range(num_blocks)]
            ))

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(*[
            nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(curr_ch, curr_ch // 2),
            self.act,
            nn.Linear(curr_ch // 2, num_classes)
        ])

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x_im, x_tsne=None):
        # Stem + Stage 0
        x = self.enc_conv_in(x_im)
        x = self.enc_blocks0(x)

        # Stages 1..K
        for down, conv, block in zip(self.downs, self.convs, self.blocks):
            x = down(x)
            x = conv(x)
            x = block(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out

def get_activation(name: str = "silu"):
    name = (name or "silu").lower()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")

class ResidualMLPBlock(nn.Module):
    """
    Pre-norm residual MLP block with optional gating (SwiGLU/GEGLU-style) and dropout.

    If gated=False:
        y = W2( act( LN(x) @ W1 ) )
    If gated=True (SwiGLU-ish):
        u, v = split( LN(x) @ W1G , 2, dim=-1)
        y = W2( act(u) * v )

    Output is added to the residual (x) directly (same width).
    """
    def __init__(self, width: int, gated: bool = False, activation: str = "silu", dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.gated = gated
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        if gated:
            # project to 2*width for gate (u,v), then back to width
            self.fc_in = nn.Linear(width, 2 * width)
        else:
            self.fc_in = nn.Linear(width, width)

        self.fc_out = nn.Linear(width, width)

        # Optional residual scaling can help deep stacks; not requested, so fixed at 1.0
        self.res_scale = 1.0

    def forward(self, x):
        h = self.norm(x)

        if self.gated:
            h = self.fc_in(h)
            u, v = torch.chunk(h, 2, dim=-1)
            h = self.act(u) * v
        else:
            h = self.fc_in(h)
            h = self.act(h)

        h = self.dropout(h)
        h = self.fc_out(h)
        return x + self.res_scale * h

class MLPDeluxe(nn.Module):
    """
    Flexible MLP with:
      - SiLU activation by default (configurable)
      - Optional gated blocks (SwiGLU/GEGLU-style)
      - Pre-norm residual MLP blocks
      - Optional dropout (default 0.0)

    Args:
        input_dim:   Input feature size
        output_dim:  Output feature size
        width:       Hidden width used across residual blocks
        num_layers:  Number of residual blocks
        activation:  'silu' | 'gelu' | 'relu' | 'tanh' (default 'silu')
        gated:       If True, uses gated blocks (project to 2*width and gate)
        dropout:     Dropout probability inside each block (default 0.0)
        head_dropout: Dropout before the output head (default 0.0)
    """
    def __init__(
        self,
        tsne: bool = False,
        output_dim: int = 10,
        width: int = 512,
        num_layers: int = 6,
        activation: str = "silu",
        gated: bool = False,
        dropout: float = 0.0,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.tsne = tsne
        input_dim = 2 if tsne else 28 * 28
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(width, gated=gated, activation=activation, dropout=dropout)
              for _ in range(num_layers)]
        )
        self.head_norm = nn.LayerNorm(width)
        self.head_dropout = nn.Dropout(head_dropout) if head_dropout and head_dropout > 0.0 else nn.Identity()
        self.output = nn.Linear(width, output_dim)

        # Non-residual activation after the input projection is common and harmless
        self.input_act = get_activation(activation)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x_im, x_tsne=None):
        if self.tsne:
            h = self.input_act(self.input(x_tsne))
        else:
            h = self.input_act(self.input(x_im.view(x_im.size(0), -1)))
        h = self.blocks(h)
        h = self.head_norm(h)
        h = self.head_dropout(h)
        return self.output(h)

str_to_class = {"MLP": MLP, "models.MLP": MLP,
                "CNN": CNN, "models.CNN": CNN,
                "MLPDeluxe": MLPDeluxe, "models.MLPDeluxe": MLPDeluxe,
                "CNNDeluxe": CNNDeluxe, "models.CNNDeluxe": CNNDeluxe}

def get_models_dict(tsne=False):
    models = {
            "mlp0": lambda: MLP(tsne=tsne, hidden_dim=64, num_hidden_layers=2, output_dim=10),
            "mlp1": lambda: MLP(tsne=tsne, hidden_dim=128, num_hidden_layers=4, output_dim=10),
            "mlp2": lambda: MLP(tsne=tsne, hidden_dim=256, num_hidden_layers=6, output_dim=10),
            "mlp3": lambda: MLP(tsne=tsne, hidden_dim=512, num_hidden_layers=8, output_dim=10),
            "mlp4": lambda: MLP(tsne=tsne, hidden_dim=1024, num_hidden_layers=10, output_dim=10),

            "cnn0": lambda: CNN(base_channels=4, num_blocks=1),
            "cnn1": lambda: CNN(base_channels=8, num_blocks=2),
            "cnn2": lambda: CNN(base_channels=16, num_blocks=3),
            "cnn3": lambda: CNN(base_channels=32, num_blocks=4),
            "cnn4": lambda: CNN(base_channels=64, num_blocks=5),

            "MLP0": lambda: MLPDeluxe(tsne=tsne, width=64, num_layers=2),
            "MLP1": lambda: MLPDeluxe(tsne=tsne, width=128, num_layers=4),
            "MLP2": lambda: MLPDeluxe(tsne=tsne, width=256, num_layers=6),
            "MLP3": lambda: MLPDeluxe(tsne=tsne, width=512, num_layers=8),
            "MLP4": lambda: MLPDeluxe(tsne=tsne, width=1024, num_layers=10),

            "CNN0": lambda: CNNDeluxe(base_channels=8,num_blocks=1,num_downsamples=2),
            "CNN1": lambda: CNNDeluxe(base_channels=16,num_blocks=2,num_downsamples=3),
            "CNN2": lambda: CNNDeluxe(base_channels=32,num_blocks=4,num_downsamples=3),
            "CNN3": lambda: CNNDeluxe(base_channels=32,num_blocks=4,num_downsamples=4),
            "CNN4": lambda: CNNDeluxe(base_channels=32,num_blocks=4,num_downsamples=5)
    }
    return models

