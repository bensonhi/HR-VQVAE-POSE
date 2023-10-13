import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv3d(nn.Conv3d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)

    def create_mask(self, mask_type):
        _, _, kh, kw, kd = self.weight.size()
        if mask_type == 'A':
            self.mask[:, :, :kh//2, :, :] = 1
            self.mask[:, :, kh//2, :kw//2, :] = 1
            self.mask[:, :, kh//2, kw//2, :kd//2] = 1
        else:
            # Modify the else block to create a raster scan ordering mask
            h, w, d = self.weight.shape[-3:]
            diagonal_len = h + w - 1 + d
            mask = torch.zeros(diagonal_len, h, w, d, dtype=torch.float32)
            for i in range(diagonal_len):
                for j in range(max(0, i-d+1), min(i+1, h), 1):
                    for k in range(max(0, i-j-d+1), min(i-j+1, w), 1):
                        l = i-j-k
                        if 0 <= l < d:
                            mask[i, j, k, l] = 1
            self.mask = mask.sum(0).unsqueeze(0).unsqueeze(0).repeat(self.out_channels, self.in_channels, 1, 1, 1)
            self.mask[:, :, h//2, w//2, d//2+1:] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

    
class MultiheadAttention3d(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_heads):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.qkv = nn.Linear(input_channels, hidden_channels * 3)
        self.proj = nn.Linear(hidden_channels, input_channels)
        
    def forward(self, x):
        b, c, d, h, w = x.size()
        qkv = self.qkv(x).view(b, d, h, w, 3, self.num_heads, self.hidden_channels // self.num_heads)
        q, k, v = qkv.permute(4, 0, 5, 1, 6, 2, 3).contiguous().view(3, -1, d * h * w, self.hidden_channels // self.num_heads)
        attn_weights = torch.softmax((q @ k.transpose(-2, -1)) / (self.hidden_channels ** 0.5), dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(b, self.num_heads, d, h, w, self.hidden_channels // self.num_heads)
        attn_output = self.proj(attn_output.view(b, -1, self.hidden_channels))
        attn_output = attn_output.view(b, c, d, h, w)
        return attn_output

    
class SpatioTemporalPixelCNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Multi-head Attention layer
        self.attention = nn.MultiheadAttention(hidden_channels, num_heads=4, dropout=0.1)

        # Layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # First layer
                layer = MaskedConv3d('A', input_channels, hidden_channels, kernel_size=3, padding=1)
            else:
                # Intermediate layers
                layer = MaskedConv3d('B', hidden_channels, hidden_channels, kernel_size=3, padding=1)
                if (i+1) % 5 == 0:
                    layer = nn.Sequential(layer, nn.BatchNorm3d(hidden_channels), nn.ReLU(), self.attention)

            self.layers.append(layer)

        # Output convolution
        self.conv_out = nn.Conv3d(hidden_channels, input_channels, kernel_size=1, padding=0)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))

        x = self.conv_out(h)
        return x

    
#Prediction Spatio-temporal PixelCNN
class SpaTempPixelCNN_VP(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers, seq_len, img_shape):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.img_shape = img_shape

        # Spatio-temporal PixelCNN model
        self.st_pixelcnn = SpatioTemporalPixelCNN(input_channels, hidden_channels, num_layers)

        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Convolutional LSTM
        self.lstm = nn.ConvLSTM2d(
            input_size=(img_shape[0] // 2, img_shape[1] // 2),
            hidden_size=(img_shape[0] // 2, img_shape[1] // 2),
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=True
        )

    def forward(self, x):
        # Split input into a sequence of frames
        x = x.view(-1, self.seq_len, self.input_channels, self.img_shape[0], self.img_shape[1])

        # Encode spatio-temporal features with SpatioTemporalPixelCNN
        h = self.st_pixelcnn(x)

        # Upsample the features
        h = self.upsample(h)

        # Pass the features through a Convolutional LSTM
        h, _ = self.lstm(h)

        return h

