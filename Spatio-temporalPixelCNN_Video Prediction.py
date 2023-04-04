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
            self.mask[:, :, kh//2+1:, :, :] = 1
            self.mask[:, :, kh//2, kw//2+1:, :] = 1
            self.mask[:, :, kh//2, kw//2, kd//2+1:] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class SpatioTemporalPixelCNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # first layer
                layer = MaskedConv3d('A', input_channels, hidden_channels, kernel_size=3, padding=1)
            else:
                # intermediate layers
                layer = MaskedConv3d('B', hidden_channels, hidden_channels, kernel_size=3, padding=1)
            self.layers.append(layer)
            
        self.conv_out = nn.Conv3d(hidden_channels, input_channels, kernel_size=1, padding=0)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))
        x = self.conv_out(h)
        return x

#Spatio-temporal PixelCNN for anticipation
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
        # Split input into sequence of frames
        x = x.view(-1, self.seq_len, self.input_channels, self.img_shape[0], self.img_shape[1])

        # Encode spatio-temporal features with SpatioTemporalPixelCNN

