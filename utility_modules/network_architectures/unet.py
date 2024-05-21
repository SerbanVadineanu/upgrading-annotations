import torch
from torch import nn


def pad_tensor(small_tensor, large_tensor):
    '''
    :param small_tensor: tensor with smaller shape
    :param large_tensor: tensor with larger shape
    :return: padded x1
    '''

    y_diff = large_tensor.shape[2] - small_tensor.shape[2]
    x_diff = large_tensor.shape[3] - small_tensor.shape[3]

    small_tensor = nn.functional.pad(small_tensor, (x_diff // 2, x_diff - x_diff // 2,
                                y_diff // 2, y_diff - y_diff // 2))

    return small_tensor


class AttentionGate(nn.Module):
    def __init__(self, input_size, output_size):
        super(AttentionGate, self).__init__()
        self.conv_x = nn.Conv2d(in_channels=input_size, out_channels=output_size,
                                kernel_size=1)
        self.conv_g = nn.Conv2d(in_channels=input_size, out_channels=output_size,
                                kernel_size=1)
        self.conv_psi = nn.Conv2d(in_channels=input_size, out_channels=output_size,
                                  kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        x_out = self.conv_x(x)
        g_out = nn.functional.upsample(self.conv_g(g), size=x_out.shape[2:], mode='bilinear')
        psi = self.relu(x_out.add(g_out))
        psi_out = self.conv_psi(psi)
        alpha = self.sigmoid(psi_out)

        return x * alpha


class DoubleConv(nn.Module):
    def __init__(self, input_size, output_size):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=output_size,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=output_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_size, out_channels=output_size,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=output_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InputConv(nn.Module):
    def __init__(self, input_size, output_size):
        super(InputConv, self).__init__()
        self.conv = DoubleConv(input_size, output_size)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(input_size, output_size)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, input_size, output_size, bilinear=True):
        super(UpBlock, self).__init__()

        if bilinear:
            self.block = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.block = nn.ConvTranspose2d(in_channels=input_size // 2,
                                            out_channels=input_size // 2,
                                            kernel_size=2, stride=2)

        self.conv = DoubleConv(input_size, output_size)

    def forward(self, x1, x2):
        # x1 is the output of the previous layer
        x1 = self.block(x1)
        x1 = pad_tensor(x1, x2)

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutputConv(nn.Module):
    def __init__(self, input_size, output_size):
        super(OutputConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_size, out_channels=output_size,
                              kernel_size=1)

    def forward(self, x):
        x = self.conv(x)

        return x


class GeneralUNet(nn.Module):
    def __init__(self, n_channels, n_classes, divider):
        super(GeneralUNet, self).__init__()
        self.inp = InputConv(n_channels, 64//divider)
        self.down1 = DownBlock(64//divider, 128//divider)
        self.down2 = DownBlock(128//divider, 256//divider)
        self.down3 = DownBlock(256//divider, 512//divider)
        self.down4 = DownBlock(512//divider, 512//divider)
        self.up1 = UpBlock(1024//divider, 256//divider, False)
        self.up2 = UpBlock(512//divider, 128//divider, False)
        self.up3 = UpBlock(256//divider, 64//divider, False)
        self.up4 = UpBlock(128//divider, 64//divider, False)
        self.outp = OutputConv(64//divider, n_classes)

    def forward(self, x):
        pass


class UNet(GeneralUNet):
    def __init__(self, n_channels, n_classes, divider=1):
        super(UNet, self).__init__(n_channels=n_channels, n_classes=n_classes, divider=divider)

    def forward(self, x):
        # Encoder
        x1 = self.inp(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outp(x)

        return x


# A variant of UNet which receives as input both the input image and a perturbed label
# However, the perturbed label is also forwarded to the last layer
class UNetLabelPass(GeneralUNet):
    def __init__(self, n_channels, n_classes, divider=1):
        super(UNetLabelPass, self).__init__(n_channels=n_channels, n_classes=n_classes, divider=divider)
        self.outp = OutputConv(65 // divider, n_classes)

    def forward(self, x):
        perturbed_label = x[:, -1, :, :]
        perturbed_label = perturbed_label.unsqueeze(dim=1)
        # Encoder
        x1 = self.inp(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = torch.cat([x, perturbed_label], dim=1)
        x = self.outp(x)

        return x


class AttentionUNet(GeneralUNet):
    def __init__(self, n_channels, n_classes):
        divider = 2
        super(AttentionUNet, self).__init__(n_channels=n_channels, n_classes=n_classes, divider=divider)
        self.att1 = AttentionGate(512 // divider, 512 // divider)
        self.att2 = AttentionGate(256 // divider, 256 // divider)
        self.att3 = AttentionGate(128 // divider, 128 // divider)
        self.att4 = AttentionGate(64 // divider, 64 // divider)

    def forward(self, x):
        # Encoding
        x1 = self.inp(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoding
        x4 = self.att1(x4, x5)
        x = self.up1(x5, x4)
        x3 = self.att2(x3, x)
        x = self.up2(x, x3)
        x2 = self.att3(x2, x)
        x = self.up3(x, x2)
        x1 = self.att4(x1, x)
        x = self.up4(x, x1)
        x = self.outp(x)

        return x


class RecurrentUnetUnit(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(RecurrentUnetUnit, self).__init__()
        self.unet = UNet(n_channels + 2, n_classes, divider=2)

    def forward(self, x, prev_state=None):
        
        if prev_state is None:
            prev_state = torch.zeros(size=[x.shape[0], 2, *list(x.shape[2:])])
            prev_state = prev_state.to(x.get_device())

        x = torch.cat([x, prev_state], dim=1)

        return self.unet(x)


class RecurrentUnet(nn.Module):
    def __init__(self, n_channels, n_classes, reps):
        super(RecurrentUnet, self).__init__()
        self.reps = reps
        self.unit = RecurrentUnetUnit(n_channels, n_classes)

    def forward(self, x):
        states = []
        crt_state = None
        for _ in range(self.reps):
            crt_state = self.unit(x, crt_state)
            states.append(crt_state)

        return states

    
# UNet used for classification
class ClassUNet(nn.Module):
    def __init__(self, n_channels, n_classes, divider=1):
        super(ClassUNet, self).__init__()
        self.unet = UNet(n_channels=n_channels, n_classes=n_classes, divider=divider)
        self.linear = nn.Linear(n_classes, n_classes, bias=False)
    
    def forward(self, x):
        x = self.unet(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.linear(x)
        
        return x