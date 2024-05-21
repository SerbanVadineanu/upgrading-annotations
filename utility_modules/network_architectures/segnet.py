from torch import nn


class CBNR(nn.Module):
    def __init__(self, input_size, output_size):
        super(CBNR, self).__init__()
        self.cbnr = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=output_size,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=output_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.cbnr(x)


class DoubleDownConv(nn.Module):

    def __init__(self, input_size, output_size):
        super(DoubleDownConv, self).__init__()
        self.block = nn.Sequential(
            CBNR(input_size, output_size),
            CBNR(output_size, output_size)
            )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.block(x)
        size_before = x.size()
        x, ind = self.maxpool(x)

        return x, ind, size_before


class DoubleUpConv(nn.Module):

    def __init__(self, input_size, output_size):
        super(DoubleUpConv, self).__init__()
        self.block = nn.Sequential(
            CBNR(input_size, input_size),
            CBNR(input_size, output_size)
        )
        self.maxunpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x, ind, size_before):
        x = self.maxunpool(x, indices=ind, output_size=size_before)

        return self.block(x)


class TripleDownConv(nn.Module):

    def __init__(self, input_size, output_size):
        super(TripleDownConv, self).__init__()
        self.block = nn.Sequential(
            CBNR(input_size, output_size),
            CBNR(output_size, output_size),
            CBNR(output_size, output_size)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.block(x)
        size_before = x.size()
        x, ind = self.maxpool(x)

        return x, ind, size_before


class TripleUpConv(nn.Module):
    def __init__(self, input_size, output_size):
        super(TripleUpConv, self).__init__()
        self.block = nn.Sequential(
            CBNR(input_size, input_size),
            CBNR(input_size, input_size),
            CBNR(input_size, output_size)
        )
        self.maxunpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x, ind, size_before):
        x = self.maxunpool(x, indices=ind, output_size=size_before)

        return self.block(x)


class SegNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SegNet, self).__init__()
        self.ddownconv1 = DoubleDownConv(n_channels, 64)
        self.ddownconv2 = DoubleDownConv(64, 128)
        self.tdownconv1 = TripleDownConv(128, 256)
        self.tdownconv2 = TripleDownConv(256, 512)
        self.tdownconv3 = TripleDownConv(512, 512)
        self.tupconv1 = TripleUpConv(512, 512)
        self.tupconv2 = TripleUpConv(512, 256)
        self.tupconv3 = TripleUpConv(256, 128)
        self.dupconv1 = DoubleUpConv(128, 64)
        self.dupconv2 = DoubleUpConv(64, n_classes)

    def forward(self, x):
        x, ind1, sb1 = self.ddownconv1(x)
        x, ind2, sb2 = self.ddownconv2(x)
        x, ind3, sb3 = self.tdownconv1(x)
        x, ind4, sb4 = self.tdownconv2(x)
        x, ind5, sb5 = self.tdownconv3(x)
        x = self.tupconv1(x, ind5, sb5)
        x = self.tupconv2(x, ind4, sb4)
        x = self.tupconv3(x, ind3, sb3)
        x = self.dupconv1(x, ind2, sb2)
        x = self.dupconv2(x, ind1, sb1)

        return x

