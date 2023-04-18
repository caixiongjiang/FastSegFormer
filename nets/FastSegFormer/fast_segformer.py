from nets.FastSegFormer.poolformer import poolformer_s12_feat as create_model_pool_s12
from nets.FastSegFormer.poolformer import poolformer_s24_feat as create_model_pool_s24
from nets.FastSegFormer.poolformer import poolformer_s36_feat as create_model_pool_s36
from nets.FastSegFormer.efficientformer_v2 import efficientformerv2_s0_feat as create_model_efficientV2_s0
from nets.UNet.swin_transformer import swin_tiny_patch4_window7_224 as create_model_T_224

import torch.nn as nn
import torch
import torch.nn.functional as F




class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)




class _DWConv(nn.Module):
    """
    Depthwise Convolutions(DW Conv)
    """
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)




class _DSConv(nn.Module):
    """
    Depthwise Separable Convolutions(DS Conv)
    """

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)





class LinearBottleneck(nn.Module):
    """
    LinearBottleneck(LB) used in MobileNetV2
    """

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out




class PyramidPooling(nn.Module):
    """Pyramid pooling module(PPM)"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        # print(size) 7*7(input:224*224)  16*16(input:512*512)
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class MultiscalePyramid(nn.Module):
    """
    Multi-scale Pyramid(MSP)
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(MultiscalePyramid, self).__init__()
        # 保持输入前后图片的尺寸保持不变 stride = 1, padding = (k-1)/2
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, kernel_size=1, padding=0)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, kernel_size=3, padding=1)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, kernel_size=5, padding=2)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, kernel_size=7, padding=3)
        self.conv5 = _ConvBNReLU(inter_channels, inter_channels, 1, **kwargs)
        self.conv6 = _ConvBNReLU(inter_channels, inter_channels, 1, **kwargs)
        self.conv7 = _ConvBNReLU(inter_channels, inter_channels, 1, **kwargs)
        self.conv8 = _ConvBNReLU(inter_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, kernel_size=1)


    def forward(self, x):

        x1 = self.conv5(self.conv1(x))
        x2 = self.conv6(self.conv2(x))
        x3 = self.conv7(self.conv3(x))
        x4 = self.conv8(self.conv4(x))
        x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = self.out(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), bottleneck_num=3, Pyramid="pooling", **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        if Pyramid == "pooling":
            if bottleneck_num == 1:
                self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 1)
                self.ppm = PyramidPooling(block_channels[0], out_channels)
            if bottleneck_num == 2:
                self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 1)
                self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 1)
                self.ppm = PyramidPooling(block_channels[1], out_channels)
            elif bottleneck_num == 3:
                self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 1)
                self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 1)
                self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
                self.ppm = PyramidPooling(block_channels[2], out_channels)
        elif Pyramid == "multiscale":
            if bottleneck_num == 1:
                self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 1)
                self.ppm = MultiscalePyramid(block_channels[0], out_channels)
            if bottleneck_num == 2:
                self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 1)
                self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 1)
                self.ppm = MultiscalePyramid(block_channels[1], out_channels)
            elif bottleneck_num == 3:
                self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 1)
                self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 1)
                self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
                self.ppm = MultiscalePyramid(block_channels[2], out_channels)

        self.bottleneck_num = bottleneck_num

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        if self.bottleneck_num == 2:
            x = self.bottleneck2(x)
        elif self.bottleneck_num == 3:
            x = self.bottleneck2(x)
            x = self.bottleneck3(x)
        x = self.ppm(x)
        return x



class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, higher_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(higher_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)
        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)




class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x




class FastSegFormer(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone="poolformer_s12", Pyramid="pooling", fork_feat=False, cnn_branch=False):
        """
        backbone poolformer out_channel = 512 and the number of channels in each layer of different versions of network
        models is the same, but the number of blocks is different.

        if the network is used in knowledge distillation training, set the option fork_feat=True.

        if cnn_branch = True, the parameters of the network will increase
        """


        super(FastSegFormer, self).__init__()

        if cnn_branch:
            if backbone == "efficientformerV2_s0" or backbone == "poolformer_s12":
                self.cnn1 = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32)
                )
                self.cnn2 = _DSConv(32, 32, stride=1)
                self.cnn_up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(32, num_classes, 1)
                )
            elif backbone == "poolformer_s24" or backbone == "poolformer_s36" or backbone == "swin_T_224" or backbone == "convnext_t":
                self.cnn1 = nn.Sequential(
                    nn.Conv2d(3, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256)
                )
                self.cnn2 = _ConvBNReLU(256, 256, 3, padding=1)
                self.cnn_up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    _ConvBNReLU(256, 128, 3, padding=1),
                    nn.Conv2d(128, num_classes, 1)
                )

        if backbone == "poolformer_s12":
            self.common_backbone = create_model_pool_s12()
            self.convert_channel = _ConvBNReLU(512, 256, kernel_size=1)
            if Pyramid == "pooling":
                self.global_feature = GlobalFeatureExtractor(in_channels=256, block_channels=(256,), out_channels=512, t=3, num_blocks=(2,),
                                                             bottleneck_num=1, Pyramid="pooling")
            elif Pyramid == "multiscale":
                self.global_feature = GlobalFeatureExtractor(in_channels=256, block_channels=(256,), out_channels=512, t=3, num_blocks=(2,),
                                                             bottleneck_num=1, Pyramid="multiscale")
            self.FFM = FeatureFusionModule(higher_in_channels=128, lower_in_channels=512, out_channels=512) # lower代表特征图的大小更小（16*16）
            if cnn_branch:
                self.classify = Classifer(512, 32, stride=1)
            else:
                self.classify = Classifer(512, num_classes, stride=1)

        elif backbone == "poolformer_s24":
            self.common_backbone = create_model_pool_s24()
            self.convert_channel = _ConvBNReLU(512, 256, kernel_size=1)
            if Pyramid == "pooling":
                self.global_feature = GlobalFeatureExtractor(in_channels=256, block_channels=(256, 320), out_channels=512, t=3, num_blocks=(3, 3),
                                                             bottleneck_num=2, Pyramid="pooling")
            elif Pyramid == "multiscale":
                self.global_feature = GlobalFeatureExtractor(in_channels=256, block_channels=(256, 320), out_channels=512, t=3, num_blocks=(3, 3),
                                                             bottleneck_num=2, Pyramid="multiscale")
            self.FFM = FeatureFusionModule(higher_in_channels=128, lower_in_channels=512, out_channels=512)
            if cnn_branch:
                self.classify = Classifer(512, 256, stride=1)
            else:
                self.classify = Classifer(512, num_classes, stride=1)

        elif backbone == "poolformer_s36":
            self.common_backbone = create_model_pool_s36()
            self.convert_channel = _ConvBNReLU(512, 256, kernel_size=1)
            if Pyramid == "pooling":
                self.global_feature = GlobalFeatureExtractor(in_channels=256, block_channels=(256, 320, 384), out_channels=512, t=4, num_blocks=(3, 3, 3),
                                                             bottleneck_num=3, Pyramid="pooling")
            elif Pyramid == "multiscale":
                self.global_feature = GlobalFeatureExtractor(in_channels=256, block_channels=(256, 320, 384), out_channels=512, t=4, num_blocks=(3, 3, 3),
                                                             bottleneck_num=3, Pyramid="multiscale")
            self.FFM = FeatureFusionModule(higher_in_channels=128, lower_in_channels=512, out_channels=512)
            if cnn_branch:
                self.classify = Classifer(512, 256, stride=1)
            else:
                self.classify = Classifer(512, num_classes, stride=1)

        elif backbone == "efficientformerV2_s0":
            self.common_backbone = create_model_efficientV2_s0()
            self.convert_channel = _ConvBNReLU(176, 128, kernel_size=1)
            if Pyramid == "pooling":
                self.global_feature = GlobalFeatureExtractor(in_channels=128, block_channels=(128, 160), out_channels=320, t=3, num_blocks=(3, 3),
                                                         bottleneck_num=2, Pyramid="pooling")
            elif Pyramid == "multiscale":
                self.global_feature = GlobalFeatureExtractor(in_channels=128, block_channels=(128, 160), out_channels=320, t=3, num_blocks=(3, 3),
                                                             bottleneck_num=2, Pyramid="multiscale")
            self.FFM = FeatureFusionModule(higher_in_channels=48, lower_in_channels=320, out_channels=320) # 注意因为包含分组卷积，out_channel必须是lower_in_channels的倍数
            if cnn_branch:
                self.classify = Classifer(320, 32, stride=1)
            else:
                self.classify = Classifer(320, num_classes, stride=1)

        elif backbone == "swin_T_224":
            self.common_backbone = create_model_T_224()

            # 将swin transfomer的分类头去掉
            remove_head = nn.Sequential()
            self.common_backbone.avgpool = remove_head
            self.common_backbone.head = remove_head

            self.convert_channel = _ConvBNReLU(768, 256, kernel_size=1)
            if Pyramid == "pooling":
                self.global_feature = GlobalFeatureExtractor(in_channels=256, block_channels=(256, 320, 384), out_channels=512, t=4, num_blocks=(3, 3, 3),
                                                             bottleneck_num=3, Pyramid="pooling")
            elif Pyramid == "multiscale":
                self.global_feature = GlobalFeatureExtractor(in_channels=256, block_channels=(256, 320, 384), out_channels=512, t=4, num_blocks=(3, 3, 3),
                                                             bottleneck_num=3, Pyramid="multiscale")
            self.FFM = FeatureFusionModule(higher_in_channels=192, lower_in_channels=512, out_channels=512)
            if cnn_branch:
                self.classify = Classifer(512, 256, stride=1)
            else:
                self.classify = Classifer(512, num_classes, stride=1)

        self.backbone = backbone
        self.fork_feat = fork_feat
        self.cnn_branch = cnn_branch


    def forward(self, x):
        size = x.size()[2:]
        if self.cnn_branch:
            half_input = F.interpolate(x, (112, 112), mode='bilinear', align_corners=True)
        features_out = []
        if self.backbone == "swin_T_224":
            x, H, W, feat1, feat2, feat3 = self.common_backbone(x)

            # 转回卷积网络所需要尺寸
            _, size1, C1 = feat1.shape
            feat1 = feat1.permute(0, 2, 1).contiguous().view(-1, C1, size1 // (8 * H), 8 * H)

            _, size2, C2 = feat2.shape
            feat2 = feat2.permute(0, 2, 1).contiguous().view(-1, C2, size2 // (4 * H), 4 * H)

            _, size3, C3 = feat3.shape
            feat3 = feat3.permute(0, 2, 1).contiguous().view(-1, C3, size3 // (2 * H), 2 * H)

            x = x.view(-1, 768, H, W)

        else:
            feat1, feat2, feat3, x = self.common_backbone(x)
        """
        if input == 224*224(poolformer)
        """
        # print(feat1.shape)  # [B, 64, 56, 56]
        # print(feat2.shape)  # [B, 128, 28, 28]
        # print(feat3.shape)  # [B, 320, 14, 14]
        # print(x.shape)      # [B, 512, 7, 7]
        # print(x.size())

        features_out.append(feat1)
        features_out.append(feat2)
        features_out.append(feat3)
        features_out.append(x)



        # 如果骨干为Poolformer_s12,图像为[B, 512, 16, 16](input=512)
        # 如果骨干为EfficientformerV2_s0,图像为[B, 176, 7, 7](input=224)
        x = self.convert_channel(x)
        x = self.global_feature(x)
        features_out.append(x)
        x = self.FFM(feat2, x)
        features_out.append(x)
        x = self.classify(x)
        if self.cnn_branch:
            x = F.interpolate(x, (112, 112), mode='bilinear', align_corners=True)
            x1 = x + self.cnn1(half_input)
            x = self.cnn2(x1)
            features_out.append(x)
            x = self.cnn_up(x)
            features_out.append(x)
        else:
            x = F.interpolate(x, size, mode='bilinear', align_corners=True)
            features_out.append(x)
        # print(x.shape)
        if self.fork_feat:
            return features_out


        return x


    def freeze_backbone(self):
        if self.backbone == "poolformer_s12":
            for param in self.common_backbone.parameters():
                param.requires_grad = False
        elif self.backbone == "poolformer_s24":
            for param in self.common_backbone.parameters():
                param.requires_grad = False
        elif self.backbone == "poolformer_s36":
            for param in self.common_backbone.parameters():
                param.requires_grad = False
        elif self.backbone == "efficientformerV2_s0":
            for param in self.common_backbone.parameters():
                param.requires_grad = False
        elif self.backbone == "swin_T_224":
            for param in self.common_backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "poolformer_s12":
            for param in self.common_backbone.parameters():
                param.requires_grad = True
        elif self.backbone == "poolformer_s24":
            for param in self.common_backbone.parameters():
                param.requires_grad = True
        elif self.backbone == "poolformer_s36":
            for param in self.common_backbone.parameters():
                param.requires_grad = True
        elif self.backbone == "efficientformerV2_s0":
            for param in self.common_backbone.parameters():
                param.requires_grad = True
        elif self.backbone == "swin_T_224":
            for param in self.common_backbone.parameters():
                param.requires_grad = True


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('The model size is ：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


if __name__ == '__main__':
    # student
    # model = FastSegFormer(num_classes=4, pretrained=False, backbone="poolformer_s12", Pyramid="pooling")
    # model = FastSegFormer(num_classes=4, pretrained=False, backbone="poolformer_s12", Pyramid="multiscale")
    # model = FastSegFormer(num_classes=4, pretrained=False, backbone="poolformer_s12", Pyramid="multiscale", cnn_branch=True)
    model = FastSegFormer(num_classes=4, pretrained=False, backbone="poolformer_s12", Pyramid="multiscale", fork_feat=True, cnn_branch=True)
    # model = FastSegFormer(num_classes=4, pretrained=False, backbone="efficientformerV2_s0", Pyramid="pooling")
    # model = FastSegFormer(num_classes=4, pretrained=False, backbone="efficientformerV2_s0", Pyramid="multiscale", fork_feat=True)
    # model = FastSegFormer(num_classes=4, pretrained=False, backbone="efficientformerV2_s0", Pyramid="multiscale", fork_feat=True, cnn_branch=True)


    # teacher
    # model = FastSegFormer(num_classes=4, pretrained=False, backbone="poolformer_s24", Pyramid="multiscale", fork_feat=True, cnn_branch=True)
    # model = FastSegFormer(num_classes=4, pretrained=False, backbone="poolformer_s36", Pyramid="multiscale", fork_feat=True, cnn_branch=True)
    # model = FastSegFormer(num_classes=4, pretrained=False, backbone="swin_T_224", Pyramid="multiscale")
    a, b, c, d, e = getModelSize(model)
    img = torch.randn((1, 3, 224, 224))
    # print(model)
    outputs = model(img)
    # print(outputs.shape)
    for output in outputs:
        print(output.shape)