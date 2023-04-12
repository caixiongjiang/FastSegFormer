'''
pip install thop
'''
import torch
from thop import profile
import argparse

from nets.UNet.swinTS_Att_Unet import swinTS_Att_Unet
from nets.FastSegFormer.fast_segformer import FastSegFormer
from nets.FANet.fa_net import FANet
from nets.FastSCNN.fast_scnn import FastSCNN
from nets.SwiftNet.swiftnet import SwiftNet
from nets.BiseNet.bisenet import BiSeNet
from nets.ENet.enet import ENet
from nets.PIDNet.pidnet import get_pred_model



device = torch.device("cpu")
#input_shape of model,batch_size=1

# net = ENet(4)
# net = FANet(4, backbone='resnet18')
# net = FANet(4, backbone='resnet34')
# net = FastSCNN(num_classes=4)
# net = SwiftNet(num_classes=4)
# net = BiSeNet(nclass=4)
net = get_pred_model('pidnet_s', num_classes=4)
# net = get_pred_model('pidnet_m', num_classes=4)
# net = get_pred_model('pidnet_l', num_classes=4)

# net = FastSegFormer(num_classes=4, pretrained=False, backbone="efficientformerV2_s0", Pyramid="pooling")
# net = FastSegFormer(num_classes=4, pretrained=False, backbone="efficientformerV2_s0", Pyramid="multiscale")
# net = FastSegFormer(num_classes=4, pretrained=False, backbone="efficientformerV2_s0", Pyramid="multiscale", cnn_branch=True)
# net = FastSegFormer(num_classes=4, pretrained=False, backbone="poolformer_s24", Pyramid="multiscale", cnn_branch=True)
# net = FastSegFormer(num_classes=4, pretrained=False, backbone="poolformer_s36", Pyramid="multiscale", cnn_branch=True)
# net = FastSegFormer(num_classes=4, pretrained=False, backbone="poolformer_s12", Pyramid="pooling")
# net = FastSegFormer(num_classes=4, pretrained=False, backbone="poolformer_s12", Pyramid="multiscale")
# net = FastSegFormer(num_classes=4, pretrained=False, backbone="poolformer_s12", Pyramid="multiscale", cnn_branch=True)
# net = FastSegFormer(num_classes=4, pretrained=False, backbone="poolformer_s24", Pyramid="multiscale")
# net = FastSegFormer(num_classes=4, pretrained=False, backbone="poolformer_s36", Pyramid="multiscale")
# net = FastSegFormer(num_classes=4, pretrained=False, backbone="swin_T_224", Pyramid="multiscale")
# net = swinTS_Att_Unet(num_classes=4, pretrained=False, backbone="swin_T_224")

input = torch.randn(1, 3, 224, 224)
flops, params = profile(net, inputs=(input, ))

print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
print("params=", str(params/1e6)+'{}'.format("M"))