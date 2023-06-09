import torch
import time

from nets.FANet.fa_net import FANet
from nets.FastSCNN.fast_scnn import FastSCNN
from nets.SwiftNet.swiftnet import SwiftNet
from nets.BiseNet.bisenet import BiSeNet
from nets.FastSegFormer.fast_segformer import FastSegFormer
from nets.UNet.swinTS_Att_Unet import swinTS_Att_Unet
from nets.ENet.enet import ENet
from nets.PIDNet.pidnet import get_pred_model
from nets.BiseNetV2.bisenetV2 import BiSeNetV2



def run(model, size, name):
    model.cuda()
    model.eval()
    t_cnt = 0.0
    with torch.no_grad():
        input = torch.randn(size).cuda()
        torch.cuda.synchronize()
        x = model(input)
        x = model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_ts = time.time()

        for i in range(100):
            x = model(input)
        torch.cuda.synchronize()
        end_ts = time.time()

        t_cnt = end_ts - start_ts  # t_cnt + (end_ts-start_ts)
    print("=======================================")
    print("Model Name: " + name)
    print("FPS: %f" % (100 / t_cnt))
    # print("=======================================")


if __name__ == "__main__":

    ENet = ENet(4)
    run(ENet, size=(1, 3, 224, 224), name='ENet')

    FANet18 = FANet(4, backbone='resnet18')
    run(FANet18, size=(1, 3, 224, 224), name='FANet-18')

    FANet34 = FANet(4, backbone='resnet34')
    run(FANet34, size=(1, 3, 224, 224), name='FANet-34')

    FastSCNN = FastSCNN(4)
    run(FastSCNN, size=(1, 3, 224, 224), name='Fast-SCNN')

    SwiftNet = SwiftNet(4)
    run(SwiftNet, size=(1, 3, 224, 224), name='SwiftNetRN18')

    BiSeNet = BiSeNet(4)
    run(BiSeNet, size=(1, 3, 224, 224), name='BiSeNet')

    PIDNet_S = get_pred_model('pidnet_s', 4)
    run(PIDNet_S, size=(1, 3, 224, 224), name='PIDNet-S')

    PIDNet_M = get_pred_model('pidnet_m', 4)
    run(PIDNet_M, size=(1, 3, 224, 224), name='PIDNet-M')

    PIDNet_L = get_pred_model('pidnet_l', 4)
    run(PIDNet_L, size=(1, 3, 224, 224), name='PIDNet-L')

    FastSegFormer_1 = FastSegFormer(4, backbone='efficientformerV2_s0')
    run(FastSegFormer_1, size=(1, 3, 224, 224), name='FastSegFormer_EF_P')

    FastSegFormer_2 = FastSegFormer(4, backbone='efficientformerV2_s0', Pyramid='multiscale')
    run(FastSegFormer_2, size=(1, 3, 224, 224), name='FastSegFormer_EF_M')

    FastSegFormer_3 = FastSegFormer(4, backbone='efficientformerV2_s0', Pyramid='multiscale', cnn_branch=True)
    run(FastSegFormer_3, size=(1, 3, 224, 224), name='FastSegFormer_EF_M_H')

    FastSegFormer_4 = FastSegFormer(4, backbone='poolformer_s12')
    run(FastSegFormer_4, size=(1, 3, 224, 224), name='FastSegFormer_PF_P')

    FastSegFormer_5 = FastSegFormer(4, backbone='poolformer_s12', Pyramid='multiscale')
    run(FastSegFormer_5, size=(1, 3, 224, 224), name='FastSegFormer_PF_M')

    FastSegFormer_6 = FastSegFormer(4, backbone='poolformer_s12', Pyramid='multiscale', cnn_branch=True)
    run(FastSegFormer_6, size=(1, 3, 224, 224), name='FastSegFormer_PF_M_H')

    Swin_T_Att_UNet = swinTS_Att_Unet(4, backbone='swin_T_224')
    run(Swin_T_Att_UNet, size=(1, 3, 224, 224), name='Swin_T_Att_UNet')

    """
    For related work:
    Paper:Real-Time Grading of Defect Apples Using Semantic Segmentation Combination with a Pruned YOLO V4 Network
    models:BiSeNetV2 + pruned YOLO V4
    num_classes: defect + stem + background
    """

    BiSeNetV2 = BiSeNetV2(n_classes=3)
    run(BiSeNetV2, size=(1, 3, 416, 416), name='Related work: BiSeNetV2(with pruned YOLO V4)')



    print("=======================================")
    print("Note: ")
    print("In the paper, a single RTX 3060 GPU is adopted for evaluation.")
    print("=======================================")
