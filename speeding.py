import torch
import time
import argparse

from nets.FANet.fa_net import FANet
from nets.FastSCNN.fast_scnn import FastSCNN
from nets.SwiftNet.swiftnet import SwiftNet
from nets.BiseNet.bisenet import BiSeNet
from nets.FastSegFormer.fast_segformer import FastSegFormer
from nets.UNet.swinTS_Att_Unet import swinTS_Att_Unet
from nets.ENet.enet import ENet
from nets.ESPNetV2.espnetv2_seg import ESPNetv2Segmentation



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

    parser = argparse.ArgumentParser(description='Testing')
    args = parser.parse_args()
    args.s = 2.0
    ESPNetV2_Seg = ESPNetv2Segmentation(args, 4)
    run(ESPNetV2_Seg, size=(1, 3, 224, 224), name='ESPNetV2_Seg')

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



    print("=======================================")
    print("Note: ")
    print("In the paper, a single RTX 3060 GPU is adopted for evaluation.")
    print("=======================================")
