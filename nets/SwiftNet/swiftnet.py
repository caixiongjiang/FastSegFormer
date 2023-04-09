import torch
import torch.nn as nn
from itertools import chain

from nets.SwiftNet.resnet.resnet_single_scale import resnet18
from nets.SwiftNet.util import _BNReluConv, upsample


class SwiftNet(nn.Module):
    def __init__(self,  num_classes=19, use_bn=False):
        super(SwiftNet, self).__init__()
        self.backbone = resnet18(pretrained=True, use_bn=False)
        self.num_classes = num_classes
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)

    def forward(self, pyramid):
        features = self.backbone(pyramid)
        logits = self.logits.forward(features[0])
        return logits

    def prepare_data(self, batch, image_size, device=torch.device('cuda')):
        if image_size is None:
            image_size = batch['target_size']
        pyramid = [p.clone().detach().requires_grad_(False).to(device) for p in batch['pyramid']]
        return {
            'pyramid': pyramid,
            'image_size': image_size,
            'target_size': batch['target_size_feats']
        }

    def do_forward(self, batch, image_size=None):
        data = self.prepare_data(batch, image_size)
        return self.forward(**data)

    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()

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
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)
    model = SwiftNet(4)
    output = model(img)
    a, b, c, d, e = getModelSize(model)
    print(output.shape)
