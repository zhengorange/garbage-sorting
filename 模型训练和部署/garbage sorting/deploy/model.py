from transformers import ViTModel, SwinModel, ResNetModel, ResNetConfig
from torch import nn

'''
微调 google/vit-base-patch16-224-in21k
微调 microsoft/swin-large-patch4-window12-384-in22k
微调 microsoft/resnet-50
训练 resnet 50
'''


class Vit(nn.Module):
    def __init__(self, num_labels=5, fine_tune=True):
        super(Vit, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        if fine_tune:
            self.vit.requires_grad_(False)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        print("vit模型声明成功")

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classifier(outputs.pooler_output)
        return logits


class SwinTransformer(nn.Module):
    def __init__(self, num_labels=5, fine_tune=True):
        super(SwinTransformer, self).__init__()
        self.swin = SwinModel.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
        if fine_tune:
            self.swin.requires_grad_(False)
        self.classifier = nn.Linear(self.swin.config.hidden_size, num_labels)
        print("swin模型声明成功")

    def forward(self, pixel_values):
        outputs = self.swin(pixel_values=pixel_values)
        logits = self.classifier(outputs.pooler_output)
        return logits


class ResNet50Pretrain(nn.Module):
    def __init__(self, num_labels=5, fine_tune=True):
        super(ResNet50Pretrain, self).__init__()
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-50')
        if fine_tune:
            self.resnet.requires_grad_(False)
        self.classifier = nn.Linear(self.resnet.config.hidden_sizes[-1], num_labels)
        print("resnet50pretrain模型声明成功")

    def forward(self, pixel_values):
        outputs = self.resnet(pixel_values=pixel_values)
        # (batch * hidden_size * 1 * 1) -->  (batch * hidden_size)
        outputs = outputs.pooler_output.squeeze(dim=3)
        outputs = outputs.squeeze(dim=2)
        logits = self.classifier(outputs)
        return logits


class ResNet50InitRandom(nn.Module):
    def __init__(self, num_labels=5, fine_tune=True):
        super(ResNet50InitRandom, self).__init__()
        self.resnet = ResNetModel(ResNetConfig())
        if fine_tune:
            pass
            # self.resnet.requires_grad_(False)
        # 2048 <---> 5
        self.classifier = nn.Linear(self.resnet.config.hidden_sizes[-1], num_labels)
        print("resnet50random模型声明成功")

    def forward(self, pixel_values):
        outputs = self.resnet(pixel_values=pixel_values)
        # (batch * hidden_size * 1 * 1) -->  (batch * hidden_size)
        outputs = outputs.pooler_output.squeeze(dim=3)
        outputs = outputs.squeeze(dim=2)
        logits = self.classifier(outputs)
        return logits


if __name__ == '__main__':
    pass
    # model = ResNet50InitRandom()
    # # 估计参数量
    # # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # # print(pytorch_total_params)
    #
    # ims = torch.randn(2, 3, 224, 224)
    # out = model(ims)
    # print(ims.shape)
    # print(out.shape)
