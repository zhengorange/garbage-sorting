import torch
from torchvision import transforms
from model import Vit, SwinTransformer, ResNet50Pretrain, ResNet50InitRandom
import torch.nn.functional as F
import datetime


class Model:
    def __init__(self):
        self.device = torch.device("cpu")
        self.net = [Vit(), SwinTransformer(), ResNet50Pretrain(), ResNet50InitRandom()]
        self.net[0].load_state_dict(torch.load("./weights/vit-0-0.8427337865147896.pth", map_location='cpu'))
        self.net[1].load_state_dict(torch.load("./weights/swin-4-0.7837589564578358.pth", map_location='cpu'))
        self.net[2].load_state_dict(torch.load("./weights/resnet_pre-8-0.7833915120338049.pth", map_location='cpu'))
        self.net[3].load_state_dict(torch.load("./weights/resnet_ran-22-0.7064119051993386.pth", map_location='cpu'))
        self.net[0].to(self.device)
        self.net[1].to(self.device)
        self.net[2].to(self.device)
        self.net[3].to(self.device)
        self.net[0].eval()
        self.net[1].eval()
        self.net[2].eval()
        self.net[3].eval()
        self.test_transforms = transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.id2label = {
            0: "其他垃圾",
            1: "厨余垃圾",
            2: "可回收垃圾",
            3: "可回收物",
            4: "有害垃圾"
        }

    @torch.no_grad()
    def getLabel(self, im_data, met: int):
        starttime = datetime.datetime.now()
        inputs = self.test_transforms(im_data)
        inputs = torch.unsqueeze(inputs, dim=0)
        inputs = inputs.to(self.device)
        outputs = self.net[met](inputs)
        confidence, predict = torch.max(F.softmax(outputs.data, dim=1), dim=1)
        id = predict.cpu().numpy()[0]
        endtime = datetime.datetime.now()
        print("推理时间:{}", endtime - starttime)
        if id in self.id2label:
            return self.id2label[id], confidence
        else:
            return "none", confidence
