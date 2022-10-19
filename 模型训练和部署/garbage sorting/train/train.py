import tensorboardX
import torch
from model import Vit, SwinTransformer, ResNet50Pretrain, ResNet50InitRandom
import torch.nn as nn
import math
from utils import train_one_epoch, evaluate, MyDataLoader
from torchvision import transforms

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    epoch_num = 30
    lr = 0.01
    lrf = 0.01
    batch_size = 16
    config = {
        "vit": Vit,
        "swin": SwinTransformer,
        "resnet_pre": ResNet50Pretrain,
        "resnet_ran": ResNet50InitRandom
    }
    model_name = 'resnet_pre'
    model_creator = config[model_name]

    train_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    train_loader = MyDataLoader(batch_size, train_transforms, "../../../../垃圾分类数据集/train/*/*")
    test_loader = MyDataLoader(batch_size, test_transforms, "../../../../垃圾分类数据集/test/*/*")

    model = model_creator(num_labels=5, fine_tune=True).to(device)
    parameters_update = [p for p in model.parameters() if p.requires_grad]
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(parameters_update, lr=lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / epoch_num)) / 2) * (1 - lrf) + lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    tb_writer = tensorboardX.SummaryWriter("log")

    acc = 0
    for epoch in range(epoch_num):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()

        val_loss, val_acc = evaluate(model=model,
                                     data_loader=test_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if val_acc > acc:
            torch.save(model.state_dict(), "./weights/{}-{}-{}.pth".format(model_name, epoch, val_acc))
            acc = val_acc
