import torch
import torchvision
vgg16 = torchvision.models.vgg16(pretrained = False)

# 保存方式1 保存网络模型结构和参数
torch.save(vgg16,"vgg16_method1.pth")

# 保存方21 保存vgg16参数为字典 不保存结构 官方推荐保存方式
torch.save(vgg16.state_dict(),"vgg16_method2.pth")