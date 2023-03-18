from PIL.Image import Image
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path ="dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)
writer = SummaryWriter("logs")
# 传入图片地址
# print(img)
# 输出 PIL 的image类型 尺寸
# transform 如何使用
tensor_trans = transforms.ToTensor()
# 将 PIL 的 image 转化成 tensor的数据类型
tensor_img = tensor_trans(img)
writer.add_image("Tensor_image", tensor_img)
writer.close()
# 可以在终端中：tensorboard --logdir=logs 查看效果