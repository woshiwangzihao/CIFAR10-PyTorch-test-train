from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


writer = SummaryWriter("logs")

img = Image.open("dataset/train/ants/7759525_1363d24e88.jpg")
print(img)
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
# 变成tensor数据类型的image
writer.add_image("ToTense",img_tensor)


# normalize 归一化
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([6,3,2],[9,3,5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("normalize",img_norm,3)



# resize
print(img.size)
trans_resize = transforms.Resize((512,512))
# img PIL_> resize ->img_resize PIL类型
img_resize = trans_resize(img)
# img _resize PIL ->totensor ->img_resize ->tensor 类型
img_resize = trans_totensor(img_resize)

writer.add_image("resize",img_resize,0)
print(img_resize)


# compose - resize
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("resize",img_resize_2,1)

writer.close()
