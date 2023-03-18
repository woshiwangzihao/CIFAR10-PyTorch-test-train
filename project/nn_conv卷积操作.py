import torch
import torch.nn.functional as F
# tensor 实例化一个张量
# 输入图像
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
# 卷积核
kernel = torch.tensor([
    [1,2,1],
    [0,1,0],
    [2,1,0]
])

input = torch.reshape(input,(1,1,5,5))
# 1数据 1通道 高度5 宽度5
kernel = torch.reshape(kernel,(1,1,3,3))
print(input.shape)
print(kernel.shape)

output = F.conv2d(input,kernel,stride=1)
print(output)
output2 = F.conv2d(input,kernel,stride=2)
print(output2)
output3 = F.conv2d(input,kernel,stride=1)
print(output3)