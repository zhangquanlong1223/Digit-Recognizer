# 张权龙
import torch
import torch.nn as nn #导入网络模型
import torch.nn.functional as F
import torch.optim as optim #导入优化器
from torchvision import datasets,transforms #对数据进行操作和支持transformation
from torch.utils.data import DataLoader#处理数据的库
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset


#定义超参 参数是通过训练出来的 超参是自己定义的
BATCH_SIZE = 16 #每批处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu') #如果有GPU使用GPU训练,如果没有的话使用cpu训练
EPOCHS= 20#训练数据集的轮次
transform = transforms.Compose([#构建转换transformation,随图像做处理(拉伸、旋转等) 类似opencv
    transforms.ToTensor(),#将图片转换成tensor
    transforms.Normalize((0.1307,), (0.3081,))#传入标准差 进行正则化降低模型复杂度,防止过拟合,把图片通道（一个通道）中的数据整理到[-1, 1]区间。
])

class DatasetFromTrainCSV(Dataset):#读取训练数据集
    def __init__(self, csv_path, height, width, transforms=None):
        self.data = pd.read_csv(csv_path)#读取数据部分
        self.labels = np.asarray(self.data.iloc[:, 0])#读取标签部分
        self.height = height#设置图片高
        self.width = width#设置图片宽
        self.transforms = transforms#对图片进行变换

    def __getitem__(self, index):
        single_image_label = self.labels[index]#实例对象P可以这样P[key]取值
        # 读取所有像素值，并将 1D array ([784]) reshape 成为 2D array ([28,28])
        img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(28, 28).astype(float)#将图片转换为28*28 单通道图像
        # 把 numpy array 格式的图像转换成灰度 PIL image
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # 将图像转换成 tensor 便于进行数据训练
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
            # 返回图像及其 label
        return (img_as_tensor, single_image_label)

    def __len__(self):#返回数据集的长度
        return len(self.data.index)

class DatasetFromTestCSV(Dataset):#读取测试数据集
    def __init__(self, csv_path, height, width, transforms=None):
        self.data = pd.read_csv(csv_path)
        self.height = height
        self.width = width
        self.transforms = transforms

    def __getitem__(self, index):
        # 读取所有像素值，并将 1D array ([784]) reshape 成为 2D array ([28,28])
        img_as_np = np.asarray(self.data.iloc[index][0:]).reshape(28, 28).astype(float)#没有标签,从零开始
        # 把 numpy array 格式的图像转换成灰度 PIL image
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # 将图像转换成 tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
            # 返回图像及其 label
        return (img_as_tensor)
    def __len__(self):
        return len(self.data.index)#返回数据集的长度



train_data = DatasetFromTrainCSV('./data/train.csv', 28, 28, transform)#构造对象
test_data = DatasetFromTestCSV("./data/test.csv", 28, 28, transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)#获取训练数据集的数据迭代器
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

#构建网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,10,5)#二维卷积 输入通道,因为是灰度图像的通道,所以是通道数是一 10:输出通道 5:kernel(卷积核)
        """
        二维卷积原理:
        self.conv1 = nn.Conv2d(in_channels=in_channel,#输入通道数，即输入数据的通道数 这里是单通道
                       out_channels=10,#输出通道数，即输出数据的通道数 
                       kernel_size=kernel_size,#卷积核大小，一般是int，也可tuple，如3【表示3x3】；（5，4）【表示5x4】
                       stride=1,#卷积移动的步长
                       padding=padding)# 是否用0填充数据四周
        """
        self.conv2=nn.Conv2d(10,20,3)#10输入通道,20是输出通道 3:kernel大小
        self.fc1=nn.Linear(20*10*10,500)#全连接层:20*10*10:输入通道 500:输出通道
        self.fc2=nn.Linear(500,10)#500:输入通道 10:输出通道
    def forward(self,x):#前向传播
        input_size=x.size(0)#batch_size*1*28*28  0代表取到batch_size
        x=self.conv1(x)#输入:batch_size*1*28*28 输出batch_size10*24*24(24=28-5+1)
        x=F.relu(x)#激活函数,保持不变 输出:batch_size*10*24*24
        x=F.max_pool2d(x,2,2)#降采样 输入:batch_ kernel:2*2  输入:batch*10*24*24 输出batch*10*12*12 减少运算量,提高运算的速度

        x=self.conv2(x)#回到卷积 输入:batch*10*12*12 输出:batch*20*10*10(12-3+1=10)
        x=F.relu(x)#激活函数

        x=x.view(input_size,-1)#拉平:flatten -1自动计算维度 20*10*10=2000

        x=self.fc1(x)#送入全连接层 输入batch*2000 输出:batch*500
        x=F.relu(x)#激活 保持shape不变

        x=self.fc2(x)#输入batch*500 输出:batch*10

        output=F.log_softmax(x,dim=1)#损失函数 计算分类后每个数字的概率值

        return output
#定义优化器
model=Digit().to(DEVICE)

optimizer=optim.Adam(model.parameters())#优化器


#定义训练方法
def train_model(model,device,train_loader,optimizer,epoch):
    #模型训练
    model.train()
    for batch_index,(data,target) in enumerate(train_loader):
        #数据部署到DEVICE上去
        data,target=data.to(device),target.to(device)
        #梯度初始化为0
        optimizer.zero_grad()
        #训练后的结果
        output=model(data)
        #计算损失
        loss=F.cross_entropy(output,target)
        #反向传播
        loss.backward()
        #参数优化
        optimizer.step()
        if batch_index % 3000 == 0:
            print('Train Epoch : {} \t loss :{: .6f}'.format(epoch,loss.item()))

# 定义测试方法
def test_model(model, device, test_loader):
    # 模型验证
    result=[]
    model.eval()
    with torch.no_grad():  # 不会计算梯度,也不会进行反向传播
        for data in test_loader:
            # 部署到device上
            data= data.to(device)
            # 测试数据
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # 或者pred=torch.max(output,dim=1)或者pred=torch.argmax(dim=1) #值  索引
            result.append(pred)
    result_arr=np.array([i.cpu().numpy().flatten() for i in result])
    final_arr=result_arr.flatten()
    ser_obj=pd.Series(
        index=[i for i in np.arange(1,28001)],
        data=final_arr
    )
    ser_obj.to_csv(r"C:\Users\zhangquanlong\Desktop\result_2.csv")



#调用方法
for epoch in range(1,EPOCHS+1):#EPOCHS+1
    train_model(model,DEVICE,train_loader,optimizer,epoch)
#torch.save(model,r"C:\Users\zhangquanlong\Desktop\Machine_learning\model.pt")#保存模型
test_model(model,DEVICE,test_loader)