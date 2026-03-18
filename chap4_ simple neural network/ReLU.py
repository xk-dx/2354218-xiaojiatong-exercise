import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 定义函数
def taget_function(x):
    return torch.sin(x)

# 准备数据
x = torch.linspace(-np.pi,np.pi,100).view(-1,1)
y = taget_function(x)

# 划分训练集和测试集
train_size = int(0.8*len(x))
indices = torch.randperm(len(x))#随机生成排列
train_indices, test_indeces = indices[:train_size], indices[:train_size]
x_train, y_train = x[train_indices], y[train_indices]
x_test, y_test = x[test_indeces], y[test_indeces]

#构建两层relu网络
class RELUNet(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        #第一层线性变化+relu激活
        self.hidden = nn.Linear(1,hidden_size)
        self.relu=nn.ReLU()
        self.predict = nn.Linear(hidden_size,1)
        
    def forward(self,x):
        x = self.hidden(x)
        x= self.relu(x)
        x=self.predict(x)
        return x
    
model = RELUNet(hidden_size=64)
optimizer = optim.Adam(model.parameters(),lr=0.01)
criterion = nn.MSELoss()

# 开始训练
epochs = 100
for epoch in range(epochs):
    prediction = model(x_train)
    loss = criterion(prediction,y_train)
    
    optimizer.zero_grad()#清空上一步残留梯度
    loss.backward()#反向传播，计算梯度
    optimizer.step()#更新参数

    if (epoch+1)%200 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# 开始推理
model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    test_loss = criterion(y_pred, y_test)
    print(f'\nFinal Test Loss: {test_loss.item():.6f}')
# 下面开始画图
plt.scatter(x_test.numpy(), y_test.numpy(), label='Real Data', s=10, color='gray')
plt.scatter(x_test.numpy(), y_pred.numpy(), label='ReLU Prediction', s=10, color='red')
plt.title('Function Fitting with ReLU Network')
plt.legend()
plt.show()