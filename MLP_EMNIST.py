import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#* 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

#* 模型保存函数
def save_model(model, optimizer, num_epochs, opt: int = 0):
    '''
    `opt:int = 0` <br> 
    是否完整地保存模型 
    '''
    # 仅保存模型参数
    if opt == 0:
        torch.save(model.state_dict(), 'mlp_letter_params.pth')
        print("模型参数已保存至 mlp_letter_params.pth")
    # 保存完整状态(含优化器)
    else:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': num_epochs
        }
        torch.save(checkpoint, 'mlp_checkpoint.pth')
        print("完整训练状态已保存至 mlp_checkpoint.pth")

# 主函数：包含训练和测试逻辑
def main():
    # 设置超参数 - 字母预测是26个类别(A-Z)
    input_size = 784    # 图片 28x28=784
    hidden_size = 128   # 隐藏层神经元数量
    num_classes = 26    # 输出类别（A-Z共26个）
    num_epochs = 5      # 训练轮次
    batch_size = 100    # 每批数据量
    learning_rate = 0.001

    # 加载EMNIST 字母子数据集
    train_dataset = torchvision.datasets.EMNIST(
        root='./data',
        split='letters',
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST标准化参数
            transforms.Lambda(lambda x: x.view(-1))      # 展平图片为向量
        ]),
        download=True
    )

    test_dataset = torchvision.datasets.EMNIST(
        root='./data',
        split='letters',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ]),
    )

    # 创建数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = MLP(input_size, hidden_size, num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            labels = labels - 1  #! 转换标签为0-25
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 每100步打印训练信息
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

    # 测试模型
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            labels = labels - 1  #! 转换标签为0-25

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # 保存模型（默认仅保存参数，可改为1保存完整状态）
    save_model(model, optimizer, num_epochs, opt=0)

# 主程序入口
if __name__ == '__main__':
    main()
