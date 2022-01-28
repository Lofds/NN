import torch.nn as nn
import torchvision
import torch
import torchvision.transforms as transforms
import torch.optim as optim



# 定义神经网络  
class Net(nn.Module):  
    def __init__(self):
        super(Net, self).__init__()
        # 五个卷积层
        self.features = nn.Sequential(

            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),   
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), 
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        )
        # 全连接层
        self.dense = nn.Sequential(
            nn.Linear(128, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128)
        x = self.dense(x)
        return x
    

def loadData(batchSize):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    trainData = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    testData = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False)
    return trainData,testData


# 训练函数
def train(func_name,trainData,niter,lr,flag):

        #定义损失函数
        net = Net()
        criterion = nn.CrossEntropyLoss() 
        print(func_name.__name__)
        if(func_name.__name__=='SGD'):
                optimizer = func_name(net.parameters(), lr,momentum=0.9)  
        else:
                optimizer = func_name(net.parameters(), lr)   
        if(flag):
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.5) 
        iter=[]
        tmp_loss = []
        tmp_accuracy = []

        # 训练网络
        num = 1
        for epoch in range(niter):  # loop over the dataset multiple times
                running_loss = 0.0
                running_correct = 0.0
                running_total = 0.0
                for i, data in enumerate(trainData, 0):
                
                        # 取数据
                    inputs, labels = data
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()   
                    optimizer.step()
                    
                    #计算loss
                    running_loss += loss.item()
                    #计算准确率
                    _, predicted = torch.max(outputs.data, 1)  
                    running_total += labels.size(0)
                    running_correct += (predicted == labels).sum().item()

                    #记录每1000个的平均loss和准确率
                    if (i+1) % 1000 == 0:  
                            running_accuracy = 100 * running_correct / running_total
                            running_loss = running_loss / 1000
                            print('epoch: %d\t batch: %d\t loss: %.6f' % (epoch + 1, i + 1, running_loss))
                            print('Accuracy of the network on the 10000 train images: %d %%' % running_accuracy)  
                            iter.append(num)
                            tmp_loss.append(running_loss)
                            tmp_accuracy.append(running_accuracy)
                            num = num + 1
                            running_loss = 0.0                            
                            running_correct = 0
                            running_total = 0
                
                if(flag):
                    print('{} scheduler: {}'.format(epoch, lr_scheduler.get_last_lr()[0]))
                    lr_scheduler.step()

        return net,iter,tmp_loss,tmp_accuracy


# 使用测试数据测试网络
def test_accuracy(net,testData):
    correct = 0
    total = 0
    with torch.no_grad():  
        for data in testData:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return 100.0 * correct / total




