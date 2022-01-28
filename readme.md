## 实验环境

该实验基于pytorch平台，需要安装的依赖包有：

```python
torch
torchvision
matplotlib
```

## 数据集下载

使用了CIFAR10数据集，下载方法如下：

```python
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
trainData = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
testData = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False)
```

## 运行方式

```python
python .\run.py
```

## 实验结果

| 实验结果                                 | 存放文件夹       |
| ------------------------------------ | ----------- |
| SGD学习率调优得到的模型和loss,accuracy函数        | res_SGD     |
| SGD学习率进一步调优得到的模型和loss,accuracy函数     | res_SGD_1   |
| SGD批量大小调优得到的模型和loss,accuracy函数       | res_SGD_2   |
| Adagrad学习率调优得到的模型和loss,accuracy函数    | res_Adagrad |
| Adam学习率调优得到的模型和loss,accuracy函数       | res_Adam    |
| 三种算法性能对比得到的模型和loss,accuracy函数        | res         |
| SGD算法使用逐渐衰减的学习率得到的模型和loss,accuracy函数 | res_dl      |






