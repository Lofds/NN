
import torch
import matplotlib.pyplot as plt
from optimizer import loadData,train,test_accuracy


batchSize = 5
func_list = [torch.optim.SGD, torch.optim.Adagrad, torch.optim.Adam]
name_list=['SGD','Adagrad','Adam']
lr_list = [0.1,0.01,0.001,0.0001]
trainData,testData = loadData(batchSize)
#三种优化算法的学习率调优
for i in range(3):
        iter=[[],[],[],[]]
        loss=[[],[],[],[]]
        accuracy=[[],[],[],[]]
        for j in range(4):
                func_name = func_list[i]
                print('optimizer:', name_list[i],lr_list[j])
                net,tmp_iter,tmp_loss,tmp_accuracy=train(func_name,trainData,2,lr_list[j],0)
                testAccuracy = test_accuracy(net,testData)
                print('Accuracy of the network on the 10000 test images: %d %%' % testAccuracy)
                iter[j]=tmp_iter
                loss[j]=tmp_loss
                accuracy[j]=tmp_accuracy
                torch.save(net, './res_'+name_list[i]+'/model'+str(j)+'.pkl')
        plt.title(name_list[i])
        plt.plot(iter[0],loss[0],label='lr=0.1',color='r')
        plt.plot(iter[1],loss[1],label='lr=0.01',color='g')
        plt.plot(iter[2],loss[2],label='lr=0.001',color='b')
        plt.plot(iter[3],loss[3],label='lr=0.0001',color='c')
        plt.legend()
        plt.savefig('./res_'+name_list[i]+'/loss.jpg')
        plt.close()
        plt.title(name_list[i])
        plt.plot(iter[0],accuracy[0],label='lr=0.1',color='r')
        plt.plot(iter[1],accuracy[1],label='lr=0.01',color='g')
        plt.plot(iter[2],accuracy[2],label='lr=0.001',color='b')
        plt.plot(iter[3],accuracy[3],label='lr=0.0001',color='c')
        plt.legend()
        plt.savefig('./res_'+name_list[i]+'/accuracy.jpg')
        plt.close()


#SGD算法使用逐渐衰减的学习率
net1,iter1,loss1,accuracy1=train(func_list[0],trainData,20,0.001,0)
net2,iter2,loss2,accuracy2=train(func_list[0],trainData,20,0.002,1)
torch.save(net1, './res_dl/model0.pkl')
torch.save(net2, './res_dl/model1.pkl')

plt.title('loss')
plt.plot(iter1,loss1,label='SGD',color='r')
plt.plot(iter2,loss2,label='decay_lr',color='g')
plt.legend()
plt.savefig('./res_dl/loss.jpg')
plt.close()
plt.title('accuracy')
plt.plot(iter1,accuracy1,label='SGD',color='r')
plt.plot(iter2,accuracy2,label='decay_lr',color='g')
plt.legend()
plt.savefig('./res_dl/accuracy.jpg')
plt.close()