this_Epoch = 10000
epoch_num = 10000
'''
for epoch in range(epoch_num) :
    if (epoch+1) % 100 == 0:
        print('./TimHu/models256/'+'Epoch_'+str(this_Epoch + epoch)+'.pth')
'''
for epoch in range(1,epoch_num) :
    if epoch % 100 == 0:
        print('./TimHu/models256/'+'Epoch_'+str(this_Epoch + epoch)+'.pth')
