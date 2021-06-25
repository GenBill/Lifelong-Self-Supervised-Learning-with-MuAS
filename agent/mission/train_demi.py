from .dataset import PlainDataset
from .dataset import JointDataset
from .m_plain import plainloader
from .train_joint import jointloader

from tqdm import tqdm
import torch
import torch.nn.parallel
from tensorboardX import SummaryWriter

import time
import copy
import warnings
warnings.filterwarnings('ignore')

def ReluSoftmax(weight):
    # leng = weight.size(0)
    weight[weight<0] = 0
    sumw = torch.sum(weight).item()
    if sumw==0:
        return weight + 0.25    # 1/leng
    else:
        return weight / sumw

# similar to deepcopy
def matcopy(this, source):
    # this = source
    this_params = list(this.named_parameters())   # get the index by debuging
    source_params = list(source.named_parameters())   # get the index by debuging
    for i in range(len(source_params)):
        this_params[i][1].data = source_params[i][1].data
    return this

# General Code for supervised train
def demitrain(model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_contra, 
    loader_joint, loader_test, 
    # 警告：optimizer_all 不含 fc_plain
    # 警告：optimizer_0 仅优化 fc_plain
    optimizer_all, optimizer_0, optimizer_1, optimizer_2, optimizer_3, optimizer_4, 
    scheduler_all, scheduler_0, scheduler_1, scheduler_2, scheduler_3, scheduler_4, 
    criterion, device, checkpoint_path, file, saveinterval=1, last_epochs=0, num_epochs=20):
    
    since = time.time()
    best_acc = 0.0
    
    # initial weight & SummaryWriter
    weight = torch.zeros(4)
    data_path = checkpoint_path+'/../Tensorboard'
    data_writer = SummaryWriter(logdir=data_path)
    
    n_iter = 0
    for epoch in range(last_epochs, last_epochs+num_epochs):
        print('\nEpoch {}/{} \n'.format(epoch, last_epochs+num_epochs - 1))
        file.write('\nEpoch {}/{} \n'.format(epoch, last_epochs+num_epochs - 1))
        file.write('-' * 10)
        file.write('\n')
        file.flush()

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            joint_loss = 0.0
            running_loss = 0.0
            running_corrects = 0
            n_samples = 0

            # Iterate over data.
            if phase=='train':
                # Set model to training mode
                model_ft.train()
                fc_plain.train()
                fc_rota.train()
                fc_patch.train()
                fc_jigpa.train()
                fc_contra.train()
                
                # Train Part
                for _, (iter_plain, iter_valid, iter_rota, iter_patch, iter_jigpa, iter_contra) in enumerate(tqdm(loader_joint)):
                    inputs, labels = iter_plain
                    inputs, labels = inputs.to(device), labels.to(device)

                    valid_inputs, valid_labels = iter_valid
                    valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)

                    rota_in, rota_la = iter_rota
                    rota_in, rota_la = rota_in.to(device), rota_la.to(device)

                    patch_in_0, patch_in_1, patch_la = iter_patch
                    patch_in_0, patch_in_1, patch_la = patch_in_0.to(device), patch_in_1.to(device), patch_la.to(device)

                    jigpa_in_0, jigpa_in_1, jigpa_in_2, jigpa_in_3, jigpa_la = iter_jigpa
                    jigpa_in_0, jigpa_in_1, jigpa_in_2, jigpa_in_3, jigpa_la = jigpa_in_0.to(device), jigpa_in_1.to(device), jigpa_in_2.to(device), jigpa_in_3.to(device), jigpa_la.to(device)
                    
                    contra_in_0, contra_in_1, contra_in_2 = iter_contra
                    contra_in_0, contra_in_1, contra_in_2 = contra_in_0.to(device), contra_in_1.to(device), contra_in_2.to(device)
                    
                    # backup = copy.deepcopy(model_ft)
                    backup = copy.deepcopy(model_ft.state_dict())
                    batchSize = labels.size(0)
                    n_samples += batchSize

                    contra_la_0 = torch.zeros(batchSize, dtype=int, device=device)
                    contra_la_1 = torch.ones(batchSize, dtype=int, device=device)

                    # Calculate origin loss
                    model_ft.eval()
                    fc_plain.train()
                    # with torch.no_grad() :
                    loss_0 = criterion(fc_plain(model_ft(valid_inputs)), valid_labels)
                    loss_origin = loss_0.item()
                    optimizer_0.zero_grad()
                    loss_0.backward()
                    optimizer_0.step()

                    # rota main
                    model_ft.train()
                    outputs = fc_rota(model_ft(rota_in))
                    loss_1 = criterion(outputs, rota_la)
                    optimizer_1.zero_grad()
                    loss_1.backward()
                    optimizer_1.step()

                    # rota valid
                    model_ft.eval()
                    with torch.no_grad() :
                        loss_valid_1 = criterion(fc_plain(model_ft(valid_inputs)), valid_labels).item()
                    
                    # rota return
                    weight[0] = loss_origin - loss_valid_1
                    # matcopy(model_ft, backup)
                    model_ft.load_state_dict(backup)

                    # patch main
                    model_ft.train()
                    outputs = fc_patch(torch.cat(
                        (model_ft(patch_in_0), model_ft(patch_in_1)), dim = 1
                    ))
                    loss_2 = criterion(outputs, patch_la)
                    optimizer_2.zero_grad()
                    loss_2.backward()
                    optimizer_2.step()

                    # patch valid
                    model_ft.eval()
                    with torch.no_grad() :
                        loss_valid_2 = criterion(fc_plain(model_ft(valid_inputs)), valid_labels).item()
                    
                    # patch return
                    weight[1] = loss_origin - loss_valid_2
                    # matcopy(model_ft, backup)
                    model_ft.load_state_dict(backup)

                    # jigpa main
                    model_ft.train()
                    outputs = fc_jigpa(torch.cat(
                        (model_ft(jigpa_in_0), model_ft(jigpa_in_1), model_ft(jigpa_in_2), model_ft(jigpa_in_3)), dim = 1
                    ))
                    loss_3 = criterion(outputs, jigpa_la)
                    optimizer_3.zero_grad()
                    loss_3.backward()
                    optimizer_3.step()

                    # jigpa valid
                    model_ft.eval()
                    with torch.no_grad() :
                        loss_valid_3 = criterion(fc_plain(model_ft(valid_inputs)), valid_labels).item()
                    
                    # jigpa return
                    weight[2] = loss_origin - loss_valid_3
                    # matcopy(model_ft, backup)
                    model_ft.load_state_dict(backup)

                    # contra main
                    model_ft.train()
                    outputs_0 = fc_contra(torch.cat(
                        (model_ft(contra_in_0), model_ft(contra_in_1)), dim = 1
                    ))
                    outputs_1 = fc_contra(torch.cat(
                        (model_ft(contra_in_0), model_ft(contra_in_2)), dim = 1
                    ))
                    loss_4 = criterion(outputs_0, contra_la_0) + criterion(outputs_1, contra_la_1)
                    optimizer_4.zero_grad()
                    loss_4.backward()
                    optimizer_4.step()

                    # contra valid
                    model_ft.eval()
                    with torch.no_grad() :
                        loss_valid_4 = criterion(fc_plain(model_ft(valid_inputs)), valid_labels).item()
                    
                    # contra return
                    weight[3] = loss_origin - loss_valid_4
                    # matcopy(model_ft, backup)
                    model_ft.load_state_dict(backup)

                    # fine train
                    model_ft.train()
                    weight = ReluSoftmax(weight)
                    # weight = torch.softmax(weight, 0)
                    # if iter_index%5==0 :
                    #     print('Weight : ', weight)

                    loss_1 = criterion(fc_rota(model_ft(rota_in)), rota_la)
                    loss_2 = criterion(fc_patch(torch.cat(
                        (model_ft(patch_in_0), model_ft(patch_in_1)), dim = 1
                    )), patch_la)

                    loss_3 = criterion(fc_jigpa(torch.cat(
                        (model_ft(jigpa_in_0), model_ft(jigpa_in_1), model_ft(jigpa_in_2), model_ft(jigpa_in_3)), dim = 1
                    )), jigpa_la)

                    # Contra
                    outputs_0 = fc_contra(torch.cat(
                        (model_ft(contra_in_0), model_ft(contra_in_1)), dim = 1
                    ))
                    outputs_1 = fc_contra(torch.cat(
                        (model_ft(contra_in_0), model_ft(contra_in_2)), dim = 1
                    ))
                    loss_4 = criterion(outputs_0, contra_la_0) + criterion(outputs_1, contra_la_1)

                    loss_ft = weight[0]*loss_1 + weight[1]*loss_2 + weight[2]*loss_3 + weight[3]*loss_4
                    loss_all = (1-weight[0])*loss_1 + (1-weight[1])*loss_2 + (1-weight[2])*loss_3 + (1-weight[3])*loss_4
                    # if iter_index%5==0 :
                    #     print('Joint Loss : {:4f}, {:4f}, {:4f}, {:4f}'.format(loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item()))
                    data_writer.add_scalars('data/StepLoss_Group', {
                        'OriginLoss': loss_origin,
                        'RotaLoss': loss_1.item(),
                        'PatchLoss': loss_2.item(),
                        'JigpaLoss': loss_3.item(),
                        'ContraLoss': loss_4.item(),
                        'LossWeight_0': weight[0],
                        'LossWeight_1': weight[1],
                        'LossWeight_2': weight[2],
                        'LossWeight_3': weight[3],
                    }, n_iter)
                    n_iter += 1

                    optimizer_all.zero_grad()
                    loss_ft.backward(retain_graph=True)
                    loss_all.backward()
                    optimizer_all.step()

                    # plain 仅仅训练全连接层
                    fc_plain.train()
                    model_ft.eval()
                    outputs = fc_plain(model_ft(inputs))
                    loss_0 = criterion(outputs, labels)
                    # if iter_index%5==0 :
                    #     print('Main Loss : {:4f}'.format(loss_0.item()))
                    optimizer_0.zero_grad()
                    loss_0.backward()
                    optimizer_0.step()

                    # statistics
                    # loss_ft.item() is float
                    joint_loss += loss_ft.item() * labels.size(0)
                    running_loss += loss_0.item() * labels.size(0)
                    pred_top_1 = torch.topk(outputs, k=1, dim=1)[1]
                    running_corrects += pred_top_1.eq(labels.view_as(pred_top_1)).int().sum().item()

                # Metrics
                top_1_acc = running_corrects / n_samples
                epoch_loss = running_loss / n_samples
                joint_loss = joint_loss / n_samples

                data_writer.add_scalars('data/TrainLoss_Group', {
                    'Acc': top_1_acc,
                    'EpochLoss': epoch_loss,
                    'JointLoss': joint_loss
                }, epoch)

                print('{} Loss: {:.6f} , Joint Loss: {:.6f} , Top 1 Acc: {:.6f} \n'.format('Train', epoch_loss, joint_loss, top_1_acc))
                file.write('{} Loss: {:.6f} , Joint Loss: {:.6f} , Top 1 Acc: {:.6f} \n'.format('Train', epoch_loss, joint_loss, top_1_acc))
                file.flush()

            else :
                # Test Part
                model_ft.eval()  # Set model to evaluate mode
                fc_plain.eval()
                for _, (inputs, labels) in enumerate(tqdm(loader_test)):
                    inputs, labels = inputs.to(device), labels.to(device)
                    batchSize = labels.size(0)
                    n_samples += batchSize
                    with torch.no_grad() :
                        outputs = fc_plain(model_ft(inputs))
                        loss_0 = criterion(outputs, labels)
                    # statistics
                    running_loss += loss_0.item() * labels.size(0)
                    pred_top_1 = torch.topk(outputs, k=1, dim=1)[1]
                    running_corrects += pred_top_1.eq(labels.view_as(pred_top_1)).int().sum().item()

                # Metrics
                top_1_acc = running_corrects / n_samples
                epoch_loss = running_loss / n_samples

                data_writer.add_scalars('data/TestLoss_Group', {
                    'Acc': top_1_acc,
                    'EpochLoss': epoch_loss
                }, epoch)

                print('{} Loss: {:.6f} Top 1 Acc: {:.6f} \n'.format(phase, epoch_loss, top_1_acc))
                file.write('{} Loss: {:.6f} Top 1 Acc: {:.6f} \n'.format(phase, epoch_loss, top_1_acc))
                file.flush()

                # deep copy the model
                if top_1_acc > best_acc:
                    best_acc = top_1_acc
                    best_model_wts = copy.deepcopy(model_ft.state_dict())
                    best_plain_wts = copy.deepcopy(fc_plain.state_dict())
                    best_rota_wts = copy.deepcopy(fc_rota.state_dict())
                    best_patch_wts = copy.deepcopy(fc_patch.state_dict())
                    best_jigpa_wts = copy.deepcopy(fc_jigpa.state_dict())
                    best_contra_wts = copy.deepcopy(fc_contra.state_dict())
        
        scheduler_all.step()
        scheduler_0.step()
        scheduler_1.step()
        scheduler_2.step()
        scheduler_3.step()
        scheduler_4.step()

        if (epoch+1) % saveinterval == 0:
            torch.save(model_ft.state_dict(), '%s/model_epoch_%d.pth' % (checkpoint_path, epoch))
            torch.save(fc_plain.state_dict(), '%s/fc_plain_epoch_%d.pth' % (checkpoint_path, epoch))
            torch.save(fc_rota.state_dict(), '%s/fc_rota_epoch_%d.pth' % (checkpoint_path, epoch))
            torch.save(fc_patch.state_dict(), '%s/fc_patch_epoch_%d.pth' % (checkpoint_path, epoch))
            torch.save(fc_jigpa.state_dict(), '%s/fc_jigpa_epoch_%d.pth' % (checkpoint_path, epoch))
            torch.save(fc_contra.state_dict(), '%s/fc_contra_epoch_%d.pth' % (checkpoint_path, epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f} \n'.format(best_acc))
    file.write('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
    file.write('Best test Acc: {:4f} \n'.format(best_acc))
    file.flush()

    # load best model weights
    model_ft.load_state_dict(best_model_wts)
    fc_plain.load_state_dict(best_plain_wts)
    fc_rota.load_state_dict(best_rota_wts)
    fc_patch.load_state_dict(best_patch_wts)
    fc_jigpa.load_state_dict(best_jigpa_wts)
    fc_contra.load_state_dict(best_contra_wts)
    
    data_writer.close()
    
    return model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_contra

