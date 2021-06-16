from .dataset import JointDataset
from .train_joint import jointloader
from tqdm import tqdm
import torch
import torch.nn.parallel
import time
import copy
import warnings
warnings.filterwarnings('ignore')

# similar to deepcopy
def matcopy(model_0, model_ft):
    return

# General Code for supervised train
def demitrain(model_ft, model_1, model_2, model_3, model_4, 
    fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro, loader_joint, 
    optimizer_0, optimizer_1, optimizer_2, optimizer_3, optimizer_4, 
    scheduler_0, scheduler_1, scheduler_2, scheduler_3, scheduler_4, 
    criterion, device, checkpoint_path, file, saveinterval=1, last_epochs=0, num_epochs=20):
    
    since = time.time()
    best_acc = 0.0

    for epoch in range(last_epochs, last_epochs+num_epochs):
        print('\nEpoch {}/{} \n'.format(epoch, last_epochs+num_epochs - 1))
        file.write('\nEpoch {}/{} \n'.format(epoch, last_epochs+num_epochs - 1))
        file.write('-' * 10)
        file.write('\n')
        file.flush()

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model_ft.train()  # Set model to training mode
                model_1.train()
                model_2.train()
                model_3.train()
                model_4.train()
                fc_plain.train()
                fc_rota.train()
                fc_patch.train()
                fc_jigpa.train()
                fc_jigro.train()
                
            else:
                model_ft.eval()  # Set model to evaluate mode
                fc_plain.eval()
                fc_rota.eval()
                fc_patch.eval()
                fc_jigpa.eval()
                fc_jigro.eval()

            running_loss = 0.0
            running_corrects = 0
            n_samples = 0

            # Iterate over data.
            for _, (iter_plain, iter_rota, iter_patch, iter_jigpa, iter_jigro) in enumerate(tqdm(loader_joint[phase])):
                inputs, labels = iter_plain
                inputs, labels = inputs.to(device), labels.to(device)

                rota_in, rota_la = iter_rota
                rota_in, rota_la = rota_in.to(device), rota_la.to(device)

                patch_in_0, patch_in_1, patch_la = iter_patch
                patch_in_0, patch_in_1, patch_la = patch_in_0.to(device), patch_in_1.to(device), patch_la.to(device)

                jigpa_in_0, jigpa_in_1, jigpa_in_2, jigpa_in_3, jigpa_la = iter_jigpa
                jigpa_in_0, jigpa_in_1, jigpa_in_2, jigpa_in_3, jigpa_la = jigpa_in_0.to(device), jigpa_in_1.to(device), jigpa_in_2.to(device), jigpa_in_3.to(device), jigpa_la.to(device)
                
                jigro_in_0, jigro_in_1, jigro_in_2, jigro_in_3, jigro_la = iter_jigro
                jigro_in_0, jigro_in_1, jigro_in_2, jigro_in_3, jigro_la = jigro_in_0.to(device), jigro_in_1.to(device), jigro_in_2.to(device), jigro_in_3.to(device), jigro_la.to(device)

                # zero the parameter gradients
                optimizer_0.zero_grad()
                optimizer_1.zero_grad()
                optimizer_2.zero_grad()
                optimizer_3.zero_grad()
                optimizer_4.zero_grad()
                batchSize = labels.size(0)
                n_samples += batchSize

                if phase == 'train':
                    # 警告：不可使用深拷贝，应遍历权值矩阵赋值
                    matcopy(model_1, model_ft)
                    matcopy(model_2, model_ft)
                    matcopy(model_3, model_ft)
                    matcopy(model_4, model_ft)

                # forward
                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):
                    plain_outputs = fc_plain(model_ft(inputs))
                    loss_0 = criterion(plain_outputs, labels)

                    outputs = fc_rota(model_1(rota_in))
                    loss_1 = criterion(outputs, rota_la)

                    outputs = fc_patch(torch.cat(
                        (model_2(patch_in_0), model_2(patch_in_1)), dim = 1
                    ))
                    loss_2 = criterion(outputs, patch_la)

                    outputs = fc_jigpa(torch.cat(
                        (model_3(jigpa_in_0), model_3(jigpa_in_1), model_3(jigpa_in_2), model_3(jigpa_in_3)), dim = 1
                    ))
                    loss_3 = criterion(outputs, jigpa_la)

                    outputs = fc_jigro(torch.cat(
                        (model_4(jigro_in_0), model_4(jigro_in_1), model_4(jigro_in_2), model_4(jigro_in_3)), dim = 1
                    ))
                    loss_4 = criterion(outputs, jigro_la)

                    # loss = loss_1 + loss_2 + loss_3 + loss_4
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_1.backward()
                        loss_2.backward()
                        loss_3.backward()
                        loss_4.backward()

                        optimizer_1.step()
                        optimizer_2.step()
                        optimizer_3.step()
                        optimizer_4.step()

                        scheduler_1.step()
                        scheduler_2.step()
                        scheduler_3.step()
                        scheduler_4.step()

                        model_ft.eval()     # Set model to evaluate mode
                        model_1.eval()
                        model_2.eval()
                        model_3.eval()
                        model_4.eval()
                        fc_plain.eval()
                        with torch.no_grad :
                            weight = torch.zeros(4)
                            oracle = criterion(fc_plain(model_ft(inputs)), labels).item()
                            weight[0] = oracle - criterion(fc_plain(model_1(inputs)), labels).item()
                            weight[1] = oracle - criterion(fc_plain(model_2(inputs)), labels).item()
                            weight[2] = oracle - criterion(fc_plain(model_3(inputs)), labels).item()
                            weight[3] = oracle - criterion(fc_plain(model_4(inputs)), labels).item()
                            weight = torch.softmax(weight, 0)
                        model_ft.train()    # Set model to training mode
                        model_1.train()
                        model_2.train()
                        model_3.train()
                        model_4.train()
                        fc_plain.train()

                        # 警告：loss_i 连接的是 model_i 无法更新 model_ft
                        # 考虑矩阵更新方法（同FSLL）
                        loss = weight[0]*loss_1 + weight[1]*loss_2 + weight[2]*loss_3 + weight[3]*loss_4

                # statistics
                running_loss += loss.item() * labels.size(0)
                pred_top_1 = torch.topk(plain_outputs, k=1, dim=1)[1]
                running_corrects += pred_top_1.eq(labels.view_as(pred_top_1)).int().sum().item()

            # Metrics
            top_1_acc = running_corrects / n_samples
            epoch_loss = running_loss / n_samples
            print('{} Loss: {:.6f} Top 1 Acc: {:.6f} \n'.format(phase, epoch_loss, top_1_acc))

            file.write('{} Loss: {:.6f} Top 1 Acc: {:.6f} \n'.format(phase, epoch_loss, top_1_acc))
            file.flush()

            # deep copy the model
            if phase == 'test' and top_1_acc > best_acc:
                best_acc = top_1_acc
                best_model_wts = copy.deepcopy(model_ft.state_dict())
                best_plain_wts = copy.deepcopy(fc_plain.state_dict())
                best_rota_wts = copy.deepcopy(fc_rota.state_dict())
                best_patch_wts = copy.deepcopy(fc_patch.state_dict())
                best_jigpa_wts = copy.deepcopy(fc_jigpa.state_dict())
                best_jigro_wts = copy.deepcopy(fc_jigro.state_dict())

        if (epoch+1) % saveinterval == 0:
            torch.save(model_ft.state_dict(), '%s/model_epoch_%d.pth' % (checkpoint_path, epoch))
            torch.save(fc_plain.state_dict(), '%s/fc_plain_epoch_%d.pth' % (checkpoint_path, epoch))
            torch.save(fc_rota.state_dict(), '%s/fc_rota_epoch_%d.pth' % (checkpoint_path, epoch))
            torch.save(fc_patch.state_dict(), '%s/fc_patch_epoch_%d.pth' % (checkpoint_path, epoch))
            torch.save(fc_jigpa.state_dict(), '%s/fc_jigpa_epoch_%d.pth' % (checkpoint_path, epoch))
            torch.save(fc_jigro.state_dict(), '%s/fc_jigro_epoch_%d.pth' % (checkpoint_path, epoch))

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
    fc_jigro.load_state_dict(best_jigro_wts)
    return model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro

