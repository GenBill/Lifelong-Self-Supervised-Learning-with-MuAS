from tqdm import tqdm
import torch
import torch.nn.parallel
import time
import copy
import warnings
warnings.filterwarnings('ignore')

# General Code for supervised train
def jointtrain(loader_plain, loader_rota, loader_patch, loader_jigpa, loader_jigro, 
    model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro, 
    optimizer, scheduler, criterion, 
    device, checkpoint_path, file, saveinterval=1, last_epochs=0, num_epochs=20):
    
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
            for index, (inputs, labels) in enumerate(tqdm(loader_plain[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                rota_in, rota_la = loader_rota[phase][index]
                patch_in_0, patch_in_1, patch_la = loader_patch[phase][index]
                jigpa_in_0, jigpa_in_1, jigpa_in_2, jigpa_in_3, jigpa_la = loader_jigpa[phase][index]
                jigro_in_0, jigro_in_1, jigro_in_2, jigro_in_3, jigro_la = loader_jigro[phase][index]

                # zero the parameter gradients
                optimizer.zero_grad()
                batchSize = labels.size(0)
                n_samples += batchSize

                # forward
                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):
                    plain_outputs = fc_plain(model_ft(inputs))
                    loss = criterion(plain_outputs, labels)

                    outputs = fc_rota(model_ft(rota_in))
                    loss += criterion(outputs, rota_la)

                    outputs = fc_patch(torch.cat(
                        (model_ft(patch_in_0), model_ft(patch_in_1)), dim = 1
                    ))
                    loss += criterion(outputs, patch_la)

                    outputs = fc_jigpa(torch.cat(
                        (model_ft(jigpa_in_0), model_ft(jigpa_in_1), model_ft(jigpa_in_2), model_ft(jigpa_in_3)), dim = 1
                    ))
                    loss += criterion(outputs, jigpa_la)

                    outputs = fc_jigro(torch.cat(
                        (model_ft(jigro_in_0), model_ft(jigro_in_1), model_ft(jigro_in_2), model_ft(jigro_in_3)), dim = 1
                    ))
                    loss += criterion(outputs, jigro_la)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                running_loss += loss.item() * labels.size(0)
                pred_top_1 = torch.topk(plain_outputs, k=1, dim=1)[1]
                running_corrects += pred_top_1.eq(labels.view_as(pred_top_1)).int().sum().item()

            # Metrics
            top_1_acc = running_corrects / n_samples
            epoch_loss = running_loss / n_samples
            print('{} Loss: {:.4f} Top 1 Acc: {:.4f} \n'.format(phase, epoch_loss, top_1_acc))

            file.write('{} Loss: {:.4f} Top 1 Acc: {:.4f} \n'.format(phase, epoch_loss, top_1_acc))
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



