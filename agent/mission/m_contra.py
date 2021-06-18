from .dataset import ContrastiveDataset
from tqdm import tqdm
import torch
import torch.nn.parallel
import time
import copy
import warnings
warnings.filterwarnings('ignore')

# General Code for supervised train
def contratrain(model, fc_layer, dataloaders, criterion, optimizer, scheduler, 
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
                model.train()  # Set model to training mode
                fc_layer.train()
            else:
                model.eval()  # Set model to evaluate mode
                fc_layer.eval()

            running_loss = 0.0
            running_corrects = 0
            n_samples = 0

            # Iterate over data.
            for _, (input_0, input_1, input_2) in enumerate(tqdm(dataloaders[phase])):
                # 0 源，1 同类，2 负样本
                input_0 = input_0.to(device)
                input_1 = input_1.to(device)
                input_2 = input_2.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                batchSize = input_0.size(0)
                n_samples += batchSize

                label_0 = torch.zeros(batchSize, dtype=int, device=device)
                label_1 = torch.ones(batchSize, dtype=int, device=device)

                # forward
                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):
                    fea_output_0 = model(input_0)
                    fea_output_1 = model(input_1)
                    fea_output_2 = model(input_2)

                    outputs_0 = fc_layer(torch.cat((fea_output_0, fea_output_1), dim=1))
                    outputs_1 = fc_layer(torch.cat((fea_output_0, fea_output_2), dim=1))
                    loss = criterion(outputs_0, label_0) + criterion(outputs_1, label_1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                running_loss += loss.item() * input_0.size(0)
                pred_top = torch.topk(outputs_0, k=1, dim=1)[1]
                running_corrects += pred_top.eq(label_0.view_as(pred_top)).int().sum().item()
                pred_top = torch.topk(outputs_1, k=1, dim=1)[1]
                running_corrects += pred_top.eq(label_1.view_as(pred_top)).int().sum().item()

            # Metrics
            top_1_acc = running_corrects / n_samples / 2
            epoch_loss = running_loss / n_samples
            print('{} Loss: {:.6f} Top 1 Acc: {:.6f} \n'.format(phase, epoch_loss, top_1_acc))
            file.write('{} Loss: {:.6f} Top 1 Acc: {:.6f} \n'.format(phase, epoch_loss, top_1_acc))
            file.flush()

            # deep copy the model
            if phase == 'test' and top_1_acc > best_acc:
                best_acc = top_1_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_fc_wts = copy.deepcopy(fc_layer.state_dict())
        if (epoch+1) % saveinterval == 0:
            torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (checkpoint_path, epoch))
            torch.save(fc_layer.state_dict(), '%s/fc_epoch_%d.pth' % (checkpoint_path, epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f} \n'.format(best_acc))
    file.write('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
    file.write('Best test Acc: {:4f} \n'.format(best_acc))
    file.flush()

    # load best model weights
    model.load_state_dict(best_model_wts)
    fc_layer.load_state_dict(best_fc_wts)
    return model, fc_layer

def contraloader(patch_dim, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers):

    image_datasets = {
        x: ContrastiveDataset(x, data_root, patch_dim, jitter, 
        preTransform = data_pre_transforms[x], postTransform=data_post_transforms[x])
        for x in ['train', 'test']
    }
    assert image_datasets
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size,
            pin_memory=True, shuffle=True, num_workers=num_workers
        ) for x in ['train', 'test']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    return dataloaders

