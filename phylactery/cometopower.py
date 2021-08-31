from agent import *

def cometopower(powerword, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers):
    patch_dim = 96
    contra_dim = 128
    gap = 6
    jitter = 6

    if powerword=='none':
        image_datasets = {
            x: PlainDataset(x, data_root, data_pre_transforms[x], data_post_transforms[x])
            for x in ['train', 'test'] }
        return len(image_datasets['train']), len(image_datasets['test'])

    if powerword=='rota':
        return rotaloader(data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
    
    if powerword=='patch':
        return patchloader(patch_dim, gap, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
    
    if powerword=='jigpa':
        return jigpaloader(patch_dim, gap, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
        
    if powerword=='jigro':
        return jigroloader(patch_dim, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
        
    if powerword=='contra':
        return contraloader(contra_dim, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
    
    if powerword=='plain':
        return plainloader(data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
    
    return None