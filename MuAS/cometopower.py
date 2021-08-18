def cometopower(powerword):
    if powerword=='rota':
        return rotaloader(data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)


    loader_plain = plainloader(data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
    loader_rota = 
    loader_patch = patchloader(patch_dim, gap, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
    loader_jigpa = jigpaloader(patch_dim, gap, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
    loader_jigro = jigroloader(patch_dim, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
    loader_contra = contraloader(patch_dim, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
    loader_joint = jointloader(patch_dim, gap, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
