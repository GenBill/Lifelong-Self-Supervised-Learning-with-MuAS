# subs python ../main_demi.py
# Use lr = 1e-4

# On RTX3090
python main_demi.py --cuda '3' --batchsize 256 --numworkers 4 --pretrain 0 --momentum 0.99 --lr_net 0.001 --lr_fc 0.002 --epochs_0 500 --epochs_1 100 
--netCont ../Joint_256/Demino/models/model_epoch_499.pth --plainCont ../Joint_256/Demino/models/fc_plain_epoch_499.pth --rotaCont ../Joint_256/Demino/models/fc_rota_epoch_499.pth --patchCont ../Joint_256/Demino/models/fc_patch_epoch_499.pth --jigpaCont ../Joint_256/Demino/models/fc_jigpa_epoch_499.pth --contraCont ../Joint_256/Demino/models/fc_contra_epoch_499.pth
