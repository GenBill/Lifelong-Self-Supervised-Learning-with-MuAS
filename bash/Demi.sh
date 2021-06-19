# subs python ../main_demi.py
# Use lr = 1e-4


# On Tesla_V100
python main_demi.py --cuda '4,5,6,7' --batchsize 128 --numworkers 2 --pretrain 0 --lr_net 0.0001 --lr_fc 0.0001 --epochs_0 400 --epochs_1 80 &
python main_demi.py --cuda '1' --batchsize 256 --numworkers 2 --pretrain 0 --lr_net 0.0001 --lr_fc 0.0001 --epochs_0 400 --epochs_1 80 &
python main_demi.py --cuda '2' --batchsize 256 --numworkers 4 --pretrain 1 --lr_net 0.0001 --lr_fc 0.0001 --epochs_0 400 --epochs_1 80 &


# On RTX3090
python main_demi.py --cuda '1' --batchsize 256 --numworkers 4 --pretrain 1 --lr_net 0.005 --lr_fc 0.01 --epochs_0 400 --epochs_1 80

