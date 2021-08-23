# subs python ../main_demi.py
# Use lr = 1e-4


# On Tesla_V100
python main_demi.py --cuda '4,5,6,7' --batchsize 128 --numworkers 2 --pretrain 0 --momentum 0.9 --lr_net 0.001 --lr_fc 0.00002 --epochs_0 400 --epochs_1 80 &
python main_demi.py --cuda '1' --batchsize 256 --numworkers 2 --pretrain 1 --momentum 0.99 --lr_net 0.001 --lr_fc 0.002 --epochs_0 500 --epochs_1 100 &
python main_demi.py --cuda '2' --batchsize 256 --numworkers 2 --pretrain 0 --momentum 0.99 --lr_net 0.001 --lr_fc 0.002 --epochs_0 500 --epochs_1 100 &


# On RTX3090
nohup python main_demi.py --cuda '1' --batchsize 128 --numworkers 4 --pretrain 0 --momentum 0.9 --lr_net 0.01 --lr_fc 0.01 --epochs_0 500 --epochs_1 100 >/dev/null 2>&1 &
python main_demi.py --cuda '1' --batchsize 128 --numworkers 4 --pretrain 0 --momentum 0.9 --lr_net 0.01 --lr_fc 0.01 --epochs_0 500 --epochs_1 100 &
nohup python main_demi.py --cuda '0' --batchsize 100 --numworkers 4 --pretrain 0 --momentum 0.9 --lr_net 0.01 --lr_fc 0.01 --epochs_0 500 --epochs_1 100 >/dev/null 2>&1 &

