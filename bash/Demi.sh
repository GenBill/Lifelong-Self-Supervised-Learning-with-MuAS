# On Tesla_V100
# python ../main_demi.py
python main_demi.py --cuda '0' --batchsize 256 --numworkers 2 --pretrain 0 --lr_net 0.001 --lr_fc 0.001 --epochs_0 400 --epochs_1 80 &
python main_demi.py --cuda '1' --batchsize 256 --numworkers 2 --pretrain 0 --lr_net 0.001 --lr_fc 0.002 --epochs_0 400 --epochs_1 80 &
python main_demi.py --cuda '2' --batchsize 256 --numworkers 4 --pretrain 0 --lr_net 0.005 --lr_fc 0.01 --epochs_0 400 --epochs_1 80 &

# On RTX3090
python main_demi.py --cuda '1' --batchsize 256 --numworkers 4 --pretrain 1 --lr_net 0.005 --lr_fc 0.01 --epochs_0 400 --epochs_1 80

# CPU 代码测试
# python main_demi.py --cuda '' --batchsize 16 --numworkers 2 --pretrain 0 --epochs_0 2 --epochs_1 2