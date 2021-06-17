# On Tesla_V100
python ../main_demi.py --cuda '4,5,6,7' --batchsize 256 --numworkers 2 --pretrain 0 --epochs_0 2 --epochs_1 2
python ../main_demi.py --cuda '4,5,6,7' --batchsize 256 --numworkers 2 --pretrain 1 --epochs_0 2 --epochs_1 2

# On RTX3090
python main_demi.py --cuda '4,5' --batchsize 256 --numworkers 2 --pretrain 0 --epochs_0 2 --epochs_1 2

# CPU 代码测试
# python main_demi.py --cuda '' --batchsize 16 --numworkers 2 --pretrain 0 --epochs_0 2 --epochs_1 2