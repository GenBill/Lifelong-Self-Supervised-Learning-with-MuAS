python main_rotavap.py --cuda '0' --batchsize 512 --numworkers 16 --momentum 0.6 --pretrain 1

python main_evap.py --cuda '1' --batchsize 512 --numworkers 16 --momentum 0.6 --pretrain 1

nohup python main_rotavap.py --cuda '0' --batchsize 512 --numworkers 16 --momentum 0.6 --pretrain False --alpha 0.2 --num_epoch_0 100 --num_epoch_1 200 --lr 0.1
