python ../main_joint.py --cuda '0' --batchsize 256 --numworkers 2  --joint 0 --pretrain 0 &
python ../main_joint.py --cuda '1' --batchsize 256 --numworkers 2  --joint 0 --pretrain 1
python ../main_joint.py --cuda '0' --batchsize 256 --numworkers 2  --joint 1 --pretrain 0 &
python ../main_joint.py --cuda '1' --batchsize 256 --numworkers 2  --joint 1 --pretrain 1

python ../main_joint.py --cuda '0' --batchsize 256 --numworkers 2  --joint 2 --pretrain 0 &
python ../main_joint.py --cuda '1' --batchsize 256 --numworkers 2  --joint 2 --pretrain 1
python ../main_joint.py --cuda '0' --batchsize 256 --numworkers 2  --joint 3 --pretrain 0 &
python ../main_joint.py --cuda '1' --batchsize 256 --numworkers 2  --joint 3 --pretrain 1

python ../main_joint.py --cuda '0,1' --batchsize 256 --numworkers 2  --joint 4 --pretrain 0
python ../main_joint.py --cuda '0,1' --batchsize 256 --numworkers 2  --joint 4 --pretrain 1


nohup python main_joint.py --cuda '1' --batchsize 128 --numworkers 4  --joint 1 --pretrain 0 --StepLeng 10 >/dev/null 2>&1 &

nohup python main_joint.py --cuda '5' --batchsize 256 --numworkers 4  --joint 1 --pretrain 0 --StepLeng 10 >/dev/null 2>&1 &
nohup python main_joint.py --cuda '6' --batchsize 256 --numworkers 4  --joint 1 --pretrain 0 --StepLeng 20 >/dev/null 2>&1 &
nohup python main_joint.py --cuda '7' --batchsize 256 --numworkers 4  --joint 1 --pretrain 0 --StepLeng 40 >/dev/null 2>&1 &

