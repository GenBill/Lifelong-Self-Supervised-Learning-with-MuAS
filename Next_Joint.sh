# 训练补偿
python main_joint.py --cuda '0' --batchsize 512 --numworkers 2  --joint 2 --pretrain 0 &
python main_joint.py --cuda '1' --batchsize 512 --numworkers 2  --joint 2 --pretrain 1 &
python main_joint.py --cuda '2' --batchsize 256 --numworkers 2  --joint 2 --pretrain 0 &
python main_joint.py --cuda '3' --batchsize 256 --numworkers 2  --joint 2 --pretrain 1

# 新任务
python main_joint.py --cuda '0' --batchsize 512 --numworkers 2  --joint 3 --pretrain 0 &
python main_joint.py --cuda '1' --batchsize 512 --numworkers 2  --joint 3 --pretrain 1 &
python main_joint.py --cuda '2' --batchsize 256 --numworkers 2  --joint 3 --pretrain 0 &
python main_joint.py --cuda '3' --batchsize 256 --numworkers 2  --joint 3 --pretrain 1

python main_joint.py --cuda '0' --batchsize 256 --numworkers 2  --joint 0 --pretrain 0 &
python main_joint.py --cuda '1' --batchsize 256 --numworkers 2  --joint 0 --pretrain 1 &
python main_joint.py --cuda '2' --batchsize 256 --numworkers 2  --joint 1 --pretrain 0 &
python main_joint.py --cuda '3' --batchsize 256 --numworkers 2  --joint 1 --pretrain 1

# Add From Big_Joint.sh
python main_joint.py --cuda '0,1' --batchsize 256 --numworkers 2  --joint 4 --pretrain 0 &
python main_joint.py --cuda '2,3' --batchsize 256 --numworkers 2  --joint 4 --pretrain 1 &

python main_joint.py --cuda '0,1,2,3' --batchsize 512 --numworkers 2  --joint 4 --pretrain 1
python main_joint.py --cuda '0,1,2,3' --batchsize 512 --numworkers 2  --joint 4 --pretrain 0 