python main_joint.py --cuda '0,1' --batchsize 256 --numworkers 2  --joint 4 --pretrain 0 &
python main_joint.py --cuda '2,3' --batchsize 256 --numworkers 2  --joint 4 --pretrain 1 &

python main_joint.py --cuda '0,1,2,3' --batchsize 512 --numworkers 2  --joint 4 --pretrain 1
python main_joint.py --cuda '0,1,2,3' --batchsize 512 --numworkers 2  --joint 4 --pretrain 0 
