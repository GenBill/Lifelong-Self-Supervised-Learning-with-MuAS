python main_single.py --cuda '4' --batchsize 256 --numworkers 2 --pretrain 0 --powerword 'rota' &
python main_single.py --cuda '5' --batchsize 256 --numworkers 2 --pretrain 1 --powerword 'rota' &

python main_single.py --cuda '6' --batchsize 256 --numworkers 2 --pretrain 0 --powerword 'patch' &
python main_single.py --cuda '7' --batchsize 256 --numworkers 2 --pretrain 1 --powerword 'patch' &

python main_single.py --cuda '0' --batchsize 256 --numworkers 2 --pretrain 0 --powerword 'jigpa' &
python main_single.py --cuda '1' --batchsize 256 --numworkers 2 --pretrain 1 --powerword 'jigpa' &

python main_single.py --cuda '2' --batchsize 256 --numworkers 2 --pretrain 1 --powerword 'jigro' &
python main_single.py --cuda '3' --batchsize 256 --numworkers 2 --pretrain 0 --powerword 'jigro'