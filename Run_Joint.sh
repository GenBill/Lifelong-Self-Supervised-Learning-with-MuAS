python joint_main.py --cuda '0' --joint 2 --pretrain 0 &
python joint_main.py --cuda '1' --joint 2 --pretrain 1

python joint_main.py --cuda '2' --joint 0 --pretrain 1 &
python joint_main.py --cuda '3' --joint 1 --pretrain 1
python joint_main.py --cuda '0' --joint 0 --pretrain 0 &
python joint_main.py --cuda '1' --joint 1 --pretrain 0