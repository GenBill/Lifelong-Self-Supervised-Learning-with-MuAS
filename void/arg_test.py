import os
import argparse
import random
import numpy as np
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--joint', type=bool, default=True, help="joint on")
parser.add_argument('--pretrain', type=bool, default=True, help="pretrain on")

# opt = parser.parse_args(args=[])
opt = parser.parse_args()

if opt.joint==True :
    if opt.pretrain==True :
        print('1 1')
    else :
        print('1 0')
else:
    if opt.pretrain==True :
        print('0 1')
    else :
        print('0 0')
