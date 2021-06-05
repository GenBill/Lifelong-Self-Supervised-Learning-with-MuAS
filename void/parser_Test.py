import argparse

parser = argparse.ArgumentParser(description='FSLL Valid')
parser.add_argument('--cuda', '-c', help='cuda Num', default='0')
parser.add_argument('--seed', '-s', help='manual Seed', default=2077)
parser.add_argument('--frost', '-f', help='Frost Stone', default=0.8)
args = parser.parse_args()


print(str(args.cuda))

print(args.cuda)
print(args.seed)
print(args.frost)
