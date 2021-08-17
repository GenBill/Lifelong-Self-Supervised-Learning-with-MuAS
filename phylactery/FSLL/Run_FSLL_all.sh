python FSLL_Valid_00.py -c '0' &
python FSLL_Valid_99.py -c '1' &
python FSLL_Valid.py -c '2' -f 0.5 &
python FSLL_Valid.py -c '3' -f 0.3

python FSLL_Valid.py -c '0' -f 0.1 &
python FSLL_Valid.py -c '1' -f 0.2 &
python FSLL_Valid.py -c '2' -f 0.4 &
python FSLL_Valid.py -c '3' -f 0.6

python FSLL_Valid.py -c '0' -f 0.7 &
python FSLL_Valid.py -c '1' -f 0.8 &
python FSLL_Valid.py -c '2' -f 0.9

