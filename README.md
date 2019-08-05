# CRF
Conditional Random Field for POS Tagging
# useage
python run.py -h <br>
<br>
        usage: run.py [-h] [--bigdata] [--anneal] [--regularize] [--seed SEED] 
<br>
        Create Conditional Random Field(CRF) for POS Tagging. 
<br>
        optional arguments: 
        -h, --help            show this help message and exit 
        --bigdata, -b         use big data 
        --anneal, -a          use simulated annealing 
        --regularize, -r      use L2 regularization 
        --seed SEED, -s SEED  set the seed for generating random numbers 
<br>
        eg: use Simulated annealing and set random seed as 1234 on the big data set 
>python run.py -b -u -s 1234 <br>

# results

|     | epoch | dev(%) | test(%) | time(s) |
|:-------:|:-------:|:-------:|:-------:|:--------|
| small data | 9/15 | 88.86 | 86.47 | 12.39 |
| big data | 31/42 | 94.29 | 94.03 | 640.14 |
