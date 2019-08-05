# CRF
Conditional Random Field for POS Tagging (using partial feature optimization)
# usage
>python run.py -h
```
usage: run.py [-h] [--bigdata] [--anneal] [--regularize] [--seed SEED]

Create Conditional Random Field(CRF) for POS Tagging.

optional arguments:
  -h, --help            show this help message and exit
  --bigdata, -b         use big data
  --anneal, -a          use simulated annealing
  --regularize, -r      use L2 regularization
  --seed SEED, -s SEED  set the seed for generating random numbers
```
eg: use Simulated annealing and set random seed as 1234 on the big data set 
>python run.py -b -u -s 1234 <br>

# results

| data set | anneal | epoch | dev(%) | test(%) | mT(s) |
|:-------:|:-------:|:-------:|:-------:|:--------|:-------:|
| small | no | 11/17 | 88.88 | 86.49 | 12.76 |
| small | yes | 9/15 | 88.86 | 86.47 | 12.39 |
| big | no | 31/42 | 94.29 | 94.03 | 640.14 |
| big | yes | 31/42 | 94.29 | 94.03 | 640.14 |
