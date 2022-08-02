## Syntactic Experiments with Sequential Data 

We have created a synthetic sequential data generator, in which the
user can specify

1. number of tokens
2. sequence length
3. number of distinct inputs
4. number of constraints per inputs
5. number of acceptable sequences per constraints

We used this generator to get datasets where only the sequence length
was different (10 tokens, 100 distinct inputs, 5 constraints per
inputs, 20 acceptable sequences per constraints) and trained the same
model that was used for the CMT experiments with the prp-loss and the
nll-loss. Below you can see the results. Note that the prp-loss
retains a significant advantage as we vary the sequence length.


|Sequence Length | Loss type | H@1 positive | H@5 positive | H@1 negative | H@5 negative|
|--- | --- | --- | --- | --- | --- |
| 4 | prp | 0.61 | 0.93 | 0.27 | 0.87 |
| 4 | nll | 0.51 | 0.62 | 0.27 | 0.35 |
| 6 | prp | 0.91 | 0.99 | 0.0  | 0.0  |
| 6 | nll | 0.51 | 0.63 | 0.0  | 0.0  |
| 8 | prp | 0.91 | 0.97 | 0.0  | 0.0  |
| 8 | nll | 0.40 | 0.57 | 0.0  | 0.0  |
|10 | prp | 0.67 | 0.77 | 0.0  | 0.0  |
|10 | nll | 0.43 | 0.53 | 0.0  | 0.0  |