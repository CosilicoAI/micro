# Benchmark Results

| Method   |   Mean KS |   Corr Error |   Zero Error |   Train Time (s) |   Gen Time (s) |
|:---------|----------:|-------------:|-------------:|-----------------:|---------------:|
| micro    |    0.0611 |       0.106  |       0.0223 |              6.1 |            0   |
| ctgan    |    0.1997 |       0.3826 |       0.0986 |             35.5 |            0.8 |
| tvae     |    0.2459 |       0.1969 |       0.0555 |             12   |            0.6 |
| copula   |    0.2632 |       0.1756 |       0.2241 |              0.5 |            0.8 |

## Metrics Explanation

- **Mean KS**: Average Kolmogorov-Smirnov statistic across all target variables (lower is better)
- **Corr Error**: Frobenius norm of correlation matrix difference (lower is better)
- **Zero Error**: Mean absolute error in zero-fractions for zero-inflated variables (lower is better)
- **Train Time**: Time to train the model in seconds
- **Gen Time**: Time to generate synthetic samples in seconds
