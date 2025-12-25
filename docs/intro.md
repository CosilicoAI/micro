# micro

**Conditional microdata synthesis using normalizing flows.**

`micro` is a Python library for synthesizing survey microdata while preserving statistical properties and conditional relationships.

## Key Features

- **Conditional generation**: Generate target variables given demographics (not full joint distribution)
- **Zero-inflation handling**: Built-in support for variables that are 0 for many observations
- **Stable training**: Normalizing flows provide exact likelihood and stable gradients
- **Privacy-preserving**: Generates synthetic records without copying real data

## Installation

```bash
pip install micro
```

## Quick Example

```python
from micro import Synthesizer

# Initialize
synth = Synthesizer(
    target_vars=["income", "expenditure"],
    condition_vars=["age", "education", "region"],
)

# Fit on training data
synth.fit(training_data, weight_col="weight")

# Generate for new demographics
synthetic = synth.generate(new_demographics)
```

## Use Cases

| Use Case | Description |
|----------|-------------|
| Survey enhancement | Impute income variables from tax data onto census |
| Privacy synthesis | Generate synthetic data for public release |
| Data fusion | Combine variables from multiple surveys |
| Missing data | Fill in missing values conditioned on observed |

## How It Works

```
Training Data          New Demographics
     │                       │
     ▼                       ▼
┌─────────┐           ┌─────────────┐
│ Fit     │           │ Condition   │
│ Model   │           │ Variables   │
└────┬────┘           └──────┬──────┘
     │                       │
     ▼                       ▼
┌─────────────────────────────────────┐
│        Normalizing Flow             │
│  P(target | conditions)             │
└─────────────────────────────────────┘
                 │
                 ▼
         Synthetic Data
```

## Comparison to Alternatives

| Feature | micro | CT-GAN | TVAE | synthpop |
|---------|:-----:|:------:|:----:|:--------:|
| Conditional generation | ✅ | ❌ | ❌ | ❌ |
| Zero-inflation | ✅ | ❌ | ❌ | ⚠️ |
| Exact likelihood | ✅ | ❌ | ❌ | N/A |
| Stable training | ✅ | ⚠️ | ✅ | ✅ |

## Contents

```{tableofcontents}
```
