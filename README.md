# BinaryClassificationTools
Collection of useful tools for binary classification problems (F1 score, etc.). Most functions operate on the assumption of an array of Ints of 1s and 0s for actuals, and an array of floats [0,1] for predictions.

### Installation
```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/dhopp1/BinaryClassificationTools.jl"))
using BinaryClassificationTools
```

### Diagnostic functions
Most should be self explanatory, if not `?func_name` provides necessary information.

- `true_positives`
- `true_negatives`
- `false_positives`
- `false_negatives`
- `precision_classification`: precision, so called because of name clash
- `recall`
- `accuracy`
- `f1_score`
- `confusion_matrix`
- `predict_from_threshold`: given an array of predicted probabilities, get a binary prediction [0,1] based on a threshold
- `best_f1_search`: given an array of predicted probabilities, find best threshold for classification based on F1 score

### ROC and AUC
- `calc_auc`: calculates the integral of a function given arrays of its Xs and Ys using Riemann sums
- `roc_curve`: plots the [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curve and displays the [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)
