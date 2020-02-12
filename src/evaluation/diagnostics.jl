using Plots

export true_positives
export true_negatives
export false_positives
export false_negatives
export precision_classification
export recall
export accuracy
export f1_score
export confusion_matrix
export predict_from_threshold
export best_f1_search

"true positives. If rate: sensitivity (true positives / actual positives)"
true_positives(actual::Array{Int64,1}, pred::Array{Int64,1}; rate = false) =
    sum(actual .== pred .== 1) / (rate ? sum(actual .== 1) : 1)

"true negatives. If rate: specificity (true negatives / actual negatives)"
true_negatives(actual::Array{Int64,1}, pred::Array{Int64,1}; rate = false) =
    sum(actual .== pred .== 0) / (rate ? sum(actual .== 1) : 1)

"false positives. If rate: fall-out (false positives / actual negatives)"
false_positives(actual::Array{Int64,1}, pred::Array{Int64,1}; rate = false) =
    sum((actual .== 0) .& (pred .== 1)) / (rate ? sum(actual .== 0) : 1)

"true negatives. If rate: (false negatives / actual positives)"
false_negatives(actual::Array{Int64,1}, pred::Array{Int64,1}; rate = false) =
    sum((actual .== 1) .& (pred .== 0)) / (rate ? sum(actual .== 1) : 1)

"precision (false positives / (true positives + false positives))"
function precision_classification(actual::Array{Int64,1}, pred::Array{Int64,1})
    tp = true_positives(actual, pred)
    fp = false_positives(actual, pred)
    return tp / (tp + fp)
end

"recall (true positives / (true positives + false negatives))"
function recall(actual::Array{Int64,1}, pred::Array{Int64,1})
    tp = true_negatives(actual, pred)
    fn = false_negatives(actual, pred)
    return tp / (tp + fn)
end

"proportion of classifications made correctly"
accuracy(actual::Array{Int64,1}, pred::Array{Int64,1}) = sum(actual .== pred) / length(actual)

"F1 score, harmonic mean of precision and recall, 2 * ((precision * recall) / (precision + recall))"
function f1_score(actual::Array{Int64,1}, pred::Array{Int64,1})
    prec = precision_classification(actual, pred)
    rec = recall(actual, pred)
    return 2 * ((prec * rec) / (prec + rec))
end

"""
_given array of 1s and 0s, actuals and predictions, returns confusion matrix_

#### parameters:
    actual : Array{Int}
        actual labels, ∈ [0,1]
    pred : Array{Float64}
        an array of prediction values 0 <= x <= 1
    ratio : Boolean
        if true, return matrix as ratios, sum = 1, else return absolute numbers

#### returns: Array{Float64,2}
    confusion matrix of form:
    [tp fp]
    [fn tn]
    X axis = actual
    y axis = predicted
"""
function confusion_matrix(
    actual::Array{Int64,1},
    pred::Array{Int64,1};
    ratio = false,
)
    [
     true_positives(actual, pred) false_positives(actual, pred)
     false_negatives(actual, pred) true_negatives(actual, pred)
    ] ./ (ratio ? (length(actual) |> x -> round.(x, digits = 3)) : 1)
end

"given an array of predicted probabilities, return 1 or 0 depending on threshold"
predict_from_threshold(pred::Array{Float64}, threshold::Float64) = [perc > threshold ? 1 : 0 for perc in pred]

"""
_given array of true labels and % predictions, return best threshold and F1 score, plot of F1 score a>_

#### parameters:
    actual : Array{Int}
        actual labels, ∈ [0,1]
    pred : Array{Float64}
        an array of prediction values 0 <= x <= 1

#### returns: Tuple{Float64, Float64, Plots.Plot{Plots.GRBackend}}
    tuple of best threshold, best F1 score, plot of F1 scores
"""
function best_f1_search(actual::Array{Int}, pred::Array{Float64})
    x = 0:0.01:1
    accuracies = [sum(predict_from_threshold(pred, i) .== actual) / length(actual) for i in x]
    f1s = [f1_score(actual, predict_from_threshold(pred_perc, i)) for i in x]

    p1 = plot(
        x,
        accuracies,
        label = "Accuracy",
        xlabel = "threshold",
        ylabel = "F1 Score",
        title = "F1 Score",
        legend = :outerbottomright,
        ylim = (0, 1),
    )
    plot!(x, f1s, label = "F1")

    best_f1, best_threshold_index = findmax([isnan(i) ? 0 : i for i in f1s])
    return x[best_threshold_index], best_f1, p1
end
