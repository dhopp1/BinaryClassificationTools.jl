include("confusion_matrix.jl")
using Plots

export calc_f1

"""
_given array of true labels and % predictions, return best threshold and F1 score, plot of F1 score a>_

#### parameters:
    actual : Array{Int}
        actual labels, ∈ [0,1]
    pred : Array{Float64}
        an array of prediction values 0 <= x <= 1
    single : Boolean
        if true, pass pred as an array ∈ [0,1], will return a single F1 score instead

#### returns: Tuple{Float64, Float64, Plots.Plot{Plots.GRBackend}}
    tuple of best threshold, best F1 score, plot of F1 scores
"""
function calc_f1(actual::Array{Int}, pred::Array{Float64})
    x = 0:0.01:1
    overall = []
    f1 = []
    for i in x
        push!(
            overall,
            sum(actual .== [perc > i ? 1 : 0 for perc in pred]) /
            length(actual),
        )
        tp = confusion_matrix(actual, [perc > i ? 1 : 0 for perc in pred])[1, 1]
        tn = confusion_matrix(actual, [perc > i ? 1 : 0 for perc in pred])[2, 2]
        fp = confusion_matrix(actual, [perc > i ? 1 : 0 for perc in pred])[2, 1]
        fn = confusion_matrix(actual, [perc > i ? 1 : 0 for perc in pred])[1, 2]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        push!(f1, 2 * ((precision * recall) / (precision + recall)))
    end
    p1 = plot(
        x,
        overall,
        label = "Accuracy",
        size = (1400, 700),
        xlabel = "threshold",
        ylabel = "score",
        title = "F1 Score",
        legend = :outerbottomright,
        ylim = (0, 1),
    )
    plot!(x, f1, label = "F1")

    best_f1, best_threshold_index = findmax([isnan(i) ? 0 : i for i in f1])
    return x[best_threshold_index], best_f1, p1
end
