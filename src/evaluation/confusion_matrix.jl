include("diagnostics.jl")

export confusion_matrix

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
