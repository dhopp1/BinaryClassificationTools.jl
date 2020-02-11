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

#### returns: Array{Int64,2}
    confusion matrix of form:
    [tp fp]
    [fn tn]
    X axis = actual
    y axis = predicted
"""
function confusion_matrix(
    actual::Array{Int64,1},
    prediction::Array{Int64,1};
    ratio = false,
)
    true_positives = sum(actual .== prediction .== 1)
    true_negatives = sum(actual .== prediction .== 0)
    false_positives = sum((actual .== 0) .& (prediction .== 1))
    false_negatives = sum((actual .== 1) .& (prediction .== 0))
    if ratio
        return [
            true_positives false_positives
            false_negatives true_negatives
        ] ./ length(actual) |> x -> round.(x, digits = 3)
    else
        return [true_positives false_positives; false_negatives true_negatives]
    end
end
