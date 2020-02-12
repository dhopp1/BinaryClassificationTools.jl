export true_positives
export true_negatives
export false_positives
export false_negatives
export confusion_matrix

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
