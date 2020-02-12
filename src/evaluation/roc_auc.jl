include("diagnostics.jl")
using DataFrames, Plots

export calc_auc
export roc_curve

"calc the AUC of a function given xs and ys"
function calc_auc(x::Array, y::Array)
    df = names!(hcat(x, y) |> DataFrame, Symbol.(["x", "y"]))
    df = sort(df, [:x, :y])
    auc = 0
    for i in 2:nrow(df)
        # rectangular component
        x = df[i, :x][1] - df[i-1, :x][1]
        y = df[i-1, :y][1]
        auc += x*y
        # triangular component
        height = df[i, :y][1] - y
        auc += 0.5*x * height
    end
    return auc
end

"generates data necssary for ROC curve, array of true positive and false positive rates + AUC"
function roc_data(actual::Array{Int64}, pred::Array{Float64}, step = 0.01)
    x = 0:step:1
    tps = [true_positives(actual, predict_from_threshold(pred, i), rate=true) for i in x]
    fps = [false_positives(actual, predict_from_threshold(pred, i), rate=true) for i in x]
    auc = calc_auc(fps, tps)
    return tps, fps, auc
end

"""
_given array of predictions (0 ≤ x ≤ 1] and array of actuals, return ROC AUC plot_

#### parameters:
    actual : Array{Int}
        actual labels, ∈ [0,1]
    pred : Array{Float64}
        an array of prediction values 0 ≤ x ≤ 1

#### returns: Plots.Plot{Plots.GRBackend}
    plot of ROC AUC
"""
function roc_curve(actual::Array{Int}, pred::Array{Float64})
    # diagonal line
    p1 = plot(
        [0, 1],
        [0, 1],
        label = "",
        line = :dash,
        color = :black,
        ylim = (0, 1),
        xlim = (0, 1),
    )
    y, x, auc = roc_data(actual, pred)
    plot!(x, y, label = "AUC=$(round(auc, digits=4))", color = :blue)

    plot!(legend = :bottomright)
    plot!(
        ylabel = "True Positive Rate",
        xlabel = "False Positive Rate",
        title = "ROC and AUC",
    )
    return p1
end
