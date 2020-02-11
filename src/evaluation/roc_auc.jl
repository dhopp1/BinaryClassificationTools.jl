using Plots

export plot_auc

function calc_auc(xs::Array, ys::Array)
    sorted = sort([(i, j) for (i, j) in zip(xs, 1:length(xs))])
    index = [j for (i, j) in sorted]
    sorted_x = [i for (i, j) in sorted]
    sorted_y = ys[index]
    auc = 0
    for i = 2:length(sorted_x)
        # rectangular component
        x = sorted_x[i] - sorted_x[i-1]
        y = sorted_y[i-1]
        auc += x * y
        # triangular component
        height = sorted_y[i] - y
        auc += 0.5 * x * height
    end
    return auc
end

prediction(probability_positive::Array{Float64}, threshold::Float64) = [x > threshold ? 1 : 0 for x in probability_positive]

"""
returns true positive and false positive rates given arrays of actuals and predictions
"""
function tp_fp(actual::Array{Int64}, pred::Array{Int64})
    tpr = sum([if pred[i] == 1 && actual[i] == 1
        1
    else
        0
    end for i in collect(range(1, stop = length(pred)))]) / sum(actual)
    fp = sum([if pred[i] == 1 && actual[i] == 0
        1
    else
        0
    end for i in collect(range(1, stop = length(pred)))])
    tn = sum([if pred[i] == 0 && actual[i] == 0
        1
    else
        0
    end for i in collect(range(1, stop = length(pred)))])
    fpr = fp / (fp + tn)
    return tpr, fpr
end

function roc_data(actual::Array{Int64}, probability_positive::Array{Float64}, step = 0.01)
    tps = []
    fps = []
    for i in range(0, 1, step = step)
        tp, fp = tp_fp(actual, prediction(probability_positive, i))
        push!(tps, tp)
        push!(fps, fp)
    end
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
function plot_auc(actual::Array{Int}, pred::Array{Float64})
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
