include("../../src/evaluation/diagnostics.jl")
using Test

data_actual = [1,1,0,0,1,0,0,0,1,1,1]
data_pred =   [1,1,1,0,1,1,1,0,0,0,1]
data_proba = [0.9, 0.8, 0.52, 0.2, 0.55, 0.52, 0.6, 0.1, 0.45, 0.3, 0.8]

@testset "diagnostics" begin
    @test true_positives(data_actual, data_pred) == 4.0
    @test round(true_positives(data_actual, data_pred, rate=true), digits=4) == 0.6667

    @test true_negatives(data_actual, data_pred) == 2.0
    @test round(true_negatives(data_actual, data_pred, rate=true), digits=4) == 0.3333

    @test false_positives(data_actual, data_pred) == 3.0
    @test false_positives(data_actual, data_pred, rate=true) == 0.6

    @test false_negatives(data_actual, data_pred) == 2.0
    @test round(false_negatives(data_actual, data_pred, rate=true), digits=4) == 0.3333

    @test round(precision_classification(data_actual, data_pred), digits=4) == 0.5714
    @test recall(data_actual, data_pred) == 0.5
    @test round(accuracy(data_actual, data_pred), digits=4) == 0.5455
    @test round(f1_score(data_actual, data_pred), digits=4) == 0.5333
    @test confusion_matrix(data_actual, data_pred) == [4.0 3.0; 2.0 2.0]
    @test round.(confusion_matrix(data_actual, data_pred, ratio=true), digits=4) == [0.3636 0.2727; 0.1818 0.1818]
    @test predict_from_threshold(data_proba, 0.5) == data_pred

    @test best_f1_search(data_actual, data_proba)[1:2] == (0.2, 0.8)
end
