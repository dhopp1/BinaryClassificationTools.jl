include("../../src/evaluation/roc_auc.jl")
using Test

data_actual = [1,1,0,0,1,0,0,0,1,1,1]
data_pred =   [1,1,1,0,1,1,1,0,0,0,1]
data_proba = [0.9, 0.8, 0.52, 0.2, 0.55, 0.52, 0.6, 0.1, 0.45, 0.3, 0.8]
roc_data_output = roc_data(data_actual, data_proba)

@testset "ROC and AUC" begin
    @test round(sum(roc_data_output[1]), digits=4) == 63.3333
    @test sum(roc_data_output[2]) == 38.8

    @test round(calc_auc(roc_data_output[2], roc_data_output[1]), digits=4) == 0.7667

    @test !isa(try roc_curve(data_actual, data_proba) catch ex ex end, Exception)
end
