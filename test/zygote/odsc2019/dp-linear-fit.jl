using Jive
@useinside module test_zygote_odsc2019_dp_linear_fit

# https://github.com/JuliaComputing/ODSC2019/blob/master/02-DP-linear-fit.ipynb

using Test
using Zygote, LinearAlgebra

struct LinearRegression
    weights::Matrix
    bias::Array{Float64,0}
end
LinearRegression(nparams) = LinearRegression(randn(1, nparams), fill(0.0))

function predict(model::LinearRegression, X)
    return model.weights * X .+ model.bias[]
end

function loss(model::LinearRegression, X, Y)
    return norm(predict(model, X) .- Y, 2)
end

weights_gt = [1.0, 2.7, 0.3, 1.2]'
bias_gt = 0.4

X = randn(length(weights_gt), 10000)
Y = weights_gt * X .+ bias_gt
X .+= 0.01 .* randn(size(X))

model = LinearRegression(size(X, 1))

grads = Zygote.gradient(model) do m
    loss(m, X[:, 1], Y[1])
end

# (weights = [-0.116278 -0.953694 -0.836205 -1.51052], bias = -1.0)
grads = grads[1]

function sgd_update!(model::LinearRegression, grads, η = 0.001)
    model.weights .-= η .* grads.weights
    model.bias .-= η .* grads.bias
end

for idx in 1:size(X, 2)
    grads = Zygote.gradient(m -> loss(m, X[:, idx], Y[idx]), model)[1]
    sgd_update!(model, grads)
end

# LinearRegression([-0.666408 -0.157642 0.252747 0.226991], 0.0)
model

# 1×4 Adjoint{Float64,Array{Float64,1}}:
#  1.0  2.7  0.3  1.2
weights_gt

end # module test_zygote_odsc2019_dp_linear_fit
