module test_zygote_basic_gradient

using Test
using Zygote # gradient Params

W, b = rand(2, 3), rand(2)

predict(x) = W*x .+ b

g = gradient(Params([W, b])) do
    sum(predict([1, 2, 3]))
end

@test g isa Zygote.Grads
@test (g[W], g[b]) == ([1.0 2.0 3.0; 1.0 2.0 3.0], [1.0, 1.0])

end # module test_zygote_basic_gradient
