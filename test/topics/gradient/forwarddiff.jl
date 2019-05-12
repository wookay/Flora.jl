module test_topics_gradient_forwarddiff

using Test
using ForwardDiff

ForwardDiff.derivative
@test ForwardDiff.derivative(sin, 2pi) == cos(2pi) == 1.0

ForwardDiff.gradient
@test ForwardDiff.gradient(x->x[1], 5:7) == [1, 0, 0]
@test ForwardDiff.gradient(x->x[2], 5:7) == [0, 1, 0]
@test ForwardDiff.gradient(x->x[3], 5:7) == [0, 0, 1]

ForwardDiff.jacobian
@test ForwardDiff.jacobian(identity, 5:7) == [1 0 0;
                                              0 1 0;
                                              0 0 1]

ForwardDiff.hessian
@test ForwardDiff.hessian(x->1, rand(2)) == [0 0; 0 0]
@test ForwardDiff.hessian(x->1.0, rand(2)) == [0 0; 0 0]

end # module test_topics_gradient_forwarddiff
