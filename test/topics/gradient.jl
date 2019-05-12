module test_topics_forwarddiff_gradient

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

end # module test_topics_forwarddiff_gradient



module test_topics_flux_gradient

using Test
using Flux.Tracker # TrackedArray

Tracker.gradient
@test Tracker.gradient(x->x[1], 5:7) == (TrackedArray([1, 0, 0]), )
@test Tracker.gradient(x->x[2], 5:7) == (TrackedArray([0, 1, 0]), )
@test Tracker.gradient(x->x[3], 5:7) == (TrackedArray([0, 0, 1]), )

Tracker.jacobian
@test Tracker.jacobian(identity, 5:7) == [1 0 0;
                                          0 1 0;
                                          0 0 1]

@test Tracker.hessian(x->1, 5:7) == [0 0 0; 0 0 0; 0 0 0]
@test Tracker.hessian(x->1.0, 5:7) == [0 0 0; 0 0 0; 0 0 0]

end # module test_topics_flux_gradient



module test_topics_zygote_gradient

using Test
using Zygote

Zygote.gradient
@test Zygote.gradient(sin, 2pi) == (1.0, )

Zygote.forward_jacobian
@test Zygote.forward_jacobian(identity, 5:7) == ([5, 6, 7], [1 0 0; 0 1 0; 0 0 1])
@test Zygote.forward_jacobian(x->x[1], 5:7) == (5, [1 0 0]')
@test Zygote.forward_jacobian(x->x[2], 5:7) == (6, [0 1 0]')
@test Zygote.forward_jacobian(x->x[3], 5:7) == (7, [0 0 1]')

Zygote.hessian
@test Zygote.hessian(x->x[1], 5:7) == [0 0 0; 0 0 0; 0 0 0]

end # module test_topics_zygote_gradient
