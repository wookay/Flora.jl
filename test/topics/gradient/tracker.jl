module test_topics_gradient_tracker

using Test
using Tracker

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

end # module test_topics_gradient_tracker
