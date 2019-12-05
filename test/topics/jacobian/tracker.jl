module test_topics_jacobian_tracker

using Test
using Tracker

Tracker.jacobian

f(x, y) = [x^2 * y, 5x  + sin(y)]
J(x) = Tracker.jacobian(x -> f(x...), x)
@test J([1; 1]) == [2.0 1.0; 5.0 0.5403023058681398]
@test J([0.5; 0.5]) == [0.5 0.25; 5.0 0.8775825618903728]

end # module test_topics_jacobian_tracker
