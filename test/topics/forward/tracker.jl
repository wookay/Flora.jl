module test_topics_forward_tracker

using Test
using Tracker

Tracker.forward
f(x) = 2x^3
y, back = Tracker.forward(f, 3)
@test back(1) == (y,) == (54,)

Tracker.forwarddiff
@test Tracker.forwarddiff(f, 3) == y

end # module test_topics_forward_tracker
