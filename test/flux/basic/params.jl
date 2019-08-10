module test_flux_basic_params

using Test
using Tracker: TrackedArray, IdSet, Params, param

@test param([1 2;]) == TrackedArray([1 2;])

p = Params([1 2;])
@test p.order == [1, 2]
@test p.params == IdSet([1 2;])

end # module test_flux_basic_params
