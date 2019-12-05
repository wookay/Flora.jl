module test_macrotools_capture

using Test
using MacroTools

result = :((:error, 0))
@capture(result, (:ok, val_))
@test val === nothing

result = :((:ok, 1))
@capture(result, (:ok, n_))
@test n == 1

end # module test_macrotools_capture
