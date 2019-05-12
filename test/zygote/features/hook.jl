module test_zygote_features_hook

using Test
using Zygote

# Zygote.hook
vals = []
grad = Zygote.gradient(5) do a
    Zygote.hook(ā -> push!(vals, ā), a^2)
end
@test grad == ([10], )
@test vals == [1]

vals = []
grad = Zygote.gradient(2, 3) do a, e
    Zygote.hook(ê -> (push!(vals, (ê=ê,)); ê), e) * Zygote.hook(â -> (push!(vals, (â=-â,)); -â), a)
end
@test grad == (-3, 2)
@test vals == [(â = -3,), (ê = 2,)]

end # module test_zygote_features_hook
