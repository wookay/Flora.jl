module test_zygote_paper_optimising_colours

# code from https://github.com/MikeInnes/zygote-paper/blob/master/5_optimising_colours/optimise_colours.jl

using Test
using Colors # RGB colordiff
using Zygote

target = RGB(1, 0, 0)
colour = RGB(1, 1, 1)
function f(y)
    colordiff(target, y)
end
(c,) = Zygote.gradient(f, colour)
@test c == (r = -78.82052432473564, g = 174.97066510154266, b = -61.715794344462076)

end # module test_zygote_paper_optimising_colours
