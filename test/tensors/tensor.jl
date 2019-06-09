module test_tensors_tensor

using Test
using Tensors # Vec Tensor

@test Vec{3, Int}((1,2,3)) == Tensor{1, 3, Int}((1,2,3))

end # module test_tensors_tensor
