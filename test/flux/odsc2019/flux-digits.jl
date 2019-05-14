using Jive
@useinside module test_flux_odsc2019_flux_digits

using Test
using Flux
using .Flux.Data.MNIST
using Images
using UnicodePlots: spy

labels = MNIST.labels()
images = MNIST.images()

# (println ∘ spy)(images[1])
"""
      Sparsity Pattern
      ┌──────────────┐
    1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ > 0
      │⠀⠀⠀⢀⣤⣤⣶⣶⣶⣶⣶⡶⠀⠀│ < 0
      │⠀⠀⠀⠈⠻⢿⣿⡋⠛⠀⠀⠀⠀⠀│
      │⠀⠀⠀⠀⠀⠈⠻⣷⣶⣄⠀⠀⠀⠀│
      │⠀⠀⠀⠀⠀⠀⣀⣬⣽⣿⠆⠀⠀⠀│
      │⠀⠀⣀⣤⣶⣿⣿⠿⠛⠁⠀⠀⠀⠀│
   28 │⠀⠀⠉⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀│
      └──────────────┘
      1             28
         nz = 166
"""

preprocess(img) = vec(Float64.(img))
xs = preprocess.(images[1:3]);
spy(reshape(xs[1],28,28))
"""
      Sparsity Pattern
      ┌──────────────┐
    1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ > 0
      │⠀⠀⠀⢀⣤⣤⣶⣶⣶⣶⣶⡶⠀⠀│ < 0
      │⠀⠀⠀⠈⠻⢿⣿⡋⠛⠀⠀⠀⠀⠀│
      │⠀⠀⠀⠀⠀⠈⠻⣷⣶⣄⠀⠀⠀⠀│
      │⠀⠀⠀⠀⠀⠀⣀⣬⣽⣿⠆⠀⠀⠀│
      │⠀⠀⣀⣤⣶⣿⣿⠿⠛⠁⠀⠀⠀⠀│
   28 │⠀⠀⠉⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀│
      └──────────────┘
      1             28
         nz = 166
"""

ys = [Flux.onehot(label, 0:9) for label in labels[1:3]]
@test ys[1] == [false, false, false, false, false, true, false, false, false, false]
@test labels[1:3] == [5, 0, 4]

end # module test_flux_odsc2019_flux_digits
