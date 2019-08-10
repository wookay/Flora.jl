using Jive
@useinside module test_flux_odsc2019_flux_digits

# https://github.com/JuliaComputing/ODSC2019/blob/master/01-Flux-digits.ipynb
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

function create_batch(r)
    xs = [preprocess(img) for img in images[r]]
    ys = [Flux.onehot(label, 0:9) for label in labels[r]]
    return (Flux.batch(xs), Flux.batch(ys))
end

batchsize = 50 # 5000
trainbatch = create_batch(1:batchsize)
testbatch = create_batch(batchsize+1:2batchsize)

n_inputs = unique(length.(images))[] # 784
n_outputs = length(unique(labels)) # 10

model = Chain(Dense(n_inputs, n_outputs, identity), softmax)
L(x,y) = Flux.crossentropy(model(x), y)
opt = Descent()

# Training
@time Flux.train!(L, params(model), [trainbatch], opt)
@time Flux.train!(L, params(model), [trainbatch], opt)
L(trainbatch...)
@time Flux.train!(L, params(model), [trainbatch], opt)
L(trainbatch...)

callback() = @show(L(trainbatch...))

Flux.train!(L, params(model), Iterators.repeated(trainbatch, 3), opt; cb = callback)

Flux.train!(L, params(model), Iterators.repeated(trainbatch, 40), opt; cb = Flux.throttle(callback, 1))

using Printf
function show_loss()
    train_loss = L(trainbatch...)
    test_loss  = L(testbatch...)
    @printf("train loss = %.3f, test loss = %.3f\n", train_loss, test_loss)
end

Flux.train!(L, params(model), Iterators.repeated(trainbatch, 100), opt;
            cb = Flux.throttle(show_loss, 1))

function topmatch(model::Chain, img::Vector{Float64}, n=2)
    m = model(img)
    list = sort(collect(enumerate(m)), by=last, rev=true)[1:2]
    map(list) do (nth, accuracy)
        ((0:9)[nth], accuracy)
    end
end

function checkout(images, labels, nth)
    img = preprocess(images[nth])
    plot = spy(reshape(img,28,28))
    println(plot.graphics)
    @info :got labels[nth], topmatch(model, img)
end

checkout(images, labels, 5001) #

prediction(i) = findmax(model(preprocess(images[i])))[2]-1 # returns (max_value, index)
sum(prediction(i) == labels[i] for i in 1:5000)/5000
sum(prediction(i) == labels[i] for i in 5001:10000)/5000

n_hidden = 20
model = Chain(Dense(n_inputs, n_hidden, relu),
              Dense(n_hidden, n_outputs, identity), softmax)
L(x,y) = Flux.crossentropy(model(x), y)
opt = ADAM()

train_loss = Float64[]
test_loss = Float64[]
Flux.train!(L, params(model), Iterators.repeated(trainbatch, 500), opt;
            cb = Flux.throttle(show_loss, 1))

checkout(images, labels, 7010) #

using Random
p = randperm(28)
images[1][p,p]

checkout(images, labels, 1) #

end # module test_flux_odsc2019_flux_digits
