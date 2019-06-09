module test_coordinatetransformations_spherical

using Test
using CoordinateTransformations # SphericalFromCartesian
using StaticArrays # SVector

# https://en.wikipedia.org/wiki/Spherical_coordinate_system

# Spherical(r, θ, ϕ) - 3D spherical coordinates
# r: radial distance
# θ : polar angle
# φ : azimuthal angle

s_from_cart = SphericalFromCartesian()
(x, y, z) = (-1.0, 2.0, 3.0)
xyz = SVector(x, y, z)
s = Spherical(3.7416573867739413, 2.0344439357957027, 0.9302740141154721)
@test s_from_cart(xyz) == s

cart_from_s = CartesianFromSpherical()
cart = cart_from_s(s)
@test s == s_from_cart(cart)

r = sqrt(x^2 + y^2 + z^2)
@test r == s.r

polar_from_cart = PolarFromCartesian()
cart_from_polar = CartesianFromPolar()
xy = SVector(x, y)
polar = polar_from_cart(xy)
@test polar == Polar(2.23606797749979, 2.0344439357957027)
@test polar.θ == s.θ
@test cart_from_polar(polar) == xy

end # modue test_coordinatetransformations_spherical
