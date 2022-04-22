"""
    foo(x, y)

Creates a 2-element static array from the scalars `x` and `y`.
"""
function foo(x::T, y::T) where T <: Number
    SA[x, y]
end