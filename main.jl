


# This is your main workspace.


# in case you didn't open julia from this directory, we need to move you here first:
#home = ENV["HOME"]
cd("C:/Users/lucil/OneDrive/Documents/GitHub/HW-unconstrained")	# you need to change this path.


# include your code and test it:
# execute both lines if you changed your code
include("src/HW_unconstrained.jl")
include("test/runtests.jl")


# use your code in the console
# after doing include("src/maxlike.jl"),
# your module is visible, along with all objects that you decided to export. e.g. you could do
using HW_unconstrained
HW_unconstrained.runAll()
# to manually examine this function
