# Basic PyTorch Linear Regressor
# True function is y = 2x + 1

# We will try to see, given some data we will fake
# with taking the true function + adding noise, 
# whether we can recover an approximation of sorts
# for the true function

# 1) Create some fake data that follows the true
#    function + some noise
# 2) 

import torch

# number of data points
N = 10

# each data point will have 1 input feature and 1 output feature
D_in = 1
D_out = 1

# Generate input data
X = torch.randn(N, D_in)

print (f"X.shape = {X.shape}")
print (f"X = {X}")

# create true target labels based on y = 2x + 1
# true W = 2.0 and true b = 1.0

true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0) # why is this not expressed as a tensor with square brackets

y_true = X @ true_W + true_b + torch.randn(N, D_out) * 0.1 # add a little noise

# next create the models brain

# initialize parameters with random values
# shaped must be correct for matrix multiplication

W = torch.randn(D_in, D_out, requires_grad=True) # why is W setup this way with input and output feature counts?
b = torch.randn(1, requires_grad=True) # why is this called this way when W has two parameters above?

# models initial hypothesis based on random numbers which are completely wrong
# but just a starting point

print(f"Initial Weight W:\n {W}\n")
# Initial Weight W:
# tensor([[-0.1244]], requires_grad=True) - why does this tensor look this way? Is this a 1 dimensional tensor

print(f"Initial Bias b:\n {b}\n")
# Initial Bias b:
# tensor([-0.6415], requires_grad=True) - why does this tensor look like this compared to W?
# this looks like a 0 dimensional tensor = scalar

# actual training code - forward pass:

y_hat = X @ W + b

print(f"Ground Truth y_true:\n {y_true}\n")

print(f"Prediction y_hat:\n {y_hat}\n")

