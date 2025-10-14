import torch

shape = (2,3)

ones = torch.ones(shape)
zeros = torch.zeros(shape)
random = torch.randn(shape)

# print(ones)
# print(zeros)
# print(random)

some_tensor = torch.tensor([[1, 2], [3,4]])

rand_like_tensor = torch.randint_like(some_tensor, low=0, high=10)
rand_like_tensor_new_type = torch.randn_like(some_tensor, dtype=torch.float)

print(some_tensor)
print(f"same datatype", rand_like_tensor)
print(f"new newtype", rand_like_tensor_new_type)
