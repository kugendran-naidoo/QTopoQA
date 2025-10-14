import torch



# assume z = x.y and y = a + b
# compute z

x = torch.tensor(4.0, requires_grad=True)
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# define first operation
y = a + b

# define final operation
z = x * y

print(f"Result z:{z}")
