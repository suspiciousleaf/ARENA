# %%
import torch

# define two tensors
A = torch.tensor(2.0, requires_grad=True)
print("Tensor-A:", A)
B = torch.tensor(5.0, requires_grad=True)
print("Tensor-B:", B)

# define a function using above defined
# tensors
x = 5 * A
print("x:", x)

# call the backward method
x.backward()

# print the gradients using .grad
print("A.grad:", A.grad)
# print("B.grad:", B.grad)

C = 2 * A
D = 3 * A
E = 2 * C
C.backward()
print(f"{A.grad=}")
D.backward()
print(f"{A.grad=}")

# %%
