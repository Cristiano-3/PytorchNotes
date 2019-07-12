# coding: utf-8
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N, D_in, H, D_out = 64, 1000, 100, 10

# get random inputs & labels
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# random initial weights
w1 = torch.randn(D_in, H, device=device, requires_grad=True)  
w2 = torch.randn(H, D_out, device=device, requires_grad=True)  

learning_rate = 1e-6
for t in range(500):
    # forward
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # backward
    loss.backward()

    # update weights
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients 
        # after updating weights
        w1.grad.zero_()
        w2.grad.zero_()

    