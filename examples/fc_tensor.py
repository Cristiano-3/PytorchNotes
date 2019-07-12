# coding: utf-8
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N, D_in, H, D_out = 64, 1000, 100, 10

# get random inputs & labels
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# random initial weights
w1 = torch.randn(D_in, H, device=device)
w2 = torch.randn(H, D_out, device=device) # , requires_grad=True

learning_rate = 1e-6
for t in range(500):
    # forward
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # backward
    grad_y_pred = 2.0 * (y_pred - y)
    
    grad_w2 = h_relu.t().mm(grad_y_pred)

    grad_h_relu = grad_y_pred.mm(w2.t())
    
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    
    grad_w1 = x.t().mm(grad_h)

    # update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    

