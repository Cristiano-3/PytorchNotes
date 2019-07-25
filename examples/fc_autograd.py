# coding: utf-8
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N, D_in, H, D_out = 64, 1000, 100, 10

# get random inputs & labels
x = torch.randn(N, D_in, device=device)  # requires_grad=False, 不计算关于该张量的梯度;
y = torch.randn(N, D_out, device=device)

# random initial weights
# 当 x 是一个 Tensor, 若 x.requires_grad=True, 
# 那么用张量 x.grad 保存张量x 关于某些标量的梯度.
w1 = torch.randn(D_in, H, device=device, requires_grad=True)   
w2 = torch.randn(H, D_out, device=device, requires_grad=True)  

learning_rate = 1e-6
for t in range(500):
    # forward, 前向传播动态定义计算图, 沿着该计算图反向传播就能容易计算出梯度;
    # 这里由于不需要手动实现反向传播(梯度计算), 不需要使用中间变量来实现前向传播过程;
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # loss
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the a scalar value
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # backward, 使用自动微分完成神经网络中的反向传播计算(即求梯度);
    # 计算所有requires_grad=True的张量, 关于loss的梯度(即w1.grad, w2.grad);
    loss.backward()

    # update weights
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients 
        # after updating weights
        w1.grad.zero_()
        w2.grad.zero_()

    