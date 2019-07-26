# coding: utf-8
import torch

# forward: 
# output = relu(input)

# backward:
# grad_output = ▽loss/▽output
# grad_intput = ▽loss/▽input 
#             = ▽loss/▽output * ▽output/▽input
#             = grad_output * ▽output/▽input

# 其中:
# ▽output/▽input = 0 when input<0  or  1 when input>=0

class MyReLU(torch.autograd.Function):
    """
    通过继承autograd.Function类, 并实现其中的前后向传播
    实现自定义的自动微分函数
    """
    @staticmethod
    def forward(ctx, input):
        """
        前向传播接收一个输入张量, 返回一个输出张量
        ctx是上下文对象, 用于暂存信息以备后向传播计算使用
        """
        ctx.save_for_backward(input)
        output = input.clamp(min=0)
        return output


    @staticmethod
    def backward(ctx, grad_output):
        """
        后向传播接收一个loss关于output的梯度
        需要计算loss关于input的梯度
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


dtype = torch.float  
device = torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

# random dataset
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# random weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6

for t in range(500):
    # 为使用自定义Function, 使用Function.apply方法并别名为relu
    relu = MyReLU.apply

    # forward
    # 手动前向过程
    y_pred = relu(x.mm(w1)).mm(w2)

    # loss
    loss = (y - y_pred).pow(2).sum()
    print(t, loss.item())

    # backward
    # 为权值自动计算微分
    loss.backward()

    # 手动更新权值参数(应用梯度降方法)
    # 为避免更新权值时构建计算图, 使用 torch.no_grad()
    # 自动微分有赖于计算图, 若更新权值的部分加入计算图并不会被自动微分时用到
    # 所以就用 torch.no_grad()
    with torch.no_grad():

        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 权值更新完毕需手动置零梯度
        w1.grad.zero_()
        w2.grad.zero_()

