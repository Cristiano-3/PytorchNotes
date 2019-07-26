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
# ▽output/▽input = 0 or 1

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
        intput, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input