# flake8: noqa
import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
import correlation_cuda


class CorrelationFunction(Function):

    @staticmethod
    def forward(ctx,
            pad_size,
            kernel_size,
            max_displacement,
            stride1,
            stride2,
            corr_multiply,
            input1,
            input2):
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(
                input1,
                input2,
                rbot1,
                rbot2,
                output,
                ctx.pad_size,
                ctx.kernel_size,
                ctx.max_displacement,
                ctx.stride1,
                ctx.stride2,
                ctx.corr_multiply)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(
                input1,
                input2,
                rbot1,
                rbot2,
                grad_output,
                grad_input1,
                grad_input2,
                ctx.pad_size,
                ctx.kernel_size,
                ctx.max_displacement,
                ctx.stride1,
                ctx.stride2,
                ctx.corr_multiply)

        return grad_input1, grad_input2

class Correlation(Module):
    def __init__(
            self,
            pad_size=0,
            kernel_size=0,
            max_displacement=0,
            stride1=1,
            stride2=2,
            corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):

        result = CorrelationFunction.apply(
            self.pad_size,
            self.kernel_size,
            self.max_displacement,
            self.stride1,
            self.stride2,
            self.corr_multiply,
            input1,
            input2)

        return result
