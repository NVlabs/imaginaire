# flake8: noqa
from torch.nn.modules.module import Module
from torch.autograd import Function, Variable
import resample2d_cuda


class Resample2dFunction(Function):

    @staticmethod
    # def forward(ctx, input1, input2, kernel_size=1, bilinear=True):
    def forward(ctx, input1, input2, kernel_size=1):
        assert input1.is_contiguous()
        assert input2.is_contiguous()

        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size
        ctx.bilinear = True

        _, d, _, _ = input1.size()
        b, _, h, w = input2.size()
        output = input1.new(b, d, h, w).zero_()

        resample2d_cuda.forward(input1, input2, output, kernel_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        assert grad_output.is_contiguous()

        input1, input2 = ctx.saved_tensors

        grad_input1 = Variable(input1.new(input1.size()).zero_())
        grad_input2 = Variable(input1.new(input2.size()).zero_())

        # resample2d_cuda.backward(input1, input2, grad_output.data,
        #                          grad_input1.data, grad_input2.data,
        #                          ctx.kernel_size, ctx.bilinear)
        resample2d_cuda.backward(input1, input2, grad_output.data,
                                 grad_input1.data, grad_input2.data,
                                 ctx.kernel_size)

        return grad_input1, grad_input2, None, None


class Resample2d(Module):

    def __init__(self, kernel_size=1, bilinear=True):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size
        self.bilinear = bilinear

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        # return Resample2dFunction.apply(
        #     input1_c, input2, self.kernel_size, self.bilinear)
        return Resample2dFunction.apply(
            input1_c, input2, self.kernel_size)