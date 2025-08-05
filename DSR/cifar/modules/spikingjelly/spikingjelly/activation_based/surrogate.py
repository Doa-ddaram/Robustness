import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .auto_cuda import cfunction

tab4_str = '\t\t\t\t'  # used for aligning code
curly_bracket_l = '{'
curly_bracket_r = '}'


@torch.jit.script
def heaviside(x: torch.Tensor):
    '''
    * :ref:`API in English <heaviside.__init__-en>`
    .. _heaviside.__init__-cn:

    :param x: �솾占�?�솻洹μ삌?燁살쫾nsor
    :return: �솾占�?�솻洹μ삌占쎈뼇tensor

    heaviside??繹먲옙?甕곕씤�쇊�뙼�봾�뼒?占쎈쑏占쎈뉴�댚占�?占쎈뮍嶺뚋욱맆占쎄튅?筌띾톪彛�?

    .. math::
        g(x) =
        \\begin{cases}
        1, & x \\geq 0 \\\\
        0, & x < 0 \\\\
        \\end{cases}

    ??野껓옙?泳�占�? `HeavisideStepFunction <https://mathworld.wolfram.com/HeavisideStepFunction.html>`_ 嚥싳쉶瑗э쭗猿뗭죩占쏙옙�쓷�뜑???占쎌쑌占쎈빋占쎈뼕占쎈펵?�굢占�?�뇦占�????

    * :ref:`佯몌옙???嶺뚮쵑�쑇PI <heaviside.__init__-cn>`
    .. _heaviside.__init__-en:

    :param x: the input tensor
    :return: the output tensor

    The heaviside function, which is defined by

    .. math::
        g(x) =
        \\begin{cases}
        1, & x \\geq 0 \\\\
        0, & x < 0 \\\\
        \\end{cases}

    For more information, see `HeavisideStepFunction <https://mathworld.wolfram.com/HeavisideStepFunction.html>`_.

    '''
    return (x >= 0).to(x)


def check_manual_grad(primitive_function, spiking_function, *args, **kwargs):
    '''
    :param primitive_function: �솾占�??岳묕옙????占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏???占쎈１??占쎌쓨??占쎈뼒?占쎈쑏?
    :type primitive_function: callable
    :param spiking_function: �솾占�??岳묕옙????占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏?
    :type spiking_function: callable

    �솾占�??岳묕옙????占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏???占쎈１?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼤筌좑옙???占쎈섐??亦낉옙??占쎌졅??�몭占�??占쎈１輿삳뿫遊억옙�뮍占쎈쇀占쎈폀鸚룸강占썩넃占쎌뻼援⒴럴占�?占쏙옙占�?占쎈뼒?占쎈쑏???占쎌죳�솾占�????占쎈첊??占쎌졅??�몭�벝已�??岳묕옙????亦낉옙?嶺뚮씭�넪藥뀐옙壤깍옙占쎈렭?????

    占쎌녇占쎈�억옙占쏙옙?占쎈뼒?占쎈쑏占쏙옙占썼굜占�???占쎈첊�솾占�??岳묕옙????占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏占쎈쳷piking_function??占쎈１?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈���댚占�?占쎈뮋?占쎈빟??占쎌쓨??占쎈뼒?占쎈쑏占쎈쳲rimitive_function??占쎈１?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈��??�솻洹μ쪠占쎄뎀??亦낉옙?嶺뚮씭�뼸筌좑옙???占쎈뻹??�윜占�???�맱占�????占쎈뻹???�뜮占�?占쎈땬占쎈��?占쎈펵?占쎈＜佯몃돆鍮뽳옙�떚?占쎌졑佯몃돆鍮귨옙嫄�??野껓옙?占쎈１?泳�占�??�굢占�?佯몃돆鍮뽳옙逾�?占쏙옙占�?甕곤옙?占쎈�둷s???

    占쎌쎔占쎈윥占쎌쑏?�맱�떦�뒍癰귘뫚留�?占쎌뱾輿삼옙??

    .. code-block:: python

        def s2nn_apply(x, alpha, beta):
            return surrogate.s2nn.apply(x, alpha, beta)

        surrogate.check_manual_grad(surrogate.S2NN.primitive_function, s2nn_apply, alpha=4., beta=1.)
    '''
    x = torch.arange(-2, 2, 32 / 8192)
    # x = torch.as_tensor([-1., 0., 1.])
    x.requires_grad_(True)
    primitive_function(x, *args, **kwargs).sum().backward()
    x_grad_auto = x.grad.clone()
    x.grad.zero_()
    spiking_function(x, *args, **kwargs).sum().backward()
    x_grad_manual = x.grad.clone()
    print('auto   grad', x_grad_auto)
    print('manual grad', x_grad_manual)
    abs_error = (x_grad_manual - x_grad_auto).abs()
    idx = abs_error.argmax()
    print('max error', abs_error[idx], 'occurs at')
    print(f'x[{idx}] = {x[idx]}')
    print('auto   grad', x_grad_auto[idx])
    print('manual grad', x_grad_manual[idx])


def check_cuda_grad(neu, surrogate_function, device, *args, **kwargs):
    # check_cuda_grad(neuron.IFNode, surrogate.S2NN, device='cuda:1', alpha=4., beta=1.)
    for dtype in [torch.float, torch.half]:
        print(dtype)
        net = neu(surrogate_function=surrogate_function(*args, **kwargs), step_mode='m')
        net.to(device)
        x = torch.arange(-2, 2, 32 / 8192, device=device, dtype=dtype)
        x.requires_grad_(True)
        net.backend = 'torch'
        net(x.unsqueeze(0)).sum().backward()
        x_grad_py = x.grad.clone()
        x.grad.zero_()
        net.reset()
        net.backend = 'cupy'
        net(x.unsqueeze(0)).sum().backward()

        x_grad_cp = x.grad.clone()
        # print('python grad', x_grad_py)
        # print('cupy   grad', x_grad_cp)
        abs_error = (x_grad_cp - x_grad_py).abs()
        idx = abs_error.argmax()
        print('max error', abs_error[idx], 'occurs at')
        print(f'x[{idx}] = {x[idx]}')
        print('python grad', x_grad_py[idx])
        print('cupy   grad', x_grad_cp[idx])


class SurrogateFunctionBase(nn.Module):
    def __init__(self, alpha, spiking=True):
        super().__init__()
        self.spiking = spiking
        self.alpha = alpha

    def set_spiking_mode(self, spiking: bool):
        self.spiking = spiking

    def extra_repr(self):
        return f'alpha={self.alpha}, spiking={self.spiking}'

    @staticmethod
    def spiking_function(x, alpha):
        raise NotImplementedError

    @staticmethod
    def primitive_function(x, alpha):
        raise NotImplementedError

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        raise NotImplementedError

    def cuda_code_start_comments(self):
        return f'// start: spikingjelly.activation_based.surrogate.{self._get_name()}.cuda_code'

    def cuda_code_end_comments(self):
        return f'// end: spikingjelly.activation_based.surrogate.{self._get_name()}.cuda_code'

    def forward(self, x: torch.Tensor):
        if self.spiking:
            return self.spiking_function(x, self.alpha)
        else:
            return self.primitive_function(x, self.alpha)

    def cuda_codes(self, y: str, x: str, dtype: str):
        # new version
        raise NotImplementedError


class MultiArgsSurrogateFunctionBase(nn.Module):
    def __init__(self, spiking: bool, *args, **kwargs):
        super().__init__()
        self.spiking = spiking

    def set_spiking_mode(self, spiking: bool):
        self.spiking = spiking

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        raise NotImplementedError

    def cuda_code_start_comments(self):
        return f'// start: spikingjelly.activation_based.surrogate.{self._get_name()}.cuda_code'

    def cuda_code_end_comments(self):
        return f'// end: spikingjelly.activation_based.surrogate.{self._get_name()}.cuda_code'

    def cuda_codes(self, y: str, x: str, dtype: str):
        # new version
        raise NotImplementedError


@torch.jit.script
def piecewise_quadratic_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    x_abs = x.abs()
    mask = (x_abs > (1 / alpha))
    grad_x = (grad_output * (- (alpha ** 2) * x_abs + alpha)).masked_fill_(mask, 0)
    return grad_x, None


class piecewise_quadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_quadratic_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class PiecewiseQuadratic(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        '''
        * :ref:`API in English <PiecewiseQuadratic.__init__-en>`
        .. _PiecewiseQuadratic.__init__-cn:

        :param alpha: ??占쎌젞??占쎌냲?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌몴�뿆占�?岳묕옙????占쎈１占쎈릅�뙴紐껓옙占�?占쎌맮?筌륅옙?�굢罐由�???占쎈１?占쎈솿??占쎈쑏?
        :param spiking: ??亦낉옙?嶺뚮씭�뼱�몴�뇿�돦占쎌삌占쎈뼇?占쎈き???�땻�벂二곤옙遊억옙�뮖?占쎌맽嚥싲갇源듸옙占쎌뼲彛�? ``True``輿삳뿫遊억옙�뮍?筌륅옙??占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙占�? ``heaviside`` ???占쎈뮍?筌륅옙?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼦占쎄샴?占쎈뮀占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??�뇡占�?�윜諭��삌�젆�굚�렧?? ``False``
            ??占쎈묍佯몃돆鍮뽳옙逾�?占쎈��?占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??輿삳뿫遊억옙�뮍?占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌뱽�댚占�?占쎈뮋?占쎈��?占쏙옙占�??占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩???占쎈１�솾占�??岳묕옙????占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏占쎈뉼占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏?

        ?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙占�???占쎈광嚥싳쉸瑗띈キ占�?占쎈뼁���遺용퉩�젆占�?占쎈뼒?占쎈쑏???占쎈１�솾占�??岳묕옙??輿삳뿫遊억옙�럵?占쎈빑饔앸��占쏙옙?影��슢�셼??占쎈뼒?占쎈쑏占쎈뉴�댚占�?筌랃옙?占쎈１?占쎈き???�땻占�?占쎈쇀??占쎄껀???占쎈뼒?占쎈쑏???�윜諭�猷놅쭩占�?占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼤筌좑옙?

        .. math::
            g'(x) =
            \\begin{cases}
            0, & |x| > \\frac{1}{\\alpha} \\\\
            -\\alpha^2|x|+\\alpha, & |x| \\leq \\frac{1}{\\alpha}
            \\end{cases}

        ?占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏占쎈뉵筌좑옙?

        .. math::
            g(x) =
            \\begin{cases}
            0, & x < -\\frac{1}{\\alpha} \\\\
            -\\frac{1}{2}\\alpha^2|x|x + \\alpha x + \\frac{1}{2}, & |x| \\leq \\frac{1}{\\alpha}  \\\\
            1, & x > \\frac{1}{\\alpha} \\\\
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseQuadratic.*
            :width: 100%

        ?泳�怨�留�?占쎈뼒?占쎈쑏???筌륅옙?嶺뚮쵐�눓雅뚳옙? [#esser2016convolutional]_ [#STBP]_ [#LSNN]_ [#neftci2019surrogate]_ [#panda2020toward]_ 佯몌옙??略노쵐鍮섓옙�뮀占쏙옙占�????

        * :ref:`佯몌옙???嶺뚮쵑�쑇PI <PiecewiseQuadratic.__init__-cn>`
        .. _PiecewiseQuadratic.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The piecewise quadratic surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) =
            \\begin{cases}
            0, & |x| > \\frac{1}{\\alpha} \\\\
            -\\alpha^2|x|+\\alpha, & |x| \\leq \\frac{1}{\\alpha}
            \\end{cases}

        The primitive function is defined by

        .. math::
            g(x) =
            \\begin{cases}
            0, & x < -\\frac{1}{\\alpha} \\\\
            -\\frac{1}{2}\\alpha^2|x|x + \\alpha x + \\frac{1}{2}, & |x| \\leq \\frac{1}{\\alpha}  \\\\
            1, & x > \\frac{1}{\\alpha} \\\\
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseQuadratic.*
            :width: 100%

        The function is used in [#esser2016convolutional]_ [#STBP]_ [#LSNN]_ [#neftci2019surrogate]_ [#panda2020toward]_.

        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return piecewise_quadratic.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        mask0 = (x > (1.0 / alpha)).to(x)
        mask1 = (x.abs() <= (1.0 / alpha)).to(x)

        return mask0 + mask1 * (-(alpha ** 2) / 2 * x.square() * x.sign() + alpha * x + 0.5)

    @staticmethod
    def backward(grad_output, x, alpha):
        return piecewise_quadratic_backward(grad_output, x, alpha)[0]

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.PiecewiseQuadratic(alpha=1.5, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=1.5$')

    # surrogate_function = surrogate.PiecewiseQuadratic(alpha=1.5, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=1.5$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('Piecewise quadratic surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


@torch.jit.script
def piecewise_exp_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha / 2 * (- alpha * x.abs()).exp_() * grad_output, None


class piecewise_exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_exp_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class PiecewiseExp(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        '''
        * :ref:`API in English <PiecewiseExp.__init__-en>`
        .. _PiecewiseExp.__init__-cn:

        :param alpha: ??占쎌젞??占쎌냲?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌몴�뿆占�?岳묕옙????占쎈１占쎈릅�뙴紐껓옙占�?占쎌맮?筌륅옙?�굢罐由�???占쎈１?占쎈솿??占쎈쑏?
        :param spiking: ??亦낉옙?嶺뚮씭�뼱�몴�뇿�돦占쎌삌占쎈뼇?占쎈き???�땻�벂二곤옙遊억옙�뮖?占쎌맽嚥싲갇源듸옙占쎌뼲彛�? ``True``輿삳뿫遊억옙�뮍?筌륅옙??占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙占�? ``heaviside`` ???占쎈뮍?筌륅옙?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼦占쎄샴?占쎈뮀占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??�뇡占�?�윜諭��삌�젆�굚�렧?? ``False``
            ??占쎈묍佯몃돆鍮뽳옙逾�?占쎈��?占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??輿삳뿫遊억옙�뮍?占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌뱽�댚占�?占쎈뮋?占쎈��?占쏙옙占�??占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩???占쎈１�솾占�??岳묕옙????占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏占쎈뉼占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏?

        ?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙占�???占쎈광嚥싳쉸瑗띈キ占�?占쎈걠?占쎈쑏???占쎈뼒?占쎈쑏???占쎈１�솾占�??岳묕옙????占쎈１?占쎈き???�땻占�?占쎈쇀??占쎄껀???占쎈뼒?占쎈쑏???�윜諭�猷놅쭩占�?占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼤筌좑옙?

        .. math::
            g'(x) = \\frac{\\alpha}{2}e^{-\\alpha |x|}

        ?占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏占쎈뉵筌좑옙?

        .. math::
            g(x) =
            \\begin{cases}
            \\frac{1}{2}e^{\\alpha x}, & x < 0 \\\\
            1 - \\frac{1}{2}e^{-\\alpha x}, & x \\geq 0
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseExp.*
            :width: 100%

        ?泳�怨�留�?占쎈뼒?占쎈쑏???筌륅옙?嶺뚮쵐�눓雅뚳옙? [#SLAYER]_ [#neftci2019surrogate]_ 佯몌옙??略노쵐鍮섓옙�뮀占쏙옙占�????

        * :ref:`佯몌옙???嶺뚮쵑�쑇PI <PiecewiseExp.__init__-cn>`
        .. _PiecewiseExp.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The piecewise exponential surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{2}e^{-\\alpha |x|}

        The primitive function is defined by

        .. math::
            g(x) =
            \\begin{cases}
            \\frac{1}{2}e^{\\alpha x}, & x < 0 \\\\
            1 - \\frac{1}{2}e^{-\\alpha x}, & x \\geq 0
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseExp.*
            :width: 100%

        The function is used in [#SLAYER]_ [#neftci2019surrogate]_ .
        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return piecewise_exp.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        mask_nonnegative = heaviside(x)
        mask_sign = mask_nonnegative * 2. - 1.
        exp_x = (mask_sign * x * -alpha).exp_() / 2.
        return mask_nonnegative - exp_x * mask_sign

    @staticmethod
    def backward(grad_output, x, alpha):
        return piecewise_exp_backward(grad_output, x, alpha)[0]

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.PiecewiseExp(alpha=2, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=2$')

    # surrogate_function = surrogate.PiecewiseExp(alpha=2, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=2$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('Piecewise exponential surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


@torch.jit.script
def sigmoid_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    sgax = (x * alpha).sigmoid_()
    return grad_output * (1. - sgax) * sgax * alpha, None


class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return sigmoid_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


@torch.jit.script
def relu_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    mask = (x > 0).float()
    return grad_output * mask * alpha, None


class relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)  # spike generation

    @staticmethod
    def backward(ctx, grad_output):
        return relu_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class ReLU(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)
        self.alpha = alpha
        self.spiking = spiking

    @staticmethod
    def spiking_function(x, alpha):
        return relu.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return F.relu(x) * alpha

    @staticmethod
    def backward(grad_output, x, alpha):
        return relu_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {y} = ({x} > 0.0f ? {alpha} : 0.0f);
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
            {tab4_str}const half2 {y} = __hgt2({x}, __float2half2_rn(0.0f)) ? {sg_name}_alpha : __float2half2_rn(0.0f);
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.relu_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)

class CustomSurrogateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, new_grad: torch.Tensor):
        # input: v - v_threshold (shape: [...])
        # new_grad: same shape, externally computed
        ctx.save_for_backward(new_grad)
        return (input > 0).float() 

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (new_grad,) = ctx.saved_tensors
        return grad_output * new_grad, None  # None for new_grad (no grad)

class CustomSurrogate(SurrogateFunctionBase):
    def __init__(self):
        super().__init__()
        self.new_grad = None  # to be externally set

    def set_grad(self, grad: torch.Tensor):
        self.new_grad = grad

    def forward(self, x: torch.Tensor):
        assert self.new_grad is not None, "CustomSurrogate.new_grad must be set before forward"
        return CustomSurrogateFunction.apply(x, self.new_grad)

    def cuda_codes(self):
        raise NotImplementedError("CUDA support is not implemented for CustomSurrogate.")
    
class Sigmoid(SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True):
        '''
        * :ref:`API in English <Sigmoid.__init__-en>`
        .. _Sigmoid.__init__-cn:

        :param alpha: ??占쎌젞??占쎌냲?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌몴�뿆占�?岳묕옙????占쎈１占쎈릅�뙴紐껓옙占�?占쎌맮?筌륅옙?�굢罐由�???占쎈１?占쎈솿??占쎈쑏?
        :param spiking: ??亦낉옙?嶺뚮씭�뼱�몴�뇿�돦占쎌삌占쎈뼇?占쎈き???�땻�벂二곤옙遊억옙�뮖?占쎌맽嚥싲갇源듸옙占쎌뼲彛�? ``True``輿삳뿫遊억옙�뮍?筌륅옙??占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙占�? ``heaviside`` ???占쎈뮍?筌륅옙?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼦占쎄샴?占쎈뮀占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??�뇡占�?�윜諭��삌�젆�굚�렧?? ``False``
            ??占쎈묍佯몃돆鍮뽳옙逾�?占쎈��?占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??輿삳뿫遊억옙�뮍?占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌뱽�댚占�?占쎈뮋?占쎈��?占쏙옙占�??占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩???占쎈１�솾占�??岳묕옙????占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏占쎈뉼占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏?

        ?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙�뼚愿춊gmoid??占쎈１�솾占�??岳묕옙????占쎈１?占쎈き???�땻占�?占쎈쇀??占쎄껀???占쎈뼒?占쎈쑏???�윜諭�猷놅쭩占�?占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼤筌좑옙?

        .. math::
            g'(x) = \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x)

        ?占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏占쎈뉵筌좑옙?

        .. math::
            g(x) = \\mathrm{sigmoid}(\\alpha x) = \\frac{1}{1+e^{-\\alpha x}}

        .. image:: ../_static/API/activation_based/surrogate/Sigmoid.*
            :width: 100%

        ?泳�怨�留�?占쎈뼒?占쎈쑏???筌륅옙?嶺뚮쵐�눓雅뚳옙? [#STBP]_ [#roy2019scaling]_ [#SNNLSTM]_ [#SNU]_ 佯몌옙??略노쵐鍮섓옙�뮀占쏙옙占�????

        * :ref:`佯몌옙???嶺뚮쵑�쑇PI <Sigmoid.__init__-cn>`
        .. _Sigmoid.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The sigmoid surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x)

        The primitive function is defined by

        .. math::
            g(x) = \\mathrm{sigmoid}(\\alpha x) = \\frac{1}{1+e^{-\\alpha x}}

        .. image:: ../_static/API/activation_based/surrogate/Sigmoid.*
            :width: 100%

        The function is used in  [#STBP]_ [#roy2019scaling]_ [#SNNLSTM]_ [#SNU]_ .
        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return sigmoid.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return (x * alpha).sigmoid()

    @staticmethod
    def backward(grad_output, x, alpha):
        return sigmoid_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_sigmoid_ax = 1.0f / (1.0f + expf(- {alpha} * {x}));
            {tab4_str}const float {y} = (1.0f - {sg_name}_sigmoid_ax) * {sg_name}_sigmoid_ax * {alpha};
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2({sg_name}_alpha, {x}))), __float2half2_rn(1.0f)));
            {tab4_str}const half2 {y} = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_sigmoid_ax), {sg_name}_sigmoid_ax), {sg_name}_alpha);
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.sigmoid_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.Sigmoid(alpha=5, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=5$')

    # surrogate_function = surrogate.Sigmoid(alpha=5, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=5$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('Sigmoid surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


@torch.jit.script
def soft_sign_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return grad_output / (2 * alpha * (1 / alpha + x.abs()).pow_(2)), None


class soft_sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return soft_sign_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class SoftSign(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        '''
        * :ref:`API in English <SoftSign.__init__-en>`
        .. _SoftSign.__init__-cn:

        :param alpha: ??占쎌젞??占쎌냲?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌몴�뿆占�?岳묕옙????占쎈１占쎈릅�뙴紐껓옙占�?占쎌맮?筌륅옙?�굢罐由�???占쎈１?占쎈솿??占쎈쑏?
        :param spiking: ??亦낉옙?嶺뚮씭�뼱�몴�뇿�돦占쎌삌占쎈뼇?占쎈き???�땻�벂二곤옙遊억옙�뮖?占쎌맽嚥싲갇源듸옙占쎌뼲彛�? ``True``輿삳뿫遊억옙�뮍?筌륅옙??占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙占�? ``heaviside`` ???占쎈뮍?筌륅옙?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼦占쎄샴?占쎈뮀占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??�뇡占�?�윜諭��삌�젆�굚�렧?? ``False``
            ??占쎈묍佯몃돆鍮뽳옙逾�?占쎈��?占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??輿삳뿫遊억옙�뮍?占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌뱽�댚占�?占쎈뮋?占쎈��?占쏙옙占�??占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩???占쎈１�솾占�??岳묕옙????占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏占쎈뉼占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏?

        ?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙�뼚愿춐ft sign??占쎈１�솾占�??岳묕옙????占쎈１?占쎈き???�땻占�?占쎈쇀??占쎄껀???占쎈뼒?占쎈쑏???�윜諭�猷놅쭩占�?占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼤筌좑옙?

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + |\\alpha x|)^{2}} = \\frac{1}{2\\alpha(\\frac{1}{\\alpha} + |x|)^{2}}

        ?占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏占쎈뉵筌좑옙?

        .. math::
            g(x) = \\frac{1}{2} (\\frac{\\alpha x}{1 + |\\alpha x|} + 1)
            = \\frac{1}{2} (\\frac{x}{\\frac{1}{\\alpha} + |x|} + 1)

        .. image:: ../_static/API/activation_based/surrogate/SoftSign.*
            :width: 100%

        ?泳�怨�留�?占쎈뼒?占쎈쑏???筌륅옙?嶺뚮쵐�눓雅뚳옙? [#SuperSpike]_ [#neftci2019surrogate]_ 佯몌옙??略노쵐鍮섓옙�뮀占쏙옙占�????

        * :ref:`佯몌옙???嶺뚮쵑�쑇PI <SoftSign.__init__-cn>`
        .. _SoftSign.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The soft sign surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + |\\alpha x|)^{2}}

        The primitive function is defined by

        .. math::
            g(x) = \\frac{1}{2} (\\frac{\\alpha x}{1 + |\\alpha x|} + 1)

        .. image:: ../_static/API/activation_based/surrogate/SoftSign.*
            :width: 100%

        The function is used in [#SuperSpike]_ [#neftci2019surrogate]_ .
        '''
        super().__init__(alpha, spiking)
        assert alpha > 0, 'alpha must be lager than 0'

    @staticmethod
    def spiking_function(x, alpha):
        return soft_sign.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return (F.softsign(x * alpha) + 1.) / 2.

    @staticmethod
    def backward(grad_output, x, alpha):
        return soft_sign_backward(grad_output, x, alpha)[0]

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.SoftSign(alpha=3, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=3$')

    # surrogate_function = surrogate.SoftSign(alpha=3, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=3$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('SoftSign surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()

@torch.jit.script
def super_spike_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha * grad_output / torch.pow(torch.abs(x) + 1., 2), None

class super_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return super_spike_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class SuperSpike(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        '''
        * :ref:`API in English <SuperSpike.__init__-en>`
        .. _SuperSpike.__init__-cn:
    
        `SuperSpike: Supervised learning in multi-layer spiking neural networks <https://arxiv.org/abs/1705.11146>`_ ?占쎈쇀???占쎈뼇??占쎈１?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙�뼚肄뇎perSpike??占쎈１�솾占�??岳묕옙????占쎈１?占쎈き???�땻占�?占쎈쇀??占쎄껀???占쎈뼒?占쎈쑏???�윜諭�猷놅쭩占�?占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼤筌좑옙?

        .. math::
            g'(x) = \\frac{\\alpha}{(1 + (|x|))^2}


        * :ref:`佯몌옙???嶺뚮쵑�쑇PI <SuperSpike.__init__-cn>`
        .. _SuperSpike.__init__-en:

        The SuperSpike surrogate spiking function proposed by `SuperSpike: Supervised learning in multi-layer spiking neural networks <https://arxiv.org/abs/1705.11146>`_. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{(1 + (|x|))^2}
        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return atan.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        raise NotImplementedError

    @staticmethod
    def backward(grad_output, x, alpha):
        return super_spike_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        raise NotImplementedError
    def cuda_codes(self, y: str, x: str, dtype: str):
        raise NotImplementedError


@torch.jit.script
def atan_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow_(2)) * grad_output, None


class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return atan_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class ATan(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        '''
        * :ref:`API in English <ATan.__init__-en>`
        .. _ATan.__init__-cn:

        ?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙占�??占쎈쇀�몴�굞�궙壤깍옙占쎄턁占쎈괘??占쎈뼒?占쎈쑏占쎈쳝rc tangent??占쎈１�솾占�??岳묕옙????占쎈１?占쎈き???�땻占�?占쎈쇀??占쎄껀???占쎈뼒?占쎈쑏???�윜諭�猷놅쭩占�?占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼤筌좑옙?

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + (\\frac{\\pi}{2}\\alpha x)^2)}

        ?占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏占쎈뉵筌좑옙?

        .. math::
            g(x) = \\frac{1}{\\pi} \\arctan(\\frac{\\pi}{2}\\alpha x) + \\frac{1}{2}

        .. image:: ../_static/API/activation_based/surrogate/ATan.*
            :width: 100%

        * :ref:`佯몌옙???嶺뚮쵑�쑇PI <ATan.__init__-cn>`
        .. _ATan.__init__-en:

        The arc tangent surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + (\\frac{\\pi}{2}\\alpha x)^2)}

        The primitive function is defined by

        .. math::
            g(x) = \\frac{1}{\\pi} \\arctan(\\frac{\\pi}{2}\\alpha x) + \\frac{1}{2}

        .. image:: ../_static/API/activation_based/surrogate/ATan.*
            :width: 100%
        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return atan.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return (math.pi / 2 * alpha * x).atan_() / math.pi + 0.5

    @staticmethod
    def backward(grad_output, x, alpha):
        return atan_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''
        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * {alpha} * {x};
            {tab4_str}const float {y} = {alpha} / 2.0f / (1.0f + {sg_name}_M_PI_2__alpha__x * {sg_name}_M_PI_2__alpha__x);
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_alpha =  __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), {sg_name}_alpha), {x});
            {tab4_str}const half2 {y} = __h2div(__h2div({sg_name}_alpha, __float2half2_rn(2.0f)), __hfma2({sg_name}_M_PI_2__alpha__x, {sg_name}_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.atan_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.ATan(alpha=3, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=3$')

    # surrogate_function = surrogate.ATan(alpha=3, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=3$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('ATan surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


@torch.jit.script
def nonzero_sign_log_abs_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return grad_output / (1 / alpha + x.abs()), None


class nonzero_sign_log_abs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return nonzero_sign_log_abs_backward((grad_output, ctx.saved_tensors[0], ctx.alpha))


class NonzeroSignLogAbs(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        '''
        * :ref:`API in English <LogAbs.__init__-en>`
        .. _LogAbs.__init__-cn:

        :param alpha: ??占쎌젞??占쎌냲?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌몴�뿆占�?岳묕옙????占쎈１占쎈릅�뙴紐껓옙占�?占쎌맮?筌륅옙?�굢罐由�???占쎈１?占쎈솿??占쎈쑏?
        :param spiking: ??亦낉옙?嶺뚮씭�뼱�몴�뇿�돦占쎌삌占쎈뼇?占쎈き???�땻�벂二곤옙遊억옙�뮖?占쎌맽嚥싲갇源듸옙占쎌뼲彛�? ``True``輿삳뿫遊억옙�뮍?筌륅옙??占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙占�? ``heaviside`` ???占쎈뮍?筌륅옙?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼦占쎄샴?占쎈뮀占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??�뇡占�?�윜諭��삌�젆�굚�렧?? ``False``
            ??占쎈묍佯몃돆鍮뽳옙逾�?占쎈��?占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??輿삳뿫遊억옙�뮍?占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌뱽�댚占�?占쎈뮋?占쎈��?占쏙옙占�??占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩???占쎈１�솾占�??岳묕옙????占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏占쎈뉼占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏?

        .. warning::
            ??占쎌쓨??占쎈뼒?占쎈쑏???占쎈１�솾占�?�솻洹μ삌占쎈뼇??占쎄퉿??占쎌쑌占쎈릅�뙴紐꾨く?占쎈빝??亦낉옙(0, 1)??�윜諭�猷뉛옙�땯??占쎈１歷�源낅㎥�몭占�?獄�占�??亦낉옙?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠????占쎈１嚥싲갇源댐옙占쏙옙嶺뚳옙???占쎈��??筌뚳옙??占쎈섈�븨諛댄돪占쎈��??

        ?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙�뼚媛컊nzeroSignLogAbs??占쎈１�솾占�??岳묕옙????占쎈１?占쎈き???�땻占�?占쎈쇀??占쎄껀???占쎈뼒?占쎈쑏???�윜諭�猷놅쭩占�?占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼤筌좑옙?

        .. math::
            g'(x) = \\frac{\\alpha}{1 + |\\alpha x|} = \\frac{1}{\\frac{1}{\\alpha} + |x|}

        ?占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏占쎈뉵筌좑옙?

        .. math::
            g(x) = \\mathrm{NonzeroSign}(x) \\log (|\\alpha x| + 1)

        ????佯몌옙??

            .. math::
                \\mathrm{NonzeroSign}(x) =
                \\begin{cases}
                1, & x \\geq 0 \\\\
                -1, & x < 0 \\\\
                \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/NonzeroSignLogAbs.*
            :width: 100%

        * :ref:`佯몌옙???嶺뚮쵑�쑇PI <LogAbs.__init__-cn>`
        .. _LogAbs.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        .. admonition:: Warning
            :class: warning

            The output range the primitive function is not (0, 1). The advantage of this function is that computation
            cost is small when backward.

        The NonzeroSignLogAbs surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{1 + |\\alpha x|} = \\frac{1}{\\frac{1}{\\alpha} + |x|}

        The primitive function is defined by

        .. math::
            g(x) = \\mathrm{NonzeroSign}(x) \\log (|\\alpha x| + 1)

        where

        .. math::
            \\mathrm{NonzeroSign}(x) =
            \\begin{cases}
            1, & x \\geq 0 \\\\
            -1, & x < 0 \\\\
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/NonzeroSignLogAbs.*
            :width: 100%

        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return nonzero_sign_log_abs.apply(x, alpha)

    @staticmethod
    def backward(grad_output, x, alpha):
        return nonzero_sign_log_abs_backward(grad_output, x, alpha)[0]

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        # the gradient of ``(heaviside(x) * 2 - 1) * (alpha * x.abs() + 1).log()`` by autograd is wrong at ``x==0``
        mask_p = heaviside(x) * 2. - 1.
        return mask_p * (alpha * mask_p * x + 1).log()

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.NonzeroSignLogAbs(alpha=1, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=1$')

    # surrogate_function = surrogate.NonzeroSignLogAbs(alpha=1, spiking=False)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=1$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('NonzeroSignLogAbs surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


@torch.jit.script
def erf_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return grad_output * (- (x * alpha).pow_(2)).exp_() * (alpha / math.sqrt(math.pi)), None


class erf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return erf_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class Erf(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        '''
        * :ref:`API in English <Erf.__init__-en>`
        .. _Erf.__init__-cn:

        :param alpha: ??占쎌젞??占쎌냲?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌몴�뿆占�?岳묕옙????占쎈１占쎈릅�뙴紐껓옙占�?占쎌맮?筌륅옙?�굢罐由�???占쎈１?占쎈솿??占쎈쑏?
        :param spiking: ??亦낉옙?嶺뚮씭�뼱�몴�뇿�돦占쎌삌占쎈뼇?占쎈き???�땻�벂二곤옙遊억옙�뮖?占쎌맽嚥싲갇源듸옙占쎌뼲彛�? ``True``輿삳뿫遊억옙�뮍?筌륅옙??占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙占�? ``heaviside`` ???占쎈뮍?筌륅옙?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼦占쎄샴?占쎈뮀占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??�뇡占�?�윜諭��삌�젆�굚�렧?? ``False``
            ??占쎈묍佯몃돆鍮뽳옙逾�?占쎈��?占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??輿삳뿫遊억옙�뮍?占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌뱽�댚占�?占쎈뮋?占쎈��?占쏙옙占�??占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩???占쎈１�솾占�??岳묕옙????占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏占쎈뉼占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏?

        ?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙�뻼援℡쳞占�?�몭占쏙옙�젂占쎈굝爾�??�굢占�???占쎈뼒?占쎈쑏?(erf)??占쎈１�솾占�??岳묕옙????占쎈１?占쎈き???�땻占�?占쎈쇀??占쎄껀???占쎈뼒?占쎈쑏???�윜諭�猷놅쭩占�?占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼤筌좑옙?

        .. math::
            g'(x) = \\frac{\\alpha}{\\sqrt{\\pi}}e^{-\\alpha^2x^2}

        ?占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏占쎈뉵筌좑옙?

        .. math::
            :nowrap:

            \\begin{split}
            g(x) &= \\frac{1}{2}(1-\\text{erf}(-\\alpha x)) \\\\
            &= \\frac{1}{2} \\text{erfc}(-\\alpha x) \\\\
            &= \\frac{1}{\\sqrt{\\pi}}\\int_{-\\infty}^{\\alpha x}e^{-t^2}dt
            \\end{split}

        .. image:: ../_static/API/activation_based/surrogate/Erf.*
            :width: 100%

        ?泳�怨�留�?占쎈뼒?占쎈쑏???筌륅옙?嶺뚮쵐�눓雅뚳옙? [#esser2015backpropagation]_ [#STBP]_ [#SRNN]_ 佯몌옙??略노쵐鍮섓옙�뮀占쏙옙占�????

        * :ref:`佯몌옙???嶺뚮쵑�쑇PI <Erf.__init__-cn>`
        .. _Erf.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The Gaussian error (erf) surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{\\sqrt{\\pi}}e^{-\\alpha^2x^2}

        The primitive function is defined by

        .. math::
            :nowrap:

            \\begin{split}
            g(x) &= \\frac{1}{2}(1-\\text{erf}(-\\alpha x)) \\\\
            &= \\frac{1}{2} \\text{erfc}(-\\alpha x) \\\\
            &= \\frac{1}{\\sqrt{\\pi}}\\int_{-\\infty}^{\\alpha x}e^{-t^2}dt
            \\end{split}

        .. image:: ../_static/API/activation_based/surrogate/Erf.*
            :width: 100%

        The function is used in [#esser2015backpropagation]_ [#STBP]_ [#SRNN]_.
        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return erf.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return torch.erfc_(-alpha * x) / 2.

    @staticmethod
    def backward(grad_output, x, alpha):
        return erf_backward(grad_output, x, alpha)[0]

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.Erf(alpha=2, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=2$')

    # surrogate_function = surrogate.Erf(alpha=2, spiking=False)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=2$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('Gaussian error surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


@torch.jit.script
def piecewise_leaky_relu_backward(grad_output: torch.Tensor, x: torch.Tensor, w: float, c: float):
    mask_width = (x.abs() < w)
    mask_c = mask_width.logical_not()
    return grad_output * x.masked_fill(mask_width, 1 / (2*w)).masked_fill(mask_c, c), None, None


class piecewise_leaky_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w=1, c=0.01):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.w = w
            ctx.c = c
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_leaky_relu_backward(grad_output, ctx.saved_tensors[0], ctx.w, ctx.c)


class PiecewiseLeakyReLU(MultiArgsSurrogateFunctionBase):
    def __init__(self, w=1., c=0.01, spiking=True):
        '''
        * :ref:`API in English <PiecewiseLeakyReLU.__init__-en>`
        .. _PiecewiseLeakyReLU.__init__-cn:

        :param w: ``-w <= x <= w`` ?占쎄덩??占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠????占쎈１�솾占�??岳묕옙??�뇡�댙彛�? ``1 / 2w``
        :param c: ``x > w`` ??占쎈튉 ``x < -w`` ?占쎄덩??占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠????占쎈１�솾占�??岳묕옙??�뇡�댙彛�? ``c``
        :param spiking: ??亦낉옙?嶺뚮씭�뼱�몴�뇿�돦占쎌삌占쎈뼇?占쎈き???�땻�벂二곤옙遊억옙�뮖?占쎌맽嚥싲갇源듸옙占쎌뼲彛�? ``True``輿삳뿫遊억옙�뮍?筌륅옙??占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙占�? ``heaviside`` ???占쎈뮍?筌륅옙?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼦占쎄샴?占쎈뮀占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??�뇡占�?�윜諭��삌�젆�굚�렧?? ``False``
            ??占쎈묍佯몃돆鍮뽳옙逾�?占쎈��?占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??輿삳뿫遊억옙�뮍?占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌뱽�댚占�?占쎈뮋?占쎈��?占쏙옙占�??占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩???占쎈１�솾占�??岳묕옙????占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏占쎈뉼占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏?

        ??占쎈광嚥싳쉸瑗띈キ釉낃괌????占쎌젡?占쎈１?甕곤옙占쎈닱占쎈룇�땻占�?占쎈き???�땻占�?占쎈쇀??占쎄껀???占쎈뼒?占쎈쑏???�윜諭�逾댐옙�뒄占쎈뤀?�뇡�댙彛�?

        .. math::
            g'(x) =
            \\begin{cases}
            \\frac{1}{2w}, & -w \\leq x \\leq w \\\\
            c, & x < -w ~or~ x > w
            \\end{cases}

        ?占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏占쎈뉵筌좑옙?

        .. math::
            g(x) =
            \\begin{cases}
            cx + cw, & x < -w \\\\
            \\frac{1}{2w}x + \\frac{1}{2}, & -w \\leq x \\leq w \\\\
            cx - cw + 1, & x > w \\\\
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseLeakyReLU.*
            :width: 100%

        ?泳�怨�留�?占쎈뼒?占쎈쑏???筌륅옙?嶺뚮쵐�눓雅뚳옙? [#yin2017algorithm]_ [#STBP]_ [#huh2018gradient]_ [#wu2019direct]_ [#STCA]_ [#roy2019scaling]_ [#LISNN]_ [#DECOLLE]_ 佯몌옙??略노쵐鍮섓옙�뮀占쏙옙占�????

        * :ref:`佯몌옙???嶺뚮쵑�쑇PI <PiecewiseLeakyReLU.__init__-cn>`
        .. _PiecewiseLeakyReLU.__init__-en:

        :param w: when ``-w <= x <= w`` the gradient is ``1 / 2w``
        :param c: when ``x > w`` or ``x < -w`` the gradient is ``c``
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The piecewise surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) =
            \\begin{cases}
            \\frac{1}{2w}, & -w \\leq x \\leq w \\\\
            c, & x < -w ~or~ x > w
            \\end{cases}

        The primitive function is defined by

        .. math::
            g(x) =
            \\begin{cases}
            cx + cw, & x < -w \\\\
            \\frac{1}{2w}x + \\frac{1}{2}, & -w \\leq x \\leq w \\\\
            cx - cw + 1, & x > w
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseLeakyReLU.*
            :width: 100%

        The function is used in [#yin2017algorithm]_ [#STBP]_ [#huh2018gradient]_ [#wu2019direct]_ [#STCA]_ [#roy2019scaling]_ [#LISNN]_ [#DECOLLE]_.
        '''
        super().__init__(spiking)
        assert w > 0.
        self.w = w
        self.c = c
        self.spiking = spiking

    def forward(self, x):
        if self.spiking:
            f = self.spiking_function
        else:
            f = self.primitive_function

        return f(x, self.w, self.c)

    @staticmethod
    def spiking_function(x: torch.Tensor, w, c):
        return piecewise_leaky_relu.apply(x, w, c)

    @staticmethod
    def backward(grad_output, x, w, c):
        return piecewise_leaky_relu_backward(grad_output, x, w, c)[0]

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, w: float, c: float):
        mask0 = (x < -w).to(x)
        mask1 = (x > w).to(x)
        mask2 = torch.ones_like(x.data) - mask0 - mask1
        if c == 0:
            return mask2 * (x / (2 * w) + 1 / 2) + mask1
        else:
            cw = c * w
            return mask0 * (c * x + cw) + mask1 * (c * x + (- cw + 1)) \
                   + mask2 * (x / (2 * w) + 1 / 2)

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        w = str(self.w) + 'f'
        w_inv = str(1. / self.w) + 'f'
        c = str(self.c) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_x_abs = fabsf({x});
            float {y};
            if ({sg_name}_x_abs > {w})
            {curly_bracket_l}
                {y} = {c};
            {curly_bracket_r}
            else
            {curly_bracket_l}
                {y} = {w_inv};
            {curly_bracket_r}
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_x_abs = __habs2({x});
            {tab4_str}const half2 {sg_name}_x_abs_ge_w = __hge2({sg_name}_x_abs, __float2half2_rn({w}));
            {tab4_str}half2 {y} = __hadd2(__hmul2(__float2half2_rn({c}),  {sg_name}_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_x_abs_ge_w), __float2half2_rn({w_inv})));
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.piecewise_leaky_relu_backward(y=y, x=x, w=self.w, c=self.c, dtype=dtype)

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.PiecewiseLeakyReLU(w=1, c=0.1, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $w=1, c=0.1$')

    # surrogate_function = surrogate.PiecewiseLeakyReLU(w=1, c=0.1, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $w=1, c=0.1$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('PiecewiseLeakyReLU surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


class squarewave_fourier_series(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, n: int, T_period: float):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.n = n
            ctx.T_period = T_period
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = 0.
        x = ctx.saved_tensors[0]
        w = math.pi * 2. / ctx.T_period
        for i in range(1, ctx.n):
            grad_x += torch.cos_((2 * i - 1.) * w * x)

        grad_x *= 4. / ctx.T_period
        grad_x *= grad_output

        return grad_x, None, None


class SquarewaveFourierSeries(MultiArgsSurrogateFunctionBase):
    def __init__(self, n: int = 2, T_period: float = 8, spiking=True):
        super().__init__(spiking)
        assert isinstance(n, int) and T_period > 0.
        self.n = n
        self.T_period = T_period
        self.spiking = spiking

    def forward(self, x):
        if self.spiking:
            f = self.spiking_function
        else:
            f = self.primitive_function
        return f(x, self.n, self.T_period)

    @staticmethod
    def spiking_function(x: torch.Tensor, w, c):
        return squarewave_fourier_series.apply(x, w, c)

    @staticmethod
    def primitive_function(x: torch.Tensor, n: int, T_period: float):
        w = math.pi * 2. / T_period
        ret = torch.zeros_like(x.data)
        for i in range(1, n):
            c = (2 * i - 1.)
            ret += torch.sin(c * w * x) / c

        return 0.5 + 2. / math.pi * ret

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        w = str(self.w) + 'f'
        w_inv = str(1. / self.w) + 'f'
        c = str(self.c) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            raise NotImplementedError
        elif dtype == 'fp16':
            raise NotImplementedError
        else:
            raise NotImplementedError

        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    # import torch
    # from spikingjelly.activation_based import surrogate
    # from matplotlib import pyplot as plt
    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200, figsize=(6, 4))
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    #
    # c_list = []
    # for n in [2, 4, 8]:
    #     surrogate_function = surrogate.SquarewaveFourierSeries(n=n, T_period=8, spiking=False)
    #     y = surrogate_function(x)
    #     plt.plot(x.data, y.data, label=f'Primitive, $n={n}$')
    #     c_list.append(plt.gca().lines[-1].get_color())
    #
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title(f'SquarewaveFourierSeries surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # # plt.grid(linestyle='--')
    # plt.savefig('./docs/source/_static/API/activation_based/surrogate/SquarewaveFourierSeries1.pdf')
    # plt.savefig('./docs/source/_static/API/activation_based/surrogate/SquarewaveFourierSeries1.svg')
    # plt.clf()
    # for i, n in enumerate([2, 4, 8]):
    #     surrogate_function = surrogate.SquarewaveFourierSeries(n=n, T_period=8, spiking=True)
    #     x = x.detach()
    #     x.requires_grad_(True)
    #     y = surrogate_function(x)
    #     z = y.sum()
    #     z.backward()
    #     plt.plot(x.data, x.grad, label=f'Gradient, $n={n}$', c=c_list[i])
    #     x.grad.zero_()
    #
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title(f'SquarewaveFourierSeries surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # # plt.grid(linestyle='--')
    # plt.savefig('./docs/source/_static/API/activation_based/surrogate/SquarewaveFourierSeries2.pdf')
    # plt.savefig('./docs/source/_static/API/activation_based/surrogate/SquarewaveFourierSeries2.svg')


class s2nn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float, beta: float):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
            ctx.beta = beta
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sgax = torch.sigmoid(ctx.alpha * x)
        grad_x = torch.where(x < 0., ctx.alpha * sgax * (1. - sgax), ctx.beta / (x + 1.))
        return grad_x * grad_output, None, None


class S2NN(MultiArgsSurrogateFunctionBase):
    def __init__(self, alpha=4., beta=1., spiking=True):
        """
        * :ref:`API in English <S2NN.__init__-en>`
        .. _S2NN.__init__-cn:

        :param alpha: ??占쎌젞??占쎌냲 ``x < 0`` ?占쎄덩占쎌몴�뿆占�?岳묕옙????占쎈１?占쎈솿??占쎈쑏?
        :param beta: ??占쎌젞??占쎌냲 ``x >= 0`` ?占쎄덩占쎌몴�뿆占�?岳묕옙????占쎈１?占쎈솿??占쎈쑏?
        :param spiking: ??亦낉옙?嶺뚮씭�뼱�몴�뇿�돦占쎌삌占쎈뼇?占쎈き???�땻�벂二곤옙遊억옙�뮖?占쎌맽嚥싲갇源듸옙占쎌뼲彛�? ``True``輿삳뿫遊억옙�뮍?筌륅옙??占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙占�? ``heaviside`` ???占쎈뮍?筌륅옙?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼦占쎄샴?占쎈뮀占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??�뇡占�?�윜諭��삌�젆�굚�렧?? ``False``
            ??占쎈묍佯몃돆鍮뽳옙逾�?占쎈��?占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??輿삳뿫遊억옙�뮍?占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌뱽�댚占�?占쎈뮋?占쎈��?占쏙옙占�??占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩???占쎈１�솾占�??岳묕옙????占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏占쎈뉼占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏?

        `S2NN: Time Step Reduction of Spiking Surrogate Gradients for Training Energy Efficient Single-Step Neural Networks <https://arxiv.org/abs/2201.10879>`_ ?占쎈쇀???占쎈뼇??占쎈１S2NN??占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏???�윜諭�猷놅쭩占�?占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼤筌좑옙?

        .. math::
            g'(x) = \\begin{cases}
                \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x), x < 0 \\\\
                \\\\frac{beta}{(x + 1)}, x \\ge 0
            \\end{cases}

        ?占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏占쎈뉵筌좑옙?

        .. math::
            g(x) = \\begin{cases}
                \\mathrm{sigmoid} (\\alpha x), x < 0 \\\\
                \\beta \\mathrm{ln}(x + 1) + 1, x \\ge 0
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/S2NN.*
            :width: 100%


        * :ref:`佯몌옙???嶺뚮쵑�쑇PI <S2NN.__init__-cn>`
        .. _S2NN.__init__-en:

        :param alpha: the param that controls the gradient when ``x < 0``
        :param beta: the param that controls the gradient when ``x >= 0``
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The S2NN surrogate spiking function, which is proposed by `S2NN: Time Step Reduction of Spiking Surrogate Gradients for Training Energy Efficient Single-Step Neural Networks <https://arxiv.org/abs/2201.10879>`_. The gradient is defined by

        .. math::
            g'(x) = \\begin{cases}
                \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x), x < 0 \\\\
                \\beta (x + 1), x \\ge 0
            \\end{cases}

        The primitive function is defined by

        .. math::
            g(x) = \\begin{cases}
                \\mathrm{sigmoid} (\\alpha x), x < 0 \\\\
                \\beta \\mathrm{ln}(x + 1) + 1, x \\ge 0
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/S2NN.*
            :width: 100%
        """
        super().__init__(spiking)
        self.alpha = alpha
        self.beta = beta
        self.spiking = spiking

    def forward(self, x):
        if self.spiking:
            f = self.spiking_function
        else:
            f = self.primitive_function

        return f(x, self.alpha, self.beta)

    @staticmethod
    def spiking_function(x: torch.Tensor, alpha, beta):
        return s2nn.apply(x, alpha, beta)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float, beta: float):
        return torch.where(x < 0., torch.sigmoid(x * alpha), beta * torch.log((x + 1.).abs_() + 1e-5) + 0.5)
        # abs and 1e-5 are used to avoid nan

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        beta = str(self.beta) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_sigmoid_ax = 1.0f / (1.0f + expf(- {alpha} * {x}));
            {tab4_str}const float {sg_name}_mask_l = (float)({x} < 0.0f);
            {tab4_str}const float {y} = (1.0f - {sg_name}_sigmoid_ax) * {sg_name}_sigmoid_ax * {alpha} * {sg_name}_mask_l + {beta} / ({x} + 1.0f) * (1.0f - {sg_name}_mask_l);
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2({sg_name}_alpha, {x}))), __float2half2_rn(1.0f)));
            {tab4_str}const half2 {sg_name}_mask_l = __hlt2({x}, __float2half2_rn(0.0f));
            {tab4_str}const half2 {y} = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_sigmoid_ax), {sg_name}_sigmoid_ax), {sg_name}_alpha), {sg_name}_mask_l), __hmul2(__h2div(__float2half2_rn({beta}), __hadd2({x}, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), {sg_name}_mask_l)));
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.s2nn_backward(y=y, x=x, alpha=self.alpha, beta=self.beta, dtype=dtype)

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200, figsize=(6, 4))
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.S2NN(alpha=4., beta=1., spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=4, \\beta=1$')
    #
    # surrogate_function = surrogate.S2NN(alpha=4, beta=1., spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=4, \\beta=1$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('S2NN surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # # plt.show()
    # plt.savefig('./S2NN.svg')
    # plt.savefig('./S2NN.pdf')


class q_pseudo_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        x = ctx.saved_tensors[0]
        if ctx.needs_input_grad[0]:
            grad_x = ((1 + 2 / (ctx.alpha - 1) * x.abs()).pow_(-ctx.alpha)) * grad_output
        return grad_x, None


class QPseudoSpike(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        '''
        * :ref:`API in English <QPseudoSpike.__init__-en>`
        .. _QPseudoSpike.__init__-cn:

        :param alpha: ??占쎌젞??占쎌냲?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌몴�뿆占�?岳묕옙????占쎈뼒?占쎈쑏占쎈꺽占쎄퐨?�젆琉명떐???占쎌뱾岳묕옙????占쎈１?占쎈솿??占쎈쑏?
        :param spiking: ??亦낉옙?嶺뚮씭�뼱�몴�뇿�돦占쎌삌占쎈뼇?占쎈き???�땻�벂二곤옙遊억옙�뮖?占쎌맽嚥싲갇源듸옙占쎌뼲彛�? ``True``輿삳뿫遊억옙�뮍?筌륅옙??占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙占�? ``heaviside`` ???占쎈뮍?筌륅옙?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼦占쎄샴?占쎈뮀占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??�뇡占�?�윜諭��삌�젆�굚�렧?? ``False``
            ??占쎈묍佯몃돆鍮뽳옙逾�?占쎈��?占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??輿삳뿫遊억옙�뮍?占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌뱽�댚占�?占쎈뮋?占쎈��?占쏙옙占�??占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩???占쎈１�솾占�??岳묕옙????占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏占쎈뉼占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏?

        `Surrogate Gradients Design <https://arxiv.org/abs/2202.00282>`_ ?占쎈쇀???占쎈뼇??占쎈１ :math:`q`-PseudoSpike??占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏???�윜諭�猷놅쭩占�?占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼤筌좑옙?

        .. math::
            g'(x) = (1+\\frac{2|x|}{\\alpha-1})^{-\\alpha}

        ????佯몌옙?? :math:`\\alpha>1` ?占쎈늾�뛾�렯猷딉옙堉�??占쎌쓨?嶺뚮쵐�눨筌좑옙???占쎈１ :math:`q`???

        ?占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏占쎈뉵筌좑옙?

        .. math::
            g(x) =
            \\begin{cases}
            \\frac{1}{2}(1-\\frac{2x}{\\alpha-1})^{1-\\alpha}, & x < 0 \\\\
            1 - \\frac{1}{2}(1+\\frac{2x}{\\alpha-1})^{1-\\alpha}, & x \\geq 0.
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/QPseudoSpike.*
            :width: 100%

        * :ref:`佯몌옙???嶺뚮쵑�쑇PI <QPseudoSpike.__init__-cn>`
        .. _QPseudoSpike.__init__-en:

        :param alpha: parameter to control tail fatness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The :math:`q`-PseudoSpike surrogate spiking function, which is first proposed in `Surrogate Gradients Design <https://arxiv.org/abs/2202.00282>`_. The gradient is defined by

        .. math::
            g'(x) = (1+\\frac{2|x|}{\\alpha-1})^{-\\alpha}

        where :math:`\\alpha>1` corresponds to :math:`q` in paper.

        The primitive function is defined by

        .. math::
            g(x) =
            \\begin{cases}
            \\frac{1}{2}(1-\\frac{2x}{\\alpha-1})^{1-\\alpha}, & x < 0 \\\\
            1 - \\frac{1}{2}(1+\\frac{2x}{\\alpha-1})^{1-\\alpha}, & x \\geq 0.
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/QPseudoSpike.*
            :width: 100%
        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return q_pseudo_spike.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float):
        mask_nonnegative = heaviside(x)
        mask_sign = mask_nonnegative * 2. - 1.

        return mask_nonnegative - mask_sign * (0.5 * ((1. + 2. / (alpha - 1.) * x * mask_sign).pow_(1. - alpha)))

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_base = 1.0f + 2.0f / ({alpha} - 1.0f) * fabsf({x});
            {tab4_str}const float {y} = powf({sg_name}_base, -{alpha});
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2({x})), __hsub2({sg_name}_alpha, __float2half2_rn(1.0f))));
            {tab4_str}const half2 {y} = h2exp2(__hmul2(h2log2({sg_name}_base), __hneg2({sg_name}_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.q_pseudo_spike_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200, figsize=(6, 4))
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.QPseudoSpike(alpha=2, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=2$')

    # surrogate_function = surrogate.QPseudoSpike(alpha=2, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=2$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('QPseudoSpike surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # # plt.savefig('QPseudoSpike.svg')
    # # plt.savefig('QPseudoSpike.pdf')


@torch.jit.script
def leaky_k_relu_backward(grad_output: torch.Tensor, x: torch.Tensor, leak: float, k: float):
    mask1 = (x >= 0.).to(x)
    grad_x = mask1 * k + (1. - mask1) * leak
    return grad_output * grad_x, None, None


class leaky_k_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, leak, k):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.leak = leak
            ctx.k = k
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return leaky_k_relu_backward(grad_output, ctx.saved_tensors[0], ctx.leak, ctx.k)


class LeakyKReLU(MultiArgsSurrogateFunctionBase):
    def __init__(self, spiking=True, leak: float = 0., k: float = 1.):
        """
        * :ref:`API in English <LeakyKReLU.__init__-en>`
        .. _LeakyKReLU.__init__-cn:

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool
        :param leak: gradient when ``x < 0``
        :type leak: float
        :param k: gradient when ``x >= 0 ``
        :type k: float

        ?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙�뼚媛쨎akyKReLU??占쎈１�솾占�??岳묕옙????占쎈１?占쎈き???�땻占�?占쎈쇀??占쎄껀???占쎈뼒?占쎈쑏???�윜諭�猷놅쭩占�?占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼤筌좑옙?

        .. math::
            g'(x) =
            \\begin{cases}
            k, & x \\geq 0 \\\\
            leak, & x < 0 \\\\
            \\end{cases}

        ?占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏占쎈뉵筌좑옙?

        .. math::
            g(x) =
            \\begin{cases}
            k \\cdot x, & x \\geq 0 \\\\
            leak \\cdot x, & x < 0 \\\\
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/LeakyKReLU.*
            :width: 100%

        * :ref:`佯몌옙???嶺뚮쵑�쑇PI <LeakyKReLU.__init__-cn>`
        .. _LeakyKReLU.__init__-en:

        :param spiking: ??亦낉옙?嶺뚮씭�뼱�몴�뇿�돦占쎌삌占쎈뼇?占쎈き???�땻�벂二곤옙遊억옙�뮖?占쎌맽嚥싲갇源듸옙占쎌뼲彛�? ``True``輿삳뿫遊억옙�뮍?筌륅옙??占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙占�? ``heaviside`` ???占쎈뮍?筌륅옙?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼦占쎄샴?占쎈뮀占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??�뇡占�?�윜諭��삌�젆�굚�렧?? ``False``
            ??占쎈묍佯몃돆鍮뽳옙逾�?占쎈��?占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??輿삳뿫遊억옙�뮍?占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌뱽�댚占�?占쎈뮋?占쎈��?占쏙옙占�??占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩???占쎈１�솾占�??岳묕옙????占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏占쎈뉼占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏?
        :type spiking: bool
        :param leak: ``x < 0`` ?占쎄덩???占쎈１�솾占�??岳묕옙?????
        :type leak: float
        :param k: ``x >= 0 `` ?占쎄덩???占쎈１�솾占�??岳묕옙?????
        :type k: float

        The LeakyKReLU surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) =
            \\begin{cases}
            k, & x \\geq 0 \\\\
            leak, & x < 0 \\\\
            \\end{cases}

        The primitive function is defined by

        .. math::
            g(x) =
            \\begin{cases}
            k \\cdot x, & x \\geq 0 \\\\
            leak \\cdot x, & x < 0 \\\\
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/LeakyKReLU.*
            :width: 100%

        """
        super().__init__(spiking, leak, k)
        self.leak = leak
        self.k = k

    @staticmethod
    def spiking_function(x, leak, k):
        return leaky_k_relu.apply(x, leak, k)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, leak: float, k: float):
        mask1 = (x >= 0.).to(x)
        return (leak * (1. - mask1) + k * mask1) * x

    @staticmethod
    def backward(grad_output, x, leak, k):
        return leaky_k_relu_backward(grad_output, x, leak, k)[0]

    def forward(self, x):
        if self.spiking:
            f = self.spiking_function
        else:
            f = self.primitive_function

        return f(x, self.leak, self.k)

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        leak = str(self.leak) + 'f'
        k = str(self.k) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_mask1 = (float) ({x} >= 0.0f);
            {tab4_str}const float {y} = {leak} * (1.0f - {sg_name}_mask1) + {k} * {sg_name}_mask1;
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_mask1 = __hgeu2({x}, __float2half2_rn(0.0f));
            {tab4_str}const half2 {y} = __hfma2(__float2half2_rn({k}), {sg_name}_mask1, __hmul2(__float2half2_rn({leak}), __hsub2(__float2half2_rn(1.0f), {sg_name}_mask1)));
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.leaky_k_relu_backward(y=y, x=x, leak=self.leak, k=self.k, dtype=dtype)

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200, figsize=(6, 4))
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.LeakyKReLU(spiking=False, leak=0.1, k=0.5)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, leak=0.1, k=1')
    #
    # surrogate_function = surrogate.LeakyKReLU(spiking=True, leak=0.1, k=0.5)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, leak=0.1, k=1')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('LeakyKReLU surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.savefig('LeakyKReLU.svg')
    # plt.savefig('LeakyKReLU.pdf')


@torch.jit.script
def fake_numerical_gradient_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    grad_x = torch.clamp_max(((x >= 0.) * 2. - 1.) / x, alpha)
    return grad_output * grad_x, None


class fake_numerical_gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return fake_numerical_gradient_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class FakeNumericalGradient(SurrogateFunctionBase):
    def __init__(self, alpha=0.3):
        super().__init__(alpha, spiking=True)

    @staticmethod
    def spiking_function(x, alpha):
        return fake_numerical_gradient.apply(x, alpha)

    @staticmethod
    def backward(grad_output, x, alpha):
        return fake_numerical_gradient_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_sign = (float) ({x} >= 0.0f) * 2.0f - 1.0f;
            {tab4_str}const float {y} = min({sg_name}_sign / {x}, {alpha});
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_sign = __hfma2(__hgeu2({x}, __float2half2_rn(0.0f)), __float2half2_rn(2.0f), __float2half2_rn(-1.0f));
            #if (__CUDA_ARCH__ < 800)
            {tab4_str}const half2 {sg_name}_grad_x = __h2div({sg_name}_sign, {x});
            {tab4_str}const half2 {sg_name}_grad_max = __float2half2_rn({alpha});
            {tab4_str}const half2 {y} = make_half2({sg_name}_grad_x.x <= {sg_name}_grad_max.x ? {sg_name}_grad_x.x : {sg_name}_grad_max.x, {sg_name}_grad_x.y <= {sg_name}_grad_max.y ? {sg_name}_grad_x.y : {sg_name}_grad_max.y);
            #else
            {tab4_str}const half2 {y} = __hmin2(__h2div({sg_name}_sign, {x}), __float2half2_rn({alpha}));
            #endif
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code


    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.fake_numerical_gradient_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)


@torch.jit.script
def log_tailed_relu_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    mask_gt1 = x > 1.
    mask_le0 = x <= 0.
    grad_x = torch.ones_like(grad_output)
    grad_x[mask_gt1] = 1. / x[mask_gt1]
    grad_x[mask_le0] = alpha
    return grad_output * grad_x, None


class log_tailed_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return log_tailed_relu_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class LogTailedReLU(SurrogateFunctionBase):
    def __init__(self, alpha=0., spiking=True):
        '''
        * :ref:`API in English <LogTailedReLU.__init__-en>`
        .. _LogTailedReLU.__init__-cn:

        :param alpha: ??占쎌젞??占쎌냲?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌몴�뿆占�?岳묕옙????占쎈１?占쎈솿??占쎈쑏?
        :param spiking: ??亦낉옙?嶺뚮씭�뼱�몴�뇿�돦占쎌삌占쎈뼇?占쎈き???�땻�벂二곤옙遊억옙�뮖?占쎌맽嚥싲갇源듸옙占쎌뼲彛�? ``True``輿삳뿫遊억옙�뮍?筌륅옙??占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌벉占쎄샴?占쎈뮀占쏙옙占�? ``heaviside`` ???占쎈뮍?筌륅옙?占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼦占쎄샴?占쎈뮀占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??�뇡占�?�윜諭��삌�젆�굚�렧?? ``False``
            ??占쎈묍佯몃돆鍮뽳옙逾�?占쎈��?占쏙옙占�???占쎌쓡嚥싳쉶�굫占쎈쭆冶⑤떼�꼤??輿삳뿫遊억옙�뮍?占쎌졋?占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩占쎌뱽�댚占�?占쎈뮋?占쎈��?占쏙옙占�??占쎈쇀??占쎈뎨饔낅똾寃ラ댖怨쀬뒠???占쎄덩???占쎈１�솾占�??岳묕옙????占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏占쎈뉼占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏?

        `Deep Learning with Low Precision by Half-wave Gaussian Quantization <https://arxiv.org/abs/1702.00953>`_ ?占쎈쇀???占쎈뼇??占쎈１ Log-tailed ReLU??占쎌쓡嚥싳쉶�굫占쎈��?占쎈뼒?占쎈쑏???�윜諭�猷놅쭩占�?占쎈뎨饔낅똾寃ラ댖怨쀬뒠?占쎈꼤筌좑옙?

        .. math::
            g'(x) =
            \\begin{cases}
            \\alpha, & x \\leq 0 \\\\
            1, & 0 < x \\leq 0 \\\\
            \\frac{1}{x}, x > 1 \\\\
            \\end{cases}

        ?占쎈늾�뛾�렯猷딉옙堉�??占쎈１??占쎌쓨??占쎈뼒?占쎈쑏占쎈뉵筌좑옙?

        .. math::
            g(x) =
            \\begin{cases}
            \\alpha x, & x \\leq 0 \\\\
            x, & 0 < x \\leq 0 \\\\
            log(x), x > 1 \\\\
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/LogTailedReLU.*
            :width: 100%

        * :ref:`佯몌옙???嶺뚮쵑�쑇PI <LogTailedReLU.__init__-cn>`
        .. _LogTailedReLU.__init__-en:

        :param alpha: parameter to control gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The Log-tailed ReLU surrogate spiking function, which is first proposed in `Deep Learning with Low Precision by `Half-wave Gaussian Quantization <https://arxiv.org/abs/1702.00953>`_. The gradient is defined by

        .. math::
            g'(x) =
            \\begin{cases}
            \\alpha, & x \\leq 0 \\\\
            1, & 0 < x \\leq 0 \\\\
            \\frac{1}{x}, x > 1 \\\\
            \\end{cases}

        The primitive function is defined by

        .. math::
            g(x) =
            \\begin{cases}
            \\alpha x, & x \\leq 0 \\\\
            x, & 0 < x \\leq 0 \\\\
            log(x), x > 1 \\\\
            \\end{cases}

        .. image:: ../_static/API/activation_based/surrogate/LogTailedReLU.*
            :width: 100%
        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return log_tailed_relu.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        mask_ge1 = (x > 1.).to(x)
        y = (1. - mask_ge1) * F.leaky_relu(x, alpha) + mask_ge1 * (torch.log(x) + 1.)
        return y

    @staticmethod
    def backward(grad_output, x, alpha):
        return log_tailed_relu_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}float {y} = 0.0f;
            {tab4_str}if({x} <= 0.0f)
            {tab4_str}{curly_bracket_l}{y} = {alpha};{curly_bracket_r}
            {tab4_str}else if({x} <= 1.0f)
            {tab4_str}{curly_bracket_l}{y} = 1.0f;{curly_bracket_r}
            {tab4_str}else
            {tab4_str}{curly_bracket_l}{y} = 1.0f / {x};{curly_bracket_r}
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half {sg_name}_alpha = __float2half_rn({alpha});

            {tab4_str}half {sg_name}_{y}_low;
            {tab4_str}const half {sg_name}_{x}_low = __low2half({x});
            {tab4_str}if(__hle({sg_name}_{x}_low, __float2half_rn(0.0f)))
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_low = {sg_name}_alpha;{curly_bracket_r}
            {tab4_str}else if(__hle({sg_name}_{x}_low, __float2half_rn(1.0f)))
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_low = __float2half_rn(1.0f);{curly_bracket_r}
            {tab4_str}else
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_low = __hdiv(__float2half_rn(1.0f), {sg_name}_{x}_low);{curly_bracket_r}

            {tab4_str}half {sg_name}_{y}_high;
            {tab4_str}const half {sg_name}_{x}_high = __high2half({x});
            {tab4_str}if(__hle({sg_name}_{x}_high, __float2half_rn(0.0f)))
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_high = {sg_name}_alpha;{curly_bracket_r}
            {tab4_str}else if(__hle({sg_name}_{x}_high, __float2half_rn(1.0f)))
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_high = __float2half_rn(1.0f);{curly_bracket_r}
            {tab4_str}else
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_high = __hdiv(__float2half_rn(1.0f), {sg_name}_{x}_high);{curly_bracket_r}


            {tab4_str}const half2 {y} = __halves2half2({sg_name}_{y}_low, {sg_name}_{y}_high);

            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.log_tailed_relu_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200, figsize=(6, 4))
    # x = torch.arange(-5, 5, 0.01)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.LogTailedReLU(spiking=False, alpha=0.01)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha$=0.1')
    #
    # surrogate_function = surrogate.LogTailedReLU(spiking=True, alpha=0.01)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha$=0.1')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('LeakyKReLU surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.savefig('LogTailedReLU.svg')
    # plt.savefig('LogTailedReLU.pdf')


@torch.jit.script
def deterministic_pass_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return grad_output, None


class deterministic_pass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return deterministic_pass_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class DeterministicPass(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return deterministic_pass.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return x

    @staticmethod
    def backward(grad_output, x, alpha):
        return deterministic_pass_backward(grad_output, x, alpha)[0]


@torch.jit.script
def rect_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha * (x.abs() < 0.5 / alpha).to(x) * grad_output, None


class rect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return rect_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class Rect(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return rect.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return torch.clamp(alpha * x + 0.5, min=0.0, max=1.0)

    @staticmethod
    def backward(grad_output, x, alpha):
        return rect_backward(grad_output, x, alpha)[0]


class poisson_pass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.bernoulli(x).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


_has_cuda_ = [
    ATan,
    PiecewiseLeakyReLU,
    Sigmoid,
    S2NN,
    ReLU,
    QPseudoSpike,
    LeakyKReLU,
    FakeNumericalGradient
]