from abc import abstractmethod
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging

from . import surrogate, base
from .auto_cuda import neuron_kernel as ac_neuron_kernel
from .auto_cuda import ss_neuron_kernel as ss_ac_neuron_kernel
try:
    import cupy
    from . import neuron_kernel, cuda_utils

except BaseException as e:
    logging.info(f'spikingjelly.activation_based.neuron: {e}')
    cupy = None
    neuron_kernel = None
    cuda_utils = None


class SimpleBaseNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: Optional[float] = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode='s'):
        """
        A simple version of ``BaseNode``. The user can modify this neuron easily.
        """
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.detach_reset = detach_reset
        self.step_mode = step_mode
        self.register_memory(name='v', value=0.)

    def single_step_forward(self, x: torch.Tensor):

        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - self.v_threshold * spike_d

        else:
            # hard reset
            self.v = spike_d * self.v_reset + (1. - spike_d) * self.v

class SimpleIFNode(SimpleBaseNode):
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

class SimpleLIFNode(SimpleBaseNode):
    def __init__(self, tau:float = 40.0, decay_input: bool = False, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode='s'):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.tau = tau
        self.decay_input = decay_input

        self.v_acc = 0
        self.v_acc_l = 0
        self.new_grad = None
        
    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            self.v = self.v + (self.v_reset - self.v + x) / self.tau
        else:
            self.v = self.v + (self.v_reset - self.v) / self.tau + x

    def single_step_forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()

        if self.training:
            spike.register_hook(lambda grad: grad * self.new_grad)

        with torch.no_grad():
            self.v_acc += spike
            self.v_acc_l = self.v + spike
            self.v_acc += (self.v_acc < 1e-3).float()
            self.new_grad = (self.v_acc_l > 1e-3).float() + \
                            torch.log(torch.tensor(1 - 1 / self.tau)) * (self.v_acc_l / self.v_acc)

        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        # x_seq: [T, N, ...]
        T = x_seq.shape[0]
        spike_seq = []

        for t in range(T):
            spike = self.single_step_forward(x_seq[t])
            spike_seq.append(spike)

        return torch.stack(spike_seq, dim=0)

    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return self.single_step_forward(x)
        elif self.step_mode == 'm':
            return self.multi_step_forward(x)
        else:
            raise ValueError(f"Unknown step_mode: {self.step_mode}")

    def reset(self):
        super().reset()
        self.v_acc = 0
        self.v_acc_l = 0
        self.new_grad = None
class BaseNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: Optional[float] = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode='s', backend='torch', store_v_seq: bool = False):
        """
        * :ref:`API in English <BaseNode.__init__-en>`

        .. _BaseNode.__init__-cn:

        :param v_threshold: 曄욅퍘�뀇�쉪�삁��쇘뵷�럨
        :type v_threshold: float

        :param v_reset: 曄욅퍘�뀇�쉪�뇥營��뵷�럨��귛쫩�옖訝띴맏 ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇥營�訝� ``v_reset``竊�
            倻귝옖溫양쉰訝� ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇧�렮 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: �룏�릲鴉졿뮡�뿶�뵪�씎溫←츞�꼮�넳�눦�빊歟�佯��쉪�쎘餓ｅ눦�빊
        :type surrogate_function: Callable

        :param detach_reset: �삸�맔弱냨eset瓦뉒쮮�쉪溫←츞�쎗�늽獵�
        :type detach_reset: bool

        :param step_mode: 閭θ퓵與▼폀竊뚦룾餓δ맏 `'s'` (�뜒閭�) �닑 `'m'` (鸚싨��)
        :type step_mode: str

        :param backend: 鵝욜뵪�벆燁띶릮塋���귚툖�릪�쉪 ``step_mode`` �룾�꺗鴉싧를�쐣訝띶릪�쉪�릮塋���귛룾餓ι�싪퓝�돀�뜲 ``self.supported_backends`` �윥�쐦壤볟뎺
            鵝욜뵪�쉪閭θ퓵與▼폀�뵱�똻�쉪�릮塋���귛쑉�뵱�똻�쉪�깄�넻訝뗰펽鵝욜뵪 ``'cupy'`` �릮塋��삸��잌벧���恙ョ쉪
        :type backend: str

        :param store_v_seq: �쑉鵝욜뵪 ``step_mode = 'm'`` �뿶竊뚨퍢訝� ``shape = [T, N, *]`` �쉪渦볟뀯�릮竊뚧삸�맔岳앭춼訝��뿴瓦뉒쮮�쉪 ``shape = [T, N, *]``
            �쉪�릢訝ゆ뿶�뿴閭η쉪�뵷�럨��� ``self.v_seq`` ��귟�양쉰訝� ``False`` �뿶溫←츞若뚧닇�릮�룵岳앯븰����릮訝�訝ゆ뿶�댗�쉪�뵷�럨竊뚦뜵 ``shape = [N, *]`` �쉪 ``self.v`` ���
            ��싧만溫양쉰�닇 ``False`` 竊뚦룾餓θ뒄�쐛�냵耶�
        :type store_v_seq: bool

        �룾孃��늽SNN曄욅퍘�뀇�쉪�읃映사쪥瀯뤷뀇���

        * :ref:`訝��뻼API <BaseNode.__init__-cn>`

        .. _BaseNode.__init__-en:

        :param v_threshold: threshold of this neurons layer
        :type v_threshold: float

        :param v_reset: reset voltage of this neurons layer. If not ``None``, the neuron's voltage will be set to ``v_reset``
            after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :type v_reset: Optional[float]

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset in backward
        :type detach_reset: bool

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        :param backend: backend fot this neurons layer. Different ``step_mode`` may support for different backends. The user can
        print ``self.supported_backends`` and check what backends are supported by the current ``step_mode``. If supported,
        using ``'cupy'`` backend will have the fastest training speed
        :type backend: str

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption
        :type store_v_seq: bool

        This class is the base class of differentiable spiking neurons.
        """
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq

        # used in lava_exchange
        self.lava_s_cale = 1 << 6

        # used for cupy backend
        self.forward_kernel = None
        self.backward_kernel = None

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        """
         * :ref:`API in English <BaseNode.neuronal_charge-en>`

        .. _BaseNode.neuronal_charge-cn:

        若싦퉱曄욅퍘�뀇�쉪�뀉�뵷藥��늽�뼶葉뗣�귛춴映삣퓚窈삣츩�렟瓦쇾릉�눦�빊���

        * :ref:`訝��뻼API <BaseNode.neuronal_charge-cn>`

        .. _BaseNode.neuronal_charge-en:


        Define the charge difference equation. The sub-class must implement this function.
        """
        raise NotImplementedError

    def neuronal_fire(self):
        """
        * :ref:`API in English <BaseNode.neuronal_fire-en>`

        .. _BaseNode.neuronal_fire-cn:

        �졊�뜮壤볟뎺曄욅퍘�뀇�쉪�뵷�럨��곲삁��쇽펽溫←츞渦볟눣�꼮�넳���

        * :ref:`訝��뻼API <BaseNode.neuronal_fire-cn>`

        .. _BaseNode.neuronal_fire-en:


        Calculate out spikes of neurons by their current membrane potential and threshold voltage.
        """

        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        """
        * :ref:`API in English <BaseNode.neuronal_reset-en>`

        .. _BaseNode.neuronal_reset-cn:

        �졊�뜮壤볟뎺曄욅퍘�뀇�뇢�붂�쉪�꼮�넳竊뚦�배넑�뵷鵝띹퓵烏뚪뇥營����

        * :ref:`訝��뻼API <BaseNode.neuronal_reset-cn>`

        .. _BaseNode.neuronal_reset-en:


        Reset the membrane potential according to neurons' output spikes.
        """
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):
        """

        * :ref:`API in English <BaseNode.single_step_forward-en>`

        .. _BaseNode.single_step_forward-cn:

        :param x: 渦볟뀯�댆曄욅퍘�뀇�쉪�뵷�럨罌욇뇧
        :type x: torch.Tensor

        :return: 曄욅퍘�뀇�쉪渦볟눣�꼮�넳
        :rtype: torch.Tensor

        �뙃�뀱�뀉�뵷��곫붂�뵷��곲뇥營��쉪窈뷴틣瓦쏂죱�뎺�릲鴉졿뮡���

        * :ref:`訝��뻼API <BaseNode.single_step_forward-cn>`

        .. _BaseNode.single_step_forward-en:

        :param x: increment of voltage inputted to neurons
        :type x: torch.Tensor

        :return: out spikes of neurons
        :rtype: torch.Tensor

        Forward by the order of `neuronal_charge`, `neuronal_fire`, and `neuronal_reset`.

        """
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class AdaptBaseNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: Optional[float] = 0.,
                 v_rest: float = 0., w_rest: float = 0., tau_w: float = 2., a: float = 0., b: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False):
        # b: jump amplitudes
        # a: subthreshold coupling
        assert isinstance(w_rest, float)
        assert isinstance(v_rest, float)
        assert isinstance(tau_w, float)
        assert isinstance(a, float)
        assert isinstance(b, float)

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.register_memory('w', w_rest)

        self.w_rest = w_rest
        self.v_rest = v_rest
        self.tau_w = tau_w
        self.a = a
        self.b = b

    @staticmethod
    @torch.jit.script
    def jit_neuronal_adaptation(w: torch.Tensor, tau_w: float, a: float, v_rest: float, v: torch.Tensor):
        return w + 1. / tau_w * (a * (v - v_rest) - w)

    def neuronal_adaptation(self):
        """
        * :ref:`API in English <AdaptBaseNode.neuronal_adaptation-en>`

        .. _AdaptBaseNode.neuronal_adaptation-cn:

        �꼮�넳鰲��룕�쉪��귛틪��㎫뵷役곭쉪�쎍�뼭

        * :ref:`訝��뻼API <AdaptBaseNode.neuronal_adaptation-cn>`

        .. _AdaptBaseNode.neuronal_adaptation-en:

        Spike-triggered update of adaptation current.
        """
        self.w = self.jit_neuronal_adaptation(self.w, self.tau_w, self.a, self.v_rest, self.v)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, w: torch.Tensor, spike_d: torch.Tensor, v_reset: float, b: float,
                       spike: torch.Tensor):
        v = (1. - spike_d) * v + spike * v_reset
        w = w + b * spike
        return v, w

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, w: torch.Tensor, spike_d: torch.Tensor, v_threshold: float, b: float,
                       spike: torch.Tensor):
        v = v - spike_d * v_threshold
        w = w + b * spike
        return v, w

    def neuronal_reset(self, spike):
        """
        * :ref:`API in English <AdaptBaseNode.neuronal_reset-en>`

        .. _AdaptBaseNode.neuronal_reset-cn:

        �졊�뜮壤볟뎺曄욅퍘�뀇�뇢�붂�쉪�꼮�넳竊뚦�배넑�뵷鵝띹퓵烏뚪뇥營����

        * :ref:`訝��뻼API <AdaptBaseNode.neuronal_reset-cn>`

        .. _AdaptBaseNode.neuronal_reset-en:


        Reset the membrane potential according to neurons' output spikes.
        """
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v, self.w = self.jit_soft_reset(self.v, self.w, spike_d, self.v_threshold, self.b, spike)

        else:
            # hard reset
            self.v, self.w = self.jit_hard_reset(self.v, self.w, spike_d, self.v_reset, self.b, spike)

    def extra_repr(self):
        return super().extra_repr() + f', v_rest={self.v_rest}, w_rest={self.w_rest}, tau_w={self.tau_w}, a={self.a}, b={self.b}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.w_float_to_tensor(x)
        self.neuronal_charge(x)
        self.neuronal_adaptation()
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def w_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.w, float):
            w_init = self.w
            self.w = torch.full_like(x.data, fill_value=w_init)


class IFNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: Optional[float] = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False):
        """
        * :ref:`API in English <IFNode.__init__-en>`

        .. _IFNode.__init__-cn:

        :param v_threshold: 曄욅퍘�뀇�쉪�삁��쇘뵷�럨
        :type v_threshold: float

        :param v_reset: 曄욅퍘�뀇�쉪�뇥營��뵷�럨��귛쫩�옖訝띴맏 ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇥營�訝� ``v_reset``竊�
            倻귝옖溫양쉰訝� ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇧�렮 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: �룏�릲鴉졿뮡�뿶�뵪�씎溫←츞�꼮�넳�눦�빊歟�佯��쉪�쎘餓ｅ눦�빊
        :type surrogate_function: Callable

        :param detach_reset: �삸�맔弱냨eset瓦뉒쮮�쉪溫←츞�쎗�늽獵�
        :type detach_reset: bool

        :param step_mode: 閭θ퓵與▼폀竊뚦룾餓δ맏 `'s'` (�뜒閭�) �닑 `'m'` (鸚싨��)
        :type step_mode: str

        :param backend: 鵝욜뵪�벆燁띶릮塋���귚툖�릪�쉪 ``step_mode`` �룾�꺗鴉싧를�쐣訝띶릪�쉪�릮塋���귛룾餓ι�싪퓝�돀�뜲 ``self.supported_backends`` �윥�쐦壤볟뎺
            鵝욜뵪�쉪閭θ퓵與▼폀�뵱�똻�쉪�릮塋���귛쑉�뵱�똻�쉪�깄�넻訝뗰펽鵝욜뵪 ``'cupy'`` �릮塋��삸��잌벧���恙ョ쉪
        :type backend: str

        :param store_v_seq: �쑉鵝욜뵪 ``step_mode = 'm'`` �뿶竊뚨퍢訝� ``shape = [T, N, *]`` �쉪渦볟뀯�릮竊뚧삸�맔岳앭춼訝��뿴瓦뉒쮮�쉪 ``shape = [T, N, *]``
            �쉪�릢訝ゆ뿶�뿴閭η쉪�뵷�럨��� ``self.v_seq`` ��귟�양쉰訝� ``False`` �뿶溫←츞若뚧닇�릮�룵岳앯븰����릮訝�訝ゆ뿶�댗�쉪�뵷�럨竊뚦뜵 ``shape = [N, *]`` �쉪 ``self.v`` ���
            ��싧만溫양쉰�닇 ``False`` 竊뚦룾餓θ뒄�쐛�냵耶�
        :type store_v_seq: bool

        Integrate-and-Fire 曄욅퍘�뀇與▼엹竊뚦룾餓η쐦鵝쒐릤�꺍燁��늽�솳竊뚧뿞渦볟뀯�뿶�뵷�럨岳앮똻�걩若싷펽訝띴폏�깗LIF曄욅퍘�뀇�궍�졆烏겼뇧��귛끀�삁訝뗧쪥瀯뤷뒯�뒟耶��뼶葉뗤맏竊�

        .. math::
            H[t] = V[t-1] + X[t]

        * :ref:`訝��뻼API <IFNode.__init__-cn>`

        .. _IFNode.__init__-en:

        :param v_threshold: threshold of this neurons layer
        :type v_threshold: float

        :param v_reset: reset voltage of this neurons layer. If not ``None``, the neuron's voltage will be set to ``v_reset``
            after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :type v_reset: Optional[float]

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset in backward
        :type detach_reset: bool

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        :param backend: backend fot this neurons layer. Different ``step_mode`` may support for different backends. The user can
        print ``self.supported_backends`` and check what backends are supported by the current ``step_mode``. If supported,
        using ``'cupy'`` backend will have the fastest training speed
        :type backend: str

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption
        :type store_v_seq: bool

        The Integrate-and-Fire neuron, which can be seen as a ideal integrator. The voltage of the IF neuron will not decay
        as that of the LIF neuron. The sub-threshold neural dynamics of it is as followed:

        .. math::
            H[t] = V[t-1] + X[t]

        """
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch', 'cupy')
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset(x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float):
        v = v + x
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset(x: torch.Tensor, v: torch.Tensor, v_threshold: float):
        v = v + x
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                               v_reset: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                          v_reset: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                return super().multi_step_forward(x_seq)
            elif self.backend == 'cupy':
                hard_reset = self.v_reset is not None

                if x_seq.dtype == torch.float:
                    dtype = 'float'
                elif x_seq.dtype == torch.half:
                    dtype = 'half2'
                else:
                    raise NotImplementedError(x_seq.dtype)

                if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset,
                                                                                           dtype=dtype):
                    self.forward_kernel = ac_neuron_kernel.IFNodeFPTTKernel(hard_reset=hard_reset, dtype=dtype)

                if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype):
                    self.backward_kernel = ac_neuron_kernel.IFNodeBPTTKernel(
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype)

                self.v_float_to_tensor(x_seq[0])

                spike_seq, v_seq = ac_neuron_kernel.IFNodeATGF.apply(x_seq.flatten(1), self.v.flatten(0),
                                                                     self.v_threshold, self.v_reset,
                                                                     self.forward_kernel,
                                                                     self.backward_kernel)

                spike_seq = spike_seq.reshape(x_seq.shape)
                v_seq = v_seq.reshape(x_seq.shape)

                if self.store_v_seq:
                    self.v_seq = v_seq

                self.v = v_seq[-1].clone()

                return spike_seq
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x_seq[0])
            if self.v_reset is None:
                if self.store_v_seq:
                    spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq,
                                                                                                           self.v,
                                                                                                           self.v_threshold)
                else:
                    spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset(x_seq, self.v, self.v_threshold)
            else:
                if self.store_v_seq:
                    spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq,
                                                                                                           self.v,
                                                                                                           self.v_threshold,
                                                                                                           self.v_reset)
                else:
                    spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset(x_seq, self.v, self.v_threshold,
                                                                                    self.v_reset)
            return spike_seq

    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                return super().single_step_forward(x)
            elif self.backend == 'cupy':
                hard_reset = self.v_reset is not None

                if x.dtype == torch.float:
                    dtype = 'float'
                elif x.dtype == torch.half:
                    dtype = 'half2'
                else:
                    raise NotImplementedError(x.dtype)
                
                if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset,
                                                                                           dtype=dtype):
                    self.forward_kernel = ss_ac_neuron_kernel.IFNodeFPKernel(hard_reset=hard_reset, dtype=dtype)

                if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype):
                    self.backward_kernel = ss_ac_neuron_kernel.IFNodeBPKernel(
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype)

                self.v_float_to_tensor(x)

                spike, v = ss_ac_neuron_kernel.IFNodeATGF.apply(x.flatten(0), self.v.flatten(0),
                                                                     self.v_threshold, self.v_reset,
                                                                     self.forward_kernel,
                                                                     self.backward_kernel)

                spike = spike.reshape(x.shape)
                v = v.reshape(x.shape)

                self.v = v

                return spike
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None:
                spike, self.v = self.jit_eval_single_step_forward_soft_reset(x, self.v, self.v_threshold)
            else:
                spike, self.v = self.jit_eval_single_step_forward_hard_reset(x, self.v, self.v_threshold, self.v_reset)
            return spike


class LIFNode(BaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: Optional[float] = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        """
        * :ref:`API in English <LIFNode.__init__-en>`

        .. _LIFNode.__init__-cn:

        :param tau: �넑�뵷鵝띷뿶�뿴躍멩빊
        :type tau: float

        :param decay_input: 渦볟뀯�삸�맔阿잋폏�뢿訝롨“�뇧
        :type decay_input: bool

        :param v_threshold: 曄욅퍘�뀇�쉪�삁��쇘뵷�럨
        :type v_threshold: float

        :param v_reset: 曄욅퍘�뀇�쉪�뇥營��뵷�럨��귛쫩�옖訝띴맏 ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇥營�訝� ``v_reset``竊�
            倻귝옖溫양쉰訝� ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇧�렮 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: �룏�릲鴉졿뮡�뿶�뵪�씎溫←츞�꼮�넳�눦�빊歟�佯��쉪�쎘餓ｅ눦�빊
        :type surrogate_function: Callable

        :param detach_reset: �삸�맔弱냨eset瓦뉒쮮�쉪溫←츞�쎗�늽獵�
        :type detach_reset: bool

        :param step_mode: 閭θ퓵與▼폀竊뚦룾餓δ맏 `'s'` (�뜒閭�) �닑 `'m'` (鸚싨��)
        :type step_mode: str

        :param backend: 鵝욜뵪�벆燁띶릮塋���귚툖�릪�쉪 ``step_mode`` �룾�꺗鴉싧를�쐣訝띶릪�쉪�릮塋���귛룾餓ι�싪퓝�돀�뜲 ``self.supported_backends`` �윥�쐦壤볟뎺
            鵝욜뵪�쉪閭θ퓵與▼폀�뵱�똻�쉪�릮塋���귛쑉�뵱�똻�쉪�깄�넻訝뗰펽鵝욜뵪 ``'cupy'`` �릮塋��삸��잌벧���恙ョ쉪
        :type backend: str

        :param store_v_seq: �쑉鵝욜뵪 ``step_mode = 'm'`` �뿶竊뚨퍢訝� ``shape = [T, N, *]`` �쉪渦볟뀯�릮竊뚧삸�맔岳앭춼訝��뿴瓦뉒쮮�쉪 ``shape = [T, N, *]``
            �쉪�릢訝ゆ뿶�뿴閭η쉪�뵷�럨��� ``self.v_seq`` ��귟�양쉰訝� ``False`` �뿶溫←츞若뚧닇�릮�룵岳앯븰����릮訝�訝ゆ뿶�댗�쉪�뵷�럨竊뚦뜵 ``shape = [N, *]`` �쉪 ``self.v`` ���
            ��싧만溫양쉰�닇 ``False`` 竊뚦룾餓θ뒄�쐛�냵耶�
        :type store_v_seq: bool

        Leaky Integrate-and-Fire 曄욅퍘�뀇與▼엹竊뚦룾餓η쐦鵝쒏삸躍�轢뤹뵷�쉪燁��늽�솳��귛끀�삁訝뗧쪥瀯뤷뒯�뒟耶��뼶葉뗤맏竊�

        �떏 ``decay_input == True``:

            .. math::
                H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        �떏 ``decay_input == False``:

            .. math::
                H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]


        * :ref:`訝��뻼API <LIFNode.__init__-cn>`

        .. _LIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

        :param v_threshold: threshold of this neurons layer
        :type v_threshold: float

        :param v_reset: reset voltage of this neurons layer. If not ``None``, the neuron's voltage will be set to ``v_reset``
            after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :type v_reset: Optional[float]

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset in backward
        :type detach_reset: bool

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        :param backend: backend fot this neurons layer. Different ``step_mode`` may support for different backends. The user can
        print ``self.supported_backends`` and check what backends are supported by the current ``step_mode``. If supported,
        using ``'cupy'`` backend will have the fastest training speed
        :type backend: str

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption
        :type store_v_seq: bool

        The Leaky Integrate-and-Fire neuron, which can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        IF ``decay_input == True``:

            .. math::
                H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        IF ``decay_input == False``:

            .. math::
                H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        """
        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.tau = tau
        self.decay_input = decay_input

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch', 'cupy')
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_decay_input(x, self.v, self.v_reset, self.tau)

        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_no_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_no_decay_input(x, self.v, self.v_reset, self.tau)

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, tau: float):
        v = v + (x - v) / tau
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float):
        v = v + (x - (v - v_reset)) / tau
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, tau: float):
        v = v * (1. - 1. / tau) + x
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x
        return v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                            v_reset: float, tau: float):
        v = v + (x - (v - v_reset)) / tau
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                               v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                            tau: float):
        v = v + (x - v) / tau
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                               tau: float):
        v = v * (1. - 1. / tau) + x
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                           v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                      v_threshold: float, v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_no_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                              v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_no_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                         v_threshold: float, v_reset: float,
                                                                         tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                           tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                      v_threshold: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_no_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                              tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                         v_threshold: float,
                                                                         tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                return super().single_step_forward(x)
            elif self.backend == 'cupy':
                hard_reset = self.v_reset is not None

                if x.dtype == torch.float:
                    dtype = 'float'
                elif x.dtype == torch.half:
                    dtype = 'half2'
                else:
                    raise NotImplementedError(x.dtype)
                
                if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset,
                                                                                           dtype=dtype,
                                                                                           decay_input=self.decay_input):
                    self.forward_kernel = ss_ac_neuron_kernel.LIFNodeFPKernel(decay_input=self.decay_input,
                                                                              hard_reset=hard_reset, dtype=dtype)

                if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype, decay_input=self.decay_input):
                    self.backward_kernel = ss_ac_neuron_kernel.LIFNodeBPKernel(
                        decay_input=self.decay_input,
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype)

                self.v_float_to_tensor(x)

                spike, v = ss_ac_neuron_kernel.LIFNodeATGF.apply(x.flatten(0), self.v.flatten(0),
                                                                 self.v_threshold, self.v_reset, 1. / self.tau,
                                                                 self.forward_kernel,
                                                                 self.backward_kernel)

                spike = spike.reshape(x.shape)
                v = v.reshape(x.shape)

                self.v = v

                return spike
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_decay_input(x, self.v,
                                                                                             self.v_threshold, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_no_decay_input(x, self.v,
                                                                                                self.v_threshold,
                                                                                                self.tau)
            else:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_decay_input(x, self.v,
                                                                                             self.v_threshold,
                                                                                             self.v_reset, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_no_decay_input(x, self.v,
                                                                                                self.v_threshold,
                                                                                                self.v_reset,
                                                                                                self.tau)
            return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                return super().multi_step_forward(x_seq)
            elif self.backend == 'cupy':

                hard_reset = self.v_reset is not None
                if x_seq.dtype == torch.float:
                    dtype = 'float'
                elif x_seq.dtype == torch.half:
                    dtype = 'half2'
                else:
                    raise NotImplementedError(x_seq.dtype)

                if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset,
                                                                                           dtype=dtype,
                                                                                           decay_input=self.decay_input):
                    self.forward_kernel = ac_neuron_kernel.LIFNodeFPTTKernel(decay_input=self.decay_input,
                                                                             hard_reset=hard_reset, dtype=dtype)

                if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype, decay_input=self.decay_input):
                    self.backward_kernel = ac_neuron_kernel.LIFNodeBPTTKernel(decay_input=self.decay_input,
                                                                              surrogate_function=self.surrogate_function.cuda_codes,
                                                                              hard_reset=hard_reset,
                                                                              detach_reset=self.detach_reset,
                                                                              dtype=dtype)

                self.v_float_to_tensor(x_seq[0])

                spike_seq, v_seq = ac_neuron_kernel.LIFNodeATGF.apply(x_seq.flatten(1), self.v.flatten(0),
                                                                      self.v_threshold, self.v_reset, 1. / self.tau,
                                                                      self.forward_kernel,
                                                                      self.backward_kernel)

                spike_seq = spike_seq.reshape(x_seq.shape)
                v_seq = v_seq.reshape(x_seq.shape)

                if self.store_v_seq:
                    self.v_seq = v_seq

                self.v = v_seq[-1].clone()

                return spike_seq
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x_seq[0])
            if self.v_reset is None:
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_decay_input(x_seq, self.v,
                                                                                                    self.v_threshold,
                                                                                                    self.tau)
                else:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_no_decay_input(x_seq, self.v,
                                                                                                       self.v_threshold,
                                                                                                       self.tau)
            else:
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.v_reset, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset_decay_input(x_seq, self.v,
                                                                                                    self.v_threshold,
                                                                                                    self.v_reset,
                                                                                                    self.tau)
                else:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_no_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.v_reset, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset_no_decay_input(x_seq, self.v,
                                                                                                       self.v_threshold,
                                                                                                       self.v_reset,
                                                                                                       self.tau)

            return spike_seq


class ParametricLIFNode(BaseNode):
    def __init__(self, init_tau: float = 2.0, decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: Optional[float] = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        """
        * :ref:`API in English <ParametricLIFNode.__init__-en>`

        .. _ParametricLIFNode.__init__-cn:

        :param init_tau: �넑�뵷鵝띷뿶�뿴躍멩빊�쉪�닜冶뗥��
        :type init_tau: float

        :param decay_input: 渦볟뀯�삸�맔阿잋폏�뢿訝롨“�뇧
        :type decay_input: bool

        :param v_threshold: 曄욅퍘�뀇�쉪�삁��쇘뵷�럨
        :type v_threshold: float

        :param v_reset: 曄욅퍘�뀇�쉪�뇥營��뵷�럨��귛쫩�옖訝띴맏 ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇥營�訝� ``v_reset``竊�
            倻귝옖溫양쉰訝� ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇧�렮 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: �룏�릲鴉졿뮡�뿶�뵪�씎溫←츞�꼮�넳�눦�빊歟�佯��쉪�쎘餓ｅ눦�빊
        :type surrogate_function: Callable

        :param detach_reset: �삸�맔弱냨eset瓦뉒쮮�쉪溫←츞�쎗�늽獵�
        :type detach_reset: bool

        :param step_mode: 閭θ퓵與▼폀竊뚦룾餓δ맏 `'s'` (�뜒閭�) �닑 `'m'` (鸚싨��)
        :type step_mode: str

        :param backend: 鵝욜뵪�벆燁띶릮塋���귚툖�릪�쉪 ``step_mode`` �룾�꺗鴉싧를�쐣訝띶릪�쉪�릮塋���귛룾餓ι�싪퓝�돀�뜲 ``self.supported_backends`` �윥�쐦壤볟뎺
            鵝욜뵪�쉪閭θ퓵與▼폀�뵱�똻�쉪�릮塋���귛쑉�뵱�똻�쉪�깄�넻訝뗰펽鵝욜뵪 ``'cupy'`` �릮塋��삸��잌벧���恙ョ쉪
        :type backend: str

        :param store_v_seq: �쑉鵝욜뵪 ``step_mode = 'm'`` �뿶竊뚨퍢訝� ``shape = [T, N, *]`` �쉪渦볟뀯�릮竊뚧삸�맔岳앭춼訝��뿴瓦뉒쮮�쉪 ``shape = [T, N, *]``
            �쉪�릢訝ゆ뿶�뿴閭η쉪�뵷�럨��� ``self.v_seq`` ��귟�양쉰訝� ``False`` �뿶溫←츞若뚧닇�릮�룵岳앯븰����릮訝�訝ゆ뿶�댗�쉪�뵷�럨竊뚦뜵 ``shape = [N, *]`` �쉪 ``self.v`` ���
            ��싧만溫양쉰�닇 ``False`` 竊뚦룾餓θ뒄�쐛�냵耶�
        :type store_v_seq: bool

        :param cupy_fp32_inference: �떏訝� `True`竊뚦쑉 `eval` 與▼폀訝뗰펽鵝욜뵪float32竊뚦뜶�쑉GPU訝딂퓧烏뚳펽亮뜸툝 `cupy` 藥꿰퍘若됭즳竊뚦닕鴉싪눎�뒯鵝욜뵪 `cupy` 瓦쏂죱�뒥��잆��
            瓦쇾릉��됮」�쉪鴉섇뀍�쓢遙섆틢 ``backend``
        :type cupy_fp32_inference: bool

        `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_
        �룓�눣�쉪 Parametric Leaky Integrate-and-Fire (PLIF)曄욅퍘�뀇與▼엹竊뚦룾餓η쐦鵝쒏삸躍�轢뤹뵷�쉪燁��늽�솳��귛끀�삁訝뗧쪥瀯뤷뒯�뒟耶��뼶葉뗤맏竊�

        �떏 ``decay_input == True``:

            .. math::
                H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        �떏 ``decay_input == False``:

            .. math::
                H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        �끀訝� :math:`\\frac{1}{\\tau} = {\\rm Sigmoid}(w)`竊�:math:`w` �삸�룾耶╊튌�쉪�뢿�빊���

        * :ref:`訝��뻼API <ParametricLIFNode.__init__-cn>`

        .. _ParametricLIFNode.__init__-en:

        :param init_tau: the initial value of membrane time constant
        :type init_tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

        :param v_threshold: threshold of this neurons layer
        :type v_threshold: float

        :param v_reset: reset voltage of this neurons layer. If not ``None``, the neuron's voltage will be set to ``v_reset``
            after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :type v_reset: Optional[float]

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset in backward
        :type detach_reset: bool

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        :param backend: backend fot this neurons layer. Different ``step_mode`` may support for different backends. The user can
        print ``self.supported_backends`` and check what backends are supported by the current ``step_mode``. If supported,
        using ``'cupy'`` backend will have the fastest training speed
        :type backend: str

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption
        :type store_v_seq: bool

        :param cupy_fp32_inference: If `True`, if this module is in `eval` mode, using float32, running on GPU, and `cupy` is installed, then this
            module will use `cupy` to accelerate. This option has priority over ``backend``
        :type cupy_fp32_inference: bool

        The Parametric Leaky Integrate-and-Fire (PLIF) neuron, which is proposed by `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_ and can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        IF ``decay_input == True``:

            .. math::
                H = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        IF ``decay_input == False``:

            .. math::
                H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        where :math:`\\frac{1}{\\tau} = {\\rm Sigmoid}(w)`, :math:`w` is a learnable parameter.
        """

        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.decay_input = decay_input
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / self.w.sigmoid()
        return super().extra_repr() + f', tau={tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()
        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - self.w.sigmoid()) + x
            else:
                self.v = self.v - (self.v - self.v_reset) * self.w.sigmoid() + x

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'cupy':
            hard_reset = self.v_reset is not None
            if x_seq.dtype == torch.float:
                dtype = 'float'
            elif x_seq.dtype == torch.half:
                dtype = 'half2'
            else:
                raise NotImplementedError(x_seq.dtype)

            if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset,
                                                                                       dtype=dtype,
                                                                                       decay_input=self.decay_input):
                self.forward_kernel = ac_neuron_kernel.ParametricLIFNodeFPTTKernel(decay_input=self.decay_input,
                                                                                   hard_reset=hard_reset, dtype=dtype)

            if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                    surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                    detach_reset=self.detach_reset, dtype=dtype, decay_input=self.decay_input):
                self.backward_kernel = ac_neuron_kernel.ParametricLIFNodeBPTTKernel(decay_input=self.decay_input,
                                                                                    surrogate_function=self.surrogate_function.cuda_codes,
                                                                                    hard_reset=hard_reset,
                                                                                    detach_reset=self.detach_reset,
                                                                                    dtype=dtype)

            self.v_float_to_tensor(x_seq[0])

            spike_seq, v_seq = ac_neuron_kernel.ParametricLIFNodeATGF.apply(
                x_seq.flatten(1), self.v.flatten(0), self.v_threshold, self.v_reset, self.w.sigmoid().to(x_seq),
                self.forward_kernel, self.backward_kernel)

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)


class QIFNode(BaseNode):
    def __init__(self, tau: float = 2., v_c: float = 0.8, a0: float = 1., v_threshold: float = 1., v_rest: float = 0.,
                 v_reset: Optional[float] = -0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False):
        """
        * :ref:`API in English <QIFNode.__init__-en>`

        .. _QIFNode.__init__-cn:

        :param tau: �넑�뵷鵝띷뿶�뿴躍멩빊
        :type tau: float

        :param v_c: �뀽�뵰�뵷�럨
        :type v_c: float

        :param a0:
        :type a0: float

        :param v_threshold: 曄욅퍘�뀇�쉪�삁��쇘뵷�럨
        :type v_threshold: float

        :param v_rest: �쓾�겘�뵷鵝�
        :type v_rest: float

        :param v_reset: 曄욅퍘�뀇�쉪�뇥營��뵷�럨��귛쫩�옖訝띴맏 ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇥營�訝� ``v_reset``竊�
            倻귝옖溫양쉰訝� ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇧�렮 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: �룏�릲鴉졿뮡�뿶�뵪�씎溫←츞�꼮�넳�눦�빊歟�佯��쉪�쎘餓ｅ눦�빊
        :type surrogate_function: Callable

        :param detach_reset: �삸�맔弱냨eset瓦뉒쮮�쉪溫←츞�쎗�늽獵�
        :type detach_reset: bool

        :param step_mode: 閭θ퓵與▼폀竊뚦룾餓δ맏 `'s'` (�뜒閭�) �닑 `'m'` (鸚싨��)
        :type step_mode: str

        :param backend: 鵝욜뵪�벆燁띶릮塋���귚툖�릪�쉪 ``step_mode`` �룾�꺗鴉싧를�쐣訝띶릪�쉪�릮塋���귛룾餓ι�싪퓝�돀�뜲 ``self.supported_backends`` �윥�쐦壤볟뎺
            鵝욜뵪�쉪閭θ퓵與▼폀�뵱�똻�쉪�릮塋���귛쑉�뵱�똻�쉪�깄�넻訝뗰펽鵝욜뵪 ``'cupy'`` �릮塋��삸��잌벧���恙ョ쉪
        :type backend: str

        :param store_v_seq: �쑉鵝욜뵪 ``step_mode = 'm'`` �뿶竊뚨퍢訝� ``shape = [T, N, *]`` �쉪渦볟뀯�릮竊뚧삸�맔岳앭춼訝��뿴瓦뉒쮮�쉪 ``shape = [T, N, *]``
            �쉪�릢訝ゆ뿶�뿴閭η쉪�뵷�럨��� ``self.v_seq`` ��귟�양쉰訝� ``False`` �뿶溫←츞若뚧닇�릮�룵岳앯븰����릮訝�訝ゆ뿶�댗�쉪�뵷�럨竊뚦뜵 ``shape = [N, *]`` �쉪 ``self.v`` ���
            ��싧만溫양쉰�닇 ``False`` 竊뚦룾餓θ뒄�쐛�냵耶�
        :type store_v_seq: bool


        Quadratic Integrate-and-Fire 曄욅퍘�뀇與▼엹竊뚥��燁띺씆瀛욘�㎫㎝�늽�룕�붂曄욅퍘�뀇與▼엹竊뚥튋�삸�뙁�빊燁��늽�룕�붂曄욅퍘�뀇(Exponential Integrate-and-Fire)�쉪瓦묇세�뎵�쑍��귛끀�삁訝뗧쪥瀯뤷뒯�뒟耶��뼶葉뗤맏竊�

        .. math::
            H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] + a_0 (V[t-1] - V_{rest})(V[t-1] - V_c))

        * :ref:`訝��뻼API <QIFNode.__init__-cn>`

        .. _QIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param v_c: critical voltage
        :type v_c: float

        :param a0:
        :type a0: float

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of this neurons layer. If not ``None``, the neuron's voltage will be set to ``v_reset``
            after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :type v_reset: Optional[float]

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset in backward
        :type detach_reset: bool

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        :param backend: backend fot this neurons layer. Different ``step_mode`` may support for different backends. The user can
        print ``self.supported_backends`` and check what backends are supported by the current ``step_mode``. If supported,
        using ``'cupy'`` backend will have the fastest training speed
        :type backend: str

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption
        :type store_v_seq: bool

        The Quadratic Integrate-and-Fire neuron is a kind of nonlinear integrate-and-fire models and also an approximation of the Exponential Integrate-and-Fire model.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] + a_0 (V[t-1] - V_{rest})(V[t-1] - V_c))
        """

        assert isinstance(tau, float) and tau > 1.
        if v_reset is not None:
            assert v_threshold > v_reset
            assert v_rest >= v_reset
        assert a0 > 0

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.tau = tau
        self.v_c = v_c
        self.v_rest = v_rest
        self.a0 = a0

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, v_c={self.v_c}, a0={self.a0}, v_rest={self.v_rest}'

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x + self.a0 * (self.v - self.v_rest) * (self.v - self.v_c)) / self.tau

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'cupy':
            self.v_float_to_tensor(x_seq[0])

            spike_seq, v_seq = neuron_kernel.MultiStepQIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.tau, self.v_threshold, self.v_reset, self.v_rest,
                self.v_c, self.a0, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)


class EIFNode(BaseNode):
    def __init__(self, tau: float = 2., delta_T: float = 1., theta_rh: float = .8, v_threshold: float = 1.,
                 v_rest: float = 0., v_reset: Optional[float] = -0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False):
        """
        * :ref:`API in English <EIFNode.__init__-en>`

        .. _EIFNode.__init__-cn:

        :param tau: �넑�뵷鵝띷뿶�뿴躍멩빊
        :type tau: float

        :param delta_T: �솪約�佯��뢿�빊
        :type delta_T: float

        :param theta_rh: �읃凉뷴벧�뵷�럨�삁���
        :type theta_rh: float

        :param v_threshold: 曄욅퍘�뀇�쉪�삁��쇘뵷�럨
        :type v_threshold: float

        :param v_reset: 曄욅퍘�뀇�쉪�뇥營��뵷�럨��귛쫩�옖訝띴맏 ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇥營�訝� ``v_reset``竊�
            倻귝옖溫양쉰訝� ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇧�렮 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: �룏�릲鴉졿뮡�뿶�뵪�씎溫←츞�꼮�넳�눦�빊歟�佯��쉪�쎘餓ｅ눦�빊
        :type surrogate_function: Callable

        :param detach_reset: �삸�맔弱냨eset瓦뉒쮮�쉪溫←츞�쎗�늽獵�
        :type detach_reset: bool

        :param step_mode: 閭θ퓵與▼폀竊뚦룾餓δ맏 `'s'` (�뜒閭�) �닑 `'m'` (鸚싨��)
        :type step_mode: str

        :param backend: 鵝욜뵪�벆燁띶릮塋���귚툖�릪�쉪 ``step_mode`` �룾�꺗鴉싧를�쐣訝띶릪�쉪�릮塋���귛룾餓ι�싪퓝�돀�뜲 ``self.supported_backends`` �윥�쐦壤볟뎺
            鵝욜뵪�쉪閭θ퓵與▼폀�뵱�똻�쉪�릮塋���귛쑉�뵱�똻�쉪�깄�넻訝뗰펽鵝욜뵪 ``'cupy'`` �릮塋��삸��잌벧���恙ョ쉪
        :type backend: str

        :param store_v_seq: �쑉鵝욜뵪 ``step_mode = 'm'`` �뿶竊뚨퍢訝� ``shape = [T, N, *]`` �쉪渦볟뀯�릮竊뚧삸�맔岳앭춼訝��뿴瓦뉒쮮�쉪 ``shape = [T, N, *]``
            �쉪�릢訝ゆ뿶�뿴閭η쉪�뵷�럨��� ``self.v_seq`` ��귟�양쉰訝� ``False`` �뿶溫←츞若뚧닇�릮�룵岳앯븰����릮訝�訝ゆ뿶�댗�쉪�뵷�럨竊뚦뜵 ``shape = [N, *]`` �쉪 ``self.v`` ���
            ��싧만溫양쉰�닇 ``False`` 竊뚦룾餓θ뒄�쐛�냵耶�
        :type store_v_seq: bool


        Exponential Integrate-and-Fire 曄욅퍘�뀇與▼엹竊뚥��燁띺씆瀛욘�㎫㎝�늽�룕�붂曄욅퍘�뀇與▼엹竊뚧삸�뵳HH曄욅퍘�뀇與▼엹(Hodgkin-Huxley model)嶸��뙑�릮�렓野쇔눣�쉪訝�瀯닸Æ�엹��귛쑉 :math:`\\Delta_T\\to 0` �뿶����뙑訝튛IF與▼엹��귛끀�삁訝뗧쪥瀯뤷뒯�뒟耶��뼶葉뗤맏竊�

        .. math::
            H[t] = V[t-1] + \\frac{1}{\\tau}\\left(X[t] - (V[t-1] - V_{rest}) + \\Delta_T\\exp\\left(\\frac{V[t-1] - \\theta_{rh}}{\\Delta_T}\\right)\\right)

        * :ref:`訝��뻼API <EIFNode.__init__-cn>`

        .. _EIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param delta_T: sharpness parameter
        :type delta_T: float

        :param theta_rh: rheobase threshold
        :type theta_rh: float

        :param v_threshold: threshold of this neurons layer
        :type v_threshold: float

        :param v_reset: reset voltage of this neurons layer. If not ``None``, the neuron's voltage will be set to ``v_reset``
            after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :type v_reset: Optional[float]

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset in backward
        :type detach_reset: bool

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        :param backend: backend fot this neurons layer. Different ``step_mode`` may support for different backends. The user can
        print ``self.supported_backends`` and check what backends are supported by the current ``step_mode``. If supported,
        using ``'cupy'`` backend will have the fastest training speed
        :type backend: str

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption
        :type store_v_seq: bool

        The Exponential Integrate-and-Fire neuron is a kind of nonlinear integrate-and-fire models and also an one-dimensional model derived from the Hodgkin-Huxley model. It degenerates to the LIF model when :math:`\\Delta_T\\to 0`.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            H[t] = V[t-1] + \\frac{1}{\\tau}\\left(X[t] - (V[t-1] - V_{rest}) + \\Delta_T\\exp\\left(\\frac{V[t-1] - \\theta_{rh}}{\\Delta_T}\\right)\\right)
        """

        assert isinstance(tau, float) and tau > 1.
        if v_reset is not None:
            assert v_threshold > v_reset
            assert v_rest >= v_reset
        assert delta_T > 0

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.tau = tau
        self.delta_T = delta_T
        self.v_rest = v_rest
        self.theta_rh = theta_rh

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, delta_T={self.delta_T}, theta_rh={self.theta_rh}'

    def neuronal_charge(self, x: torch.Tensor):
        with torch.no_grad():
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.as_tensor(self.v, device=x.device)

        self.v = self.v + (x + self.v_rest - self.v + self.delta_T * torch.exp(
            (self.v - self.theta_rh) / self.delta_T)) / self.tau

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'cupy':
            self.v_float_to_tensor(x_seq[0])

            spike_seq, v_seq = neuron_kernel.MultiStepEIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.tau, self.v_threshold, self.v_reset, self.v_rest,
                self.theta_rh, self.delta_T, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)


class IzhikevichNode(AdaptBaseNode):
    def __init__(self, tau: float = 2., v_c: float = 0.8, a0: float = 1., v_threshold: float = 1.,
                 v_reset: Optional[float] = 0., v_rest: float = -0.1, w_rest: float = 0., tau_w: float = 2., a: float = 0.,
                 b: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False):
        assert isinstance(tau, float) and tau > 1.
        assert a0 > 0

        super().__init__(v_threshold, v_reset, v_rest, w_rest, tau_w, a, b, surrogate_function, detach_reset, step_mode,
                         backend, store_v_seq)
        self.tau = tau
        self.v_c = v_c
        self.a0 = a0

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, v_c={self.v_c}, a0={self.a0}'

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x + self.a0 * (self.v - self.v_rest) * (self.v - self.v_c) - self.w) / self.tau

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'cupy':
            self.v_float_to_tensor(x_seq[0])
            self.w_float_to_tensor(x_seq[0])

            spike_seq, v_seq, w_seq = neuron_kernel.MultiStepIzhikevichNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.w.flatten(0), self.tau, self.v_threshold, self.v_reset,
                self.v_rest, self.a, self.b, self.tau_w,
                self.v_c, self.a0, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)
            w_seq = w_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()
            self.w = w_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)


class LIAFNode(LIFNode):
    def __init__(self, act: Callable, threshold_related: bool, *args, **kwargs):
        """
        * :ref:`API in English <LIAFNode.__init__-en>`

        .. _LIAFNode.__init__-cn:

        :param act: 嚥�域삣눦�빊
        :type act: Callable
        :param threshold_related: �삸�맔鵝욜뵪�삁��쇌풚壅뽪Æ凉� (TR mode). �떏訝� ``True`` �닕 ``y = act(h - v_th)``竊�
            �맔�닕 ``y = act(h)``
        :type threshold_related: bool

        `LIAF-Net: Leaky Integrate and Analog Fire Network for Lightweight and Efficient Spatiotemporal Information Processing <https://arxiv.org/abs/2011.06176>`_ �룓�눣�쉪LIAF曄욅퍘�뀇��괢IAFNode�뭽LIFNode�쉪烏뚥맏�쎑�릪竊뚥퐜渦볟눣�삸 ``self.act(...)`` ��뚪씆�꼮�넳���

        .. Warning::

            The outputs of this neurons layer are not binary spikes.


        * :ref:`訝��뻼API <LIAFNode.__init__-cn>`

        .. _LIAFNode.__init__-en:

        :param act: the activation function
        :type act: Callable
        :param threshold_related: whether the neuron uses threshold related (TR mode). If ``True``, ``y = act(h - v_th)``,
            otherwise ``y = act(h)``
        :type threshold_related: bool

        Other parameters in `*args, **kwargs` are same with :class:`LIFNode`.

        The LIAF neuron proposed in `LIAF-Net: Leaky Integrate and Analog Fire Network for Lightweight and Efficient Spatiotemporal Information Processing <https://arxiv.org/abs/2011.06176>`_. LIAFNode has the same behavior as LIFNode, but outputs ``self.act(...)``
        rather than spikes.

        .. admonition:: Warning
            :class: warning

            The outputs of this neurons layer are not binary spikes.

        """
        super().__init__(*args, **kwargs)
        self.act = act
        self.threshold_related = threshold_related

        assert self.backend == 'torch', "LIAFNode only supports for backend='torch'!"
        assert self.single_step_cupy_fp32_inference == False, "LIAFNode does not support for single_step_cupy_fp32_inference!"

    @property
    def supported_backends(self):
        return ('torch',)

    def single_step_forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        if self.threshold_related:
            y = self.act(self.v - self.v_threshold)
        else:
            y = self.act(self.v)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return y


class KLIFNode(BaseNode):
    def __init__(self, scale_reset: bool = False, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: Optional[float] = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        """
        * :ref:`API in English <KLIFNode.__init__-en>`

        .. _KLIFNode.__init__-cn:

        :param scale_reset: �삸�맔�쑉 ``neuronal_reset`` �뿶弱� ``v`` 瓦쏂죱煐⒵붂
        :type scale_reset: bool

        :param tau: �넑�뵷鵝띷뿶�뿴躍멩빊
        :type tau: float

        :param decay_input: 渦볟뀯�삸�맔阿잋폏�뢿訝롨“�뇧
        :type decay_input: bool

        :param v_threshold: 曄욅퍘�뀇�쉪�삁��쇘뵷�럨
        :type v_threshold: float

        :param v_reset: 曄욅퍘�뀇�쉪�뇥營��뵷�럨��귛쫩�옖訝띴맏 ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇥營�訝� ``v_reset``竊�
            倻귝옖溫양쉰訝� ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇧�렮 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: �룏�릲鴉졿뮡�뿶�뵪�씎溫←츞�꼮�넳�눦�빊歟�佯��쉪�쎘餓ｅ눦�빊
        :type surrogate_function: Callable

        :param detach_reset: �삸�맔弱냨eset瓦뉒쮮�쉪溫←츞�쎗�늽獵�
        :type detach_reset: bool

        :param step_mode: 閭θ퓵與▼폀竊뚦룾餓δ맏 `'s'` (�뜒閭�) �닑 `'m'` (鸚싨��)
        :type step_mode: str

        :param backend: 鵝욜뵪�벆燁띶릮塋���귚툖�릪�쉪 ``step_mode`` �룾�꺗鴉싧를�쐣訝띶릪�쉪�릮塋���귛룾餓ι�싪퓝�돀�뜲 ``self.supported_backends`` �윥�쐦壤볟뎺
            鵝욜뵪�쉪閭θ퓵與▼폀�뵱�똻�쉪�릮塋���귛쑉�뵱�똻�쉪�깄�넻訝뗰펽鵝욜뵪 ``'cupy'`` �릮塋��삸��잌벧���恙ョ쉪
        :type backend: str

        :param store_v_seq: �쑉鵝욜뵪 ``step_mode = 'm'`` �뿶竊뚨퍢訝� ``shape = [T, N, *]`` �쉪渦볟뀯�릮竊뚧삸�맔岳앭춼訝��뿴瓦뉒쮮�쉪 ``shape = [T, N, *]``
            �쉪�릢訝ゆ뿶�뿴閭η쉪�뵷�럨��� ``self.v_seq`` ��귟�양쉰訝� ``False`` �뿶溫←츞若뚧닇�릮�룵岳앯븰����릮訝�訝ゆ뿶�댗�쉪�뵷�럨竊뚦뜵 ``shape = [N, *]`` �쉪 ``self.v`` ���
            ��싧만溫양쉰�닇 ``False`` 竊뚦룾餓θ뒄�쐛�냵耶�
        :type store_v_seq: bool

        `KLIF: An optimized spiking neuron unit for tuning surrogate gradient slope and membrane potential <https://arxiv.org/abs/2302.09238>`_ �룓�눣�쉪K-based Leaky Integrate-and-Fire 曄욅퍘�뀇與▼엹竊뚦룾餓η쐦鵝쒏삸躍�轢뤹뵷�쉪燁��늽�솳��귛끀�삁訝뗧쪥瀯뤷뒯�뒟耶��뼶葉뗤맏竊�

        �떏 ``decay_input == True``:

            .. math::
                H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        �떏 ``decay_input == False``:

            .. math::
                H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        力ⓩ꼷竊똊LIF曄욅퍘�뀇�쉪�붂�뵷�뭽�뇥營�訝롦솹��싩쉪曄욅퍘�뀇訝띶릪竊뚥맏竊�

            .. math::

                F[t] &= \\mathrm{ReLU}(kH[t])

                S[t] &= \\Theta(F[t] - V_{th})

        倻귝옖 ``scale_reset == False``竊뚦닕

            .. math::
                V[t] = \\begin{cases}
                    F[t](1-S[t]) + V_{reset}S[t], hard~~reset \\\\
                    F[t] - S[t]V_{th}, soft~~reset
                \\end{cases}

        倻귝옖 ``scale_reset == True``竊뚦닕

            .. math::
                V[t] = \\begin{cases}
                    \\frac{F[t]}{k}(1-S[t]) + V_{reset}S[t], hard~~reset \\\\
                    \\frac{1}{k}(F[t] - S[t]V_{th}), soft~~reset
                \\end{cases}



        * :ref:`訝��뻼API <KLIFNode.__init__-cn>`

        .. _KLIFNode.__init__-en:

        :param scale_reset: whether scale ``v`` in ``neuronal_reset``
        :type scale_reset: bool

        :param tau: membrane time constant
        :type tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

        :param v_threshold: threshold of this neurons layer
        :type v_threshold: float

        :param v_reset: reset voltage of this neurons layer. If not ``None``, the neuron's voltage will be set to ``v_reset``
            after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :type v_reset: Optional[float]

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset in backward
        :type detach_reset: bool

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        :param backend: backend fot this neurons layer. Different ``step_mode`` may support for different backends. The user can
        print ``self.supported_backends`` and check what backends are supported by the current ``step_mode``. If supported,
        using ``'cupy'`` backend will have the fastest training speed
        :type backend: str

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption
        :type store_v_seq: bool

        The K-based Leaky Integrate-and-Fire neuron proposed by `KLIF: An optimized spiking neuron unit for tuning surrogate gradient slope and membrane potential <https://arxiv.org/abs/2302.09238>`_, which can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        IF ``decay_input == True``:

            .. math::
                H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        IF ``decay_input == False``:

            .. math::
                H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        Note that the neuronal fire and reset of the KLIF neuron is different from native neurons:

            .. math::

                F[t] &= \\mathrm{ReLU}(kH[t])

                S[t] &= \\Theta(F[t] - V_{th})

        If ``scale_reset == False``, then

            .. math::
                V[t] = \\begin{cases}
                    F[t](1-S[t]) + V_{reset}S[t], hard~~reset \\\\
                    F[t] - S[t]V_{th}, soft~~reset
                \\end{cases}

        Elif ``scale_reset == True``, then

            .. math::
                V[t] = \\begin{cases}
                    \\frac{F[t]}{k}(1-S[t]) + V_{reset}S[t], hard~~reset \\\\
                    \\frac{1}{k}(F[t] - S[t]V_{th}), soft~~reset
                \\end{cases}


        """
        assert isinstance(tau, float) and tau > 1.
        if backend == 'cupy':
            raise NotImplementedError("The CuPy backend for the KLIF neuron has not been implemented!")

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.scale_reset = scale_reset
        self.tau = tau
        self.decay_input = decay_input

        self.k = nn.Parameter(torch.as_tensor(1.))

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float, k: torch.Tensor):
        v = v + (x - (v - v_reset)) / tau
        v = torch.relu_(k * v)
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float, k: torch.Tensor):
        v = v - (v - v_reset) / tau + x
        v = torch.relu_(k * v)
        return v

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            v_reset = 0.
        else:
            v_reset = self.v_reset
        if self.decay_input:
            self.v = self.neuronal_charge_decay_input(x, self.v, v_reset, self.tau, self.k)

        else:
            self.v = self.neuronal_charge_no_decay_input(x, self.v, v_reset, self.tau, self.k)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.scale_reset:
            if self.v_reset is None:
                # soft reset
                self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold) / self.k

            else:
                # hard reset
                self.v = self.jit_hard_reset(self.v / self.k, spike_d, self.v_reset)

        else:

            if self.v_reset is None:
                # soft reset
                self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

            else:
                # hard reset
                self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)


class PSN(nn.Module, base.MultiStepModule):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan()):
        """
        :param T: the number of time-steps
        :type T: int
        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        The Parallel Spiking Neuron proposed in `Parallel Spiking Neurons with High Efficiency and Long-term Dependencies Learning Ability <https://arxiv.org/abs/2304.12760>`_. The neuronal dynamics are defined as

        .. math::

            H &= WX, ~~~~~~~~~~~~~~~W \\in \\mathbb{R}^{T \\times T}, X \\in \\mathbb{R}^{T \\times N} \\label{eq psn neuronal charge}\\\\
            S &= \\Theta(H - B), ~~~~~B \\in \\mathbb{R}^{T}, S\\in \\{0, 1\\}^{T \\times N}

        where :math:`W` is the learnable weight matrix, and :math:`B` is the learnable threshold.

        .. admonition:: Note
            :class: note

            The PSN only supports the multi-step mode.
        """
        super().__init__()
        self.T = T
        self.surrogate_function = surrogate_function
        weight = torch.zeros([T, T])
        bias = torch.zeros([T, 1])

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, -1.)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1))
        spike_seq = self.surrogate_function(h_seq)
        return spike_seq.view(x_seq.shape)

    def extra_repr(self):
        return super().extra_repr() + f'T={self.T}, '


class MaskedPSN(base.MemoryModule):
    @staticmethod
    @torch.jit.script
    def gen_masked_weight(lambda_: torch.Tensor, mask0: torch.Tensor, mask1: torch.Tensor, weight: torch.Tensor):
        return (lambda_ * mask0 + (1. - lambda_) * mask1) * weight

    def masked_weight(self):
        if self.lambda_ >= 1.:
            return self.weight * self.mask0
        else:
            return self.gen_masked_weight(self.lambda_, self.mask0, self.mask1, self.weight)

    def __init__(self, k: int, T: int, lambda_init: float = 0.,
                 surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan(), step_mode: str = 's'):
        """
        :param k: the order of the Masked PSN
        :type k: int
        :param T: the number of time-steps
        :type T: int
        :param lambda_init: the initial value of :math:`\\lambda` to adjust the progressive masking process
        :type lambda_init: float
        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable
        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        The Masked Parallel Spiking Neuron proposed in `Parallel Spiking Neurons with High Efficiency and Long-term Dependencies Learning Ability <https://arxiv.org/abs/2304.12760>`_. The neuronal dynamics are defined as

        .. math::

            H &= (W \\cdot {M}_{k})X, ~~~~~~~~~~~~~~~W \\in \\mathbb{R}^{T \\times T}, {M}_{k} \\in \\mathbb{R}^{T \\times T}, X \\in \\mathbb{R}^{T \\times N} \\\\
            S &= \\Theta(H - B), ~~~~~B \\in \\mathbb{R}^{T}, S\\in \\{0, 1\\}^{T \\times N}

        where :math:`W` is the learnable weight matrix, :math:`B` is the learnable threshold, and :math:`{M}_{k}` is defined as

        .. math::

            {M}_{k}[i][j] = \\begin{cases}
                1, ~~ j \\leq i \\leq j + k - 1 \\\\
                0, \\mathrm{otherwise}
            \\end{cases}.

        :math:`\\lambda` is used to adjust the progressive masking process, which is

        .. math::

            M_{k}(\\lambda) = \\lambda \\cdot M_{k} + (1 - \\lambda) \\cdot J,

        where :math:`J` is an all-one matrix.

        The user can set :math:`\\lambda` during training by calling ``self.lambda_ = ...``.

        .. admonition:: Note
            :class: note

            The masked PSN supports both single-step and multi-step mode. But using the multi-step mode is much faster than the single-step mode.

        """
        super().__init__()
        self.register_memory('time_step', 0)
        self.register_memory('queue', [])
        self.step_mode = step_mode
        self.k = k
        self.T = T
        self.surrogate_function = surrogate_function
        weight = torch.zeros([T, T])
        bias = torch.zeros([T, 1])
        self.register_buffer('_lambda_', torch.as_tensor(lambda_init))

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, -1.)

        mask1 = torch.ones([T, T])
        mask0 = torch.tril(mask1) * torch.triu(mask1, -(self.k - 1))
        self.register_buffer('mask0', mask0)
        self.register_buffer('mask1', mask1)


    def single_step_forward(self, x: torch.Tensor):
        if self.lambda_ < 1.:
            raise ValueError("The masked PSN can not work in single-step mode when k < 1!")

        self.queue.append(x.flatten())
        if self.queue.__len__() > self.k:
            self.queue.pop(0)

        if self.time_step + 1 > self.T:
            raise OverflowError(f"The MaskedPSN(T={self.T}) has run {self.time_step + 1} time-steps!")


        weight = self.masked_weight()[self.time_step, self.time_step + 1 - self.queue.__len__(): self.time_step + 1]
        x_seq = torch.stack(self.queue)



        for i in range(x.dim()):
            weight = weight.unsqueeze(-1)


        h = torch.sum(weight * x_seq, 0)
        spike = self.surrogate_function(h + self.bias[self.time_step])

        self.time_step += 1
        return spike.view(x.shape)

    def multi_step_forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        assert x_seq.shape[0] == self.T
        h_seq = torch.addmm(self.bias, self.masked_weight(), x_seq.flatten(1))
        spike_seq = self.surrogate_function(h_seq).view(x_seq.shape)
        return spike_seq

    @property
    def lambda_(self):
        return self._lambda_

    @lambda_.setter
    def lambda_(self, value: float):
        torch.fill_(self.lambda_, value)

    def extra_repr(self):
        return super().extra_repr() + f', lambda_={self.lambda_}, T={self.T}'


class SlidingPSN(base.MemoryModule):

    @property
    def supported_backends(self):
        return 'gemm', 'conv'

    def gen_gemm_weight(self, T: int):
        weight = torch.zeros([T, T], device=self.weight.device)
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - self.k)
            length = min(end - start, self.k)
            weight[i][start: end] = self.weight[self.k - length: self.k]

        return weight

    def __init__(self, k: int, exp_init: bool = True,
                 surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan(), step_mode: str = 's',
                 backend: str = 'gemm'):
        """
        :param k: the order of the Sliding PSN
        :type k: int
        :param exp_init: if ``True``, the weight will be initialized as ``(..., 1/4, 1/2, 1)``. If ``False``, the weight    will be initialized by the kaiming uniform
        :type exp_init: bool
        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable
        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str
        :param backend: backend fot this neuron layer, which can be "gemm" or "conv". This option only works for the multi-step mode
        :type backend: str

        The Sliding Parallel Spiking Neuron proposed in `Parallel Spiking Neurons with High Efficiency and Long-term Dependencies Learning Ability <https://arxiv.org/abs/2304.12760>`_. The neuronal dynamics are defined as

        .. math::

            H[t] &= \\sum_{i=0}^{k-1}W_{i}\\cdot X[t - k + 1 + i], \\\\
	        S[t] &= \\Theta(H[t] - B),


        where :math:`W = [W_{0}, W_{1}, ..., W_{k-1}] \\in \\mathbb{R}^{T}` is the learnable weight, and :math:`B` is the learnable threshold.


        .. admonition:: Note
            :class: note

            The Sliding PSN supports both single-step and multi-step mode. But using the multi-step mode is much faster than the single-step mode.


        """

        super().__init__()
        self.register_memory('queue', [])
        self.step_mode = step_mode
        self.k = k
        self.surrogate_function = surrogate_function
        self.backend = backend

        if exp_init:
            weight = torch.ones([k])
            for i in range(k - 2, -1, -1):
                weight[i] = weight[i + 1] / 2.
        else:
            weight = torch.ones([1, k])
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight = weight[0]

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.as_tensor(-1.))

    def single_step_forward(self, x: torch.Tensor):
        self.queue.append(x.flatten())
        if self.queue.__len__() > self.k:
            self.queue.pop(0)

        weight = self.weight[self.k - self.queue.__len__(): self.k]
        x_seq = torch.stack(self.queue)

        weight = weight.unsqueeze(-1)

        h = torch.sum(weight * x_seq, 0)
        spike = self.surrogate_function(h + self.bias)

        return spike.view(x.shape)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'gemm':

            weight = self.gen_gemm_weight(x_seq.shape[0])
            h_seq = torch.addmm(self.bias, weight, x_seq.flatten(1)).view(x_seq.shape)
            return self.surrogate_function(h_seq)
        elif self.backend == 'conv':

            # x_seq.shape = [T, N, *]
            x_seq_shape = x_seq.shape
            # [T, N, *] -> [T, N] -> [N, T] -> [N, 1, T]
            x_seq = x_seq.flatten(1).t().unsqueeze(1)

            x_seq = F.pad(x_seq, pad=(self.k - 1, 0))
            x_seq = F.conv1d(x_seq, self.weight.view(1, 1, -1), stride=1)

            x_seq = x_seq.squeeze(1).t().view(x_seq_shape)
            return self.surrogate_function(x_seq + self.bias)

        else:
            raise NotImplementedError(self.backend)

    def extra_repr(self):
        return super().extra_repr() + f', order={self.k}'

class GatedLIFNode(base.MemoryModule):
    def __init__(self, T: int, inplane = None,
                 init_linear_decay = None, init_v_subreset = None, init_tau: float = 0.25, init_v_threshold: float = 0.5, init_conduct: float = 0.5,
                 surrogate_function: Callable = surrogate.Sigmoid(), step_mode='m', backend='torch'):
        """
        * :ref:`訝��뻼API <GatedLIFNode.__init__-cn>`

        .. _GatedLIFNode.__init__-cn:

        :param T: �뿶�뿴閭ι빣
        :type T: int

        :param inplane: 渦볟뀯tensor�쉪��싮걪�빊��귚툖溫양쉰inplane竊뚦닕容섋�ㅴ슴�뵪layer-wise GLIF
        :type inplane: int

        :param init_linear_decay: �넑�뵷鵝띸봇��㎬“�뇧躍멩빊�닜冶뗥�쇽펽訝띹�양쉰弱깁퍡溫ㅴ맏init_v_threshold/(T * 2)
        :type init_linear_decay: float

        :param init_v_subreset: �넑�뵷鵝띶쨳鵝띸뵷�럨�닜冶뗥��
        :type init_v_subreset: float

        :param init_tau: �넑�뵷鵝띷뿶�뿴躍멩빊�쉪�닜冶뗥��
        :type init_tau: float

        :param init_v_threshold: 曄욅퍘�뀇�쉪�삁��쇘뵷�럨�닜冶뗥��
        :type init_v_threshold: float

        :param init_conduct: �넑�뵷鵝띸뵷野쇘럤�닜冶뗥��
        :type init_conduct: float

        :param surrogate_function: �룏�릲鴉졿뮡�뿶�뵪�씎溫←츞�꼮�넳�눦�빊歟�佯��쉪�쎘餓ｅ눦�빊
        :type surrogate_function: Callable

        :param step_mode: 閭θ퓵與▼폀竊뚦룵�뵱�똻 `'m'` (鸚싨��)
        :type step_mode: str

        :param backend: 鵝욜뵪�벆燁띶릮塋���귚툖�릪�쉪 ``step_mode`` �룾�꺗鴉싧를�쐣訝띶릪�쉪�릮塋���귛룾餓ι�싪퓝�돀�뜲 ``self.supported_backends`` �윥�쐦壤볟뎺
            鵝욜뵪�쉪閭θ퓵與▼폀�뵱�똻�쉪�릮塋���귛쑉�뵱�똻�쉪�깄�넻訝뗰펽鵝욜뵪 ``'cupy'`` �릮塋��삸��잌벧���恙ョ쉪��괾ated-LIF�룵�뵱�똻torch
        :type backend: str


        與▼엹�눣鸚꾬폏`GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks <https://openreview.net/forum?id=UmFSx2c4ubT>`
        GLIF訝�����쐣�쉪�넑�뵷鵝띶뢿�빊�꺗�삸�룾耶��쉪竊뚦똿�떖�뼭凉뺝뀯�쉪�뿨�렒楹삥빊���

        * :ref:`API in English <GatedLIFNode.__init__-en>`

        .. _GatedLIFNode.__init__-en:

        :param T: time-step
        :type T: int

        :param inplane: input tensor channel number, default: None(layer-wise GLIF). If set, otherwise(channel-wise GLIF)
        :type inplane: int

        :param init_linear_decay: initial linear-decay constant竊똡efault: init_v_threshold/(T * 2)
        :type init_linear_decay: float

        :param init_v_subreset: initial soft-reset constant
        :type init_v_subreset: float

        :param init_tau: initial exponential-decay constant
        :type init_tau: float

        :param init_v_threshold: initial menbrane potential threshold
        :type init_v_threshold: float

        :param init_conduct: initial conduct
        :type init_conduct: float

        :param surrogate_function: surrogate gradient
        :type surrogate_function: Callable

        :param step_mode: step mode, only support `'m'` (multi-step)
        :type step_mode: str

        :param backend: backend fot this neuron layer, which can be "gemm" or "conv". This option only works for the multi-step mode
        :type backend: str


        Gated LIF neuron refers to `GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks <https://openreview.net/forum?id=UmFSx2c4ubT>`
        All membrane-related parameters are learnable, including the gates.
        """

        assert isinstance(init_tau, float) and init_tau < 1.
        assert isinstance(T, int) and T is not None
        assert isinstance(inplane, int) or inplane is None
        assert (isinstance(init_linear_decay, float) and init_linear_decay < 1.) or init_linear_decay is None
        assert (isinstance(init_v_subreset, float) and init_v_subreset < 1.) or init_v_subreset is None

        assert step_mode == 'm'
        super().__init__()
        self.surrogate_function = surrogate_function
        self.backend = backend
        self.step_mode = step_mode
        self.T = T
        self.register_memory('v', 0.)
        self.register_memory('u', 0.)
        self.channel_wise = inplane is not None
        if self.channel_wise: #channel-wise learnable params
            self.alpha, self.beta, self.gamma = [nn.Parameter(torch.tensor(0.2 * (np.random.rand(inplane) - 0.5), dtype=torch.float)) for i in range(3)]
            self.tau = nn.Parameter(- math.log(1 / init_tau - 1) * torch.ones(inplane, dtype=torch.float))
            self.v_threshold = nn.Parameter(- math.log(1 / init_v_threshold - 1) * torch.ones(inplane, dtype=torch.float))
            init_linear_decay = init_v_threshold / (T * 2) if init_linear_decay is None else init_linear_decay
            self.linear_decay = nn.Parameter(- math.log(1 / init_linear_decay - 1) * torch.ones(inplane, dtype=torch.float))
            init_v_subreset = init_v_threshold if init_v_subreset is None else init_v_subreset
            self.v_subreset = nn.Parameter(- math.log(1 / init_v_subreset - 1) * torch.ones(inplane, dtype=torch.float))
            self.conduct = nn.Parameter(- math.log(1 / init_conduct - 1) * torch.ones((T, inplane), dtype=torch.float))

        else:   #layer-wise learnable params
            self.alpha, self.beta, self.gamma = [nn.Parameter(torch.tensor(0.2 * (np.random.rand() - 0.5), dtype=torch.float)) for i in range(3)]
            self.tau = nn.Parameter(torch.tensor(- math.log(1 / init_tau - 1), dtype=torch.float))
            self.v_threshold = nn.Parameter(torch.tensor(- math.log(1 / init_v_threshold - 1), dtype=torch.float))
            init_linear_decay = init_v_threshold / (T * 2) if init_linear_decay is None else init_linear_decay
            self.linear_decay = nn.Parameter(torch.tensor(- math.log(1 / init_linear_decay - 1), dtype=torch.float))
            init_v_subreset = init_v_threshold if init_v_subreset is None else init_v_subreset
            self.v_subreset = nn.Parameter(torch.tensor(- math.log(1 / init_v_subreset - 1), dtype=torch.float))
            self.conduct = nn.Parameter(- math.log(1 / init_conduct - 1) * torch.ones(T, dtype=torch.float))

    @property
    def supported_backends(self):
        return 'torch'

    def extra_repr(self):
        with torch.no_grad():
            tau = self.tau
            v_subreset = self.v_subreset
            linear_decay = self.linear_decay
            conduct = self.conduct
        return super().extra_repr() + f', tau={tau}' + f', v_subreset={v_subreset}' + f', linear_decay={linear_decay}' + f', conduct={conduct}'

    def neuronal_charge(self, x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, t):
        input = x * (1 - beta * (1 - self.conduct[t].view(1, -1, 1, 1).sigmoid()))
        self.u = ((1 - alpha * (1 - self.tau.view(1, -1, 1, 1).sigmoid())) * self.v \
                  - (1 - alpha) * self.linear_decay.view(1, -1, 1, 1).sigmoid()) \
                 + input

    def neuronal_reset(self, spike, alpha: torch.Tensor, gamma: torch.Tensor):
        self.u = self.u - (1 - alpha * (1 - self.tau.view(1, -1, 1, 1).sigmoid())) * self.v * gamma * spike \
                 - (1 - gamma) * self.v_subreset.view(1, -1, 1, 1).sigmoid() * spike

    def neuronal_fire(self):
        return self.surrogate_function(self.u - self.v_threshold.view(1, -1, 1, 1).sigmoid())

    def multi_step_forward(self, x_seq: torch.Tensor):
        alpha, beta, gamma = self.alpha.view(1, -1, 1, 1).sigmoid(), self.beta.view(1, -1, 1, 1).sigmoid(), self.gamma.view(1, -1, 1, 1).sigmoid()
        y_seq = []
        spike = torch.zeros(x_seq.shape[1:], device=x_seq.device)
        for t in range(self.T):
            self.neuronal_charge(x_seq[t], alpha, beta, t)
            self.neuronal_reset(spike, alpha, gamma)
            spike = self.neuronal_fire()
            self.v = self.u
            y_seq.append(spike)
        return torch.stack(y_seq)


##########################################################################################################
# DSR modules
##########################################################################################################

import torch.distributed as dist

class DSRIFNode(base.MemoryModule):
    def __init__(self, T: int = 20, v_threshold: float = 6., alpha: float = 0.5, v_threshold_training: bool = True,
                 v_threshold_grad_scaling: float = 1.0, v_threshold_lower_bound: float = 0.01, step_mode='m',
                 backend='torch', **kwargs):

        """
        * :ref:`訝��뻼API <DSRIFNode.__init__-cn>`

        .. _DSRIFNode.__init__-cn:

        :param T: �뿶�뿴閭ι빣
        :type T: int

        :param v_threshold: 曄욅퍘�뀇�쉪�삁��쇘뵷�럨�닜冶뗥��
        :type v_threshold: float

        :param alpha: �붂�뵷�삁��쇘쉪煐⒵붂�썱耶�
        :type alpha: float

        :param v_threshold_training: �삸�맔弱녽삁��쇘뵷�럨溫양쉰訝뷴룾耶╊튌�뢿�빊竊뚪퍡溫ㅴ맏`'True'`
        :type v_threshold_training: bool

        :param v_threshold_grad_scaling: 野방붂�뵷�삁��쇘쉪歟�佯�瓦쏂죱煐⒵붂�쉪煐⒵붂�썱耶�
        :type v_threshold_grad_scaling: float

        :param v_threshold_lower_bound: 溫�瀯껇퓝葉뗤릎竊뚪삁��쇘뵷�럨�꺗�룚�댆�쉪���弱뤷��
        :type v_threshold_lower_bound: float

        :param step_mode: 閭θ퓵與▼폀竊뚦룵�뵱�똻 `'m'` (鸚싨��)
        :type step_mode: str

        :param backend: 鵝욜뵪�벆燁띶릮塋���귚툖�릪�쉪 ``step_mode`` �룾�꺗鴉싧를�쐣訝띶릪�쉪�릮塋���귛룾餓ι�싪퓝�돀�뜲 ``self.supported_backends`` �윥�쐦壤볟뎺
            鵝욜뵪�쉪閭θ퓵與▼폀�뵱�똻�쉪�릮塋���귛쑉�뵱�똻�쉪�깄�넻訝뗰펽鵝욜뵪 ``'cupy'`` �릮塋��삸��잌벧���恙ョ쉪��괗SR-IF�룵�뵱�똻torch
        :type backend: str

        與▼엹�눣鸚꾬폏`Training High-Performance Low-Latency Spiking Neural Networks by Differentiation on Spike Representation
         <https://arxiv.org/pdf/2205.00459.pdf>`.


        * :ref:`API in English <DSRIFNode.__init__-en>`

        .. _DSRIFNode.__init__-en:

        :param T: time-step
        :type T: int

        :param v_threshold: initial menbrane potential threshold
        :type v_threshold: float

        :param alpha: the scaling factor for the menbrane potential threshold
        :type alpha: float

        :param v_threshold_training: whether the menbrane potential threshold is trained, default: `'True'`
        :type v_threshold_training: bool

        :param v_threshold_grad_scaling: the scaling factor for the gradient of the menbrane potential threshold
        :type v_threshold_grad_scaling: float

        :param v_threshold_lower_bound: the minimum of the menbrane potential threshold during training
        :type v_threshold_lower_bound: float

        :param step_mode: step mode, only support `'m'` (multi-step)
        :type step_mode: str

        :param backend: backend fot this neuron layer, which can be "gemm" or "conv". This option only works for the multi-step mode
        :type backend: str


        DSR IF neuron refers to `Training High-Performance Low-Latency Spiking Neural Networks by Differentiation on Spike Representation
         <https://arxiv.org/pdf/2205.00459.pdf>`.
        """

        assert isinstance(T, int) and T is not None
        assert isinstance(v_threshold, float) and v_threshold >= v_threshold_lower_bound
        assert isinstance(alpha, float) and alpha > 0.0 and alpha <= 1.0
        assert isinstance(v_threshold_lower_bound, float) and v_threshold_lower_bound > 0.0
        assert step_mode == 'm'

        super().__init__()
        self.backend = backend
        self.step_mode = step_mode
        self.T = T
        if v_threshold_training:
            self.v_threshold = nn.Parameter(torch.tensor(v_threshold))
        else:
            self.v_threshold = torch.tensor(v_threshold)
        self.alpha = alpha
        self.v_threshold_lower_bound = v_threshold_lower_bound
        self.v_threshold_grad_scaling = v_threshold_grad_scaling

    @property
    def supported_backends(self):
        return 'torch'

    def extra_repr(self):
        with torch.no_grad():
            T = self.T
            v_threshold = self.v_threshold
            alpha = self.alpha
            v_threshold_lower_bound = self.v_threshold_lower_bound
            v_threshold_grad_scaling = self.v_threshold_grad_scaling
        return f', T={T}' + f', init_vth={v_threshold}' + f', alpha={alpha}' + f', vth_bound={v_threshold_lower_bound}' + f', vth_g_scale={v_threshold_grad_scaling}'

    def multi_step_forward(self, x_seq: torch.Tensor):
        with torch.no_grad():
            self.v_threshold.copy_(
                F.relu(self.v_threshold - self.v_threshold_lower_bound) + self.v_threshold_lower_bound)
        iffunc = self.DSRIFFunction.apply
        y_seq = iffunc(x_seq, self.T, self.v_threshold, self.alpha, self.v_threshold_grad_scaling)
        return y_seq


    class DSRIFFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, T=10, v_threshold=1.0, alpha=0.5, v_threshold_grad_scaling=1.0):
            ctx.save_for_backward(inp)

            mem_potential = torch.zeros_like(inp[0]).to(inp.device)
            spikes = []

            for t in range(inp.size(0)):
                mem_potential = mem_potential + inp[t]
                spike = ((mem_potential >= alpha * v_threshold).float() * v_threshold).float()
                mem_potential = mem_potential - spike
                spikes.append(spike)
            output = torch.stack(spikes)

            ctx.T = T
            ctx.v_threshold = v_threshold
            ctx.v_threshold_grad_scaling = v_threshold_grad_scaling
            return output

        @staticmethod
        def backward(ctx, grad_output):
            with torch.no_grad():
                inp = ctx.saved_tensors[0]
                T = ctx.T
                v_threshold = ctx.v_threshold
                v_threshold_grad_scaling = ctx.v_threshold_grad_scaling

                input_rate_coding = torch.mean(inp, 0)
                grad_output_coding = torch.mean(grad_output, 0) * T

                input_grad = grad_output_coding.clone()
                input_grad[(input_rate_coding < 0) | (input_rate_coding > v_threshold)] = 0
                input_grad = torch.stack([input_grad for _ in range(T)]) / T

                v_threshold_grad = grad_output_coding.clone()
                v_threshold_grad[input_rate_coding <= v_threshold] = 0
                v_threshold_grad = torch.sum(v_threshold_grad) * v_threshold_grad_scaling
                if v_threshold_grad.is_cuda and torch.cuda.device_count() != 1:
                    try:
                        dist.all_reduce(v_threshold_grad, op=dist.ReduceOp.SUM)
                    except:
                        raise RuntimeWarning(
                            'Something wrong with the `all_reduce` operation when summing up the gradient of v_threshold from multiple gpus. Better check the gpu status and try DistributedDataParallel.')

                return input_grad, None, v_threshold_grad, None, None


class DSRLIFNode(base.MemoryModule):
    def __init__(self, T: int = 20, v_threshold: float = 1., tau: float = 2.0, delta_t: float = 0.05,
                 alpha: float = 0.3, v_threshold_training: bool = True,
                 v_threshold_grad_scaling: float = 1.0, v_threshold_lower_bound: float = 0.1, step_mode='m',
                 backend='torch', **kwargs):

        """
        * :ref:`訝��뻼API <DSRLIFNode.__init__-cn>`

        .. _DSRLIFNode.__init__-cn:

        :param T: �뿶�뿴閭ι빣
        :type T: int

        :param v_threshold: 曄욅퍘�뀇�쉪�삁��쇘뵷�럨�닜冶뗥��
        :type v_threshold: float

        :param tau: �넑�뵷鵝띷뿶�뿴躍멩빊
        :type tau: float

        :param delta_t: 野밧쒜�늽�뼶葉뗥숱凉뤹쉪LIF與▼엹瓦쏂죱獵삥븺�뙑�쉪閭ι빣
        :type delta_t: float

        :param alpha: �붂�뵷�삁��쇘쉪煐⒵붂�썱耶�
        :type alpha: float

        :param v_threshold_training: �삸�맔弱녽삁��쇘뵷�럨溫양쉰訝뷴룾耶╊튌�뢿�빊竊뚪퍡溫ㅴ맏`'True'`
        :type v_threshold_training: bool

        :param v_threshold_grad_scaling: 野방붂�뵷�삁��쇘쉪歟�佯�瓦쏂죱煐⒵붂�쉪煐⒵붂�썱耶�
        :type v_threshold_grad_scaling: float

        :param v_threshold_lower_bound: 溫�瀯껇퓝葉뗤릎竊뚪삁��쇘뵷�럨�꺗�룚�댆�쉪���弱뤷��
        :type v_threshold_lower_bound: float

        :param step_mode: 閭θ퓵與▼폀竊뚦룵�뵱�똻 `'m'` (鸚싨��)
        :type step_mode: str

        :param backend: 鵝욜뵪�벆燁띶릮塋���귚툖�릪�쉪 ``step_mode`` �룾�꺗鴉싧를�쐣訝띶릪�쉪�릮塋���귛룾餓ι�싪퓝�돀�뜲 ``self.supported_backends`` �윥�쐦壤볟뎺
            鵝욜뵪�쉪閭θ퓵與▼폀�뵱�똻�쉪�릮塋���귛쑉�뵱�똻�쉪�깄�넻訝뗰펽鵝욜뵪 ``'cupy'`` �릮塋��삸��잌벧���恙ョ쉪��괗SR-IF�룵�뵱�똻torch
        :type backend: str

        與▼엹�눣鸚꾬폏`Training High-Performance Low-Latency Spiking Neural Networks by Differentiation on Spike Representation
         <https://arxiv.org/pdf/2205.00459.pdf>`.


        * :ref:`API in English <DSRLIFNode.__init__-en>`

        .. _DSRLIFNode.__init__-en:

        :param T: time-step
        :type T: int

        :param v_threshold: initial menbrane potential threshold
        :type v_threshold: float

        :param tau: membrane time constant
        :type tau: float

        :param delta_t: discretization step for discretizing the ODE version of the LIF model
        :type delta_t: float

        :param alpha: the scaling factor for the menbrane potential threshold
        :type alpha: float

        :param v_threshold_training: whether the menbrane potential threshold is trained, default: `'True'`
        :type v_threshold_training: bool

        :param v_threshold_grad_scaling: the scaling factor for the gradient of the menbrane potential threshold
        :type v_threshold_grad_scaling: float

        :param v_threshold_lower_bound: the minimum of the menbrane potential threshold during training
        :type v_threshold_lower_bound: float

        :param step_mode: step mode, only support `'m'` (multi-step)
        :type step_mode: str

        :param backend: backend fot this neuron layer, which can be "gemm" or "conv". This option only works for the multi-step mode
        :type backend: str


        DSR LIF neuron refers to `Training High-Performance Low-Latency Spiking Neural Networks by Differentiation on Spike Representation
         <https://arxiv.org/pdf/2205.00459.pdf>`.
        """

        assert isinstance(T, int) and T is not None
        assert isinstance(v_threshold, float) and v_threshold >= v_threshold_lower_bound
        assert isinstance(alpha, float) and alpha > 0.0 and alpha <= 1.0
        assert isinstance(v_threshold_lower_bound, float) and v_threshold_lower_bound > 0.0
        assert step_mode == 'm'

        super().__init__()
        self.backend = backend
        self.step_mode = step_mode
        self.T = T
        if v_threshold_training:
            self.v_threshold = nn.Parameter(torch.tensor(v_threshold))
        else:
            self.v_threshold = torch.tensor(v_threshold)
        self.tau = tau
        self.delta_t = delta_t
        self.alpha = alpha
        self.v_threshold_lower_bound = v_threshold_lower_bound
        self.v_threshold_grad_scaling = v_threshold_grad_scaling

    @property
    def supported_backends(self):
        return 'torch'

    def extra_repr(self):
        with torch.no_grad():
            T = self.T
            v_threshold = self.v_threshold
            tau = self.tau
            delta_t = self.delta_t
            alpha = self.alpha
            v_threshold_lower_bound = self.v_threshold_lower_bound
            v_threshold_grad_scaling = self.v_threshold_grad_scaling
        return f', T={T}' + f', init_vth={v_threshold}' + f', tau={tau}' + f', dt={delta_t}' + f', alpha={alpha}' + \
               f', vth_bound={v_threshold_lower_bound}' + f', vth_g_scale={v_threshold_grad_scaling}'

    def multi_step_forward(self, x_seq: torch.Tensor):
        with torch.no_grad():
            self.v_threshold.copy_(
                F.relu(self.v_threshold - self.v_threshold_lower_bound) + self.v_threshold_lower_bound)
        liffunc = self.DSRLIFFunction.apply
        y_seq = liffunc(x_seq, self.T, self.v_threshold, self.tau, self.delta_t, self.alpha,
                        self.v_threshold_grad_scaling)
        return y_seq

    @classmethod
    def weight_rate_spikes(cls, data, T, tau, delta_t):
        chw = data.size()[1:]
        data_reshape = data.view(T, -1, *chw).permute(list(range(1, len(chw) + 2)) + [0])
        weight = torch.tensor([math.exp(-1 / tau * (delta_t * T - ii * delta_t)) for ii in range(1, T + 1)]).to(
            data_reshape.device)
        return (weight * data_reshape).sum(dim=len(chw) + 1) / weight.sum()

    class DSRLIFFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, T, v_threshold, tau, delta_t=0.05, alpha=0.3, v_threshold_grad_scaling=1.0):
            ctx.save_for_backward(inp)

            chw = inp.size()[1:]
            input_reshape = inp.view(T, -1, *chw)
            mem_potential = torch.zeros_like(input_reshape.size(1), *chw).to(input_reshape.device)
            beta = math.exp(-delta_t / tau)

            spikes = []
            for t in range(input_reshape.size(0)):
                mem_potential = beta * mem_potential + (1 - beta) * input_reshape[t]
                spike = ((mem_potential >= alpha * v_threshold).float() * v_threshold).float()
                mem_potential = mem_potential - spike
                spikes.append(spike / delta_t)
            output = torch.cat(spikes, 0)

            ctx.T = T
            ctx.v_threshold = v_threshold
            ctx.tau = tau
            ctx.delta_t = delta_t
            ctx.v_threshold_grad_scaling = v_threshold_grad_scaling
            return output

        
        @staticmethod
        def backward(ctx, grad_output):
            inp = ctx.saved_tensors[0]
            T = ctx.T
            v_threshold = ctx.v_threshold
            delta_t = ctx.delta_t
            tau = ctx.tau
            # v_threshold_grad_scaling = ctx.v_threshold_grad_scaling

            input_rate_coding = DSRLIFNode.weight_rate_spikes(inp, T, tau, delta_t)
            grad_output_coding = DSRLIFNode.weight_rate_spikes(grad_output, T, tau, delta_t) * T

            indexes = (input_rate_coding > 0) & (input_rate_coding < v_threshold / delta_t * tau)
            input_grad = torch.zeros_like(grad_output_coding)
            input_grad[indexes] = grad_output_coding[indexes].clone() / tau
            input_grad = torch.cat([input_grad for _ in range(T)], 0) / T

            v_threshold_grad = grad_output_coding.clone()
            v_threshold_grad[input_rate_coding <= v_threshold / delta_t * tau] = 0
            v_threshold_grad = torch.sum(v_threshold_grad) * delta_t 
            if v_threshold_grad.is_cuda and torch.cuda.device_count() != 1:
                try:
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(v_threshold_grad, op=dist.ReduceOp.SUM)
                except:
                    raise RuntimeWarning('Something wrong with the `all_reduce` operation when summing up the gradient of v_threshold from multiple gpus. Better check the gpu status and try DistributedDataParallel.')

            return input_grad, None, v_threshold_grad, None, None, None, None


##########################################################################################################
# OTTT modules
##########################################################################################################

class OTTTLIFNode(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: Optional[float] = None, surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = True, step_mode='s', backend='torch', store_v_seq: bool = False):
        """
        * :ref:`API in English <OTTTLIFNode.__init__-en>`

        .. _OTTTLIFNode.__init__-cn:

        :param tau: �넑�뵷鵝띷뿶�뿴躍멩빊
        :type tau: float

        :param decay_input: 渦볟뀯�삸�맔阿잋폏�뢿訝롨“�뇧
        :type decay_input: bool

        :param v_threshold: 曄욅퍘�뀇�쉪�삁��쇘뵷�럨
        :type v_threshold: float

        :param v_reset: 曄욅퍘�뀇�쉪�뇥營��뵷�럨��귛쫩�옖訝띴맏 ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇥營�訝� ``v_reset``竊�
            倻귝옖溫양쉰訝� ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇧�렮 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: �룏�릲鴉졿뮡�뿶�뵪�씎溫←츞�꼮�넳�눦�빊歟�佯��쉪�쎘餓ｅ눦�빊
        :type surrogate_function: Callable

        :param detach_reset: �삸�맔弱냨eset瓦뉒쮮�쉪溫←츞�쎗�늽獵삠�귟�ε뢿�빊�쑉�쑍與▼쓼訝�訝띹돈鵝쒐뵪竊뚥퍎訝뷰퓷�똻餓ｇ쟻瀯잋����뚥퓷�븰
        :type detach_reset: bool

        :param step_mode: 閭θ퓵與▼폀竊뚥맏雅녵퓷瑥곭쪥瀯뤷뀇�쉪�샑耶섇뜝�뵪弱륅펽餓끻룾餓δ맏 `'s'` (�뜒閭�)
        :type step_mode: str

        :param backend: 鵝욜뵪�벆燁띶릮塋���귚툖�릪�쉪 ``step_mode`` �룾�꺗鴉싧를�쐣訝띶릪�쉪�릮塋���귛룾餓ι�싪퓝�돀�뜲 ``self.supported_backends`` �윥�쐦壤볟뎺
            鵝욜뵪�쉪閭θ퓵與▼폀�뵱�똻�쉪�릮塋���귛쑉�뵱�똻�쉪�깄�넻訝뗰펽鵝욜뵪 ``'cupy'`` �릮塋��삸��잌벧���恙ョ쉪
        :type backend: str

        :param store_v_seq: �쑉鵝욜뵪 ``step_mode = 'm'`` �뿶竊뚨퍢訝� ``shape = [T, N, *]`` �쉪渦볟뀯�릮竊뚧삸�맔岳앭춼訝��뿴瓦뉒쮮�쉪 ``shape = [T, N, *]``
            �쉪�릢訝ゆ뿶�뿴閭η쉪�뵷�럨��� ``self.v_seq`` ��귟�양쉰訝� ``False`` �뿶溫←츞若뚧닇�릮�룵岳앯븰����릮訝�訝ゆ뿶�댗�쉪�뵷�럨竊뚦뜵 ``shape = [N, *]`` �쉪 ``self.v`` ���
            ��싧만溫양쉰�닇 ``False`` 竊뚦룾餓θ뒄�쐛�냵耶�
        :type store_v_seq: bool

        曄욅퍘�뀇與▼엹�눣鸚꾬폏`Online Training Through Time for Spiking Neural Networks <https://arxiv.org/pdf/2210.04195.pdf>`
        與▼엹閭ｅ릲鴉졿뮡�뭽Leaky Integrate-and-Fire曄욅퍘�뀇�쎑�릪竊쏁뵪雅롩쉹�뿶�뿴�쑉瀛욤��瀯�


        * :ref:`訝��뻼API <OTTTLIFNode.__init__-cn>`

        .. _OTTTLIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

        :param v_threshold: threshold of this neurons layer
        :type v_threshold: float

        :param v_reset: reset voltage of this neurons layer. If not ``None``, the neuron's voltage will be set to ``v_reset``
            after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :type v_reset: Optional[float]

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset in backward. this parameter does not take any effect in
            the module, and is retained solely for code consistency
        :type detach_reset: bool

        :param step_mode: the step mode, which can solely be `s` (single-step) to guarantee the memory-efficient computation
        :type step_mode: str

        :param backend: backend fot this neurons layer. Different ``step_mode`` may support for different backends. The user can
        print ``self.supported_backends`` and check what backends are supported by the current ``step_mode``. If supported,
        using ``'cupy'`` backend will have the fastest training speed
        :type backend: str

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption
        :type store_v_seq: bool

        OTTT LIF neuron refers to `Online Training Through Time for Spiking Neural Networks <https://arxiv.org/pdf/2210.04195.pdf>`
        The forward propagation is the same as the Leaky Integrate-and-Fire neuron; used for online training through time

        """

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        assert step_mode == 's', "Please use single-step mode to enable memory-efficient training."
        """
        �넑�뵷鵝띶컛�쑉�뎺�릲鴉졿뮡瓦뉒쮮訝��뇥�뼭�쇉溫겻맏煐볟춼竊뚥빳�뵱�똻鸚싧뜞�늽躍껃폀溫�瀯껆쉪�깄�넻訝뗤퓷�븰岳→겘�쑉�릢�뿶�댗瓦쏂죱鸚싨А�룏�릲鴉졿뮡

        membrane potential will be registered as buffer during forward, to support multiple backpropagation for all time steps with 
        reserved informtion under distributed training on multiple GPUs
        """
        self._memories.pop('v')

    def reset(self):
        super().reset()
        if hasattr(self, 'v'):
            del self.v
        if hasattr(self, 'trace'):
            del self.trace

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v.detach()

        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_decay_input(x, self.v, self.v_reset, self.tau)

        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_no_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_no_decay_input(x, self.v, self.v_reset, self.tau)

    @staticmethod
    @torch.jit.script
    def track_trace(spike: torch.Tensor, trace: torch.Tensor, tau: float):
        with torch.no_grad():
            trace = trace * (1. - 1. / tau) + spike
        return trace


    def single_step_forward(self, x: torch.Tensor):
        """
        溫�瀯껅뿶竊뚩풏�눣�꼮�넳�뭽瓦뱄폑�렓�릤�뿶竊뚩풏�눣�꼮�넳
        溫�瀯껅뿶���誤곩컛�릮瀯��뢿�빊與▼쓼�뵪layer.py訝�若싦퉱�쉪GradwithTrace瓦쏂죱�똿獒낉펽�졊�뜮瓦배�←츞歟�佯�
        
        output spike and trace during training; output spike during inference
        during training, successive parametric modules shoule be wrapped by GradwithTrace defined in layer.py, to calculate gradients with traces
        """

        if not hasattr(self, 'v'):
            if self.v_reset is None:
                self.register_buffer('v', torch.zeros_like(x))
            else:
                self.register_buffer('v', torch.ones_like(x) * self.v_reset)

        if self.training:
            if not hasattr(self, 'trace'):
                self.register_buffer('trace', torch.zeros_like(x))
    
            if self.backend == 'torch':
                self.neuronal_charge(x)
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)

                self.trace = self.track_trace(spike, self.trace, self.tau)

                return [spike, self.trace]
            else:
                raise ValueError(self.backend)
        else:
            if self.v_reset is None:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_decay_input(x, self.v,
                                                                                             self.v_threshold, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_no_decay_input(x, self.v,
                                                                                                self.v_threshold,
                                                                                                self.tau)
            else:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_decay_input(x, self.v,
                                                                                             self.v_threshold,
                                                                                             self.v_reset, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_no_decay_input(x, self.v,
                                                                                                self.v_threshold,
                                                                                                self.v_reset,
                                                                                                self.tau)
            return spike


##########################################################################################################
# SLTT modules
##########################################################################################################

class SLTTLIFNode(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: Optional[float] = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = True, step_mode='s', backend='torch', store_v_seq: bool = False):
        """
        * :ref:`API in English <SLTTLIFNode.__init__-en>`

        .. _SLTTLIFNode.__init__-cn:

        :param tau: �넑�뵷鵝띷뿶�뿴躍멩빊
        :type tau: float

        :param decay_input: 渦볟뀯�삸�맔阿잋폏�뢿訝롨“�뇧
        :type decay_input: bool

        :param v_threshold: 曄욅퍘�뀇�쉪�삁��쇘뵷�럨
        :type v_threshold: float

        :param v_reset: 曄욅퍘�뀇�쉪�뇥營��뵷�럨��귛쫩�옖訝띴맏 ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇥營�訝� ``v_reset``竊�
            倻귝옖溫양쉰訝� ``None``竊뚦퐪曄욅퍘�뀇�뇢�붂�꼮�넳�릮竊뚨뵷�럨鴉싪˙�뇧�렮 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: �룏�릲鴉졿뮡�뿶�뵪�씎溫←츞�꼮�넳�눦�빊歟�佯��쉪�쎘餓ｅ눦�빊
        :type surrogate_function: Callable

        :param detach_reset: �삸�맔弱냨eset瓦뉒쮮�쉪溫←츞�쎗�늽獵삠�귟�ε뢿�빊�쑉�쑍與▼쓼訝�訝띹돈鵝쒐뵪竊뚥퍎訝뷰퓷�똻餓ｇ쟻瀯잋����뚥퓷�븰
        :type detach_reset: bool

        :param step_mode: 閭θ퓵與▼폀竊뚥맏雅녵퓷瑥곭쪥瀯뤷뀇�쉪�샑耶섇뜝�뵪弱륅펽餓끻룾餓δ맏 `'s'` (�뜒閭�)
        :type step_mode: str

        :param backend: 鵝욜뵪�벆燁띶릮塋���귚툖�릪�쉪 ``step_mode`` �룾�꺗鴉싧를�쐣訝띶릪�쉪�릮塋���귛룾餓ι�싪퓝�돀�뜲 ``self.supported_backends`` �윥�쐦壤볟뎺
            鵝욜뵪�쉪閭θ퓵與▼폀�뵱�똻�쉪�릮塋���귛쑉�뵱�똻�쉪�깄�넻訝뗰펽鵝욜뵪 ``'cupy'`` �릮塋��삸��잌벧���恙ョ쉪
        :type backend: str

        :param store_v_seq: �쑉鵝욜뵪 ``step_mode = 'm'`` �뿶竊뚨퍢訝� ``shape = [T, N, *]`` �쉪渦볟뀯�릮竊뚧삸�맔岳앭춼訝��뿴瓦뉒쮮�쉪 ``shape = [T, N, *]``
            �쉪�릢訝ゆ뿶�뿴閭η쉪�뵷�럨��� ``self.v_seq`` ��귟�양쉰訝� ``False`` �뿶溫←츞若뚧닇�릮�룵岳앯븰����릮訝�訝ゆ뿶�댗�쉪�뵷�럨竊뚦뜵 ``shape = [N, *]`` �쉪 ``self.v`` ���
            ��싧만溫양쉰�닇 ``False`` 竊뚦룾餓θ뒄�쐛�냵耶�
        :type store_v_seq: bool

        曄욅퍘�뀇與▼엹�눣鸚꾬폏`Towards Memory- and Time-Efficient Backpropagation for Training Spiking Neural Networks
        <https://arxiv.org/pdf/2302.14311.pdf>`.與▼엹閭ｅ릲鴉졿뮡�뭽Leaky Integrate-and-Fire曄욅퍘�뀇�쎑�릪.


        * :ref:`訝��뻼API <SLTTLIFNode.__init__-cn>`

        .. _SLTTLIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

        :param v_threshold: threshold of this neurons layer
        :type v_threshold: float

        :param v_reset: reset voltage of this neurons layer. If not ``None``, the neuron's voltage will be set to ``v_reset``
            after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :type v_reset: Optional[float]

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset in backward. this parameter does not take any effect in
            the module, and is retained solely for code consistency
        :type detach_reset: bool

        :param step_mode: the step mode, which can solely be `s` (single-step) to guarantee the memory-efficient computation
        :type step_mode: str

        :param backend: backend fot this neurons layer. Different ``step_mode`` may support for different backends. The user can
        print ``self.supported_backends`` and check what backends are supported by the current ``step_mode``. If supported,
        using ``'cupy'`` backend will have the fastest training speed
        :type backend: str

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption
        :type store_v_seq: bool

        SLTT LIF neuron refers to `Towards Memory- and Time-Efficient Backpropagation for Training Spiking Neural Networks
        <https://arxiv.org/pdf/2302.14311.pdf>`. The forward propagation is the same as the Leaky Integrate-and-Fire neuron's.

        """

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        assert step_mode == 's', "Please use single-step mode to enable memory-efficient training."
        self._memories.pop('v')

    def reset(self):
        super().reset()
        if hasattr(self, 'v'):
            del self.v

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v.detach()

        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_decay_input(x, self.v, self.v_reset, self.tau)

        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_no_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_no_decay_input(x, self.v, self.v_reset, self.tau)

    def single_step_forward(self, x: torch.Tensor):

        if not hasattr(self, 'v'):
            if self.v_reset is None:
                self.register_buffer('v', torch.zeros_like(x))
            else:
                self.register_buffer('v', torch.ones_like(x) * self.v_reset)

        if self.training:
            if self.backend == 'torch':
                self.neuronal_charge(x)
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                return spike
            else:
                raise ValueError(self.backend)
        else:
            if self.v_reset is None:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_decay_input(x, self.v,
                                                                                             self.v_threshold, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_no_decay_input(x, self.v,
                                                                                                self.v_threshold,
                                                                                                self.tau)
            else:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_decay_input(x, self.v,
                                                                                             self.v_threshold,
                                                                                             self.v_reset, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_no_decay_input(x, self.v,
                                                                                                self.v_threshold,
                                                                                                self.v_reset,
                                                                                                self.tau)
            return spike


##########################################################################################################
# Current-based LIF (CLIF) modules
##########################################################################################################

class CLIFNode(BaseNode):
    def __init__(self, c_decay: float = 0.5, v_decay: float = 0.75, v_threshold: float = 0.5,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Rect()):

        super().__init__(v_threshold, v_reset, surrogate_function)

        self.register_memory('c', 0.)

        self.c_decay = c_decay
        self.v_decay = v_decay

    def neuronal_charge(self, x: torch.Tensor):
        self.c = self.c * self.c_decay + x
        self.v = self.v * self.v_decay + self.c

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.c_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        spike_seq = []

        for t in range(T):
            spike = self.single_step_forward(x_seq[t])
            spike_seq.append(spike)

        return torch.stack(spike_seq)

    def c_float_to_tensor(self, c: torch.Tensor):
        if isinstance(self.c, float):
            c_init = self.c
            self.c = torch.full_like(c.data, fill_value=c_init)


##########################################################################################################
# Noisy modules for exploration of RL
##########################################################################################################

"""Generate colored noise."""

from typing import Union, Iterable, Optional
from numpy import sqrt, newaxis, integer
from numpy.fft import irfft, rfftfreq
from numpy.random import default_rng, Generator, RandomState
from numpy import sum as npsum


def powerlaw_psd_gaussian(
        exponent: float, 
        size: Union[int, Iterable[int]], 
        fmin: float = 0.0, 
        random_state: Optional[Union[int, Generator, RandomState]] = None
    ):
    """Gaussian (1/f)**beta noise.

    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

    Normalised to unit variance

    Parameters:
    -----------

    exponent : float
        The power-spectrum of the generated noise is proportional to

        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2

        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.

    size : Union[int, Iterable[int]]
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.

    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. 
        
        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.

    random_state :  int, numpy.integer, numpy.random.Generator, numpy.random.RandomState, 
                    optional
        Optionally sets the state of NumPy's underlying random number generator.
        Integer-compatible values or None are passed to np.random.default_rng.
        np.random.RandomState or np.random.Generator are used directly.
        Default: None.

    Returns
    -------
    out : array
        The samples.


    Examples:
    ---------

    # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """
    
    # Make sure size is a list so we can iterate it and assign to it.
    if isinstance(size, (integer, int)):
        size = [size]
    elif isinstance(size, Iterable):
        size = list(size)
    else:
        raise ValueError("Size must be of type int or Iterable[int]")
    
    # The number of samples in each time series
    samples = size[-1]
    
    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples) # type: ignore # mypy 1.5.1 has problems here 
    
    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1./samples) # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")
    
    # Build scaling factors for all frequencies
    s_scale = f    
    ix   = npsum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)
    
    # Calculate theoretical output standard deviation from scaling
    w      = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2. # correct f = +-0.5
    sigma = 2 * sqrt(npsum(w**2)) / samples
    
    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale     = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]
    
    # prepare random number generator
    normal_dist = _get_normal_distribution(random_state)

    # Generate scaled random power + phase
    sr = normal_dist(scale=s_scale, size=size)
    si = normal_dist(scale=s_scale, size=size)
    
    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si[..., -1] = 0
        sr[..., -1] *= sqrt(2)    # Fix magnitude
    
    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0
    sr[..., 0] *= sqrt(2)    # Fix magnitude
    
    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si
    
    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma
    
    return y


def _get_normal_distribution(random_state: Optional[Union[int, Generator, RandomState]]):
    normal_dist = None
    if isinstance(random_state, (integer, int)) or random_state is None:
        random_state = default_rng(random_state)
        normal_dist = random_state.normal
    elif isinstance(random_state, (Generator, RandomState)):
        normal_dist = random_state.normal
    else:
        raise ValueError(
            "random_state must be one of integer, numpy.random.Generator, "
            "numpy.random.Randomstate"
        )
    return normal_dist

class NoisyBaseNode(nn.Module, base.MultiStepModule):
    def __init__(self, num_node, is_training: bool = True, T: int = 5, sigma_init: float = 0.5, 
                 beta: float = 0.0, v_threshold: float = 0.5, v_reset: Optional[float] = 0.,
                 surrogate_function: Callable = surrogate.Rect()):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        super().__init__()

        self.num_node = num_node
        self.is_training = is_training
        self.T = T
        self.beta = beta

        self.sigma_v = sigma_init / math.sqrt(num_node)
        self.cn_v = None

        self.sigma_s = sigma_init / math.sqrt(num_node)
        self.cn_s = None

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.v_reset is None:
            self.v = self.v - spike * self.v_threshold
        else:
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def init_tensor(self, data: torch.Tensor):
        self.v = torch.full_like(data, fill_value=self.v_reset)

    def forward(self, x_seq: torch.Tensor):
        self.init_tensor(x_seq[0].data)
        
        y = []

        if self.is_training:
            if self.cn_v is None or self.cn_s is None:
                self.noise_step += 1

            for t in range(self.T):      
                if self.cn_v is None:
                    self.neuronal_charge(x_seq[t] + self.sigma_v * self.eps_v_seq[self.noise_step][t].to(x_seq.device))
                else:
                    self.neuronal_charge(x_seq[t] + self.sigma_v * self.cn_v[:, t])
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                if self.cn_s is None:
                    spike = spike + self.sigma_s * self.eps_s_seq[self.noise_step][t].to(x_seq.device)
                else:
                    spike = spike + self.sigma_s * self.cn_s[:, t]
                y.append(spike)
            
        else:
            for t in range(self.T):
                self.neuronal_charge(x_seq[t])
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                y.append(spike)

        return torch.stack(y)
        
    def reset_noise(self, num_rl_step):
        eps_shape = [self.num_node, num_rl_step * self.T]
        per_order = [1, 2, 0]
        # (nodes, steps * T) -> (nodes, steps, T) -> (steps, T, nodes)
        self.eps_v_seq = torch.FloatTensor(powerlaw_psd_gaussian(self.beta, eps_shape).reshape(self.num_node, num_rl_step, self.T)).permute(per_order)
        self.eps_s_seq = torch.FloatTensor(powerlaw_psd_gaussian(self.beta, eps_shape).reshape(self.num_node, num_rl_step, self.T)).permute(per_order)
        self.noise_step = -1

    def get_colored_noise(self):
        cn = [self.eps_v_seq[self.noise_step], self.eps_s_seq[self.noise_step]]
        return torch.cat(cn, dim=1)

    def load_colored_noise(self, cn):
        self.cn_v = cn[:, :, :self.num_node]
        self.cn_s = cn[:, :, self.num_node:]

    def cancel_load(self):
        self.cn_v = None
        self.cn_s = None


class NoisyCLIFNode(NoisyBaseNode):
    def __init__(self, num_node, c_decay: float = 0.5, v_decay: float = 0.75, is_training: bool = True, 
                 T: int = 5, sigma_init: float = 0.5, beta: float = 0.0, v_threshold: float = 0.5, 
                 v_reset: Optional[float] = 0., surrogate_function: Callable = surrogate.Rect()):
        super().__init__(num_node, is_training, T, sigma_init, beta, v_threshold, 
                         v_reset, surrogate_function)

        self.c_decay = c_decay
        self.v_decay = v_decay

    def neuronal_charge(self, x: torch.Tensor):
        self.c = self.c * self.c_decay + x
        self.v = self.v * self.v_decay + self.c

    def init_tensor(self, data: torch.Tensor):
        self.c = torch.full_like(data, fill_value=0.0)
        self.v = torch.full_like(data, fill_value=self.v_reset)


##########################################################################################################
# Inter-Layer Connections (ILC) modules for population-coded spiking actor network
##########################################################################################################

class ILCBaseNode(nn.Module, base.MultiStepModule):
    def __init__(self, act_dim, dec_pop_dim, v_threshold: float = 1.0, v_reset: Optional[float] = 0.,
                 surrogate_function: Callable = surrogate.Rect()):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        super().__init__()

        self.act_dim = act_dim
        self.out_pop_dim = act_dim * dec_pop_dim
        self.dec_pop_dim = dec_pop_dim

        self.conn = nn.Conv1d(act_dim, self.out_pop_dim, dec_pop_dim, groups=act_dim)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.v_reset is None:
            self.v = self.v - spike * self.v_threshold
        else:
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def init_tensor(self, data: torch.Tensor):
        self.v = torch.full_like(data, fill_value=self.v_reset)

    def forward(self, x_seq: torch.Tensor):
        self.init_tensor(x_seq[0].data)

        T = x_seq.shape[0]
        spike_seq = []

        for t in range(T):
            self.neuronal_charge(x_seq[t])
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            spike_seq.append(spike)
            if t < T - 1:
                x_seq[t + 1] = x_seq[t + 1] + self.conn(spike.view(-1, self.act_dim, self.dec_pop_dim)).view(-1, self.out_pop_dim)

        return torch.stack(spike_seq)


class ILCCLIFNode(ILCBaseNode):
    def __init__(self, act_dim, dec_pop_dim, c_decay: float = 0.5, v_decay: float = 0.75,
                 v_threshold: float = 0.5, v_reset: Optional[float] = 0.,
                 surrogate_function: Callable = surrogate.Rect()):

        super().__init__(act_dim, dec_pop_dim, v_threshold, v_reset, surrogate_function)

        self.c_decay = c_decay
        self.v_decay = v_decay

    def neuronal_charge(self, x: torch.Tensor):
        self.c = self.c * self.c_decay + x
        self.v = self.v * self.v_decay + self.c

    def init_tensor(self, data: torch.Tensor):
        self.c = torch.full_like(data, fill_value=0.0)
        self.v = torch.full_like(data, fill_value=self.v_reset)


class ILCLIFNode(ILCBaseNode):
    def __init__(self, act_dim, dec_pop_dim, v_decay: float = 0.75,
                 v_threshold: float = 1.0, v_reset: Optional[float] = 0.,
                 surrogate_function: Callable = surrogate.Rect()):

        super().__init__(act_dim, dec_pop_dim, v_threshold, v_reset, surrogate_function)

        self.v_decay = v_decay

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v * self.v_decay + x


class ILCIFNode(ILCBaseNode):
    def __init__(self, act_dim, dec_pop_dim, v_threshold: float = 1.0, v_reset: Optional[float] = 0.,
                 surrogate_function: Callable = surrogate.Rect()):

        super().__init__(act_dim, dec_pop_dim, v_threshold, v_reset, surrogate_function)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x


##########################################################################################################
# Noisy modules with Inter-Layer Connections (ILC)
##########################################################################################################

class NoisyILCBaseNode(nn.Module, base.MultiStepModule):
    def __init__(self, act_dim, dec_pop_dim, is_training: bool = True, T: int = 5, 
                 sigma_init: float = 0.5, beta: float = 0.0, v_threshold: float = 1.0, 
                 v_reset: Optional[float] = 0., surrogate_function: Callable = surrogate.Rect()):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        super().__init__()

        self.act_dim = act_dim
        self.num_node = act_dim * dec_pop_dim
        self.dec_pop_dim = dec_pop_dim

        self.conn = nn.Conv1d(act_dim, self.num_node, dec_pop_dim, groups=act_dim)

        self.is_training = is_training
        self.T = T
        self.beta = beta

        self.sigma_v = sigma_init / math.sqrt(self.num_node)
        self.cn_v = None

        self.sigma_s = sigma_init / math.sqrt(self.num_node)
        self.cn_s = None

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.v_reset is None:
            self.v = self.v - spike * self.v_threshold
        else:
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def init_tensor(self, data: torch.Tensor):
        self.v = torch.full_like(data, fill_value=self.v_reset)

    def forward(self, x_seq: torch.Tensor):
        self.init_tensor(x_seq[0].data)

        y = []

        if self.is_training:
            if self.cn_v is None or self.cn_s is None:
                self.noise_step += 1

            for t in range(self.T):      
                if self.cn_v is None:
                    self.neuronal_charge(x_seq[t] + self.sigma_v * self.eps_v_seq[self.noise_step][t].to(x_seq.device))
                else:
                    self.neuronal_charge(x_seq[t] + self.sigma_v * self.cn_v[:, t])
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                if self.cn_s is None:
                    spike = spike + self.sigma_s * self.eps_s_seq[self.noise_step][t].to(x_seq.device)
                else:
                    spike = spike + self.sigma_s * self.cn_s[:, t]
                y.append(spike)

                if t < self.T - 1:
                    x_seq[t + 1] = x_seq[t + 1] + self.conn(spike.view(-1, self.act_dim, self.dec_pop_dim)).view(-1, self.num_node)
            
        else:
            for t in range(self.T):
                self.neuronal_charge(x_seq[t])
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                y.append(spike)

                if t < self.T - 1:
                    x_seq[t + 1] = x_seq[t + 1] + self.conn(spike.view(-1, self.act_dim, self.dec_pop_dim)).view(-1, self.num_node)

        return torch.stack(y)
        
    def reset_noise(self, num_rl_step):
        eps_shape = [self.num_node, num_rl_step * self.T]
        per_order = [1, 2, 0]
        # (nodes, steps * T) -> (nodes, steps, T) -> (steps, T, nodes)
        self.eps_v_seq = torch.FloatTensor(powerlaw_psd_gaussian(self.beta, eps_shape).reshape(self.num_node, num_rl_step, self.T)).permute(per_order)
        self.eps_s_seq = torch.FloatTensor(powerlaw_psd_gaussian(self.beta, eps_shape).reshape(self.num_node, num_rl_step, self.T)).permute(per_order)
        self.noise_step = -1

    def get_colored_noise(self):
        cn = [self.eps_v_seq[self.noise_step], self.eps_s_seq[self.noise_step]]
        return torch.cat(cn, dim=1)

    def load_colored_noise(self, cn):
        self.cn_v = cn[:, :, :self.num_node]
        self.cn_s = cn[:, :, self.num_node:]

    def cancel_load(self):
        self.cn_v = None
        self.cn_s = None


class NoisyILCCLIFNode(NoisyILCBaseNode):
    def __init__(self, act_dim, dec_pop_dim, c_decay: float = 0.5, v_decay: float = 0.75,
                 is_training: bool = True, T: int = 5, sigma_init: float = 0.5, 
                 beta: float = 0.0, v_threshold: float = 1.0, v_reset: Optional[float] = 0.,
                 surrogate_function: Callable = surrogate.Rect()):
        super().__init__(act_dim, dec_pop_dim, is_training, T, sigma_init, beta, v_threshold, 
                         v_reset, surrogate_function)

        self.c_decay = c_decay
        self.v_decay = v_decay

    def neuronal_charge(self, x: torch.Tensor):
        self.c = self.c * self.c_decay + x
        self.v = self.v * self.v_decay + self.c

    def init_tensor(self, data: torch.Tensor):
        self.c = torch.full_like(data, fill_value=0.0)
        self.v = torch.full_like(data, fill_value=self.v_reset)


##########################################################################################################
# Non-spiking modules for floating-point output
##########################################################################################################

class NonSpikingBaseNode(nn.Module, base.MultiStepModule):
    def __init__(self, decode='last-mem'):
        super().__init__()

        self.decode = decode

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def forward(self, x_seq: torch.Tensor):
        self.v = torch.full_like(x_seq[0].data, fill_value=0.0)

        T = x_seq.shape[0]
        v_seq = []

        for t in range(T):
            self.neuronal_charge(x_seq[t])
            v_seq.append(self.v)

        if self.decode == 'max-mem':
            mem = torch.max(torch.stack(v_seq, 0), 0).values

        elif self.decode == 'max-abs-mem':
            v_stack = torch.stack(v_seq, 0)
            max_mem = torch.max(v_stack, 0).values
            min_mem = torch.min(v_stack, 0).values
            mem = max_mem * (max_mem.abs() > min_mem.abs()) + min_mem * (max_mem.abs() <= min_mem.abs())

        elif self.decode == 'mean-mem':
            mem = torch.mean(torch.stack(v_seq, 0), 0)

        else:  # 'last-mem'
            mem = v_seq[-1]

        return mem


class NonSpikingIFNode(NonSpikingBaseNode):
    def __init__(self, decode='last-mem'):
        super().__init__(decode)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x


class NonSpikingLIFNode(NonSpikingBaseNode):
    def __init__(self, tau: float = 2., decode='last-mem'):
        super().__init__(decode)

        self.tau = tau

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x - self.v) / self.tau


##########################################################################################################
# Noisy Non-spiking modules
##########################################################################################################

class NoisyNonSpikingBaseNode(nn.Module, base.MultiStepModule):
    def __init__(self, num_node, is_training: bool = True, T: int = 5, 
                 sigma_init: float = 0.5, beta: float = 0.0, decode: str = 'last-mem'):
        super().__init__()

        self.num_node = num_node
        self.is_training = is_training
        self.T = T
        self.beta = beta
        self.decode = decode

        self.sigma = nn.Parameter(torch.FloatTensor(num_node))
        self.sigma.data.fill_(sigma_init / math.sqrt(num_node))
        self.cn = None

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def init_tensor(self, data: torch.Tensor):
        self.v = torch.full_like(data, fill_value=0.0)

    def forward(self, x_seq: torch.Tensor):
        self.init_tensor(x_seq[0].data)

        v_seq = []

        if self.is_training:
            if self.cn is None:
                self.noise_step += 1

            for t in range(self.T):
                if self.cn is None:
                    self.neuronal_charge(x_seq[t] + self.sigma.mul(self.eps_seq[self.noise_step][t].to(x_seq.device)))
                else:
                    self.neuronal_charge(x_seq[t] + self.sigma.mul(self.cn[:, t].to(x_seq.device)))
                v_seq.append(self.v)
                
        else:
            for t in range(self.T):
                self.neuronal_charge(x_seq[t])
                v_seq.append(self.v)

        if self.decode == 'max-mem':
            mem = torch.max(torch.stack(v_seq, 0), 0).values

        elif self.decode == 'max-abs-mem':
            v_stack = torch.stack(v_seq, 0)
            max_mem = torch.max(v_stack, 0).values
            min_mem = torch.min(v_stack, 0).values
            mem = max_mem * (max_mem.abs() > min_mem.abs()) + min_mem * (max_mem.abs() <= min_mem.abs())

        elif self.decode == 'mean-mem':
            mem = torch.mean(torch.stack(v_seq, 0), 0)

        else:  # 'last-mem'
            mem = v_seq[-1]

        return mem

    def reset_noise(self, num_rl_step):
        eps_shape = [self.num_node, num_rl_step * self.T]
        per_order = [1, 2, 0]
        self.eps_seq = torch.FloatTensor(powerlaw_psd_gaussian(self.beta, eps_shape).reshape(self.num_node, num_rl_step, self.T)).permute(per_order)
        self.noise_step = -1

    def get_colored_noise(self):
        return self.eps_seq[self.noise_step]

    def load_colored_noise(self, cn):
        self.cn = cn

    def cancel_load(self):
        self.cn = None


class NoisyNonSpikingIFNode(NoisyNonSpikingBaseNode):
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x
