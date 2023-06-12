""" 
copied from https://github.com/HazyResearch/state-spaces/blob/main/src/models/nn/components.py
Utility nn components, in particular handling activations, initializations, and normalization layers """

from functools import partial
import math
import numpy as np
from typing import ForwardRef
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from einops import rearrange, repeat, reduce



class SequenceDecoder(nn.Module):
    def __init__(
        self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last", decoder_mlp=False, min_context=None, transposed=True
    ):
        super().__init__()

        self.min_context = min_context

        if d_output is None:
            self.output_transform = nn.Identity()
        elif decoder_mlp:
            self.output_transform = Mlp(d_model,4*d_model,d_output, transposed=transposed)
        else:
            self.output_transform = nn.Linear(d_model, d_output) 

        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False

        self.use_lengths = use_lengths
        self.mode = mode

        if mode == 'ragged':
            assert not use_lengths

    def forward(self, x, state=None, lengths=None, l_output=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """

        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(-2)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        
        if self.min_context is not None:
            restrict = lambda x: x[..., self.min_context:-l_output, :]
        elif self.mode == "last":
            restrict = lambda x: x[..., -l_output:, :]
        elif self.mode == "first":
            restrict = lambda x: x[..., :l_output, :]
        elif self.mode == "pool":
            restrict = lambda x: (
                torch.cumsum(x, dim=-2)
                / torch.arange(
                    1, 1 + x.size(-2), device=x.device, dtype=x.dtype
                ).unsqueeze(-1)
            )[..., -l_output:, :]

            def restrict(x):
                L = x.size(-2)
                denom = torch.arange(
                    L - l_output + 1, L + 1, dtype=x.dtype, device=x.device
                )
                s = x.sum(dim=-2, keepdim=True)
                if l_output > 1:
                    c = torch.cumsum(x[..., -(l_output - 1) :, :].flip(-2), dim=-2)
                    c = F.pad(c, (0, 0, 1, 0))
                    s = s - c  # (B, l_output, D)
                    s = s.flip(-2)
                    denom = denom.view(1,-1,1)
                
                s = s / denom
                return s

        elif self.mode == "sum":
            restrict = lambda x: torch.cumsum(x, dim=-2)[..., -l_output:, :]
            # TODO use same restrict function as pool case
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"
            # remove any additional padding (beyond max length of any sequence in the batch)
            restrict = lambda x: x[..., : max(lengths), :]
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'sum']"
            )

        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(-2) == 1
            x = x.squeeze(-2)

        x = self.output_transform(x)

        return x


class Residual(nn.Module):
    """ Residual connection with constant affine weights. Can simulate standard residual, no residual, and "constant gates". """

    def __init__(self, i_layer, d_input, d_model, alpha=1.0, beta=1.0):
        # print("ConstantResidual extra kwargs", kwargs)
        super().__init__()
        assert (d_input == d_model) or alpha == 0.0
        self.i_layer = i_layer
        self.d_input = d_input
        self.d_model = d_model
        self.alpha   = alpha
        self.beta    = beta

    @property
    def d_output(self):
        return self.d_model

    def forward(self, x, y, transposed): # TODO documentation of transposed
        y = self.beta*y if self.beta != 1.0 else y
        return self.alpha * x + y if self.alpha else y


class DownAvgPool(nn.Module):
    def __init__(self, d_input, stride=1, expand=None, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

        if self.expand is not None:
            self.linear = LinearActivation(
                d_input,
                d_input * expand,
                transposed=transposed,
            )

    def forward(self, x):
        if not self.transposed:
            x = rearrange(x, 'b ... d -> b d ...')

        if self.stride > 1:
            # einops appears slower than F
            if x.ndim == 3:
                x = F.avg_pool1d(x, self.stride, self.stride)
            elif x.ndim == 4:
                x = F.avg_pool2d(x, self.stride, self.stride)
            else:
                # Reduction string e.g. "b d (l1 2) (l2 2) -> b d l1 l2"
                reduce_str = "b d " + " ".join([f"(l{i} {self.stride})" for i in range(x.ndim-2)]) \
                        + " -> b d " + " ".join([f"l{i}" for i in range(x.ndim-2)])
                x = reduce(x, reduce_str, 'mean')

        # if self.expand > 1:
        #     x = repeat(x, 'b d ... -> b (d e) ...', e=self.expand)

        if not self.transposed:
            x = rearrange(x, 'b d ... -> b ... d')
        if self.expand is not None:
            x = self.linear(x)
        return x, None

    def step(self, x, state, **kwargs):
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        if self.expand is None:
            return self.d_input
        else:
            return self.d_input * self.expand



def stochastic_depth(input: torch.tensor, p: float, mode: str, training: bool = True):
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.
    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``
    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(
            "drop probability has to be between 0 and 1, but got {}".format(p)
        )
    if mode not in ["batch", "row"]:
        raise ValueError(
            "mode has to be either 'batch' or 'row', but got {}".format(mode)
        )
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate).div_(survival_rate)
    return input * noise


class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str) -> None:
        # TODO(karan): need to upgrade to torchvision==0.11.0 to use StochasticDepth directly
        # from torchvision.ops import StochasticDepth
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "p=" + str(self.p)
        tmpstr += ", mode=" + str(self.mode)
        tmpstr += ")"
        return tmpstr


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                "dropout probability has to be in [0, 1), " "but got {}".format(p)
            )
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X):
        """ X: (batch, dim, lengths...) """
        if self.training:
            if not self.transposed:
                X = rearrange(X, "b d ... -> b ... d")
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, "b ... d -> b d ...")
            return X
        return X


def Activation(activation=None, size=None, dim=-1):
    if activation in [None, "id", "identity", "linear"]:
        return nn.Identity()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation in ["swish", "silu"]:
        return nn.SiLU()
    elif activation == "glu":
        return nn.GLU(dim=dim)
    elif activation == "sigmoid":
        return nn.Sigmoid()
    # elif activation == 'modrelu':
    #     return Modrelu(size)
    # elif activation == 'sqrelu':
    #     return SquaredReLU()
    # elif activation == 'ln':
    #     return TransposedLN(dim)
    else:
        raise NotImplementedError(
            "hidden activation '{}' is not implemented".format(activation)
        )


def LinearActivation(
    d_input,
    d_output,
    bias=True,
    zero_bias_init=False,
    transposed=False,
    initializer=None,
    activation=None,
    activate=False,  # Apply activation as part of this module
    weight_norm=False,
    **kwargs,
):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Construct core module
    # linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation == "glu":
        d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, d_output, dim=1 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear

# Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/mlp.py
from torch.nn.modules.utils import _pair

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 act_fn=None, drop=0., transposed=True, device=None, dtype=None):
        """TD [2021-10-27] act_fn takes precedence over act_layer if set.
        This is to support Pytorch 1.10 Transformer interface that construct the activation
        *function*, not the activation *layer*.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        linear_cls = TransposedLinear if transposed else nn.Linear
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = _pair(drop)
        self.fc1 = linear_cls(in_features, hidden_features)
        self.act = act_layer() if act_fn is None else act_fn
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_cls(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransposedLinear(nn.Module):
    """ Linear module on the second-to-last dimension
    Assumes shape (B, D, L), where L can be 1 or more axis
    """

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_output, d_input))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # nn.Linear default init
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear') # should be equivalent

        if bias:
            self.bias = nn.Parameter(torch.empty(d_output))
            bound = 1 / math.sqrt(d_input)
            nn.init.uniform_(self.bias, -bound, bound)
            setattr(self.bias, "_optim", {"weight_decay": 0.0})
        else:
            self.bias = 0.0

    def forward(self, x):
        num_axis = len(x.shape[2:])  # num_axis in L, for broadcasting bias
        y = contract("b u ..., v u -> b v ...", x, self.weight) + self.bias.view(
            -1, *[1] * num_axis
        )
        return y


class TransposedLN(nn.Module):
    """ LayerNorm module over second dimension
    Assumes shape (B, D, L), where L can be 1 or more axis
    This is slow and a dedicated CUDA/Triton implementation shuld provide substantial end-to-end speedup
    """

    def __init__(self, d, scalar=True):
        super().__init__()
        self.scalar = scalar
        if self.scalar:
            self.m = nn.Parameter(torch.zeros(1))
            self.s = nn.Parameter(torch.ones(1))
            setattr(self.m, "_optim", {"weight_decay": 0.0})
            setattr(self.s, "_optim", {"weight_decay": 0.0})
        else:
            self.ln = nn.LayerNorm(d)

    def forward(self, x):
        if self.scalar:
            # calc. stats over D dim / channels
            s, m = torch.std_mean(x, dim=1, unbiased=False, keepdim=True)
            y = (self.s / s) * (x - m + self.m)
        else:
            # move channel to last axis, apply layer_norm, then move channel back to second axis
            _x = self.ln(rearrange(x, "b d ... -> b ... d"))
            y = rearrange(_x, "b ... d -> b d ...")
        return y


class Normalization(nn.Module):
    def __init__(
        self,
        d,
        transposed=False,  # Length dimension is -1 or -2
        _name_="layer",
        **kwargs,
    ):
        super().__init__()
        self.transposed = transposed
        self._name_ = _name_

        if _name_ == "layer":
            self.channel = True  # Normalize over channel dimension
            if self.transposed:
                self.norm = TransposedLN(d, **kwargs)
            else:
                self.norm = nn.LayerNorm(d, **kwargs)
        elif _name_ == "instance":
            self.channel = False
            norm_args = {"affine": False, "track_running_stats": False}
            norm_args.update(kwargs)
            self.norm = nn.InstanceNorm1d(
                d, **norm_args
            )  # (True, True) performs very poorly
        elif _name_ == "batch":
            self.channel = False
            norm_args = {"affine": True, "track_running_stats": True}
            norm_args.update(kwargs)
            self.norm = nn.BatchNorm1d(d, **norm_args)
        elif _name_ == "group":
            self.channel = False
            self.norm = nn.GroupNorm(1, d, *kwargs)
        elif _name_ == "none":
            self.channel = True
            self.norm = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x):
        # Handle higher dimension logic
        shape = x.shape
        if self.transposed:
            x = rearrange(x, "b d ... -> b d (...)")
        else:
            x = rearrange(x, "b ... d -> b (...)d ")

        # The cases of LayerNorm / no normalization are automatically handled in all cases
        # Instance/Batch Norm work automatically with transposed axes
        if self.channel or self.transposed:
            x = self.norm(x)
        else:
            x = x.transpose(-1, -2)
            x = self.norm(x)
            x = x.transpose(-1, -2)

        x = x.view(shape)
        return x

    def step(self, x, **kwargs):
        assert self._name_ in ["layer", "none"]
        if self.transposed:
            x = x.unsqueeze(-1)
        x = self.forward(x)
        if self.transposed:
            x = x.squeeze(-1)
        return x


def get_initializer(name, activation=None):
    if activation in [None, "id", "identity", "linear", "modrelu"]:
        nonlinearity = "linear"
    elif activation in ["relu", "tanh", "sigmoid"]:
        nonlinearity = activation
    elif activation in ["gelu", "swish", "silu"]:
        nonlinearity = "relu"  # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(
            f"get_initializer: activation {activation} not supported"
        )

    if name == "uniform":
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == "normal":
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == "xavier":
        initializer = torch.nn.init.xavier_normal_
    elif name == "zero":
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == "one":
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        raise NotImplementedError(
            f"get_initializer: initializer type {name} not supported"
        )

    return initializer



""" Misc functional utilities """

def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i
    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A) # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)







""" HiPPO utilities """

def transition(measure, N):
    """ A, B transition matrices for different measures """
    # Legendre (translated)
    if measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A

        # Halve again for timescale correctness
        A *= 0.5
        B *= 0.5
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'legsd':
        # Essentially equivalent to S4D-LegS
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
        A += .5 * B*B[None, :, 0]
        B = B / 2.0
    elif measure in ['fourier_diag', 'foud']:
        # Essentially equivalent to S4D-Lin
        freqs = np.arange(N//2)
        d = np.stack([freqs, np.zeros(N//2)], axis=-1).reshape(-1)[:-1]
        A = 2*np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        A = A - .5 * np.eye(N)
        B = np.zeros(N)
        B[0::2] = 2**.5
        B[0] = 1
        B = B[:, None]
    elif measure in ['fourier', 'fout']:
        freqs = np.arange(N//2)
        d = np.stack([np.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2**.5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :]
        B = B[:, None]
    else:
        raise NotImplementedError

    return A, B

def rank_correction(measure, N, rank=1, dtype=torch.float):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        P = torch.sqrt(.5+torch.arange(N, dtype=dtype)).unsqueeze(0) # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        P = torch.sqrt(1+2*torch.arange(N, dtype=dtype)) # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = torch.stack([P0, P1], dim=0) # (2 N)
        P *= 2**(-0.5) # Halve the rank correct just like the original matrix was halved
    elif measure in ['fourier', 'fout']:
        P = torch.zeros(N)
        P[0::2] = 2**.5
        P[0] = 1
        P = P.unsqueeze(0)
    elif measure in ['fourier_diag', 'foud', 'legsd']:
        P = torch.zeros(1, N, dtype=dtype)
    else: raise NotImplementedError

    d = P.size(0)
    if rank > d:
        P = torch.cat([P, torch.zeros(rank-d, N, dtype=dtype)], dim=0) # (rank N)
    return P

def nplr(measure, N, rank=1, dtype=torch.float, diagonalize_precision=True):
    """ Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == torch.float or torch.double
    cdtype = torch.cfloat if dtype == torch.float else torch.cdouble

    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype) # (N, N)
    B = torch.as_tensor(B, dtype=dtype)[:, 0] # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype) # (r N)
    AP = A + torch.sum(P.unsqueeze(-2)*P.unsqueeze(-1), dim=-3)

    # We require AP to be nearly skew-symmetric
    _A = AP + AP.transpose(-1, -2)
    if (err := torch.sum((_A - _A[0,0]*torch.eye(N))**2) / N) > 1e-5: # if not torch.allclose(_A - _A[0,0]*torch.eye(N), torch.zeros(N, N), atol=1e-5):
        print("WARNING: HiPPO matrix not skew symmetric", err)


    # Take advantage of identity + skew-symmetric form to calculate real and imaginary parts separately
    # Imaginary part can use eigh instead of eig
    w_re = torch.mean(torch.diagonal(AP), -1, keepdim=True)

    # Diagonalize in double precision
    if diagonalize_precision: AP = AP.to(torch.double)
    w_im, V = torch.linalg.eigh(AP*-1j) # (..., N) (..., N, N)
    if diagonalize_precision: w_im, V = w_im.to(cdtype), V.to(cdtype)
    w = w_re + 1j * w_im
    # Check: V w V^{-1} = A
    # print("check", V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2))


    # Only keep half of each conjugate pair
    _, idx = torch.sort(w.imag)
    w_sorted = w[idx]
    V_sorted = V[:, idx]

    # There is an edge case when eigenvalues can be 0, which requires some machinery to handle
    # We use a huge hack here: Assume only one pair is 0, and that it is the first row/column of A (only happens in Fourier case)
    V = V_sorted[:, :N//2]
    w = w_sorted[:N//2]
    assert w[-2].abs() > 1e-4, "Only 1 zero eigenvalue allowed in diagonal part of A"
    if w[-1].abs() < 1e-4:
        V[:, -1] = 0.
        V[0, -1] = 2**-0.5
        V[1, -1] = 2**-0.5 * 1j

    _AP = V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2)
    if ((err := torch.sum((2*_AP.real-AP)**2)/N) > 1e-5):
        print("Warning: Diagonalization of A matrix not numerically precise - error", err)
    # print("check", V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2))

    V_inv = V.conj().transpose(-1, -2)

    B = contract('ij, j -> i', V_inv, B.to(V)) # V^* B
    P = contract('ij, ...j -> ...i', V_inv, P.to(V)) # V^* P

    return w, P, B, V

def dplr(scaling, N, rank=1, H=1, dtype=torch.float, real_scale=1.0, imag_scale=1.0, random_real=False, random_imag=False, normalize=False, diagonal=True, random_B=False):
    assert dtype == torch.float or torch.double
    dtype = torch.cfloat if dtype == torch.float else torch.cdouble

    pi = torch.tensor(math.pi)
    if random_real:
        real_part = torch.rand(H, N//2)
    else:
        real_part = .5 * torch.ones(H, N//2)
    if random_imag:
        imag_part = N//2 * torch.rand(H, N//2)
    else:
        imag_part = repeat(torch.arange(N//2), 'n -> h n', h=H)

    real_part = real_scale * real_part
    if scaling == 'random':
        imag_part = torch.randn(H, N//2)
    elif scaling == 'real':
        imag_part = 0 * imag_part
        real_part = 1 + repeat(torch.arange(N//2), 'n -> h n', h=H)
    elif scaling in ['linear', 'lin']:
        imag_part = pi * imag_part
    elif scaling in ['inverse', 'inv']: # Based on asymptotics of the default HiPPO matrix
        imag_part = 1/pi * N * (N/(1+2*imag_part)-1)
    elif scaling in ['inverse2', 'inv2']:
        imag_part = 1/pi * N * (N/(1+imag_part)-1)
    elif scaling in ['quadratic', 'quad']:
        imag_part = 1/pi * (1+2*imag_part)**2
    elif scaling in ['legs', 'hippo']:
        w, _, _, _ = nplr('legsd', N)
        imag_part = w.imag

    else: raise NotImplementedError
    imag_part = imag_scale * imag_part
    w = -real_part + 1j * imag_part

    # Initialize B
    if random_B:
        B = torch.randn(H, N//2, dtype=dtype)
    else:
        B = torch.ones(H, N//2, dtype=dtype)

    if normalize:
        norm = -B/w # (H, N) # Result if you integrate the kernel with constant 1 function
        zeta = 2*torch.sum(torch.abs(norm)**2, dim=-1, keepdim=True) # Variance with a random C vector
        B = B / zeta**.5

    P = torch.randn(rank, H, N//2, dtype=dtype)
    if diagonal: P = P * 0.0
    V = torch.eye(N, dtype=dtype)[: :N//2] # Only used in testing
    V = repeat(V, 'n m -> h n m', h=H)

    return w, P, B, V

def ssm(measure, N, R, H, **ssm_args):
    """Dispatcher to create single SSM initialization
    N: state size
    R: rank (for DPLR parameterization)
    H: number of independent SSM copies
    """

    if measure == "dplr":
        w, P, B, V = dplr(N=N, rank=R, H=H, **ssm_args)
    elif measure.startswith("diag"):
        args = measure.split("-")
        assert args[0] == "diag" and len(args) > 1
        scaling = args[1]
        w, P, B, V = dplr(scaling=scaling, N=N, rank=R, H=H, diagonal=True, **ssm_args)
    else:
        w, P, B, V = nplr(measure, N, R, **ssm_args)
        w = repeat(w, 'n -> s n', s=H)
        P = repeat(P, 'r n -> r s n', s=H)
        B = repeat(B, 'n -> s n', s=H)
        V = repeat(V, 'n m -> s n m', s=H)
    return w, P, B, V

combinations = {
    'hippo': ['legs', 'fourier'],
    'diag': ['diag-inv', 'diag-lin'],
    'all': ['legs', 'fourier', 'diag-inv', 'diag-lin'],
}

def combination(measures, N, R, S, **ssm_args):
    if isinstance(measures, str):
        measures = combinations[measures] if measures in combinations else [measures]

    assert S % len(measures) == 0, f"{S} independent trainable SSM copies must be multiple of {len(measures)} different measures"
    w, P, B, V = zip(
        *[ssm(measure, N, R, S // len(measures), **ssm_args) for measure in measures]
    )
    w = torch.cat(w, dim=0) # (S N)
    P = torch.cat(P, dim=1) # (R S N)
    B = torch.cat(B, dim=0) # (S N)
    V = torch.cat(V, dim=0) # (S N N)
    return w, P, B, V