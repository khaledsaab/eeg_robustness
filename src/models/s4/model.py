""" 
copied from https://github.com/HazyResearch/state-spaces/blob/main/src/models/sequence

Isotropic deep sequence model backbone, in the style of ResNets / Transformers.
The SequenceModel class implements a generic (batch, length, d_input) -> (batch, length, d_output) transformation
"""

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from src.utils import to_list, to_dict
from src.models.s4.utils import (
    Normalization,
    DropoutNd,
    StochasticDepth,
    Residual,
    DownAvgPool,
    SequenceDecoder,
    Mlp
)
import hydra


class S4Model(nn.Module):
    def __init__(
        self,
        d_input,  # dim of input
        d_model,  # Resize input (useful for deep models with residuals)
        d_output,  # dim of output
        l_output=0,  # output length
        n_layers=1,  # Number of layers
        transposed=False,  # Transpose inputs so each layer receives (batch, dim, length)
        dropout=0.0,  # Dropout parameter applied on every residual and every layer
        tie_dropout=False,  # Tie dropout mask across sequence like nn.Dropout1d/nn.Dropout2d
        prenorm=True,  # Pre-norm vs. post-norm
        n_repeat=1,  # Each layer is repeated n times per stage before applying pooling
        layer=None,  # Layer config, must be specified
        residual=None,  # Residual config
        norm=None,  # Normalization config (e.g. layer vs batch)
        pool=None,  # Config for pooling layer per stage
        track_norms=True,  # Log norms of each layer output
        dropinp=0.0,  # Input dropout
        decoder_mode="last",  # mode of decoder
        decoder_min_context=None, # minimum context decoder assumes
        decoder_mlp=False,
        age_input=False,
        mlp=False,
    ):
        super().__init__()
        # Save arguments needed for forward pass
        self.d_model = d_model
        self.d_input = d_input
        self.transposed = transposed
        self.track_norms = track_norms
        self.age_input = age_input

        # Input dropout (not really used)
        dropout_fn = (
            partial(DropoutNd, transposed=self.transposed)
            if tie_dropout
            else nn.Dropout
        )
        self.drop = dropout_fn(dropinp) if dropinp > 0.0 else nn.Identity()

        # Encoders
        self.encoder = nn.Linear(d_input, d_model)
        if self.age_input:
            self.age_encoder = nn.Linear(1,d_model)


        # Decoder
        self.decoder = SequenceDecoder(
            d_model, d_output=d_output, mode=decoder_mode, l_output=l_output, decoder_mlp=decoder_mlp, min_context=decoder_min_context, transposed=transposed
        )

        layer = to_list(layer, recursive=False)

        # Some special arguments are passed into each layer
        for _layer in layer:
            # If layers don't specify dropout, add it
            if _layer.get("dropout", None) is None:
                _layer["dropout"] = dropout
            # Ensure all layers are shaped the same way
            _layer["transposed"] = transposed

        # Duplicate layers
        layers = layer * n_layers * n_repeat

        # Instantiate layers
        _layers = []
        d = d_model
        for l, layer in enumerate(layers):
            # Pool at the end of every n_repeat blocks
            pool_cfg = pool if (l + 1) % n_repeat == 0 else None
            block = S4ResidualBlock(
                d,
                l + 1,
                prenorm=prenorm,
                dropout=dropout,
                tie_dropout=tie_dropout,
                transposed=transposed,
                layer=layer,
                residual=residual,
                norm=norm,
                pool=pool_cfg,
                mlp=mlp,
            )
            _layers.append(block)
            d = block.d_output

        self.d_output = d
        self.layers = nn.ModuleList(_layers)
        if prenorm:
            if norm is None:
                self.norm = None
            elif isinstance(norm, str):
                self.norm = Normalization(
                    self.d_output, transposed=self.transposed, _name_=norm
                )
            else:
                self.norm = Normalization(
                    self.d_output, transposed=self.transposed, **norm
                )
        else:
            self.norm = nn.Identity()

    def forward(self, inputs, *args, state=None, **kwargs):
        """Inputs assumed to be (batch, sequence, dim)"""

        inputs = self.encoder(inputs)
        if self.age_input:
            ages = args[0].view(-1,1).float() / 100 
            age_embedding = self.age_encoder(ages).unsqueeze(-2) 
            inputs = inputs + age_embedding

        if self.transposed:
            inputs = rearrange(inputs, "b ... d -> b d ...")
        inputs = self.drop(inputs)

        # Track norms
        if self.track_norms:
            output_norms = [torch.mean(inputs.detach() ** 2)]

        # Apply layers
        outputs = inputs
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            outputs, state = layer(outputs, state=prev_state, **kwargs)
            next_states.append(state)
            if self.track_norms:
                output_norms.append(torch.mean(outputs.detach() ** 2))
        if self.norm is not None:
            outputs = self.norm(outputs)

        if self.transposed:
            outputs = rearrange(outputs, "b d ... -> b ... d")

        if self.track_norms:
            metrics = to_dict(output_norms, recursive=False)
            self.metrics = {f"norm/{i}": v for i, v in metrics.items()}

        #outputs = self.patch_decoder(outputs)
        #outputs = outputs.view(inputs.shape[0],-1,self.d_input)
        outputs = self.decoder(outputs)

        # return outputs, next_states
        return outputs

    @property
    def d_state(self):
        d_states = [layer.d_state for layer in self.layers]
        return sum([d for d in d_states if d is not None])

    @property
    def state_to_tensor(self):
        # Slightly hacky way to implement this in a curried manner (so that the function can be extracted from an instance)
        # Somewhat more sound may be to turn this into a @staticmethod and grab subclasses using hydra.utils.get_class
        def fn(state):
            x = [
                _layer.state_to_tensor(_state)
                for (_layer, _state) in zip(self.layers, state)
            ]
            x = [_x for _x in x if _x is not None]
            return torch.cat(x, dim=-1)

        return fn

    def default_state(self, *batch_shape, device=None):
        return [
            layer.default_state(*batch_shape, device=device) for layer in self.layers
        ]

    def step(self, x, state, **kwargs):
        # Apply layers
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            x, state = layer.step(x, state=prev_state, **kwargs)
            next_states.append(state)

        x = self.norm(x)

        return x, next_states


class S4ResidualBlock(nn.Module):
    def __init__(
        self,
        d_input,
        i_layer=None,  # Only needs to be passed into certain residuals like Decay
        prenorm=True,
        dropout=0.0,
        tie_dropout=False,
        transposed=False,
        layer=None,  # Config for black box module
        residual=None,  # Config for residual function
        norm=None,  # Config for normalization layer
        pool=None,
        drop_path=0.0,
        mlp=False,
    ):
        super().__init__()

        self.i_layer = i_layer
        self.d_input = d_input
        self.layer = hydra.utils.instantiate(layer, _recursive_=False)
        self.prenorm = prenorm
        self.transposed = transposed
        self.mlp=mlp

        # Residual
        # d_residual is the output dimension after residual
        if residual is None:
            self.residual = None
            self.d_residual = self.layer.d_output
        else:
            self.residual = Residual(i_layer, d_input, self.layer.d_output)
            self.d_residual = self.residual.d_output

        # Normalization
        d_norm = d_input if self.prenorm else self.d_residual
        # We don't use config to directly instantiate since Normalization has some special cases
        if norm is None:
            self.norm = None
            self.norm2 = None
        elif isinstance(norm, str):
            self.norm = Normalization(d_norm, transposed=self.transposed, _name_=norm)
            self.norm2 = Normalization(d_norm, transposed=self.transposed, _name_=norm)
        else:
            self.norm = Normalization(d_norm, transposed=self.transposed, **norm)
            self.norm2 = Normalization(d_norm, transposed=self.transposed, **norm)

        # Pool
        self.pool = DownAvgPool(self.d_residual, transposed=self.transposed)

        # Dropout
        dropout_cls = (
            partial(DropoutNd, transposed=self.transposed)
            if tie_dropout
            else nn.Dropout1d # used to be nn.Dropout
        )
        self.drop = dropout_cls(dropout) if dropout > 0.0 else nn.Identity()

        # Stochastic depth
        self.drop_path = (
            StochasticDepth(drop_path, mode="row") if drop_path > 0.0 else nn.Identity()
        )

        # FFN
        if self.mlp:
            self.ff = Mlp(self.d_output,4*self.d_output,self.d_output,drop=dropout, transposed=transposed)

    @property
    def d_output(self):
        return self.pool.d_output if self.pool is not None else self.d_residual

    @property
    def d_state(self):
        return self.layer.d_state

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def forward(self, x, state=None, **kwargs):
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm:
            y = self.norm(y)

        # Black box layer
        y, state = self.layer(y, state=state, **kwargs)

        # Residual
        if self.residual is not None:
            y = self.residual(x, self.drop_path(self.drop(y)), self.transposed)

        # Post-norm
        if self.norm is not None and not self.prenorm:
            y = self.norm(y)

        # Pool
        if self.pool is not None:
            y, _ = self.pool(y)

        # MLP
        if self.mlp:
            y = y + self.ff(y)
            if self.norm2 is not None:
                y = self.norm2(y)


        return y, state

    def step(self, x, state, **kwargs):
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm:
            y = self.norm.step(y)

        # Black box layer
        y, state = self.layer.step(y, state, **kwargs)

        # Residual
        if self.residual is not None:
            y = self.residual(
                x, y, transposed=False
            )  # NOTE this would not work with concat residual function (catformer)

        # Post-norm
        if self.norm is not None and not self.prenorm:
            y = self.norm.step(y)

        # Pool
        if self.pool is not None:
            y, _ = self.pool(y)

        return y, state
