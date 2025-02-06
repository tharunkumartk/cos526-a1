import torch
import torch.nn as nn


# Positional encoding (section 5.1)
class PositionalEncoder(nn.Module):
    def __init__(self, multires, i=0, input_dims=3, include_input=True, log_sampling=True, periodic_fns=[torch.sin, torch.cos]):
        super(PositionalEncoder, self).__init__()
        self.i_embed = i

        max_freq = multires - 1
        N_freqs = multires

        embed_fns = []
        d = input_dims
        out_dim = 0
        if include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        #------------------------------------------------------
        #----- PLEASE FILL IN COMPUTATIONS FOR 
        #----- freq_bands: [N_freqs]. Various frequency bands required for positional encoding
        #----- embed_fn: Lambda input, **kwargs: encoding. Wrapper function that applies periodic functions to specific frequency bands.
        #----- out_dim: int. Output dimension of the encoding
        #------------------------------------------------------
        
        assert False, 'Please finish the code before removing this assertion'
            
        if log_sampling:
            freq_bands = ...
        else:
            freq_bands = ...

        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fn = ...
                embed_fns.append(embed_fn)
                out_dim += ...

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        if self.i_embed == -1:
            return nn.Identity(), 3
        else:
            return torch.cat([fn(inputs) for fn in self.embed_fns], -1)