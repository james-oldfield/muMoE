import torch
from entmax import entmax15 as entmax
import torch.nn as nn
import einops
from einops import reduce, rearrange
import tensorly as tl
tl.set_backend('pytorch')

class MuMoE(nn.Module):
    def __init__(self, MuMoE_layer, input_dim, output_dim, expert_dims, ranks, second_dim=0, normalization='bn', act=entmax, use_bias=True, hierarchy=1, apply_ln=False):
        """
        Wrapper around MuMoE layers to handle the expert gating and normalization, and any second layers.

        MuMoE_layer: nn.Module, the MuMoE layer to use (e.g. CPMuMoE, TRMuMoE)
        input_dim: int, the input dimension of the vector (or the # channels for token-based networks)
        output_dim: int, the output dimension of the vector
        expert_dims: list of ints, the dimensions of the expert modes
        ranks: int (CP), or list of list of ints (TR) (example format: R, or: [[r1, n, r2], [r2, i, r3] [r3, o, r1]])
        second_dim: int, if>0, uses a 2-layer MuMoE block.
        normalization: str, the normalization to use before expert coefficients' activation (bn/ln)
        act: activation function for the expert gating (e.g.entmax)
        use_bias: bool, whether to use a bias term (folded into the second-to-last "input" mode)
        hierarchy: int, the number of levels of hierarchy
        apply_ln: bool, whether to apply layer normalization to the input before the layer's computation (independent of gating normalization)
        """
        super(MuMoE, self).__init__()
        self.MuMoE = MuMoE_layer(input_dim, output_dim, expert_dims, ranks, use_bias=use_bias, hierarchy=hierarchy)
        self.expert_dims = expert_dims
        self.normalization = normalization
        self.act = act
        self.second_dim = second_dim

        if self.second_dim > 0:
            self.MuMoE2 = MuMoE_layer(output_dim, second_dim, expert_dims, ranks, use_bias=use_bias, hierarchy=hierarchy)
            self.gelu = nn.GELU()

        self.projs = []
        ############ expert gating params
        for di, d in enumerate(expert_dims):
            W = torch.nn.Parameter(torch.zeros(input_dim, d), requires_grad=True)
            self.projs += [W]

        self.input_ln = torch.nn.LayerNorm(input_dim) if apply_ln else nn.Identity()
        self.projs = nn.ParameterList(self.projs)

        if self.normalization == 'bn': self.bn = nn.ModuleList([torch.nn.BatchNorm1d(dim, affine=False) for dim in expert_dims])
        if self.normalization == 'ln': self.ln = nn.ModuleList([torch.nn.LayerNorm(dim, elementwise_affine=False, bias=False) for dim in expert_dims])

    def forward(self, x):
        if len(x.shape) == 2: x = x.view(x.shape[0], 1, x.shape[1])

        # generate the expert coefficients
        if self.normalization == 'bn': a = [self.act(rearrange(self.bn[e](rearrange(x@self.projs[e], 'b t n -> b n t')), 'b n t -> b t n')) for e in range(len(self.expert_dims))]
        if self.normalization == 'ln': a = [self.act(self.ln[e](x@self.projs[e])) for e in range(len(self.expert_dims))]

        self.a = a

        x = self.input_ln(x)

        x = self.MuMoE(x, a=a)
        if self.second_dim:
            x = self.gelu(x)
            x = self.MuMoE2(x, a=a)
        return x

class CPMuMoE(nn.Module):
    def __init__(self, input_dim, output_dim, expert_dims, rank, use_bias=True, hierarchy=1):
        """
        input_dim: int, the input dimension of the vector (or the # channels for token-based networks)
        output_dim: int, the output dimension of the vector
        expert_dims: list of ints, the dimensions of the expert modes
        rank: int, the CP rank of the tensor
        use_bias: bool, whether to use a bias term (folded into the second-to-last "input" mode)
        hierarchy: int, the number of levels of hierarchy
        """
        super(CPMuMoE, self).__init__()

        self.factors = []
        self.hierarchy = hierarchy
        self.use_bias = use_bias
        self.expert_dims = expert_dims
        dims = expert_dims + [input_dim+1, output_dim] if self.use_bias else expert_dims + [input_dim, output_dim]

        # factors
        for fi, (d, r) in enumerate(zip(dims, [rank]*len(dims))):
            if fi < len(expert_dims):
                W = nn.Parameter(torch.randn(d, r)*1.00 + 1.0, requires_grad=True) # N(1,1)

                if self.hierarchy > 1 and fi != 0: # N(1,0) for additional expert modes
                    torch.nn.init.ones_(W)
            else:
                W = nn.Parameter(torch.randn(d, r), requires_grad=True)

                effective_input_dim = input_dim if fi == len(dims)-2 else rank
                k = 1/effective_input_dim
                torch.nn.init.uniform_(W, -k**0.5, k**0.5)  # init by U(-\sqrt{1/d}, \sqrt{1/d}) 

            self.factors += [W]
        
        self.factors = nn.ParameterList(self.factors)

    def forward(self, z, a):
        """
        Performs a forward pass through the CP-MuMoE layer in factorized form (Eq. 2 in the main paper).

        Code is written to operate on (batch, tokens, channels)-dim input tensors, to handle token-based architectures.
        """
        if self.hierarchy == 1:
            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1, device=z.device)], dim=-1)
            f1 = einops.einsum(a[0], self.factors[0], 'b t n1, n1 r -> b t r')
            final = einops.einsum(z, self.factors[1], 'b t i, i r -> b t r')
            out = einops.einsum(f1, final, self.factors[2], 'b t r, b t r, o r -> b t o')

            return out

        elif self.hierarchy == 2:
            f1 = einops.einsum(a[0], self.factors[0], 'b t n1, n1 r -> b t r')
            f2 = einops.einsum(a[1], self.factors[1], 'b t n1, n1 r -> b t r')

            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1, device=z.device)], dim=-1)
            final = einops.einsum(z, self.factors[2], 'b t i, i r -> b t r')
            out = einops.einsum(f1, f2, final, self.factors[3], 'b t r, b t r, b t r, o r -> b t o') # trilinear

            return out
        
        else:
            raise NotImplementedError


class TRMuMoE(nn.Module):
    def __init__(self, input_dim, output_dim, expert_dims, ranks, act=entmax, use_bias=True, hierarchy=1):
        """
        input_dim: int, the input dimension of the vector (or the # channels for token-based networks)
        output_dim: int, the output dimension of the vector
        expert_dims: list of ints, the dimensions of the expert modes
        ranks: list of list of ints, (example format: [[r1, n, r2], [r2, i, r3] [r3, o, r1]]) the TT/TR-ranks of the tensor
        act: activation function for the expert gating (e.g.entmax/softmax)
        use_bias: bool, whether to use a bias term (folded into the second-to-last "input" mode)
        second_dim: int, if >0, uses a 2-layer MuMoE.
        hierarchy: int, the number of levels of hierarchy
        """
        super(TRMuMoE, self).__init__()

        self.act = act
        self.factors = []
        self.hierarchy = hierarchy
        self.use_bias = use_bias
        self.expert_dims = expert_dims
        dims = expert_dims + [input_dim+1, output_dim] if self.use_bias else expert_dims + [input_dim, output_dim]

        # factors
        for fi, rank in enumerate(ranks):
            r = rank.copy()
            r[1] = dims[fi] # populate the TT-rank tensor with the specified dims of weight tensor

            if fi < len(expert_dims):
                var = 1.0 if fi == 0 else 0.0
                W = nn.Parameter(torch.zeros(*[r]), requires_grad=True)
                # init each slice with a diagonal matrix with elements N(1,var) to replicate linear layers along expert dimensions
                for dim in range(r[1]):
                    W.data[:, dim, :] = torch.eye(r[0], r[2]) * (torch.randn(r[0], r[2]) * var + 1.0)
            else: # input/output cores
                W = nn.Parameter(torch.randn(*[r]), requires_grad=True) # TT/TR core

                # input_dims is the product of remaining TT-ranks for dims not contracted over, else the current input vector dimension
                # i.e. if [output_dim] else [input_dim]
                effective_input_dim = r[0]*r[2] if fi == len(ranks)-1 else input_dim

                k = 1/effective_input_dim
                torch.nn.init.uniform_(W, -k**0.5, k**0.5)  # init by U(-\sqrt{1/d}, \sqrt{1/d})
                #####################################################################

            self.factors += [W]
            
        self.factors = nn.ParameterList(self.factors)

    def forward(self, z, a):
        """
        Performs a forward pass through the TT-MuMoE layer in factorized form (Eq. 4 in the main paper).

        Code is written to operate on (batch, tokens, channels)-dim input tensors, to handle token-based architectures.
        """
        if self.hierarchy == 1:
            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1, device=z.device)], dim=-1)
            train = einops.einsum(a[0], self.factors[0], 'b t x, r1 x r2 -> b t r1 r2')  # batched mode-2 product [expert mode]
            train = einops.einsum(
                einops.einsum(z, self.factors[1], 'b t x, r2 x r3 -> b t r2 r3'), # batched mode-2 product [input mode]
                train,
                'b t r2 r3, b t r1 r2 -> b t r1 r3'
            ) # matrix multiplication with the two contracted tt/tr cores
            train = einops.einsum(self.factors[2], train, 'r3 o r1, b t r1 r3 -> b t o') # final batched mode-2 product with the output core

            return train

        elif self.hierarchy == 2:
            ################## trilinear (2 levels of experts)
            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1, device=z.device)], dim=-1)
            train = einops.einsum(a[0], self.factors[0], 'b t x, r1 x r2 -> b t r1 r2') # batched mode-2 product [1st expert mode]
            train = einops.einsum(
                train,
                einops.einsum(a[1], self.factors[1], 'b t x, r2 x r3 -> b t r2 r3'), # batched mode-2 product [2nd expert mode]
                'b t r1 r2, b t r2 r3 -> b t r1 r3')
            train = einops.einsum(
                train,
                einops.einsum(z, self.factors[2], 'b t x, r3 x r4 -> b t r3 r4'),
                'b t r1 r3, b t r3 r4 -> b t r1 r4')
            train = einops.einsum(
                train,
                self.factors[3],
                'b t r1 r4, r4 o r1 -> b t o')

            return train

        else:
            raise NotImplementedError