import torch
from functools import partial
from entmax import entmax15 as entmax
import numpy as np
import torch.nn as nn
import einops
from einops import reduce, rearrange
import tensorly as tl
tl.set_backend('pytorch')

"""
NOTE: this code orders the tensor modes as [expert_dim_1, ..., expert_dim_n, input_dim, output_dim] to be consistent with the PyTorch linear layer ordering.
"""

class TTMMoE(nn.Module):
    def __init__(self, input_dim, output_dim, expert_dims, ranks, act=entmax, use_bias=1, hierarchy=1):
        """
        input_dim: int, the input dimension of the vector (or the # channels for token-based networks)
        output_dim: int, the output dimension of the vector
        expert_dims: list of ints, the dimensions of the expert modes
        ranks: list of list of ints, (example format: [[r1, n, r2], [r2, i, r3] [r3, o, r1]]) the TT/TR-ranks of the tensor
        act: activation function for the expert gating (e.g.entmax/softmax)
        use_bias: bool, whether to use a bias term (folded into the second-to-last "input" mode)
        hierarchy: int, the number of levels of hierarchy
        """
        super(TTMMoE, self).__init__()

        self.act = act
        self.factors = []
        self.projs = []
        self.hierarchy = hierarchy
        self.expert_dims = expert_dims
        self.use_bias = use_bias
        dims = expert_dims + [input_dim, output_dim]

        # factors
        for fi, rank in enumerate(ranks):
            r = rank.copy()
            r[1] = dims[fi] # populate the TT-rank tensor with the specified dims of weight tensor
            if fi == len(ranks)-2 and self.use_bias:
                r[1] += 1 # add bias dimension to the second-to-last "input" mode

            if fi < len(expert_dims):
                var = 1.0 if fi == 2 else 0.0
                W = nn.Parameter(torch.zeros(*[r]), requires_grad=True)
                # init each slice with a diagonal matrix with elements N(1,var) to replicate linear layers along expert dimensions
                for dim in range(r[1]):
                    W.data[:, dim, :] = torch.eye(r[0], r[2]) * (torch.randn(r[0], r[2]) * var + 1.0)
            else: # input/output cores
                W = nn.Parameter(torch.randn(*[r]), requires_grad=True) # TT/TR core
                W.data = torch.nn.functional.normalize(W.data, p=2, dim=[0,2])

            self.factors += [W]

        ############ expert gating matrices
        for di, dim in enumerate(expert_dims):
            W = torch.nn.Parameter(torch.zeros(input_dim, dim), requires_grad=True)
            self.projs += [W]

        self.factors = nn.ParameterList(self.factors)
        self.projs = nn.ParameterList(self.projs)
        self.bn = nn.ModuleList([torch.nn.BatchNorm1d(dim, affine=False) for dim in expert_dims])

    def forward(self, z):
        """
        Performs a forward pass through the TT-MMoE layer in factorized form (Eq. 4 in the main paper).

        Code is written to operate on (batch, tokens, channels)-dim input tensors, to handle token-based architectures.
        """
        if len(z.shape) == 2: z = z.view(z.shape[0], 1, z.shape[1]) # if not using token-based network, add a dummy token dimension

        if self.hierarchy == 1:
            # expert coefficients \phi(G'@z)
            a1 = self.act(rearrange(self.bn[0](rearrange(einops.einsum(z, self.projs[0], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))

            ################## Bilinear (1 levels of experts)
            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1)], dim=-1)
            train = einops.einsum(a1, self.factors[0], 'b t x, r1 x r2 -> b t r1 r2')  # batched mode-2 product [expert mode]
            train = einops.einsum(
                einops.einsum(z, self.factors[1], 'b t x, r2 x r3 -> b t r2 r3'), # batched mode-2 product [input mode]
                train,
                'b t r2 r3, b t r1 r2 -> b t r1 r3'
            ) # matrix multiplication with the two contracted tt/tr cores
            train = einops.einsum(self.factors[2], train, 'r3 o r1, b t r1 r3 -> b t o') # final batched mode-2 product with the output core

        elif self.hierarchy == 2:
            # expert coefficients
            a1 = self.act(rearrange(self.bn[0](rearrange(einops.einsum(z, self.projs[0], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            a2 = self.act(rearrange(self.bn[1](rearrange(einops.einsum(z, self.projs[1], 'b t k, k n2 -> b t n2'), 'b t n2 -> b n2 t')), 'b n2 t -> b t n2'))

            ################## trilinear (2 levels of experts)
            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1)], dim=-1)
            train = einops.einsum(a1, self.factors[0], 'b t x, r1 x r2 -> b t r1 r2') # batched mode-2 product [1st expert mode]
            train = einops.einsum(
                train,
                einops.einsum(a2, self.factors[1], 'b t x, r2 x r3 -> b t r2 r3'), # batched mode-2 product [2nd expert mode]
                'b t r1 r2, b t r2 r3 -> b t r1 r3')
            train = einops.einsum(
                train,
                einops.einsum(z, self.factors[2], 'b t x, r3 x r4 -> b t r3 r4'),
                'b t r1 r3, b t r3 r4 -> b t r1 r4')
            train = einops.einsum(
                train,
                self.factors[3],
                'b t r1 r4, r4 o r1 -> b t o')

        elif self.hierarchy == 3:
            # expert coefficients
            a1 = self.act(rearrange(self.bn[0](rearrange(einops.einsum(z, self.projs[0], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            a2 = self.act(rearrange(self.bn[1](rearrange(einops.einsum(z, self.projs[1], 'b t k, k n2 -> b t n2'), 'b t n2 -> b n2 t')), 'b n2 t -> b t n2'))
            a3 = self.act(rearrange(self.bn[2](rearrange(einops.einsum(z, self.projs[2], 'b t k, k n3 -> b t n3'), 'b t n3 -> b n3 t')), 'b n3 t -> b t n3'))

            ################## multilinear (3 levels of experts)
            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1)], dim=-1)
            train = einops.einsum(
                a1,
                self.factors[0],
                'b t x, r1 x r2 -> b t r1 r2') 
            train = einops.einsum(
                train,
                einops.einsum(a2, self.factors[1], 'b t x, r2 x r3 -> b t r2 r3'),
                'b t r1 r2, b t r2 r3 -> b t r1 r3')
            train = einops.einsum(
                train,
                einops.einsum(a3, self.factors[2], 'b t x, r3 x r4 -> b t r3 r4'),
                'b t r1 r3, b t r3 r4 -> b t r1 r4')
            train = einops.einsum(
                train,
                einops.einsum(z, self.factors[3], 'b t x, r4 x r5 -> b t r4 r5'),
                'b t r1 r4, b t r4 r5 -> b t r1 r5')
            train = einops.einsum(
                train,
                self.factors[4],
                'b t r1 r5, r5 o r1 -> b t o')
        elif self.hierarchy == 4:
            a1 = self.act(rearrange(self.bn[0](rearrange(einops.einsum(z, self.projs[0], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            a2 = self.act(rearrange(self.bn[1](rearrange(einops.einsum(z, self.projs[1], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            a3 = self.act(rearrange(self.bn[2](rearrange(einops.einsum(z, self.projs[2], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            a4 = self.act(rearrange(self.bn[3](rearrange(einops.einsum(z, self.projs[3], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            ################## multilinear (3 levels of experts)
            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1)], dim=-1)
            train = einops.einsum(
                a1,
                self.factors[0],
                'b t x, r1 x r2 -> b t r1 r2') 
            train = einops.einsum(
                train,
                einops.einsum(a2, self.factors[1], 'b t x, r2 x r3 -> b t r2 r3'),
                'b t r1 r2, b t r2 r3 -> b t r1 r3')
            train = einops.einsum(
                train,
                einops.einsum(a3, self.factors[2], 'b t x, r3 x r4 -> b t r3 r4'),
                'b t r1 r3, b t r3 r4 -> b t r1 r4')
            train = einops.einsum(
                train,
                einops.einsum(a4, self.factors[3], 'b t x, r4 x r5 -> b t r4 r5'),
                'b t r1 r4, b t r4 r5 -> b t r1 r5')
            train = einops.einsum(
                train,
                einops.einsum(z, self.factors[4], 'b t x, r5 x r6 -> b t r5 r6'),
                'b t r1 r5, b t r5 r6 -> b t r1 r6')
            train = einops.einsum(
                train,
                self.factors[5],
                'b t r1 r6, r6 o r1 -> b t o')

        elif self.hierarchy > 4:
            print('ERROR: hierarchy > 4 not yet implemented')
            raise NotImplementedError

        return train.squeeze(dim=1)

class CPMMoE(nn.Module):
    def __init__(self, input_dim, output_dim, expert_dims, rank, act=entmax, use_bias=True, hierarchy=1):
        """
        input_dim: int, the input dimension of the vector (or the # channels for token-based networks)
        output_dim: int, the output dimension of the vector
        expert_dims: list of ints, the dimensions of the expert modes
        rank: int, the CP rank of the tensor
        act: activation function for the expert gating (e.g.entmax/softmax)
        use_bias: bool, whether to use a bias term (folded into the second-to-last "input" mode)
        hierarchy: int, the number of levels of hierarchy
        """
        super(CPMMoE, self).__init__()

        self.act = act
        self.factors = []
        self.projs = []
        self.hierarchy = hierarchy
        self.use_bias = use_bias
        dims = expert_dims + [input_dim+1, output_dim] if self.use_bias else expert_dims + [input_dim, output_dim]
        self.expert_dims = expert_dims

        # factors
        for fi, (d, r) in enumerate(zip(dims, [rank]*len(dims))):
            if fi < len(expert_dims):
                W = nn.Parameter(torch.randn(d, r)*1.00 + 1.0, requires_grad=True) # N(1,1)
                if self.hierarchy > 1 and fi != 0: # N(1,0) for additional expert modes
                    torch.nn.init.ones_(W)
            else:
                W = nn.Parameter(torch.randn(d, r), requires_grad=True)
                W.data = torch.nn.functional.normalize(W.data, p=2, dim=1)

            self.factors += [W]

        ############ expert gating params
        for di, d in enumerate(expert_dims):
            W = torch.nn.Parameter(torch.zeros(input_dim, d), requires_grad=True)
            self.projs += [W]

        self.factors = nn.ParameterList(self.factors)
        self.projs = nn.ParameterList(self.projs)
        self.bn = nn.ModuleList([torch.nn.BatchNorm1d(dim, affine=False) for dim in expert_dims])

    def forward(self, z):
        """
        Performs a forward pass through the CP-MMoE layer in factorized form (Eq. 2 in the main paper).

        Code is written to operate on (batch, tokens, channels)-dim input tensors, to handle token-based architectures.
        """
        if len(z.shape) == 2: z = z.view(z.shape[0], 1, z.shape[1])

        if self.hierarchy == 1:
            similarity = einops.einsum(z, self.projs[0], 'b t k, k n1 -> b t n1')
            a1 = self.act(rearrange(self.bn[0](rearrange(similarity, 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))

            f1 = einops.einsum(a1, self.factors[0], 'b t n1, n1 r -> b t r')
            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1)], dim=-1)
            final = einops.einsum(z, self.factors[1], 'b t i, i r -> b t r')
            out = einops.einsum(f1, final, self.factors[2], 'b t r, b t r, o r -> b t o')

        elif self.hierarchy == 2:
            a1 = self.act(rearrange(self.bn[0](rearrange(einops.einsum(z, self.projs[0], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            f1 = einops.einsum(a1, self.factors[0], 'b t n1, n1 r -> b t r')

            a2 = self.act(rearrange(self.bn[1](rearrange(einops.einsum(z, self.projs[1], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            f2 = einops.einsum(a2, self.factors[1], 'b t n1, n1 r -> b t r')

            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1)], dim=-1)
            final = einops.einsum(z, self.factors[2], 'b t i, i r -> b t r')
            out = einops.einsum(f1, f2, final, self.factors[3], 'b t r, b t r, b t r, o r -> b t o') # trilinear

        elif self.hierarchy == 3:
            a1 = self.act(rearrange(self.bn[0](rearrange(einops.einsum(z, self.projs[0], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            f1 = einops.einsum(a1, self.factors[0], 'b t n1, n1 r -> b t r')

            a2 = self.act(rearrange(self.bn[1](rearrange(einops.einsum(z, self.projs[1], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            f2 = einops.einsum(a2, self.factors[1], 'b t n1, n1 r -> b t r')

            a3 = self.act(rearrange(self.bn[2](rearrange(einops.einsum(z, self.projs[2], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            f3 = einops.einsum(a3, self.factors[2], 'b t n1, n1 r -> b t r')

            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1)], dim=-1)
            final = einops.einsum(z, self.factors[3], 'b t i, i r -> b t r')
            out = einops.einsum(f1, f2, f3, final, self.factors[4], 'b t r, b t r, b t r, b t r, o r -> b t o')
            # ##################

        elif self.hierarchy == 4:
            a1 = self.act(rearrange(self.bn[0](rearrange(einops.einsum(z, self.projs[0], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            f1 = einops.einsum(a1, self.factors[0], 'b t n1, n1 r -> b t r')

            a2 = self.act(rearrange(self.bn[1](rearrange(einops.einsum(z, self.projs[1], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            f2 = einops.einsum(a2, self.factors[1], 'b t n1, n1 r -> b t r')

            a3 = self.act(rearrange(self.bn[2](rearrange(einops.einsum(z, self.projs[2], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            f3 = einops.einsum(a3, self.factors[2], 'b t n1, n1 r -> b t r')

            a4 = self.act(rearrange(self.bn[3](rearrange(einops.einsum(z, self.projs[3], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            f4 = einops.einsum(a4, self.factors[3], 'b t n1, n1 r -> b t r')

            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1)], dim=-1)
            final = einops.einsum(z, self.factors[4], 'b t i, i r -> b t r')
            out = einops.einsum(f1, f2, f3, f4, final, self.factors[5], 'b t r, b t r, b t r, b t r, b t r, o r -> b t o')

        elif self.hierarchy > 4:
            print('ERROR: hierarchy > 4 not yet implemented for CP')
            raise NotImplementedError

        return out.squeeze(dim=1)

class TuckerMMoE(nn.Module):
    def __init__(self, input_dim, output_dim, expert_dims, ranks, act=entmax, use_bias=True, hierarchy=1):
        """
        input_dim: int, the input dimension of the vector (or the # channels for token-based networks)
        output_dim: int, the output dimension of the vector
        expert_dims: list of ints, the dimensions of the expert modes
        ranks: list of ints, (example format: [128, 128, 128]) the Tucker ranks of the tensor
        act: activation function for the expert gating (e.g.entmax/softmax)
        use_bias: bool, whether to use a bias term (folded into the second-to-last "input" mode)
        hierarchy: int, the number of levels of hierarchy
        """
        super(TuckerMMoE, self).__init__()

        self.act = act
        self.factors = []
        self.projs = []
        self.hierarchy = hierarchy
        self.use_bias = use_bias
        self.expert_dims = expert_dims

        dims = expert_dims + [input_dim+1, output_dim]

        # factors
        for fi, (d, r) in enumerate(zip(dims, ranks)):
            if fi < len(expert_dims):
                W = nn.Parameter(torch.randn(d, r)*1.0 + 1.0, requires_grad=True) # N(1,1)
                if self.hierarchy > 1 and fi != 0: 
                    torch.nn.init.ones_(W) # N(1,0) for additional expert modes
            else:
                W = torch.nn.Parameter(torch.randn(d, r), requires_grad=True)
                W.data = torch.nn.functional.normalize(W.data, p=2, dim=1)

            self.factors += [W]

        ############ expert gating matrices
        for d in expert_dims:
            W = torch.nn.Parameter(torch.zeros(input_dim, d), requires_grad=True)
            self.projs += [W]

        self.factors = nn.ParameterList(self.factors)
        self.projs = nn.ParameterList(self.projs)

        # core tensor (\mathcal{Z} in paper)
        self.core = nn.Parameter(torch.randn(*ranks), requires_grad=True)
        self.core.data = torch.nn.functional.normalize(self.core.data, p=2, dim=tuple(range(self.core.dim()))) 

        self.factors = nn.ParameterList(self.factors)
        self.bn = nn.ModuleList([torch.nn.BatchNorm1d(dim, affine=False) for dim in expert_dims])

    def forward(self, z):
        """
        Performs a forward pass through the Tucker-MMoE layer in factorized form (Eq. 3 in the main paper).

        Code is written to operate on (batch, tokens, channels)-dim input tensors, to handle token-based architectures.
        """
        if len(z.shape) == 2: z = z.view(z.shape[0], 1, z.shape[1])

        if self.hierarchy == 1:
            a1 = self.act(rearrange(self.bn[0](rearrange(einops.einsum(z, self.projs[0], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))

            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1)], dim=-1)

            # contract over "input" mode first, confusing order but more RAM-efficient
            out = einops.einsum(self.core, z@self.factors[1], 'rn1 ri ro, b t ri -> b t rn1 ro')
            # then: contract over the expert dim
            out = einops.einsum(out, a1@self.factors[0], 'b t rn1 ro, b t rn1 -> b t ro')
            # then contract over output mode
            out = einops.einsum(out, self.factors[2], 'b t ro, o ro -> b t o')

        elif self.hierarchy == 2:
            a1 = self.act(rearrange(self.bn[0](rearrange(einops.einsum(z, self.projs[0], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            a2 = self.act(rearrange(self.bn[1](rearrange(einops.einsum(z, self.projs[1], 'b t k, k n1 -> b t n1'), 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))

            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1)], dim=-1)
            out = einops.einsum(self.core, a1@ self.factors[0], 'rn1 rn2 ri ro, b t rn1 -> b t rn2 ri ro')
            out = einops.einsum(out, a2@ self.factors[1], 'b t rn2 ri ro, b t rn2 -> b t ri ro')
            out = einops.einsum(out, z@self.factors[2], 'b t ri ro, b t ri -> b t ro')
            out = einops.einsum(out, self.factors[3], 'b t ro, o ro -> b t o')

        elif self.hierarchy == 3:
            a1 = self.act(rearrange(self.bn[0](rearrange(z @ self.projs[0], 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            a2 = self.act(rearrange(self.bn[1](rearrange(z @ self.projs[1], 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            a3 = self.act(rearrange(self.bn[2](rearrange(z @ self.projs[2], 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))

            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1)], dim=-1)
            out = einops.einsum(self.core, a1@self.factors[0], 'rn1 rn2 rn3 ri ro, b t rn1 -> b t rn2 rn3 ri ro')
            out = einops.einsum(out, a2@self.factors[1], 'b t rn2 rn3 ri ro, b t rn2 -> b t rn3 ri ro')
            out = einops.einsum(out, a3@self.factors[2], 'b t rn3 ri ro, b t rn3 -> b t ri ro')
            out = einops.einsum(out, z@self.factors[3], 'b t ri ro, b t ri -> b t ro')
            out = einops.einsum(out, self.factors[4], 'b t ro, o ro -> b t o')

        elif self.hierarchy == 4:
            a1 = self.act(rearrange(self.bn[0](rearrange(z @ self.projs[0], 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            a2 = self.act(rearrange(self.bn[1](rearrange(z @ self.projs[1], 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            a3 = self.act(rearrange(self.bn[2](rearrange(z @ self.projs[2], 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))
            a4 = self.act(rearrange(self.bn[3](rearrange(z @ self.projs[3], 'b t n1 -> b n1 t')), 'b n1 t -> b t n1'))

            if self.use_bias: z = torch.cat([z, torch.ones(z.shape[0], z.shape[1], 1)], dim=-1)
            out = einops.einsum(self.core, a1@self.factors[0], 'rn1 rn2 rn3 rn4 ri ro, b t rn1 -> b t rn2 rn3 rn4 ri ro')
            out = einops.einsum(out, a2@self.factors[1], 'b t rn2 rn3 rn4 ri ro, b t rn2 -> b t rn3 rn4 ri ro')
            out = einops.einsum(out, a3@self.factors[2], 'b t rn3 rn4 ri ro, b t rn3 -> b t rn4 ri ro')
            out = einops.einsum(out, a4@self.factors[3], 'b t rn4 ri ro, b t rn4 -> b t ri ro')
            out = einops.einsum(out, z@self.factors[4], 'b t ri ro, b t ri -> b t ro')
            out = einops.einsum(out, self.factors[5], 'b t ro, o ro -> b t o')

        elif self.hierarchy > 4:
            print('ERROR: hierarchy > 4 not yet implemented for Tucker')
            raise NotImplementedError

        return out.squeeze(dim=1)