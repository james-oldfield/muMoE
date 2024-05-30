from MuMoE import MuMoE, CPMuMoE, TRMuMoE
import tensorly as tl
from torch import nn
from einops import einsum
import torch
tl.set_backend('pytorch')

from entmax import entmax15
act = lambda x: entmax15(x, dim=-1)

n_experts = 16
batch_size = 4
in_dim = 4
out_dim = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

# test the layer's factorised forward pass computation matches that of the fully materialised tensor.

print('Testing CPMMoE...')
with torch.no_grad():
    x = torch.randn(batch_size, in_dim).to(device)

    ###########################
    ########################### hierarchy E=1
    ###########################
    rank = 16

    ############ model output
    model = MuMoE(input_dim=in_dim, output_dim=out_dim, MuMoE_layer=CPMuMoE, expert_dims=[n_experts], ranks=rank, act=act, normalization='bn')
    model.eval()
    out = model(x).squeeze(1)

    ############ unfactorized output
    W = tl.cp_tensor.cp_to_tensor((torch.ones(rank).to(device), model.MuMoE.factors))
    a = model.act(model.bn[0](x @ model.projs[0])) # generate mixture weights

    b = W[:, -1, :] # get the bias term
    W = W[:, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a, x, 'n i o, b n, b i -> b o') + einsum(b, a, 'n o, b n -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=1) factorized==unfactorized forward pass test passed')

    ###########################
    ########################### hierarchy E=2
    ###########################

    ############ model output
    model = MuMoE(input_dim=in_dim, output_dim=out_dim, MuMoE_layer=CPMuMoE, expert_dims=[n_experts, 2], ranks=rank, act=act, normalization='bn', hierarchy=2)
    model.eval()
    out = model(x).squeeze(1)

    ############ unfactorized output
    W = tl.cp_tensor.cp_to_tensor((torch.ones(rank).to(device), model.MuMoE.factors))
    # generate expert coefficients
    a1 = model.act(model.bn[0](x @ model.projs[0]))
    a2 = model.act(model.bn[1](x @ model.projs[1]))

    b = W[:, :, -1, :] # get the bias term
    W = W[:, :, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a1, a2, x, 'n1 n2 i o, b n1, b n2, b i -> b o') + einsum(b, a1, a2, 'n1 n2 o, b n1, b n2 -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=2) factorized==unfactorized forward pass test passed')

print('Testing TRMMoE...')
with torch.no_grad():
    x = torch.randn(batch_size, in_dim).to(device)

    ###########################
    ########################### hierarchy E=1
    ###########################
    r1 = 4 ; r2 = 4 ; r3 = 4
    ranks = [[r1, n_experts, r2], [r2, in_dim, r3], [r3, out_dim, r1]]
    ############ model output
    model = MuMoE(input_dim=in_dim, output_dim=out_dim, MuMoE_layer=TRMuMoE, expert_dims=[n_experts], ranks=ranks, act=act, normalization='bn')
    model.eval()
    out = model(x).squeeze(1)

    ############ unfactorized output
    W = tl.tr_tensor.tr_to_tensor(model.MuMoE.factors)
    a = model.act(model.bn[0](x @ model.projs[0])) # generate mixture weights

    b = W[:, -1, :] # get the bias term
    W = W[:, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a, x, 'n i o, b n, b i -> b o') + einsum(b, a, 'n o, b n -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=1) factorized==unfactorized forward pass test passed')

    ###########################
    ########################### hierarchy E=2
    ###########################
    r1 = 4 ; r2 = 4 ; r3 = 4 ; r4 = 4
    ranks = [[r1, n_experts, r2], [r2, 2, r3], [r3, in_dim, r4], [r4, out_dim, r1]]
    ############ model output
    model = MuMoE(input_dim=in_dim, output_dim=out_dim, MuMoE_layer=TRMuMoE, expert_dims=[n_experts, 2], ranks=ranks, act=act, normalization='bn', hierarchy=2)
    model.eval()
    out = model(x).squeeze(1)

    ############ unfactorized output
    W = tl.tr_tensor.tr_to_tensor(model.MuMoE.factors)
    # generate expert coefficients
    a1 = model.act(model.bn[0](x @ model.projs[0]))
    a2 = model.act(model.bn[1](x @ model.projs[1]))

    b = W[:, :, -1, :] # get the bias term
    W = W[:, :, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a1, a2, x, 'n1 n2 i o, b n1, b n2, b i -> b o') + einsum(b, a1, a2, 'n1 n2 o, b n1, b n2 -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=2) factorized==unfactorized forward pass test passed')