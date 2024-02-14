from MMoE import CPMMoE, TTMMoE, TuckerMMoE
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
    model = CPMMoE(in_dim, out_dim, expert_dims=[n_experts], rank=rank, act=act).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.cp_tensor.cp_to_tensor((torch.ones(rank).to(device), model.factors))
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
    model = CPMMoE(in_dim, out_dim, expert_dims=[n_experts, 2], rank=rank, act=act, hierarchy=2).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.cp_tensor.cp_to_tensor((torch.ones(rank).to(device), model.factors))
    # generate expert coefficients
    a1 = model.act(model.bn[0](x @ model.projs[0]))
    a2 = model.act(model.bn[1](x @ model.projs[1]))

    b = W[:, :, -1, :] # get the bias term
    W = W[:, :, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a1, a2, x, 'n1 n2 i o, b n1, b n2, b i -> b o') + einsum(b, a1, a2, 'n1 n2 o, b n1, b n2 -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=2) factorized==unfactorized forward pass test passed')

    ###########################
    ########################### hierarchy E=3
    ###########################

    ############ model output
    model = CPMMoE(in_dim, out_dim, expert_dims=[n_experts, 2, 2], rank=rank, act=act, hierarchy=3).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.cp_tensor.cp_to_tensor((torch.ones(rank).to(device), model.factors))
    # generate expert coefficients
    a1 = model.act(model.bn[0](x @ model.projs[0]))
    a2 = model.act(model.bn[1](x @ model.projs[1]))
    a3 = model.act(model.bn[2](x @ model.projs[2]))

    b = W[:, :, :, -1, :] # get the bias term
    W = W[:, :, :, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a1, a2, a3, x, 'n1 n2 n3 i o, b n1, b n2, b n3, b i -> b o') + einsum(b, a1, a2, a3, 'n1 n2 n3 o, b n1, b n2, b n3 -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=3) factorized==unfactorized forward pass test passed')

    ###########################
    ########################### hierarchy E=4
    ###########################

    ############ model output
    model = CPMMoE(in_dim, out_dim, expert_dims=[n_experts, 2, 2, 2], rank=rank, act=act, hierarchy=4).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.cp_tensor.cp_to_tensor((torch.ones(rank).to(device), model.factors))
    # generate expert coefficients
    a1 = model.act(model.bn[0](x @ model.projs[0]))
    a2 = model.act(model.bn[1](x @ model.projs[1]))
    a3 = model.act(model.bn[2](x @ model.projs[2]))
    a4 = model.act(model.bn[3](x @ model.projs[3]))

    b = W[:, :, :, :, -1, :] # get the bias term
    W = W[:, :, :, :, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a1, a2, a3, a4, x, 'n1 n2 n3 n4 i o, b n1, b n2, b n3, b n4, b i -> b o') + einsum(b, a1, a2, a3, a4, 'n1 n2 n3 n4 o, b n1, b n2, b n3, b n4 -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=4) factorized==unfactorized forward pass test passed')

print('Testing TuckerMMoE...')
with torch.no_grad():
    x = torch.randn(batch_size, in_dim).to(device)

    ###########################
    ########################### hierarchy E=1
    ###########################
    ranks = [4]*3

    ############ model output
    model = TuckerMMoE(in_dim, out_dim, expert_dims=[n_experts], ranks=ranks, act=act).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.tucker_tensor.tucker_to_tensor((model.core, model.factors))
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

    ranks = [4]*4
    ############ model output
    model = TuckerMMoE(in_dim, out_dim, expert_dims=[n_experts, 2], ranks=ranks, act=act, hierarchy=2).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.tucker_tensor.tucker_to_tensor((model.core, model.factors))
    # generate expert coefficients
    a1 = model.act(model.bn[0](x @ model.projs[0]))
    a2 = model.act(model.bn[1](x @ model.projs[1]))

    b = W[:, :, -1, :] # get the bias term
    W = W[:, :, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a1, a2, x, 'n1 n2 i o, b n1, b n2, b i -> b o') + einsum(b, a1, a2, 'n1 n2 o, b n1, b n2 -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=2) factorized==unfactorized forward pass test passed')

    ###########################
    ########################### hierarchy E=3
    ###########################

    ranks = [4]*5
    ############ model output
    model = TuckerMMoE(in_dim, out_dim, expert_dims=[n_experts, 2, 2], ranks=ranks, act=act, hierarchy=3).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.tucker_tensor.tucker_to_tensor((model.core, model.factors))
    # generate expert coefficients
    a1 = model.act(model.bn[0](x @ model.projs[0]))
    a2 = model.act(model.bn[1](x @ model.projs[1]))
    a3 = model.act(model.bn[2](x @ model.projs[2]))

    b = W[:, :, :, -1, :] # get the bias term
    W = W[:, :, :, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a1, a2, a3, x, 'n1 n2 n3 i o, b n1, b n2, b n3, b i -> b o') + einsum(b, a1, a2, a3, 'n1 n2 n3 o, b n1, b n2, b n3 -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=3) factorized==unfactorized forward pass test passed')

    ###########################
    ########################### hierarchy E=4
    ###########################

    ranks = [4]*6
    ############ model output
    model = TuckerMMoE(in_dim, out_dim, expert_dims=[n_experts, 2, 2, 2], ranks=ranks, act=act, hierarchy=4).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.tucker_tensor.tucker_to_tensor((model.core, model.factors))
    # generate expert coefficients
    a1 = model.act(model.bn[0](x @ model.projs[0]))
    a2 = model.act(model.bn[1](x @ model.projs[1]))
    a3 = model.act(model.bn[2](x @ model.projs[2]))
    a4 = model.act(model.bn[3](x @ model.projs[3]))

    b = W[:, :, :, :, -1, :] # get the bias term
    W = W[:, :, :, :, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a1, a2, a3, a4, x, 'n1 n2 n3 n4 i o, b n1, b n2, b n3, b n4, b i -> b o') + einsum(b, a1, a2, a3, a4, 'n1 n2 n3 n4 o, b n1, b n2, b n3, b n4 -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=4) factorized==unfactorized forward pass test passed')


print('Testing TTMMoE...')
with torch.no_grad():
    x = torch.randn(batch_size, in_dim).to(device)

    ###########################
    ########################### hierarchy E=1
    ###########################
    r1 = 1 ; r2 = 4 ; r3 = 4
    ranks = [[r1, n_experts, r2], [r2, in_dim, r3], [r3, out_dim, r1]]
    ############ model output
    model = TTMMoE(in_dim, out_dim, expert_dims=[n_experts], ranks=ranks, act=act).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.tt_tensor.tt_to_tensor(model.factors)
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
    r1 = 1 ; r2 = 4 ; r3 = 4 ; r4 = 4
    ranks = [[r1, n_experts, r2], [r2, 2, r3], [r3, in_dim, r4], [r4, out_dim, r1]]
    ############ model output
    model = TTMMoE(in_dim, out_dim, expert_dims=[n_experts, 2], ranks=ranks, act=act, hierarchy=2).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.tt_tensor.tt_to_tensor(model.factors)
    # generate expert coefficients
    a1 = model.act(model.bn[0](x @ model.projs[0]))
    a2 = model.act(model.bn[1](x @ model.projs[1]))

    b = W[:, :, -1, :] # get the bias term
    W = W[:, :, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a1, a2, x, 'n1 n2 i o, b n1, b n2, b i -> b o') + einsum(b, a1, a2, 'n1 n2 o, b n1, b n2 -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=2) factorized==unfactorized forward pass test passed')

    ###########################
    ########################### hierarchy E=3
    ###########################
    r1 = 1 ; r2 = 4 ; r3 = 4 ; r4 = 4; r5 = 4
    ranks = [[r1, n_experts, r2], [r2, 2, r3], [r3, 2, r4], [r4, in_dim, r5], [r5, out_dim, r1]]
    ############ model output
    model = TTMMoE(in_dim, out_dim, expert_dims=[n_experts, 2, 2], ranks=ranks, act=act, hierarchy=3).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.tt_tensor.tt_to_tensor(model.factors)
    # generate expert coefficients
    a1 = model.act(model.bn[0](x @ model.projs[0]))
    a2 = model.act(model.bn[1](x @ model.projs[1]))
    a3 = model.act(model.bn[2](x @ model.projs[2]))

    b = W[:, :, :, -1, :] # get the bias term
    W = W[:, :, :, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a1, a2, a3, x, 'n1 n2 n3 i o, b n1, b n2, b n3, b i -> b o') + einsum(b, a1, a2, a3, 'n1 n2 n3 o, b n1, b n2, b n3 -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=3) factorized==unfactorized forward pass test passed')

    ###########################
    ########################### hierarchy E=4
    ###########################
    r1 = 1 ; r2 = 4 ; r3 = 4 ; r4 = 4; r5 = 4; r6 = 4
    ranks = [[r1, n_experts, r2], [r2, 2, r3], [r3, 2, r4], [r4, 2, r5], [r5, in_dim, r6], [r6, out_dim, r1]]
    ############ model output
    model = TTMMoE(in_dim, out_dim, expert_dims=[n_experts, 2, 2, 2], ranks=ranks, act=act, hierarchy=4).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.tt_tensor.tt_to_tensor(model.factors)
    # generate expert coefficients
    a1 = model.act(model.bn[0](x @ model.projs[0]))
    a2 = model.act(model.bn[1](x @ model.projs[1]))
    a3 = model.act(model.bn[2](x @ model.projs[2]))
    a4 = model.act(model.bn[3](x @ model.projs[3]))

    b = W[:, :, :, :, -1, :] # get the bias term
    W = W[:, :, :, :, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a1, a2, a3, a4, x, 'n1 n2 n3 n4 i o, b n1, b n2, b n3, b n4, b i -> b o') + einsum(b, a1, a2, a3, a4, 'n1 n2 n3 n4 o, b n1, b n2, b n3, b n4 -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=4) factorized==unfactorized forward pass test passed')

print('Testing TRMMoE...')
with torch.no_grad():
    x = torch.randn(batch_size, in_dim).to(device)

    ###########################
    ########################### hierarchy E=1
    ###########################
    r1 = 4 ; r2 = 4 ; r3 = 4
    ranks = [[r1, n_experts, r2], [r2, in_dim, r3], [r3, out_dim, r1]]
    ############ model output
    model = TTMMoE(in_dim, out_dim, expert_dims=[n_experts], ranks=ranks, act=act).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.tr_tensor.tr_to_tensor(model.factors)
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
    model = TTMMoE(in_dim, out_dim, expert_dims=[n_experts, 2], ranks=ranks, act=act, hierarchy=2).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.tr_tensor.tr_to_tensor(model.factors)
    # generate expert coefficients
    a1 = model.act(model.bn[0](x @ model.projs[0]))
    a2 = model.act(model.bn[1](x @ model.projs[1]))

    b = W[:, :, -1, :] # get the bias term
    W = W[:, :, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a1, a2, x, 'n1 n2 i o, b n1, b n2, b i -> b o') + einsum(b, a1, a2, 'n1 n2 o, b n1, b n2 -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=2) factorized==unfactorized forward pass test passed')

    ###########################
    ########################### hierarchy E=3
    ###########################
    r1 = 4 ; r2 = 4 ; r3 = 4 ; r4 = 4; r5 = 4
    ranks = [[r1, n_experts, r2], [r2, 2, r3], [r3, 2, r4], [r4, in_dim, r5], [r5, out_dim, r1]]
    ############ model output
    model = TTMMoE(in_dim, out_dim, expert_dims=[n_experts, 2, 2], ranks=ranks, act=act, hierarchy=3).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.tr_tensor.tr_to_tensor(model.factors)
    # generate expert coefficients
    a1 = model.act(model.bn[0](x @ model.projs[0]))
    a2 = model.act(model.bn[1](x @ model.projs[1]))
    a3 = model.act(model.bn[2](x @ model.projs[2]))

    b = W[:, :, :, -1, :] # get the bias term
    W = W[:, :, :, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a1, a2, a3, x, 'n1 n2 n3 i o, b n1, b n2, b n3, b i -> b o') + einsum(b, a1, a2, a3, 'n1 n2 n3 o, b n1, b n2, b n3 -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=3) factorized==unfactorized forward pass test passed')

    ###########################
    ########################### hierarchy E=4
    ###########################
    r1 = 4 ; r2 = 4 ; r3 = 4 ; r4 = 4; r5 = 4; r6 = 4
    ranks = [[r1, n_experts, r2], [r2, 2, r3], [r3, 2, r4], [r4, 2, r5], [r5, in_dim, r6], [r6, out_dim, r1]]
    ############ model output
    model = TTMMoE(in_dim, out_dim, expert_dims=[n_experts, 2, 2, 2], ranks=ranks, act=act, hierarchy=4).to(device)
    model.eval()
    out = model(x)

    ############ unfactorized output
    W = tl.tr_tensor.tr_to_tensor(model.factors)
    # generate expert coefficients
    a1 = model.act(model.bn[0](x @ model.projs[0]))
    a2 = model.act(model.bn[1](x @ model.projs[1]))
    a3 = model.act(model.bn[2](x @ model.projs[2]))
    a4 = model.act(model.bn[3](x @ model.projs[3]))

    b = W[:, :, :, :, -1, :] # get the bias term
    W = W[:, :, :, :, :-1, :] # get just the linear transformation weights

    # compute the raw Eq. (1) series of tensor contractions
    real_out = einsum(W, a1, a2, a3, a4, x, 'n1 n2 n3 n4 i o, b n1, b n2, b n3, b n4, b i -> b o') + einsum(b, a1, a2, a3, a4, 'n1 n2 n3 n4 o, b n1, b n2, b n3, b n4 -> b o')

    torch.testing.assert_close(out, real_out)
    print('...(hierarchy=4) factorized==unfactorized forward pass test passed')