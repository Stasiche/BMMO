# There should be no main() in this file!!! 
# Nothing should start running when you import this file somewhere.
# You may add other supporting functions to this file.
#
# Important rules:
# 1) Function pa_bc must return tensor which has dimensions (#a x #b x #c),
#    where #v is a number of different values of the variable v.
#    For input variables #v = how many input values of this variable you give to the function.
#    For output variables #v = number of all possible values of this variable.
#    Ex. for pb_a: #b = bmax-bmin+1,   #a is arbitrary.
# 2) Random variables in function names must be written in alphabetic order
#    e.g. pda_cb is an improper function name (pad_bc must be used instead)
# 3) Single dimension must be explicitly stated:
#    if you give only one value of a variable a to the function pb_a, i.e. #a=1, 
#    then the function pb_a must return tensor of shape (#b, 1), not (#b,).
#
# The format of all the functions for distributions is the following:
# Inputs:
# params - dictionary with keys 'amin', 'amax', 'bmin', 'bmax', 'p1', 'p2', 'p3'
# model - model number, number from 1 to 4
# all other parameters - values of the conditions (variables a, b, c, d).
#                        Numpy vectors of size (k,), where k is an arbitrary number.
#                        For variant 3: c and d must be numpy arrays of size (k,N),
#                        where N is a number of lectures.
# Outputs:
# prob, val
# prob - probabilities for different values of the output variable with different input conditions
#        prob[i,...] = p(v=val[i]|...)
# val - support of a distribution, numpy vector of size (#v,) for variable v
#
# Example 1:
#    Function pc_ab - distribution p(c|a,b)
#    Input: a of size (k_a,) and b of size (k_b,)
#    Result: prob of size (cmax-cmin+1,k_a,k_b), val of size (cmax-cmin+1,) 
#
# Example 2 (for variant 3):
#    Function pb_ad - distribution p(b|a,d_1,...,d_N)
#    Input: a of size (k_a,) and d of size (k_d,N)
#    Result: prob of size (bmax-bmin+1,k_a,k_d), val of size (bmax-bmin+1,)
#
# The format the generation function from variant 3 is the following:
# Inputs:
# N - how many points to generate
# all other inputs have the same format as earlier
# Outputs:
# d - generated values of d, numpy array of size (N,#a,#b)
import numpy as np
from scipy.stats import binom, poisson


# In variant 1 the following functions are required:
def pa(params, model):
    l, r = params['amin'], params['amax']
    val = np.arange(l, r + params['step'], step = params['step'])
    prob = 1 / (r - l + 1) * np.ones_like(val)

    return prob, val


def pb(params, model):
    l, r = params['bmin'], params['bmax']
    val = np.arange(l, r + 1, step = params['step'])
    prob = 1 / (r - l + 1) * np.ones_like(val)

    return prob, val


def pc_ab(a, b, params, model):
    c_val = np.arange(0, params['amax'] + params['bmax'] + 1, step = params['step'])
    c_val_t = c_val[:, None]
    a_val = a[None, :]
    b_val = b[None, :]
    if model == 1:
        prob_a = binom.pmf(c_val_t, a_val, params['p1'])
        prob_b = binom.pmf(c_val_t, b_val, params['p2'])
    else:
        prob_a = poisson.pmf(c_val_t, a_val * params['p1'])
        prob_b = poisson.pmf(c_val_t, b_val * params['p2'])

    prob = np.zeros((c_val.size, a.size, b.size))
    for k in range(len(c_val)):
        prob[k] = np.dot(prob_a[:k + 1].T, prob_b[:k + 1][::-1])

    return prob, c_val


def pd_c(c, params, model):
    d_l, d_r = 0, 2 * (params['amax'] + params['bmax'])
    d_val = np.arange(d_l, d_r + 1, step = params['step'])

    k = d_val[:, None] - c[None, :]
    prob = binom.pmf(k, c, params['p3'])

    return prob, d_val


##########################################

def pc_a(a, params, model):
    c_l, c_r = 0, params['amax'] + params['bmax']
    c_val = np.arange(c_l, c_r + 1, step = params['step'])

    b_l, b_r = params['bmin'], params['bmax']
    b_val = np.arange(b_l, b_r + 1, step = params['step'])
    prob = np.sum(pc_ab(a, b_val, params, model)[0] * pb(params, model)[0][0], axis=2)

    return prob, c_val


def pc_b(b, params, model):
    c_l, c_r = 0, params['amax'] + params['bmax']
    c_val = np.arange(c_l, c_r + 1, step = params['step'])

    a_l, a_r = params['amin'], params['amax']
    a_val = np.arange(a_l, a_r + 1, step = params['step'])
    prob = np.sum(pc_ab(a_val, b, params, model)[0] * pa(params, model)[0][0], axis=1)

    return prob, c_val


def pc_d(d, params, model):
    c_l, c_r = 0, params['amax'] + params['bmax']
    c_val = np.arange(c_l, c_r + 1, step = params['step'])

    prob = pd_c(c_val, params, model)[0][d] * pc(params, model)[0]
    prob /= np.sum(prob, axis=1, keepdims=True)

    return prob.transpose(1, 0), c_val


def pc(params, model):
    c_l, c_r = 0, params['amax'] + params['bmax']
    c_val = np.arange(c_l, c_r + 1, step = params['step'])

    a_l, a_r = params['amin'], params['amax']
    a_val = np.arange(a_l, a_r + 1, step = params['step'])
    prob = np.sum(pc_a(a_val, params, model)[0] * pa(params, model)[0][0], axis=1)

    return prob, c_val


def pd(params, model):
    d_l, d_r = 0, 2 * (params['amax'] + params['bmax'])
    d_val = np.arange(d_l, d_r + 1, step = params['step'])

    c_l, c_r = 0, params['amax'] + params['bmax']
    c_val = np.arange(c_l, c_r + 1, step = params['step'])

    prob = np.sum(pd_c(c_val, params, model)[0] * pc(params, model)[0][None, :], axis=1)

    return prob, d_val


def pc_abd(a, b, d, params, model):
    c_l, c_r = 0, params['amax'] + params['bmax']
    c_val = np.arange(c_l, c_r + 1, step = params['step'])

    pc_ab_, _ = pc_ab(a, b, params, model)
    pd_c_prob = pd_c(c_val, params, model)[0][d]

    LP = pd_c_prob[:, :, None, None] * pc_ab_[None, :, :, :]
    evid = np.tensordot(pd_c_prob, pc_ab_, axes=(1, 0))

    prob = (LP / evid[:, None, :, :]).transpose(1, 2, 3, 0)

    return prob, c_val
