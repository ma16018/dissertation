import sys
sys.path.append('../')

import numpy as np
import pandas as pd

import argparse
import json
import os
import pickle

from mcmc import PMMH, CorrPMMH, CorrPMMH2, CorrPMMHIID, CorrPMMHIIDComplex
from particles.mcmc import BasicRWHM

from nig_iid import NIGIID
from gamma_iid_incr import GammaIIDIncr
from ns_iid_incr import NSIIDIncr
from gbfry_iid_incr import GBFRYIIDIncr, GBFRYIIDIncrParts
from ghyperbolic_iid import GHDIIDIncr
from student_iid import StudentIIDIncr
from vgamma3_iid import VGamma3IID
from vgamma4_iid import VGamma4IID

from correlated import SMCCorrelated, SMCCorrelated2

parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str, default=None)
parser.add_argument('--model', type=str, default='gbfry', choices=['gbfry', 'gamma', 'nig', 'ns',
                                                                   'ghd', 'student', 'vgamma3', 'vgamma4'])

# for PMMH
parser.add_argument('--save_states', type=bool, default=True) #action='store_true')
parser.add_argument('--Nx', type=int, default=500)
parser.add_argument('--burnin', type=int, default=None)
parser.add_argument('--niter', type=int, default=5000)
parser.add_argument('--verbose', type=int, default=100)

# for saving
parser.add_argument('--run_name', type=str, default='trial')

args = parser.parse_args()

if args.filename is None:
    raise ValueError('You must specify data')
else:
    data = os.path.splitext(os.path.basename(args.filename))[0]

save_dir = os.path.join('results', args.model, data, args.run_name)

with open(os.path.join(args.filename), 'rb') as f:
    datafile = pickle.load(f, encoding='latin1')

# Set to false for the models using particles SMC
mh_flag = False

# Remove the hour slots where the volume is abnormal
datafile = datafile[datafile.Volume >= 200]

y = datafile['y']
y = y / np.std(y) #W

ssm_options = {}

if args.model == 'gamma':
    ssm_cls = GammaIIDIncr
elif args.model == 'gbfry':
    ssm_cls = GBFRYIIDIncr
elif args.model == 'ns':
    ssm_cls = NSIIDIncr
elif args.model == 'vgamma3':
    mh_flag = True
    ssm_cls = VGamma3IID
elif args.model == 'vgamma4':
    mh_flag = True
    ssm_cls = VGamma4IID
elif args.model == 'nig':
    ssm_cls = NIGIID
elif args.model == 'student':
    ssm_cls = StudentIIDIncr
elif args.model == 'ghd':
    ssm_cls = GHDIIDIncr
else:
    raise NotImplementedError

prior = ssm_cls.get_prior()
theta0 = ssm_cls.get_theta0(y)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# if mh_flag:
#     model = ssm_cls(data=y, prior=prior)
#     pmmh = BasicRWHM(niter=args.niter, verbose=args.niter/args.verbose,
#                       theta0=theta0, model=model)
# else:
#     pmmh = PMMH(ssm_cls=ssm_cls, data=y,
#             prior=prior, theta0=ssm_cls.get_theta0(y),
#             Nx=args.Nx, niter=args.niter, keep_states=args.save_states,
#             ssm_options=ssm_options, verbose=args.niter/args.verbose)
# pmmh.run()

# burnin = args.burnin or args.niter // 2

# print("Saving chains")
# with open(os.path.join(save_dir, 'chain.pkl'), 'wb') as f:
#     pickle.dump(pmmh.chain, f)

# if args.save_states:
#     x = np.stack(pmmh.states, 0)
#     print("Saving states")
#     with open(os.path.join(save_dir, 'states.pkl'), 'wb') as f:
#         pickle.dump(x, f)
#     print("Saving acceptance")
#     a = np.stack(pmmh.acceptance_rates, 0)
#     b = np.stack(pmmh.acceptance_nums, 0)
#     with open(os.path.join(save_dir, 'acceptance_number.pkl'), 'wb') as f:
#         pickle.dump(b, f)
#     with open(os.path.join(save_dir, 'acceptance_rate.pkl'), 'wb') as f:
#         pickle.dump(a, f)

# Correlated
# pmmh = CorrPMMH(ssm_cls=ssm_cls, data=y,
#         prior=prior, theta0=ssm_cls.get_theta0(y),
#         Nx=args.Nx, niter=args.niter,
#         verbose=args.niter/args.verbose)
# pmmh.run()
# if args.save_states:
#     x = np.stack(pmmh.states, 0)
#     print("Saving states")
#     with open(os.path.join(save_dir, 'states.pkl'), 'wb') as f:
#         pickle.dump(x, f)

# print("Saving chains")
# with open(os.path.join(save_dir, 'chain.pkl'), 'wb') as f:
#     pickle.dump(pmmh.chain, f)
    

# Correlated 2
# pmmh = CorrPMMH2(ssm_cls=ssm_cls, data=y,
#         prior=prior, theta0=ssm_cls.get_theta0(y),
#         Nx=args.Nx, niter=args.niter,
#         verbose=args.niter/args.verbose, smc_cls=SMCCorrelated2)
# pmmh.run()
# if args.save_states:
#     x = np.stack(pmmh.states, 0)
#     print("Saving states")
#     with open(os.path.join(save_dir, 'states.pkl'), 'wb') as f:
#         pickle.dump(x, f)

# print("Saving chains")
# with open(os.path.join(save_dir, 'chain.pkl'), 'wb') as f:
#     pickle.dump(pmmh.chain, f)

# Correlated 3
perc = 0.3  # do max 0.3
pmmh = CorrPMMHIID(ssm_cls=ssm_cls, data=y,
        prior=prior, theta0=ssm_cls.get_theta0(y),
        Nx=args.Nx, niter=args.niter,
        verbose=args.niter/args.verbose, perc=perc)
pmmh.run()
print("Saving chains")
with open(os.path.join(save_dir, 'chain_corr_{}.pkl'.format(perc)), 'wb') as f:
    pickle.dump(pmmh.chain, f)
print("Saving acceptance")
a = np.stack(pmmh.acceptance_rates, 0)
b = np.stack(pmmh.acceptance_nums, 0)
with open(os.path.join(save_dir, 'acceptance_rate_corr_{}.pkl'.format(perc)), 'wb') as f:
    pickle.dump(a, f)
with open(os.path.join(save_dir, 'acceptance_number_corr_{}.pkl'.format(perc)), 'wb') as f:
    pickle.dump(b, f)
print("Saving states")
x = np.stack(pmmh.states, 0)
with open(os.path.join(save_dir, 'states_corr_{}.pkl'.format(perc)), 'wb') as f:
    pickle.dump(x, f)


# Correlated 4
# perc=2
# ssm_cls = GBFRYIIDIncrParts
# pmmh = CorrPMMHIIDComplex(ssm_cls=ssm_cls, data=y,
#         prior=prior, theta0=ssm_cls.get_theta0(y),
#         Nx=args.Nx, niter=args.niter,
#         verbose=args.niter/args.verbose, parts=perc)
# pmmh.run()
# print("Saving chains")
# with open(os.path.join(save_dir, 'chain_corr_{}.pkl'.format(perc)), 'wb') as f:
#     pickle.dump(pmmh.chain, f)
# print("Saving acceptance")
# a = np.stack(pmmh.acceptance_rates, 0)
# b = np.stack(pmmh.acceptance_nums, 0)
# with open(os.path.join(save_dir, 'acceptance_rate_corr_{}.pkl'.format(perc)), 'wb') as f:
#     pickle.dump(a, f)
# with open(os.path.join(save_dir, 'acceptance_number_corr_{}.pkl'.format(perc)), 'wb') as f:
#     pickle.dump(b, f)
# print("Saving states")
# x = np.stack(pmmh.states, 0)
# with open(os.path.join(save_dir, 'states_corr_{}.pkl'.format(perc)), 'wb') as f:
#     pickle.dump(x, f)
