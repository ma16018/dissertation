import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import argparse
import os
import pickle
from scipy.stats.mstats import mquantiles

from gamma_iid_incr import GammaIIDIncr
from gbfry_iid_incr import GBFRYIIDIncr
from nig_iid import NIGIID
from ns_iid_incr import NSIIDIncr
from ghyperbolic_iid import GHDIIDIncr
from student_iid import StudentIIDIncr
from vgamma3_iid import VGamma3IID
from vgamma4_iid import VGamma4IID

from utils import logit

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default=None)
parser.add_argument('--model', type=str, default='gbfry')
parser.add_argument('--run_names', type=str, nargs='+', 
                    default=['chain1', 'chain2', 'chain3'])
                    # default=['chain4', 'chain5', 'chain6'])
                    # default=['chain7', 'chain8', 'chain9'])
parser.add_argument('--show', action='store_true')
parser.add_argument('--no_states', action='store_true')
parser.add_argument('--niter', type=int, default=2000)

args = parser.parse_args()

if args.model == 'gamma':
    ssm_cls = GammaIIDIncr
elif args.model == 'gbfry':
    ssm_cls = GBFRYIIDIncr
elif args.model == 'ns':
    ssm_cls = NSIIDIncr
elif args.model == 'vgamma3':
    ssm_cls = VGamma3IID
elif args.model == 'vgamma4':
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

if args.filename is None:
    raise ValueError('You must specify data')
else:
    data = os.path.splitext(os.path.basename(args.filename))[0]

save_dir = os.path.join('results', args.model, data)
prefix = '{}_{}'.format(data, args.model)

fig_dir = os.path.join('plots', data, args.model)
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

keys = prior.laws.keys()
chains = []
sb.set()
sb.set_style("whitegrid", {'axes.grid':False})

# Set font sizes
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

for run_name in args.run_names:
    with open(os.path.join(save_dir, run_name, 'chain.pkl'), 'rb') as f:
        chain = pickle.load(f)
        n = len(chain.theta)
        burnin = np.min((n//2, 500))
        # burnin = 0
        chains.append(chain[burnin:args.niter-1])

# particles = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000]
gathered = {}
for key in keys:
    params = r"${}$".format(ssm_cls.params_latex[key])
    gathered[params] = []
    plt.figure('{}_{}_trace'.format(prefix, ssm_cls.params_name[key]))
    plt.ylabel("Density")
    plt.xlabel(params)
    plt.title("Density by chain for {} MCMC iterations".format(args.niter))
    c_id = 1
    i=0
    for chain in chains:
        val = ssm_cls.params_transform[key](chain.theta[key])
        gathered[params].append(np.array(val))
        # plt.plot(np.arange(burnin, n), val, linewidth=1, label='{}'.format(particles[i]))
        # i += 1
        # sb.histplot(val, color='r', stat='density')
        sb.kdeplot(val, label='Chain {}'.format(c_id))
        c_id += 1
    plt.legend()
    plt.savefig(os.path.join(fig_dir, '{}_hist_plot_conv_check.png'.format(ssm_cls.params_name[key])),
            bbox_inches='tight')
