import sys
sys.path.append('../')
from metrics2 import KS
import argparse
import os
import pickle
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from gbfry_iid_incr import GBFRYIIDIncr
from utils import VaR, ecdf

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='../data/simulated/gbfry_T_5000_eta_1.0_tau_3.0_sigma_0.6_c_1.0_train.pkl')
parser.add_argument('--model', type=str, default='gbfry')
parser.add_argument('--eta_name', type=str, default='log_eta')

args = parser.parse_args()

# Set font sizes
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

if args.filename is None:
    raise ValueError('You must specify data')
else:
    data = os.path.splitext(os.path.basename(args.filename))[0]

fig_dir = os.path.join('eda', data, args.model)
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

with open(os.path.join(args.filename), 'rb') as f:
    datafile = pickle.load(f, encoding='latin1')
    datafile = datafile[datafile.Volume >= 200]
    std_true = np.std(datafile['y'])
    train_y = datafile['y'].values
    train_y = train_y / std_true  # W

test_filename = args.filename[:args.filename.rfind('train.pkl')] + 'test.pkl'
with open(os.path.join(test_filename), 'rb') as f:
    datafile = pickle.load(f, encoding='latin1')
    datafile = datafile[datafile.Volume >= 200]
    ssm_options = {}

true_y = datafile['y'].values
true_y = true_y / std_true #W

true_var = VaR(true_y)

T = len(true_y)

print('VaR of observed data: {}'.format(true_var))

# Plot ordered y^2
# low_per = (100-args.credible_mass)/2
# high_per = 100-low_per
# plt_range = int(95/100*T)
# plt.figure('Ordered y^2')
# #plt.title("Posterior predictive of ordered y^2")
# plt.xlabel("Rank")
# plt.ylabel("y^2")
# plt.loglog(range(plt_range), np.sort(true_y**2)[::-1][:plt_range], linewidth=1.5)
# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, "Ordered_y_square_test.png"), bbox_inches='tight')
# plt.close('all')

skip = 500
x_array = np.sort(np.square(true_y[true_y != 0]))
p_0 = np.sum(true_y == 0)/len(true_y)
len_x = len(x_array)
empirical_cdf = -np.log(p_0 + (1-p_0)*np.arange(1, len_x+1)/len_x)
# plt.loglog(x_array[skip:], empirical_cdf[skip:], linewidth=1.5, label="Empirical cdf")

ex = x_array[skip:49990]
x = np.log(ex)
ey = empirical_cdf[skip:49990]
y = np.log(ey)
plt.plot(x, y)


x0s = np.array([-6, -4, -2, 0, 2])
for x0 in x0s:
    i = min(range(len(x)), key=lambda j: abs(x[j] - x0))
    slope = (y[i + 100] - y[i]) / (x[i + 100] - x[i])
    print('Slope at ', x0, 'is', slope)
    def tangent_line(x_val):
        return slope * (x_val - x0) + y[i]
    plt.figure()
    plt.plot(x, y, label='Curve')
    plt.plot(x, [tangent_line(x_val) for x_val in x], label='Tangent at x = {}'.format(x0))
    plt.scatter(x[i], y[i], color='red', label='Point (x0, y0)')
    plt.legend()
    plt.savefig(os.path.join(fig_dir, 'ecdf_{}.png'.format(x0)), bbox_inches='tight')    





















# xr = np.arange(-7, 2)
# from scipy import interpolate
# a = -5
# small_t = np.arange(a-1,a+2)
# spl = interpolate.splrep(lx, ly)
# fa = interpolate.splev(a,spl,der=0)     # f(a)
# fprime = interpolate.splev(a,spl,der=1) # f'(a)

# fig = plt.figure()
# plt.loglog(x, y)
# plt.loglog(np.exp(xs), tan)
# # xs = x[1:50]
# # plt.loglog(xs, 1/xs**(2))
# #plt.loglog(a,fa,'om',small_t,tan,'--r')
# plt.show()

# plt.plot(x,y)
# plt.plot(np.log(x), np.log(y))

# draw_tangent(t,price_index,1991)
# draw_tangent(t,price_index,1998)

# plot(t,price_index,alpha=0.5)
# show()





# # Plot log F(x) no zeros with skip of Y^2
# low_per = (100-args.credible_mass)/2
# high_per = 100-low_per
# skip = 500
# x_array = np.sort(np.square(true_y[true_y != 0]))
# p_0 = np.sum(true_y == 0)/len(true_y)
# len_x = len(x_array)
# plt.figure('cdf_tail')
# #plt.title("Posterior predictive of ordered abs(y)")
# plt.ylabel("-log(P(Y^2 < x))")
# plt.xlabel("x")
# plt.ylim(-np.log(1-1/len_x), np.minimum(-np.log(p_0/10), 5))
# empirical_cdf = -np.log(p_0 + (1-p_0)*np.arange(1, len_x+1)/len_x)
# plt.loglog(x_array[skip:], empirical_cdf[skip:], linewidth=1.5, label="Empirical cdf")
# plt.tight_layout()
# plt.show()

# # Plot abs(y) log log
# n_bins = 100
# plt.figure('Histogram of true abs(y) (log-log scale)')
# bins_range = (0.08, 40.)
# logbins = np.logspace(np.log(bins_range[0]),np.log(bins_range[1]), n_bins)
# plt.xscale('log')
# plt.hist(np.abs(true_y), bins=logbins, log=True, density=1, alpha=0.5, label="True observations")
# plt.title("Abs(y)")
# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, "Histogram_true_abs_y.png"), bbox_inches='tight')

# # relate x and y?
