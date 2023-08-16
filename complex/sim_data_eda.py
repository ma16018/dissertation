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
from scipy.special import gamma
from scipy.stats import moment
import seaborn as sb

filename = '../data/simulated/gbfry_driven_sv_T_1000_eta_5.0_tau_1.5_c_1.0'
tau = 1.5
# filename = '../data/simulated/gbfry_driven_sv_T_1000_eta_5.0_tau_3.0_c_1.0'
# tau = 3

# filename = '../data/simulated/gamma_driven_sv_T_5000_eta_1.0_c_1.0'
model = 'gbfry'
# model = 'gamma'
trials = 10

# FB = '../data/data_minute_tech/FB_min_train.pkl'
# with open(os.path.join(FB), 'rb') as f:
#     datafile = pickle.load(f, encoding='latin1')
#     datafile = datafile[datafile.Volume >= 200]
#     FB_y = datafile['y']
#     FB_y = FB_y / np.std(FB_y)
#     FB_st = datafile['Open']
# FB_sim = '../data/simulated/gbfry_T_5000_eta_0.85_tau_1.97_sigma_0.25_c_1.86_train_1.pkl'
# with open(os.path.join(FB_sim), 'rb') as f:
#     datafile = pickle.load(f, encoding='latin1')
#     datafile = datafile[datafile.Volume >= 200]
#     FB_y_sim = datafile['y']
#     FB_y_sim = FB_y_sim / np.std(FB_y_sim)
#     std_true = np.std(datafile['y'])
#     datafile['y'] = datafile['y'].values / std_true
#     FB_data = datafile
    

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

data = os.path.basename(filename)
    # data_files = []
    # for i in args.num:
    #     file = ''.join([args.filename, str(i+1), '.pkl'])        
    #     data = os.path.splitext(os.path.basename(file))[0]
    #     data_files.append(data)

fig_dir = os.path.join('eda', data)
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

train_list = []
true_list = []
for i in range(trials):
    train_filename = ''.join([filename, '_train_', str(i+1), '.pkl'])
    with open(os.path.join(train_filename), 'rb') as f:
        datafile = pickle.load(f, encoding='latin1')
        datafile = datafile[datafile.Volume >= 200]
        std_true = np.std(datafile['y'])
        datafile['y'] = datafile['y'].values / std_true
        train_list.append(datafile)
    
    test_filename = ''.join([filename, '_test_', str(i+1), '.pkl'])
    with open(os.path.join(test_filename), 'rb') as f:
        datafile = pickle.load(f, encoding='latin1')
        datafile = datafile[datafile.Volume >= 200]
        datafile['y'] = datafile['y'].values / std_true
        true_list.append(datafile)
        
ssm_options = {}

# Plots

# trial = 1
# plt.figure(figsize=(10, 5))
# plt.subplot(122)
# plt.plot(FB_y.sort_index())
# plt.subplot(121)
# plt.plot(FB_y_sim[:len(FB_y)])
# plt.savefig(os.path.join(fig_dir, 'trajectories_{}.png'.format(trial)), 
#             bbox_inches='tight')


# df = FB_data
# vbar = np.stack(df['x']).squeeze()
# y = np.stack(df['y']).squeeze()
# vstar = np.cumsum(vbar)
# Xt = np.cumsum(y)
# St = np.exp(Xt)*200

# fig, axs = plt.subplots(2,2)
# axs[0,1].plot(vbar)
# axs[0,1].set_title("Integrate stochastic volatility increments")
# axs[1,1].plot(y)
# axs[1,1].set_title("Log-returns")
# axs[0,0].plot(vstar)
# axs[0,0].set_title("V*")
# axs[1,0].plot(Xt)
# #axs[1,0].plot(200*np.exp(np.cumsum(y)))
# axs[1,0].set_title("X_tk")
# plt.savefig(os.path.join(fig_dir, 'trajectories_{}.png'.format(trial)), 
#             bbox_inches='tight')

# plt.figure()
# plt.plot(FB_st.sort_index())

# Density plots

# Plots
trial = 1
df = train_list[trial-1]
vbar = np.stack(df['x']).squeeze()
y = np.stack(df['y']).squeeze()
Xt = np.cumsum(y)
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
axs[0].plot(vbar)
axs[0].set_title("Integrated stochastic volatility increments from GGP process")
axs[0].set_ylabel(r"$\bar{V}_k$")
axs[0].set_xlabel("k")
axs[1].plot(y)
axs[1].set_title(r"Log-returns given $\bar{V}_k$")
axs[1].set_ylabel(r"$Y_k$")
axs[1].set_xlabel("k")
axs[2].plot(Xt)
axs[2].set_title("Log-stock price")
axs[2].set_ylabel(r"$X_t$")
axs[2].set_xlabel("t")
fig.tight_layout()
plt.savefig(os.path.join(fig_dir, 'trajectories_train_tau_{}.png'.format(str(tau))), 
            bbox_inches='tight')


vbar_all = []
y_all = []
for i, df in enumerate(true_list):
    vbar_all.append(np.stack(df['x']).squeeze())
    y_all.append(np.stack(df['y']).squeeze())
vbar_all = np.concatenate(vbar_all)
y_all = np.concatenate(y_all)
plt.figure()
sb.kdeplot(y_all)
plt.xlabel("y")
plt.title(r"Empirical density of $Y_k$ for $\tau=1.5$ in OU model")
# sb.histplot(y_all, stat='density')
plt.savefig(os.path.join(fig_dir, 'density_y_tau_{}.png'.format(tau)), bbox_inches='tight')
plt.figure()
sb.kdeplot(vbar_all)
plt.xlabel(r"$\bar{V}_k$")
plt.title(r"Empirical density of $\bar{V}_k$ for $\tau=1.5$ in OU model")
# sb.histplot(vbar_all, stat='density')
plt.savefig(os.path.join(fig_dir, 'density_vbar_tau_{}.png'.format(tau)), bbox_inches='tight')

taus = [1.5, 3.]
trial = 1
plt.figure(figsize=(8, 6))
for i, tau in enumerate(taus):
    filename = ('../data/simulated/gbfry_driven_sv_T_1000_eta_5.0_tau_{}_c_1.0_test_{}.pkl'
                .format(tau, trial))
    with open(os.path.join(filename), 'rb') as f:
        datafile = pickle.load(f, encoding='latin1')
        datafile = datafile[datafile.Volume >= 200]
        vbar = np.stack(datafile['x']).squeeze()
        std_true = np.std(datafile['y'])
        y = datafile['y'].values / std_true
        sb.kdeplot(vbar, label=r"$\tau={}$".format(tau))
        # sb.histplot(vbar, stat='density')
plt.legend()
plt.xlabel(r"$\bar{v}_k$")
plt.title(r"Empirical density of simulated $\bar{V}_k$ from OU GBFRY")
# plt.xlim([0, 10])
plt.savefig(os.path.join(fig_dir, 'densities_vbar.png'), bbox_inches='tight')

taus = [1.5, 3.]
trial = 1
plt.figure(figsize=(8, 6))
for i, tau in enumerate(taus):
    filename = ('../data/simulated/gbfry_driven_sv_T_1000_eta_5.0_tau_{}_c_1.0_test_{}.pkl'
                .format(tau, trial))
    with open(os.path.join(filename), 'rb') as f:
        datafile = pickle.load(f, encoding='latin1')
        datafile = datafile[datafile.Volume >= 200]
        vbar = np.stack(datafile['x']).squeeze()
        std_true = np.std(datafile['y'])
        y = datafile['y'].values / std_true
        sb.kdeplot(y, label=r"$\tau={}$".format(tau))
        # sb.histplot(vbar, stat='density')
plt.legend()
plt.xlabel(r"$Y_k$")
plt.title(r"Empirical density of simulated $Y_k$ from OU GBFRY")
# plt.xlim([0, 10])
plt.savefig(os.path.join(fig_dir, 'densities_y.png'), bbox_inches='tight')

taus = [1.5, 3.]
trials = np.arange(1, 5)
plt.figure(figsize=(8, 6))
for tau in taus:
    for trial in trials:
        filename = ('../data/simulated/gbfry_driven_sv_T_1000_eta_5.0_tau_{}_c_1.0_test_{}.pkl'
                    .format(tau, trial))
        with open(os.path.join(filename), 'rb') as f:
            datafile = pickle.load(f, encoding='latin1')
            datafile = datafile[datafile.Volume >= 200]
            std_true = np.std(datafile['y'])
            y = datafile['y'].values / std_true
            Xt = np.cumsum(y)
            line_color = 'blue' if tau == 1.5 else 'green'
            plt.plot(Xt, label=r"$\tau={}$, trial={}".format(tau, trial), color=line_color)
            # sb.kdeplot(y, label=r"$\tau={}$".format(tau))
            # sb.histplot(vbar, stat='density')
# plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$X_t$")
plt.title(r"Log-stock price trajectory for OU GBFRY model")
# plt.xlim([0, 10])
plt.savefig(os.path.join(fig_dir, 'xt_taus.png'), bbox_inches='tight')

plt.figure(figsize=(8, 5))
for i, df in enumerate(true_list):
    y = np.stack(df['y']).squeeze()
    Xt = np.cumsum(y)
    plt.plot(Xt, label='GGP {}'.format(str(i)))
# plt.legend()
plt.ylabel(r"$X_t$")
plt.xlabel("t")
plt.title("Log-stock price for 10 GGP simulations")
plt.savefig(os.path.join(fig_dir, 'xt.png'), 
            bbox_inches='tight')

# Moments

def km(m, eta, sigma, tau, c, t=1):
    num = t*eta*(tau-sigma)*gamma(m-sigma)
    den = c**m*(tau-m)*gamma(1-sigma)
    return num/den

def ggp_var(eta, sigma, tau, c, t=1):
    num = t*eta*(tau-sigma)*(1-sigma)
    den = c**2*(tau-2)
    return num/den


df = true_list[0]
vbar = np.stack(df['x']).squeeze()
k1 = km(m=1, eta=5, tau=3, sigma=0, c=1)
k2 = km(m=2, eta=5, tau=3, sigma=0, c=1)
var = ggp_var(eta=5, tau=3, sigma=0, c=1)
# k3 = km(m=3, eta=1, tau=3, sigma=0.6, c=1) # gives infinity

means = []
variances = []
higher_moments = []
for df in true_list:
    vbar = np.stack(df['x']).squeeze()
    means.append(np.mean(vbar))
    variances.append(np.var(vbar))
    higher_moments.append(moment(vbar, moment=4, center=0))

print("Average mean is ", np.mean(means), "with 95% CI ", 
      np.percentile(means, [5, 95]))
print("Average variance is ", np.mean(variances), "with 95% CI ", 
      np.percentile(variances, [5, 95]))
print("Average 4th moment is ", np.mean(higher_moments), "with 95% CI ", 
      np.percentile(higher_moments, [5, 95]))

moment(vbar, moment=3, center=0)
moment(vbar, moment=4, center=0)
moment(vbar, moment=5, center=0)

# Inverse scale parameter

filename = '../data/simulated/gbfry_T_5000_eta_1.0_tau_3.0_sigma_0.6_c_2.0'
train_filename = ''.join([filename, '_test_1.pkl'])
with open(os.path.join(train_filename), 'rb') as f:
    datafile = pickle.load(f, encoding='latin1')
    datafile = datafile[datafile.Volume >= 200]
    std_true = np.std(datafile['y'])
    datafile['y'] = datafile['y'].values / std_true

vbar = np.stack(datafile['x']).squeeze()
cvbar = 2*vbar
vbar1 = np.stack(true_list[1]['x']).squeeze()
plt.figure(figsize=(10, 6))
plt.subplot(122)
plt.plot(vbar1)
plt.xlabel("k")
plt.ylabel(r"$\bar{V}_k$")
plt.title(r"$\bar{V}_k \sim GGP(\eta, \sigma, \tau, 1)$")
plt.subplot(121)
plt.plot(cvbar, alpha=0.8, label=r"$c\bar{V}_k$")
plt.plot(vbar, color='green', alpha=0.6, label=r"$\bar{V}_k$")
plt.xlabel("k")
plt.ylabel(r"$\bar{V}_k$")
plt.legend()
plt.title(r"$\bar{V}_k \sim GGP(\eta, \sigma, \tau, c)$")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'inverse_scale_parameter.png'), 
            bbox_inches='tight')

np.mean(vbar)
np.mean(vbar1)
np.var(vbar)
np.var(vbar1)

# Power-law

trial = 5
T = len(true_list[0])
true_y = true_list[trial]['y']
skip = 5000
x_array = np.sort(np.square(true_y[true_y != 0]))
p_0 = np.sum(true_y == 0)/len(true_y)
len_x = len(x_array)
empirical_survival = 1 - (p_0 + (1-p_0)*np.arange(1, len_x+1)/len_x)
plt.figure()
plt.loglog(x_array[skip:-2], empirical_survival[skip:-2], linewidth=1.5)

plt.figure()
ex = x_array[skip:-2]
x = np.log(ex)
ey = empirical_survival[skip:-2]
y = np.log(ey)
plt.plot(x, y)

x0 = 3
i = min(range(len(x)), key=lambda j: abs(x[j] - x0))
slope = (y[-1] - y[i]) / (x[-1] - x[i])
print('Slope at ', (x[-1] + x[i])/2, 'is', slope)  # -3.3635665021278665
def tangent_line(x_val):
    return slope * (x_val - x0) + y[i]
plt.figure()
plt.plot(x, y, label='Survival function')
plt.plot(x[i-5000:], [tangent_line(x_val) for x_val in x[i-5000:]], 
         label='Tangent at x = {}'.format(round((x[-1] + x[i])/2, 1)))
plt.title(r"Logarithm of empirial survival function of $Y$")
plt.ylabel(r"logP($Y$ > x)")
plt.xlabel(r"log($x^2)$")
# plt.scatter(x[i], y[i], color='red', label='Point (x0, y0)')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'esy_{}.png'.format(x0)), bbox_inches='tight')    


# Power-law try 2

trial = 1
T = len(true_list[0])
true_v = np.stack(true_list[trial]['x']).squeeze()
skip = 1000
x_array = np.sort(true_v)
len_x = len(x_array)
empirical_survival = 1-np.arange(1, len_x+1)/len_x
plt.loglog(x_array[skip:], empirical_survival[skip:], linewidth=1.5)

ex = x_array[skip:-2]
x = np.log(ex)
ey = empirical_survival[skip:-2]
y = np.log(ey)
plt.plot(x, y)

x0 = 2
i = min(range(len(x)), key=lambda j: abs(x[j] - x0))
slope = (y[-1] - y[i]) / (x[-1] - x[i])
print('Slope at ', x0, 'is', slope)  # -3.3728700867756656
def tangent_line(x_val):
    return slope * (x_val - x0) + y[i]
plt.figure()
plt.plot(x, y, label='Survival function')
plt.plot(x[i-5000:], [tangent_line(x_val) for x_val in x[i-5000:]], 
         label='Tangent at x = {}'.format(round((x[-1] + x[i])/2, 1)))
plt.title(r"Logarithm of empirial survival function of $\bar{V}_k$")
plt.ylabel(r"logP($\bar{V}_k$ > v)")
plt.xlabel(r"log(v)")
# plt.scatter(x[i], y[i], color='red', label='Point (x0, y0)')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'esurvival_{}.png'.format(x0)), bbox_inches='tight')    





# Variance

def ggp_var(eta, sigma, tau, c, t=1):
    num = t*eta*(tau-sigma)*(1-sigma)
    den = c**2*(tau-2)
    return num/den

taus = np.arange(2.1, 4.1, 0.1)
var_tau = ggp_var(eta=1, sigma=0.6, tau=taus, c=1)
plt.figure()
plt.xlabel(r"$\tau$")
plt.ylabel(r"Var($V^*_1$)")
plt.plot(taus, var_tau)

sigs = np.arange(0.1, 1, 0.1)
var_sig = ggp_var(eta=1, sigma=sigs, tau=3, c=1)
plt.figure()
plt.xlabel(r"$\sigma$")
plt.ylabel(r"Var($V^*_1$)")
plt.plot(sigs, var_sig)

# cs = np.arange(0.1, 2.1, 0.1)
# var_c = ggp_var(eta=1, sigma=0.6, tau=3, c=cs)
# plt.plot(cs, var_c)

# etas = np.arange(0.1, 2.1, 0.1)
# var_eta = ggp_var(eta=etas, sigma=0.6, tau=3, c=1)
# plt.plot(etas, var_eta)


# plots of Vk for different sigma and tau

params1 = {
    "eta": 1.0,
    "c": 1.0,
    "tau": 2.2,
    "sigma": 0.2
}
params2 = {
    "eta": 1.0,
    "c": 1.0,
    "tau": 2.2,
    "sigma": 0.9
}
params3 = {
    "eta": 1.0,
    "c": 1.0,
    "tau": 3.,
    "sigma": 0.2
}
params4 = {
    "eta": 1.0,
    "c": 1.0,
    "tau": 3.,
    "sigma": 0.9
}
params_list = [params1, params2, params3, params4]
trial=str(1)
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for i, params in enumerate(params_list):
    filename = ('../data/simulated/gbfry_'
                'T_{}_'
                'eta_{}_'
                'tau_{}_'
                'sigma_{}_'
                'c_{}'
                '_test_{}.pkl'
                .format(T, params['eta'], params['tau'], params['sigma'], 
                        params['c'], trial))
    with open(os.path.join(filename), 'rb') as f:
        datafile = pickle.load(f, encoding='latin1')
        datafile = datafile[datafile.Volume >= 200]
        vbar = np.stack(datafile['x']).squeeze()
    axs[i%2, i//2].plot(vbar)
    axs[i%2, i//2].set_title(r"$\sigma={}, \tau={}$".format(params['sigma'], 
                                                            params['tau']))
    axs[i%2, i//2].set_ylabel(r"$\bar{V}_k$")
    axs[i%2, i//2].set_xlabel("k")
    axs[i%2, i//2].set_ylim([0, 150])
fig.tight_layout()
plt.savefig(os.path.join(fig_dir, 'ggp_incrs_diff_params'), bbox_inches='tight')    


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
