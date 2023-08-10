import sys
sys.path.append('../')
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = '../data/simulated/gbfry_T_5000_eta_1.0_tau_3.0_sigma_0.6_c_1.0_train.pkl'
model = 'gbfry'
data = os.path.splitext(os.path.basename(filename))[0]
save_dir = os.path.join('results', model, data)
prefix = '{}_{}'.format(data, model)
run_names = ['particles1000', 'particles1000-2', 'particles1000-3']

acc_nums_corr = []
rates_corr = []
for run_name in run_names:
    with open(os.path.join(save_dir, run_name, 'acceptance_number_corr_0.15.pkl'), 'rb') as f:
        number = pickle.load(f)
        acc_nums_corr.append(number)
        niter = number.shape[0]
        rate = number / np.arange(1, niter+1)
        rates_corr.append(rate)
        
chains = []
for run_name in run_names:
    with open(os.path.join(save_dir, run_name, 'chain_corr_0.15.pkl'), 'rb') as f:
        chain = pickle.load(f)
        n = len(chain.theta)
        burnin = np.min((n//2, 500))
        # burnin = 0
        chains.append(chain[burnin:])
    
        
run_names = ['chain11', 'chain12', 'chain14']
acc_nums = []
rates = []
for run_name in run_names:
    with open(os.path.join(save_dir, run_name, 'acceptance_number.pkl'), 'rb') as f:
        number = pickle.load(f)
        acc_nums.append(number)
        niter = number.shape[0]
        rate = number / np.arange(1, niter+1)
        rates.append(rate)

chains_norm = []
for run_name in run_names:
    with open(os.path.join(save_dir, run_name, 'chain.pkl'), 'rb') as f:
        chain = pickle.load(f)
        n = len(chain.theta)
        burnin = np.min((n//2, 500))
        # burnin = 0
        chains_norm.append(chain[burnin:])

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

fig_dir = os.path.join('plots', data, model)
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

particles = [100, 250, 500, 750, 1000, 1500, 2000]
plt.figure(figsize=(14, 6))
plt.subplot(122)
for r, p in zip(rates, particles):
    plt.plot(r, label=str(p))
plt.title(r"Progression of acceptance rate of proposals $\phi^*$")
plt.xlabel("MCMC iteration")
plt.ylabel("Proportion of accepted proposals")
# plt.legend(title='Particles')
plt.subplot(121)
for n, p in zip(acc_nums, particles):
    plt.plot(n, label=str(p))
plt.title(r"Progression of number of accepted proposals $\phi^*$")
plt.xlabel("MCMC iteration")
plt.ylabel("Number of accepted proposals")
plt.tight_layout()
plt.legend(title='Particles')
plt.savefig(os.path.join(fig_dir, 'acceptance.png'), 
            bbox_inches='tight')

# Corr vs normal
plt.figure(figsize=(14, 6))
plt.subplot(122)
for r in rates:
    plt.plot(r, label='Normal')
for r in rates_corr:
    plt.plot(r, label='Correlated')
plt.title(r"Progression of acceptance rate of proposals $\phi^*$")
plt.xlabel("MCMC iteration")
plt.ylabel("Proportion of accepted proposals")
# plt.legend(title='Particles')
plt.subplot(121)
for r in acc_nums:
    plt.plot(r, label='Normal')
for r in acc_nums_corr:
    plt.plot(r, label='Correlated')
plt.title(r"Progression of number of accepted proposals $\phi^*$")
plt.xlabel("MCMC iteration")
plt.ylabel("Number of accepted proposals")
plt.tight_layout()
plt.legend(title='Particles')
plt.savefig(os.path.join(fig_dir, 'acceptance_corrvsnorm.png'), 
            bbox_inches='tight')

# Autocorrelation of the chain?
chain = chains[0]
col = 'red'
# chain = chains_norm[0]
# col = 'blue'
logc = pd.DataFrame(chain.theta['log_c'])
loge = pd.DataFrame(chain.theta['log_eta'])
logtaumo = pd.DataFrame(chain.theta['log_tau_minus_one'])
logitsig = pd.DataFrame(chain.theta['logit_sigma'])


plt.figure()
pd.plotting.lag_plot(logc, lag=1, c=col)
plt.figure()
pd.plotting.lag_plot(loge, lag=1, c=col)
plt.figure()
pd.plotting.lag_plot(logtaumo, lag=1, c=col)
plt.figure()
pd.plotting.lag_plot(logitsig, lag=1, c=col)





# pf.generate_particles()
# pf.reweight_particles()
# print(pf.rs_flag)
# pf.compute_summaries()
# print(pf.logLt)

# pf.setup_auxiliary_weights()
# pf.resample_move()
# pf.reweight_particles()
# print(pf.rs_flag)
# pf.compute_summaries()
# print(pf.logLt)
