import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

from gbfry_driven_sv import GBFRYDrivenSV

from utils import logit

# ------------------------------------------------------------------
# This script simulates data from the GBFRY IID model.
# ------------------------------------------------------------------

# Parameters of the simulation

# Number of observations
T = 5000
T_test = 50000
# Parameters of the IID GBFRY model
params = {
    "eta": 1.,
    "c": 1.,
    "tau": 3.,
    "mu": 0.,
    "beta":0.,
    "lam":1e-2
}
# Directory where to save the generated data
save_dir = "../data/simulated/"

# Simulate y from model
ssm_cls = GBFRYDrivenSV
theta = {
            'log_eta':np.log(params['eta']),
            'log_c':np.log(params['c']),
            'log_tau_minus_one':np.log(params['tau']-1.),
            'log_lam':np.log(params['lam']),
            'mu':params['mu'],
            'beta':params['beta']
}

model = GBFRYDrivenSV(**theta)

# Generate train
x, y = model.simulate(T)
y = np.array(y).squeeze()
x = np.array(x)[:,0,1]

data = pd.DataFrame()
data['x'] = x
data['y'] = y
data['Volume'] = 500*np.ones_like(y)

# Generate test
xt, yt = model.simulate(T_test)
yt = np.array(yt).squeeze()
xt = np.array(xt)[:,0,1]

test = pd.DataFrame()
test['x'] = xt
test['y'] = yt
test['Volume'] = 500*np.ones_like(yt)

# Generate test
# x_test, y_test = model.simulate(T_test)

# data_test= pd.DataFrame()
# data_test['x'] = x_test
# data_test['y'] = np.array(y_test).squeeze()
# data_test['Volume'] = 500*np.ones_like(y_test)

if not os.path.isdir('../data/simulated/'):
    os.makedirs('../data/simulated/')


filename = ('../data/simulated/gbfry_driven_sv_'
        'T_{}_'
        'eta_{}_'
        'tau_{}_'
        'c_{}_'
        'train.pkl'
        .format(T, params['eta'], params['tau'], params['c']))

with open(filename, 'wb') as f:
    pickle.dump(data, f)
    
filename_test = ('../data/simulated/gbfry_driven_sv_'
        'T_{}_'
        'eta_{}_'
        'tau_{}_'
        'c_{}_'
        'test.pkl'
        .format(T, params['eta'], params['tau'], params['c']))

with open(filename_test, 'wb') as f:
    pickle.dump(test, f)

data = os.path.splitext(os.path.basename(filename))[0]
fig_dir = os.path.join('plots', data, 'gbfry')
plt.figure('x')
plt.plot(x)
plt.title('Integrated Stochastic Volatility')
plt.xlabel('Time')
plt.savefig(os.path.join(fig_dir, 'int_stoc_vol.png'), bbox_inches='tight')
plt.figure('y')
plt.plot(y)
plt.title('log-returns')
plt.xlabel('Time')
plt.savefig(os.path.join(fig_dir, 'log_ret.png'), bbox_inches='tight')
plt.figure('x')
plt.plot(xt)
plt.title('Integrated Stochastic Volatility Test Data')
plt.xlabel('Time')
plt.savefig(os.path.join(fig_dir, 'int_stoc_vol_test.png'), bbox_inches='tight')
plt.figure('y')
plt.plot(yt)
plt.title('log-returns Test Data')
plt.xlabel('Time')
plt.savefig(os.path.join(fig_dir, 'log_ret_test.png'), bbox_inches='tight')

dict_muvar = {
            'mu':np.mean(xt),
            'var':np.var(xt),
            'max':np.max(xt),
            'min':np.min(xt),
            'mu_y':np.mean(yt),
            'var_y':np.var(yt),
            'max_y':np.max(yt),
            'min_y':np.min(yt),
}
with open(os.path.join(fig_dir, 'args.txt'), 'w') as f:
    json.dump(dict_muvar, f, indent=2)