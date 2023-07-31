import numpy as np
import pandas as pd
import os
import pickle
import json

from gbfry_iid_incr import GBFRYIIDIncr
import matplotlib.pyplot as plt

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
    "eta": 1.0,
    "c": 1.0,
    "tau": 1.2,
    "sigma": .6
}
# Directory where to save the generated data
save_dir = "../data/simulated/"

# Simulate y from model
ssm_cls = GBFRYIIDIncr
theta = {
            'log_eta':np.log(params['eta']),
            'log_c':np.log(params['c']),
            'log_tau_minus_one':np.log(params['tau']-1.),
            'logit_sigma':logit(params['sigma'])
}

model = GBFRYIIDIncr(**theta)

# Generate train
x, y = model.simulate(T)

data = pd.DataFrame()
data['x'] = x
data['y'] = np.array(y).squeeze()
data['Volume'] = 500*np.ones_like(y)

# Generate test
x_test, y_test = model.simulate(T_test)

data_test= pd.DataFrame()
data_test['x'] = x_test
data_test['y'] = np.array(y_test).squeeze()
data_test['Volume'] = 500*np.ones_like(y_test)

# Save data
filename = ('gbfry_'
            'T_{}_'
            'eta_{}_'
            'tau_{}_'
            'sigma_{}_'
            'c_{}'
            '_train.pkl'
            .format(T, params['eta'], params['tau'], params['sigma'], params['c']))

filename_test = ('gbfry_'
            'T_{}_'
            'eta_{}_'
            'tau_{}_'
            'sigma_{}_'
            'c_{}_test'
            '.pkl'
            .format(T, params['eta'], params['tau'], params['sigma'], params['c']))

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

with open(save_dir+filename, 'wb') as f:
    pickle.dump(data, f)

with open(save_dir+filename_test, 'wb') as f:
    pickle.dump(data_test, f)
    
    
data_name = os.path.splitext(os.path.basename(filename))[0]
fig_dir = os.path.join('plots', data_name, 'gbfry')
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)
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
plt.plot(x_test)
plt.title('Integrated Stochastic Volatility Test Data')
plt.xlabel('Time')
plt.savefig(os.path.join(fig_dir, 'int_stoc_vol_test.png'), bbox_inches='tight')
plt.figure('y')
plt.plot(y_test)
plt.title('log-returns Test Data')
plt.xlabel('Time')
plt.savefig(os.path.join(fig_dir, 'log_ret_test.png'), bbox_inches='tight')

dict_muvar = {
            'mu':np.mean(x_test),
            'var':np.var(x_test),
            'max':np.max(x_test),
            'min':np.min(x_test),
            'mu_y':np.mean(y_test),
            'var_y':np.var(y_test),
            'max_y':np.max(y_test),
            'min_y':np.min(y_test),
}
with open(os.path.join(fig_dir, 'args.txt'), 'w') as f:
    json.dump(dict_muvar, f, indent=2)
    