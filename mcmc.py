from particles import mcmc
from particles import smc_samplers as ssp
import numpy as np
from scipy import stats
from correlated import SMCCorrelated, SMCCorrelated2, SMCCorrelatedIID, SMCCorrelatedIIDComplex

class PMMH(mcmc.PMMH):
    def __init__(self, *args, **kwargs):
        self.keep_states = kwargs.pop('keep_states', False)
        self.ssm_options = kwargs.pop('ssm_options', {})
        super(PMMH, self).__init__(*args, **kwargs)
        if self.keep_states:
            self.smc_options = {'store_history':True}
            self.states = []
            self.acceptance_rates = []
            self.acceptance_nums = []
        else:
            self.smc_options = {'collect':'off'}

    def print_progress(self, n):
        params = self.chain.theta.dtype.fields.keys()
        msg = 'Iteration %i' % n
        if hasattr(self, 'nacc') and n > 0:
            msg += ', acc. rate=%.3f' % (self.nacc / n)
        for p in params:
            theta = self.chain.theta[p][n]
            if self.ssm_cls.params_name.get(p) is None:
                msg += ', %s=%.3f' % (p, theta)
            else:
                msg += ', %s=%.3f' % (
                    self.ssm_cls.params_name[p],
                    self.ssm_cls.params_transform[p](theta))
        print(msg)

    def alg_instance(self, theta):
        return self.smc_cls(fk=self.fk_cls(ssm=self.ssm_cls(**theta, **self.ssm_options),
                                           data=self.data),
                            N=self.Nx, **self.smc_options)

    def compute_post(self):
        self.prop.lpost[0] = self.prior.logpdf(self.prop.theta)
        if np.isfinite(self.prop.lpost[0]):
            pf = self.alg_instance(ssp.rec_to_dict(self.prop.theta[0]))
            pf.run()
            self.prop.lpost[0] += pf.logLt

            if self.keep_states:
                #self.states.append(np.array([m['mean'] for m in pf.summaries.moments]))
                self.states.append(np.array(pf.hist.extract_one_trajectory()))
                self.acceptance_rates.append(self.acc_rate)
                self.acceptance_nums.append(self.nacc)
    
    # def step(self, n):
    #     z = stats.norm.rvs(size=self.dim)
    #     self.prop_arr[0] = self.arr[n - 1] + np.dot(self.L, z)
    #     self.compute_post()
    #     # print('current=', self.prop.lpost[0], 'prev=', self.chain.lpost[n - 1])
    #     lp_acc = self.prop.lpost[0] - self.chain.lpost[n - 1]
    #     lu = np.log(stats.uniform.rvs())
    #     print('accept thres=', lp_acc)
    #     if lu < lp_acc:  # accept
    #         self.chain.copyto_at(n, self.prop, 0)
    #         self.nacc += 1
    #     else:  # reject
    #         self.chain.copyto_at(n, self.chain, n - 1)
    #     if self.adaptive:
    #         self.cov_tracker.update(self.arr[n])
    #         self.L = self.scale * self.cov_tracker.L
                
    # def step(self, n):
    #     z = stats.norm.rvs(size=self.dim)
    #     self.prop_arr[0] = self.arr[n - 1] + np.dot(self.L, z)
    #     self.compute_post()
    #     lp_acc = self.prop.lpost[0] - self.chain.lpost[n - 1]
    #     print(self.prop.theta[0])
    #     if np.log(stats.uniform.rvs()) < lp_acc:  # accept
    #         self.chain.copyto_at(n, self.prop, 0)
    #         self.nacc += 1
    #     else:  # reject
    #         self.chain.copyto_at(n, self.chain, n - 1)
    #     if self.adaptive:
    #         self.cov_tracker.update(self.arr[n])
    #         self.L = self.scale * self.cov_tracker.L


class CorrPMMH(PMMH):
    def __init__(self, smc_cls=SMCCorrelated, *args, **kwargs):
        super(PMMH, self).__init__(*args, **kwargs)
        self.smc_cls=smc_cls
        # self.history = self.ssm_cls(**ssp.rec_to_dict(self.theta0[0])).PX0().rvs(size=self.Nx)
        self.history = self.ssm_cls().PX0().rvs(size=self.Nx)
        self.states = []

    def alg_instance(self, theta):
        return self.smc_cls(fk=self.fk_cls(ssm=self.ssm_cls(**theta),
                                           data=self.data),
                            N=self.Nx, history=self.history, 
                            num_keep=int(self.Nx*0.5), 
                            **self.smc_options)
    
    def compute_post(self):
        self.prop.lpost[0] = self.prior.logpdf(self.prop.theta)
        if np.isfinite(self.prop.lpost[0]):
            pf = self.alg_instance(ssp.rec_to_dict(self.prop.theta[0]))
            pf.run()
            self.history = pf.hist.X[0]
            self.prop.lpost[0] += pf.logLt

            # self.states.append(np.array(pf.hist.extract_one_trajectory()))


class CorrPMMH2(PMMH):
    def __init__(self, smc_cls=SMCCorrelated2, *args, **kwargs):
        super(PMMH, self).__init__(*args, **kwargs)
        self.smc_cls=smc_cls
        # self.history = self.ssm_cls(**ssp.rec_to_dict(self.theta0[0])).PX0().rvs(size=self.Nx)
        self.history = None
        self.states = []

    def alg_instance(self, theta):
        return self.smc_cls(fk=self.fk_cls(ssm=self.ssm_cls(**theta),
                                           data=self.data),
                            N=self.Nx, historyX=self.history,
                            **self.smc_options)
    
    def compute_post(self):
        self.prop.lpost[0] = self.prior.logpdf(self.prop.theta)
        if np.isfinite(self.prop.lpost[0]):
            pf = self.alg_instance(ssp.rec_to_dict(self.prop.theta[0]))
            pf.run()
            self.history = pf.hist.X
            self.prop.lpost[0] += pf.logLt

            # self.states.append(np.array(pf.hist.extract_one_trajectory()))


class CorrPMMHIID(PMMH):
    def __init__(self, smc_cls=SMCCorrelatedIID, perc=0.1, *args, **kwargs):
        super(PMMH, self).__init__(*args, **kwargs)
        self.smc_cls=smc_cls
        # self.history = self.ssm_cls(**ssp.rec_to_dict(self.theta0[0])).PX0().rvs(size=self.Nx)
        self.history = None
        self.states = []
        self.acceptance_rates = []
        self.acceptance_nums = []
        self.perc = perc

    def alg_instance(self, theta):
        return self.smc_cls(fk=self.fk_cls(ssm=self.ssm_cls(**theta),
                                           data=self.data),
                            N=self.Nx, historyX=self.history,
                            **self.smc_options, perc=self.perc)
    
    def compute_post(self):
        self.prop.lpost[0] = self.prior.logpdf(self.prop.theta)
        if np.isfinite(self.prop.lpost[0]):
            pf = self.alg_instance(ssp.rec_to_dict(self.prop.theta[0]))
            pf.run()
            self.history = pf.hist.X
            self.prop.lpost[0] += pf.logLt
        # print('Proposal', self.prop.theta)
            
        self.acceptance_rates.append(self.acc_rate)
        self.acceptance_nums.append(self.nacc)
        self.states.append(np.array(pf.hist.extract_one_trajectory()))
        
    # def step(self, n):
    #     z = stats.norm.rvs(size=self.dim)
    #     self.prop_arr[0] = self.arr[n - 1] + np.dot(self.L, z)
    #     self.compute_post()
    #     # print('current=', self.prop.lpost[0], 'prev=', self.chain.lpost[n - 1])
    #     lp_acc = self.prop.lpost[0] - self.chain.lpost[n - 1]
    #     lu = np.log(stats.uniform.rvs())
    #     print('accept thres=', lp_acc)
    #     if lu < lp_acc:  # accept
    #         self.chain.copyto_at(n, self.prop, 0)
    #         self.nacc += 1
    #     else:  # reject
    #         self.chain.copyto_at(n, self.chain, n - 1)
    #     if self.adaptive:
    #         self.cov_tracker.update(self.arr[n])
    #         self.L = self.scale * self.cov_tracker.L

class CorrPMMHIIDComplex(PMMH):
    def __init__(self, smc_cls=SMCCorrelatedIIDComplex, parts=2, *args, **kwargs):
        super(PMMH, self).__init__(*args, **kwargs)
        self.smc_cls=smc_cls
        # self.history = self.ssm_cls(**ssp.rec_to_dict(self.theta0[0])).PX0().rvs(size=self.Nx)
        self.history = None
        self.states = []
        self.acceptance_rates = []
        self.acceptance_nums = []
        self.parts = parts

    def alg_instance(self, theta):
        return self.smc_cls(fk=self.fk_cls(ssm=self.ssm_cls(parts=self.parts, **theta),
                                           data=self.data),
                            N=self.Nx, historyX=self.history,
                            **self.smc_options, parts=self.parts)
    
    def compute_post(self):
        self.prop.lpost[0] = self.prior.logpdf(self.prop.theta)
        if np.isfinite(self.prop.lpost[0]):
            pf = self.alg_instance(ssp.rec_to_dict(self.prop.theta[0]))
            pf.run()
            self.history = pf.hist.X
            self.prop.lpost[0] += pf.logLt
        # print('Proposal', self.prop.theta)
            
        self.acceptance_rates.append(self.acc_rate)
        self.acceptance_nums.append(self.nacc)
        self.states.append(np.array(pf.hist.extract_one_trajectory()))
        
    # def step(self, n):
    #     z = stats.norm.rvs(size=self.dim)
    #     self.prop_arr[0] = self.arr[n - 1] + np.dot(self.L, z)
    #     self.compute_post()
    #     # print('current=', self.prop.lpost[0], 'prev=', self.chain.lpost[n - 1])
    #     lp_acc = self.prop.lpost[0] - self.chain.lpost[n - 1]
    #     lu = np.log(stats.uniform.rvs())
    #     print('accept thres=', lp_acc)
    #     if lu < lp_acc:  # accept
    #         self.chain.copyto_at(n, self.prop, 0)
    #         self.nacc += 1
    #     else:  # reject
    #         self.chain.copyto_at(n, self.chain, n - 1)
    #     if self.adaptive:
    #         self.cov_tracker.update(self.arr[n])
    #         self.L = self.scale * self.cov_tracker.L