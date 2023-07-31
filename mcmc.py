from particles import mcmc
from particles import smc_samplers as ssp
import numpy as np
from correlated import SMCCorrelated, SMCCorrelated2

class PMMH(mcmc.PMMH):
    def __init__(self, *args, **kwargs):
        self.keep_states = kwargs.pop('keep_states', False)
        self.ssm_options = kwargs.pop('ssm_options', {})
        super(PMMH, self).__init__(*args, **kwargs)
        if self.keep_states:
            self.smc_options = {'store_history':True}
            self.states = []
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

            self.states.append(np.array(pf.hist.extract_one_trajectory()))


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

            self.states.append(np.array(pf.hist.extract_one_trajectory()))
