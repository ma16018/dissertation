from particles import SMC
import numpy as np
from particles import resampling as rs

class SMCCorrelated(SMC):
    def __init__(self, 
                 fk=None,
                 N=100,
                 qmc=False,
                 resampling="systematic",
                 ESSrmin=0.5,
                 store_history=True,
                 verbose=False,
                 collect=None,
                 num_keep=10,
                 history=None):
        super(SMCCorrelated, self).__init__(fk=fk,
                                            N=N,
                                            qmc=qmc,
                                            resampling=resampling,
                                            ESSrmin=ESSrmin,
                                            store_history=store_history,
                                            verbose=verbose,
                                            collect=collect)
        self.random_indices = np.random.choice(N, num_keep, replace=False)
        self.remaining_indices = np.setdiff1d(np.arange(N), self.random_indices)
        self.history = history

    def generate_particles(self):
        super().generate_particles()
        self.X[self.random_indices] = self.history[self.random_indices]
        
        
class SMCCorrelated2(SMC):
    def __init__(self, 
                 fk=None,
                 N=100,
                 qmc=False,
                 resampling="systematic",
                 ESSrmin=0.5,
                 store_history=True,
                 verbose=False,
                 collect=None,
                 historyX=None):
        self.num_keep = int(N*0.2) if historyX else 0
        super(SMCCorrelated2, self).__init__(fk=fk,
                                            N=N-self.num_keep,
                                            qmc=qmc,
                                            resampling=resampling,
                                            ESSrmin=ESSrmin,
                                            store_history=store_history,
                                            verbose=verbose,
                                            collect=collect)
        self.random_indices = np.random.choice(N-self.num_keep, 
                                               self.num_keep, replace=False)
        self.historyX = historyX

    def compute_summaries(self):
        logp = self.fk.logG(self.t, self.Xp, self.X)
        if self.historyX:   
            Xphist = self.historyX[self.t-1][self.random_indices]
            Xhist = self.historyX[self.t][self.random_indices]
            logp_hist = self.fk.logG(self.t, Xphist, Xhist)
            logp = np.concatenate((logp, logp_hist))
        new_wgts = rs.Weights(lw=logp)
        
        self.logLt += new_wgts.log_mean
        if self.verbose:
            print(self)
        if self.hist:
            self.hist.save(self)
        # must collect summaries *after* history, because a collector (e.g.
        # FixedLagSmoother) may needs to access history
        if self.summaries:
            self.summaries.collect(self)
            
        
    # def generate_particles(self):
    #     self.X[num_keep:] = self.fk.M0(self.N-num_keep)
    #     self.X[0:num_keep] = history.X[0][random_indices]
    
    # def resample_move(self):
        
        
    # def reweight_particles(self):
        
    #     lw_previous = history.wgts[self.t].lw[random_indices]
    #     lw_new = self.wgts.lw
    #     lw = np.concatenate((lw_previous, lw_new))
    #     self.wgts = rs.Weights(lw=lw)
        
    # def compute_summaries(self):
        
        
    #     if self.t > 0:
    #         prec_log_mean_w = self.log_mean_w
    #     self.log_mean_w = self.wgts.log_mean
    #     if self.t == 0 or self.rs_flag:
    #         self.loglt = self.log_mean_w
    #     else:
    #         self.loglt = self.log_mean_w - prec_log_mean_w
    #     self.logLt += self.loglt
    #     if self.verbose:
    #         print(self)
    #     if self.hist:
    #         self.hist.save(self)
    #     # must collect summaries *after* history, because a collector (e.g.
    #     # FixedLagSmoother) may needs to access history
    #     if self.summaries:
    #         self.summaries.collect(self)
        
    
        
# class SMCCorrelated(SMC):
#     def __init__(self, 
#                  fk=None,
#                  N=100,
#                  qmc=False,
#                  resampling="systematic",
#                  ESSrmin=0.5,
#                  store_history=True,
#                  verbose=False,
#                  collect=None,
#                  perc_keep):
#         super(SMCCorrelated, self).__init__(fk=fk,
#                                             N=N,
#                                             qmc=qmc,
#                                             resampling=resampling,
#                                             ESSrmin=ESSrmin,
#                                             store_history=store_history,
#                                             verbose=verbose,
#                                             collect=collect)
#         self.perc_keep = perc_keep
        
#     def compute_summaries(self):
#         self.hist.save(self)
        

# def calc_logLt(mat):
        
    
class SMCCorrelatedIID(SMC):
    def __init__(self, 
                 fk=None,
                 N=100,
                 qmc=False,
                 resampling="systematic",
                 ESSrmin=0.5,
                 store_history=True,
                 verbose=False,
                 collect=None,
                 historyX=None,
                 perc=0.1):
        super(SMCCorrelatedIID, self).__init__(fk=fk,
                                            N=N,
                                            qmc=qmc,
                                            resampling=resampling,
                                            ESSrmin=ESSrmin,
                                            store_history=store_history,
                                            verbose=verbose,
                                            collect=collect)
        self.num_keep = int(N*perc) if historyX else 0
        self.random_indices = np.random.choice(N, self.num_keep, replace=False)
        self.historyX = historyX
        
    def resample_move(self):
        super().resample_move()
        if self.historyX:
            self.X[self.random_indices] = self.historyX[self.t][self.random_indices]


class SMCCorrelatedIIDComplex(SMC):
    def __init__(self, 
                 fk=None,
                 N=100,
                 qmc=False,
                 resampling="systematic",
                 ESSrmin=0.5,
                 store_history=True,
                 verbose=False,
                 collect=None,
                 historyX=None,
                 parts=2):
        super(SMCCorrelatedIIDComplex, self).__init__(fk=fk,
                                            N=N,
                                            qmc=qmc,
                                            resampling=resampling,
                                            ESSrmin=ESSrmin,
                                            store_history=store_history,
                                            verbose=verbose,
                                            collect=collect)
        self.num_keep = int(N*0.2) if historyX else 0
        self.random_indices = np.random.choice(N, self.num_keep, replace=False)
        self.historyX = historyX
        self.parts = parts
        
    def resample_move(self):
        if self.historyX and np.random.choice(2)==0:    
            self.rs_flag = self.fk.time_to_resample(self)
            if self.rs_flag:  # if resampling
                self.A = rs.resampling(self.resampling, self.aux.W, M=self.N)
                self.Xp = self.X[self.A]
                self.reset_weights()
            else:
                self.A = np.arange(self.N)
                self.Xp = self.X
            self.X = self.historyX[self.t]
            change = np.random.choice(self.parts)
            self.X[:, change] = self.fk.M(self.t, self.Xp)[:, change]
        else:
            super().resample_move()
           
    def reweight_particles(self):
        X = np.sum(self.X, axis=1)
        self.wgts = self.wgts.add(self.fk.logG(self.t, self.Xp, X))
        