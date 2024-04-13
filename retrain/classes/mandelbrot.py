from statsmodels.base.model import GenericLikelihoodModel
from scipy.special import zeta
import pickle
import numpy as np


class Mandelbrot(GenericLikelihoodModel):
    def to_pickle(self, filename, remove_data=True):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        
        if not self.fit_result:
            raise ValueError("No fit result registered yet; pickling pointless!")
        
        if remove_data:
            self.fit_result.model = None
            self.fit_result.exog = None
            self.fit_result.endog = None
            
            
        with open(filename, "wb") as handle:
            pickle.dump(self.fit_result, handle)   
            
    @classmethod
    def from_pickle(cls, filename, to_class=False, frequencies=None, 
                    ranks=None, **kwargs):
        
        if not filename.endswith(".pkl"):
            filename += ".pkl"        
        with open(filename, "rb") as handle:
            fit_res = pickle.load(handle)
            
        if not to_class:
            return fit_res
        
        if (frequencies is None) or (ranks is None):
            raise ValueError("Mandelbrot class can only be instatiated with" 
                              "frequencies and ranks given!")
            
        mandel = cls(frequencies, ranks, **kwargs)
        fit_res.model = mandel
        mandel.register_fit(fit_res)
        return mandel
            
    
    def __init__(self, frequencies, ranks, **kwargs):
        if not len(frequencies) == len(ranks):
            raise ValueError("NOT THE SAME NUMBER OF RANKS AND FREQS!")
        
        frequencies = np.asarray(frequencies)
        ranks = np.asarray(ranks)
        
        self.n_obs = np.sum(frequencies)
        
        super().__init__(endog=frequencies, exog=ranks, **kwargs)
        self.fit_result = None
        self.lg = np.log10
    

    def prob(self, params, ranks=None, log=False):
        if ranks is None:
            ranks = self.exog
        
        alpha, beta = params
        if log:
            return -alpha * self.lg(beta + ranks) - self.lg(zeta(alpha, q=beta+1.))
        else:
            return ((beta + ranks)**(-alpha)) / zeta(alpha, q=beta+1.)
    
    
    def loglike(self, params, frequencies=None, ranks=None):
        rs = self.exog if (ranks is None) else ranks 
        fs = self.endog if (frequencies is None) else frequencies
        alpha, beta = params
        
        if alpha < 1.0 or beta < 0.0:
            return -np.inf
        
        # no need to calculate P(r) when observed f(r) was zero
        log_probs = -alpha * self.lg(beta + rs) - self.lg(zeta(alpha, q=beta+1.))
        log_probs = log_probs.reshape(-1, )
        return np.sum(fs * log_probs) - beta**5
    
    
    def register_fit(self, fit_result, overwrite=False):
        if not self.fit_result is None and not overwrite:
            raise ValueError("A fit result is already registered and overwrite=False!")
            
        self.fit_result = fit_result
        self.optim_params = fit_result.params
        self.pseudo_r_squared = self.pseudo_r_squared(self.optim_params)
        self.SE, self.SE_relative = fit_result.bse, fit_result.bse/self.optim_params
        self.BIC, self.BIC_relative = fit_result.bic,\
                            (-2*self.null_loglike())/fit_result.bic
        
    
    def print_result(self, string=False):
        if self.fit_result is None:
            raise ValueError("Register a fitting result first!")

        def format_x(x):
            return float('{0:.3g}'.format(x))


        s = "="*50
        s += "\n" + "MANDELBROT"
        s += "\n" + "  Optimal Parameters " + str(tuple(map(format_x, self.optim_params)))
        
        s += "\n" + "  Standard Error [relative]: " + str(tuple(map(format_x, self.SE))) +\
              ", [" + str(tuple(map(format_x, self.SE_relative))) + "]"
        
        s += "\n" + "  Pseudo R^2: " + str(format_x(self.pseudo_r_squared))
        
        s += "\n" + "  BIC [relative]: " + str(format_x(self.BIC)) +\
              ", [" + str(format_x(self.BIC_relative)) + "]"
        s += "\n" + "="*50
        
        if string:
            return s
        
        print(s)
    
    
    def null_loglike(self, epsilon=1e-10):
        return self.loglike((1.+epsilon, 0.0))
    
    
    def pseudo_r_squared(self, params):
        return 1-self.loglike(params)/self.null_loglike()
    
    
    def predict(self, params, ranks=None, freqs=True, n_obs=None, 
                correct_for_finite_domain=True):
        if ranks is None:
            ranks = self.exog
        ranks = np.asarray(ranks)
        
        if n_obs is None:
            n_obs = self.n_obs
            
        alpha, beta = params
        pred_probs = self.prob(params, ranks=ranks, log=False)
        
        if correct_for_finite_domain:
            if not freqs:
                raise NotImplementedError("Correction for "\
                                          "finite domain not implemented with probabilities!")
            return pred_probs*(n_obs/np.sum(pred_probs))
        
        if freqs:
            return n_obs*pred_probs
        
        return pred_probs