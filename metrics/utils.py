import statsmodels.api as sm

def robust_reg_model_p(x, y):
    """Fits a robust regression model using the Huber method and returns
    the p-value of the fit.
    
    Parameters
    ----------
    x : array, shape = (n_samples,)
        The independent variable
    y : array, shape = (n_samples,)
        The dependent variable
    
    Returns
    -------
    p : float
        P-value of the robust regression fit
    """
    
    x = sm.add_constant(x)
    mdl = sm.RLM(y, x)
    mdl = mdl.fit()
    p = mdl.pvalues[1]
    
    return p