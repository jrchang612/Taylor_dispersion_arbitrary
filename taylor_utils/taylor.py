import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import pickle

def save_pickle(data, file_name):
    """
    Saves data as pickle format
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    return None

def weighted_mean_r(x, beta, a0):
    return np.sum(x/(a0 + beta*x)**2, axis = 0, keepdims = True)/np.sum(1/(a0 + beta*x)**2, axis = 0, keepdims = True)
def weighted_variance_r(x, beta, a0, weighted_x):
    return np.sum((x-weighted_x)**2/(a0 + beta*x)**2, axis = 0)/np.sum(1/(a0 + beta*x)**2, axis = 0)
def weighted_skewness_r(x, beta, a0, weighted_x, weighted_var):
    return np.sum((x-weighted_x)**3/(a0 + beta*x)**2, axis = 0)/np.sum(1/(a0 + beta*x)**2, axis = 0)/weighted_var**(3/2)
def weighted_kurtosis_r(x, beta, a0, weighted_x, weighted_var):
    return np.sum((x-weighted_x)**4/(a0 + beta*x)**2, axis = 0)/np.sum(1/(a0 + beta*x)**2, axis = 0)/weighted_var**2


def weighted_mean_r_xfunc(x, func_x):
    return np.sum(x/(func_x(x))**2, axis = 0, keepdims = True)/np.sum(1/(func_x(x))**2, axis = 0, keepdims = True)
def weighted_variance_r_xfunc(x, func_x, weighted_x):
    return np.sum((x-weighted_x)**2/(func_x(x))**2, axis = 0)/np.sum(1/(func_x(x))**2, axis = 0)
def weighted_skewness_r_xfunc(x, func_x, weighted_x, weighted_var):
    return np.sum((x-weighted_x)**3/(func_x(x))**2, axis = 0)/np.sum(1/(func_x(x))**2, axis = 0)/weighted_var**(3/2)
def weighted_kurtosis_r_xfunc(x, func_x, weighted_x, weighted_var):
    return np.sum((x-weighted_x)**4/(func_x(x))**2, axis = 0)/np.sum(1/(func_x(x))**2, axis = 0)/weighted_var**2

def weighted_mean_r_moment_xfunc(x, func_x):
    return np.sum(x, axis = 0, keepdims = True)/np.sum(np.ones_like(x), axis = 0, keepdims = True)
def weighted_variance_r_moment_xfunc(x, func_x, weighted_x):
    return np.sum((x-weighted_x)**2, axis = 0)/np.sum(np.ones_like(x), axis = 0)
def weighted_skewness_r_moment_xfunc(x, func_x, weighted_x, weighted_var):
    return np.sum((x-weighted_x)**3, axis = 0)/np.sum(np.ones_like(x), axis = 0)/weighted_var**(3/2)
def weighted_kurtosis_r_moment_xfunc(x, func_x, weighted_x, weighted_var):
    return np.sum((x-weighted_x)**4, axis = 0)/np.sum(np.ones_like(x), axis = 0)/weighted_var**2

def simulation_var_cone_moment_initialVar(Pe0, func_x, Nt0 = 500, seed = 0, sigx2_0 = 300, upper_bound = None):
    # define parameter
    U0 = 1
    a0 = 1
    A = 40
    B = A/Pe0
    dt = 1/B
    D = B/A
    Npts = 5000
    Nt = Nt0*A+1
    sig_s = np.sqrt(2*D*dt)
    np.random.seed(seed)

    # initialization
    r = np.zeros((Npts, Nt))
    theta = np.zeros((Npts, Nt))
    x = np.zeros((Npts, Nt))
    x[:, 0] = np.random.randn(Npts)*np.sqrt(sigx2_0)
    theta[:, 0] = (np.random.rand(Npts))*2*np.pi - np.pi
    r[:,0] = np.sqrt((np.random.rand(Npts))*func_x(x[:, 0])**2)
    
    if upper_bound:
        x_range_est = upper_bound
    else:
        x_range_est = Nt*dt*U0*8
    x_range_for_ur = np.linspace(-500, x_range_est, 10000+1)
    dx = (x_range_est+500)/10000
    a_x = func_x(x_range_for_ur)
    beta_for_ur = np.gradient(a_x, dx)
    beta_of_x_for_ur = interp1d(x_range_for_ur, beta_for_ur, kind='cubic')
    
    ux = lambda x, r: 2*(a0**2*U0/(func_x(x))**2)*(1 - r**2/(func_x(x))**2)
    ur = lambda x, r: 2*(a0**2*U0/(func_x(x))**2)*beta_of_x_for_ur(x)*(r/(func_x(x)) - r**3/(func_x(x))**3)
    U_x = lambda x: U0*a0**2/func_x(x)**2
    
    rand = np.random.randn(Nt-1, 3*Npts)

    # simulation
    for i in range(1, Nt):
        x[:, i] = x[:, i-1] + ux(x[:, i-1], r[:, i-1])*dt + sig_s*rand[i-1, 0:Npts]
        r_temp = r[:, i-1] + ur(x[:, i-1], r[:, i-1])*dt
        x2 = r_temp*np.cos(theta[:, i-1]) + sig_s*rand[i-1, Npts:2*Npts]
        x3 = r_temp*np.sin(theta[:, i-1]) + sig_s*rand[i-1, 2*Npts:3*Npts]
        theta[:, i] = np.arctan2(x3, x2)
        r_new = np.sqrt(x2**2 + x3**2)
        loc_pos = (r_new > func_x(x[:, i]))
        r_new[loc_pos] = 2*func_x(x[loc_pos, i]) - r_new[loc_pos]
        loc_neg = (r_new < 0)
        r_new[loc_neg] = -r_new[loc_neg]
        theta[loc_neg, i] = theta[loc_neg, i] + np.pi
        loc_pos = (r_new > func_x(x[:, i]))
        r_new[loc_pos] = 2*func_x(x[loc_pos, i]) - r_new[loc_pos]
        loc_neg = (r_new < 0)
        r_new[loc_neg] = -r_new[loc_neg]
        theta[loc_neg, i] = theta[loc_neg, i] + np.pi
        r[:, i] = r_new

    # analysis
    T = np.arange(Nt)*dt
    weighted_x = weighted_mean_r_moment_xfunc(x, func_x)
    weighted_var = weighted_variance_r_moment_xfunc(x, func_x, weighted_x)
    weighted_skewness = weighted_skewness_r_moment_xfunc(x, func_x, weighted_x, weighted_var)
    weighted_kurtosis = weighted_kurtosis_r_moment_xfunc(x, func_x, weighted_x, weighted_var)

    x_min = -500
    x_max = np.max(weighted_x)*2
    n_diff = 10000
    x_range = np.linspace(x_min, x_max, n_diff+1)
    x_mid = (x_range[0:-1] + x_range[1:])/2
    dx = (x_max - x_min)/n_diff
    
    a_x = func_x(x_range)
    beta = np.gradient(a_x, dx)
    beta_of_x = interp1d(x_range, beta, kind='cubic')
    gamma = np.gradient(beta, dx)
    gamma_of_x = interp1d(x_range, gamma, kind='cubic')
    delta = np.gradient(gamma, dx)
    delta_of_x = interp1d(x_range, delta, kind='cubic')
    
    predicted_x_bar = np.zeros_like(weighted_x.flatten(), dtype = float)
    predicted_var_heuristic = np.zeros_like(weighted_var.flatten(), dtype = float)
    predicted_x_bar[0] = weighted_x.flatten()[0]
    predicted_var_heuristic[0] = weighted_var.flatten()[0]
    
    for i in range(1,Nt):
        x_curr = np.copy(predicted_x_bar[i-1])
        predicted_x_bar[i] = x_curr + dt*(U_x(x_curr) + 2*D*beta_of_x(x_curr)/func_x(x_curr))
        
        predicted_var_heuristic[i] = predicted_var_heuristic[i-1] + dt*(
            2*D 
            + U_x(x_curr)**2*func_x(x_curr)**2/(24*D) 
            - 4*U0*a0**2*predicted_var_heuristic[i-1]*beta_of_x(x_curr)/func_x(x_curr)**3 
            #+ 1/2*predicted_var_heuristic[i-1]*(6*(beta_of_x(x_curr))**2-2*func_x(x_curr)*gamma_of_x(x_curr))/func_x(x_curr)**4 
            #+ 4*D*predicted_var_heuristic[i-1]*(func_x(x_curr)*gamma_of_x(x_curr) - (beta_of_x(x_curr))**2)/func_x(x_curr)**2 
        )
        # 
    
    #f_x_2 = func_x(x_range)**2
    #f_x_2 = (f_x_2[0:-1] + f_x_2[1:])/2
    #F_x = np.hstack([0, np.cumsum(f_x_2*dx)])/(U0*a0**2)
    #F_inv = interp1d(F_x, x_range, kind='cubic')
    #G_x = np.hstack([0, np.cumsum((f_x_2)/(1 + 1/6*((beta[0:-1]+beta[1:])/2)**2 + 1/12*func_x(x_mid)*(gamma[0:-1]+gamma[1:])/2)*dx)])/(U0*a0**2)
    #G_inv = interp1d(G_x, x_range, kind='cubic')
    
    """one_beta_agamma_o_aEN2 = (1 + 1/6*beta**2 + 1/12*a_x*gamma)/(a_x**2)
    diff_one_beta_agamma_o_aEN2 = np.gradient(one_beta_agamma_o_aEN2, dx)
    diff2_aEN2 = np.gradient(np.gradient(1/a_x**2, dx), dx)
    diff2_beta_o_a = np.gradient(np.gradient(beta/a_x, dx), dx)

    diff_one_beta_agamma_o_aEN2_of_x = interp1d(x_range, diff_one_beta_agamma_o_aEN2, kind='cubic')
    diff2_aEN2_of_x = interp1d(x_range, diff2_aEN2, kind = 'cubic')
    diff2_beta_o_a = interp1d(x_range, diff2_beta_o_a, kind = 'cubic')
    aEN2_of_x = interp1d(x_range, 1/a_x**2, kind = 'cubic')
    beta_o_a_of_x = interp1d(x_range, beta/a_x, kind = 'cubic')"""
    
    #cdx = np.sum(1/func_x(x)**2, axis = 0)
    
    #predicted_x_bar = G_inv(T)

    #ln_cdx_0 = np.log(cdx[0])
    #predicted_ln_cdx_heuristic = np.zeros_like(cdx, dtype = float)
    #predicted_ln_cdx_heuristic[0] = ln_cdx_0
    """dlnc_bar_dt = U0*a0**2*diff_one_beta_agamma_o_aEN2_of_x(predicted_x_bar) + D*(U0**2*a0**4/(48*D**2)*diff2_aEN2_of_x(predicted_x_bar) 
                                                                                  - 1/6*U0*a0**2/D*diff2_beta_o_a(predicted_x_bar))
    predicted_var_heuristic2 = np.zeros_like(weighted_var, dtype = float)
    predicted_var_heuristic2[0] = weighted_var[0]
    
    for i in range(1,Nt):
        predicted_var_heuristic2[i] = predicted_var_heuristic2[i-1] + dt*(
                +2*U0*a0**2*predicted_var_heuristic2[i-1]*diff_one_beta_agamma_o_aEN2_of_x(predicted_x_bar[i-1])
                +2*D*(1 + 1/48*U0**2*a0**4/D**2*aEN2_of_x(predicted_x_bar[i-1]) 
                      #- 1/6*U0*a0**2/D*beta_o_a_of_x(predicted_x_bar[i-1]) # marked
                     )
                #+D*predicted_var_heuristic2[i-1]*(5/48*U0**2*a0**4/D**2*diff2_aEN2_of_x(predicted_x_bar[i-1])
                #                                -5/6*U0*a0**2/D*diff2_beta_o_a(predicted_x_bar[i-1]))
        )"""
    result = {'x': x,
              'r': r,
              'theta': theta,
              'T': T,
              'weighted_x': weighted_x, 
              'weighted_var': weighted_var,
              'weighted_skewness': weighted_skewness,
              'weighted_kurtosis': weighted_kurtosis,
              'x_range': x_range,
              'x_mid': x_mid,
              'dx': dx,
              #'F_x': F_x,
              'a_x': a_x,
              'beta': beta,
              'gamma': gamma,
              'delta': delta,
              #'G_x': G_x,
              #'one_beta_agamma_o_aEN2': one_beta_agamma_o_aEN2,
              #'diff_one_beta_agamma_o_aEN2': diff_one_beta_agamma_o_aEN2,
              #'diff2_aEN2': diff2_aEN2,
              #'diff2_beta_o_a': diff2_beta_o_a,
              #'cdx': cdx,
              'predicted_x_bar': predicted_x_bar,
              #'ln_cdx_0': ln_cdx_0,
              #'predicted_ln_cdx_heuristic': predicted_ln_cdx_heuristic,
              #'dlnc_bar_dt': dlnc_bar_dt,
              #'predicted_var': predicted_var,
              'predicted_var_heuristic': predicted_var_heuristic,
              #'predicted_var_heuristic2': predicted_var_heuristic2,
              'a0': a0, 'Pe0': Pe0, 'D': D, 'U0': U0, 'dt': dt, 'Npts': Npts, 'Nt': Nt}
    return result

def simulation_var_cone_moment_skewness_initialVar(Pe0, func_x, Nt0 = 500, seed = 0, sigx2_0 = 300, upper_bound = None):
    # define parameter
    U0 = 1
    a0 = 1
    A = 40
    B = A/Pe0
    dt = 1/B
    D = B/A
    Npts = 5000
    Nt = Nt0*A+1
    sig_s = np.sqrt(2*D*dt)
    np.random.seed(seed)

    # initialization
    r = np.zeros((Npts, Nt))
    theta = np.zeros((Npts, Nt))
    x = np.zeros((Npts, Nt))
    x[:, 0] = np.random.randn(Npts)*np.sqrt(sigx2_0)
    theta[:, 0] = (np.random.rand(Npts))*2*np.pi - np.pi
    r[:,0] = np.sqrt((np.random.rand(Npts))*func_x(x[:, 0])**2)
    
    if upper_bound:
        x_range_est = upper_bound
    else:
        x_range_est = Nt*dt*U0*8
    x_range_for_ur = np.linspace(-500, x_range_est, 10000+1)
    dx = (x_range_est+500)/10000
    a_x = func_x(x_range_for_ur)
    beta_for_ur = np.gradient(a_x, dx)
    beta_of_x_for_ur = interp1d(x_range_for_ur, beta_for_ur, kind='cubic')
    
    ux = lambda x, r: 2*(a0**2*U0/(func_x(x))**2)*(1 - r**2/(func_x(x))**2)
    ur = lambda x, r: 2*(a0**2*U0/(func_x(x))**2)*beta_of_x_for_ur(x)*(r/(func_x(x)) - r**3/(func_x(x))**3)
    U_x = lambda x: U0*a0**2/func_x(x)**2
    
    rand = np.random.randn(Nt-1, 3*Npts)

    # simulation
    for i in range(1, Nt):
        x[:, i] = x[:, i-1] + ux(x[:, i-1], r[:, i-1])*dt + sig_s*rand[i-1, 0:Npts]
        r_temp = r[:, i-1] + ur(x[:, i-1], r[:, i-1])*dt
        x2 = r_temp*np.cos(theta[:, i-1]) + sig_s*rand[i-1, Npts:2*Npts]
        x3 = r_temp*np.sin(theta[:, i-1]) + sig_s*rand[i-1, 2*Npts:3*Npts]
        theta[:, i] = np.arctan2(x3, x2)
        r_new = np.sqrt(x2**2 + x3**2)
        loc_pos = (r_new > func_x(x[:, i]))
        r_new[loc_pos] = 2*func_x(x[loc_pos, i]) - r_new[loc_pos]
        loc_neg = (r_new < 0)
        r_new[loc_neg] = -r_new[loc_neg]
        theta[loc_neg, i] = theta[loc_neg, i] + np.pi
        loc_pos = (r_new > func_x(x[:, i]))
        r_new[loc_pos] = 2*func_x(x[loc_pos, i]) - r_new[loc_pos]
        loc_neg = (r_new < 0)
        r_new[loc_neg] = -r_new[loc_neg]
        theta[loc_neg, i] = theta[loc_neg, i] + np.pi
        r[:, i] = r_new

    # analysis
    T = np.arange(Nt)*dt
    weighted_x = weighted_mean_r_moment_xfunc(x, func_x)
    weighted_var = weighted_variance_r_moment_xfunc(x, func_x, weighted_x)
    weighted_skewness = weighted_skewness_r_moment_xfunc(x, func_x, weighted_x, weighted_var)
    weighted_kurtosis = weighted_kurtosis_r_moment_xfunc(x, func_x, weighted_x, weighted_var)

    x_min = -500
    x_max = np.max(weighted_x)*2
    n_diff = 10000
    x_range = np.linspace(x_min, x_max, n_diff+1)
    x_mid = (x_range[0:-1] + x_range[1:])/2
    dx = (x_max - x_min)/n_diff
    
    a_x = func_x(x_range)
    beta = np.gradient(a_x, dx)
    beta_of_x = interp1d(x_range, beta, kind='cubic')
    gamma = np.gradient(beta, dx)
    gamma_of_x = interp1d(x_range, gamma, kind='cubic')
    delta = np.gradient(gamma, dx)
    delta_of_x = interp1d(x_range, delta, kind='cubic')
    
    predicted_x_bar = np.zeros_like(weighted_x.flatten(), dtype = float)
    predicted_var_heuristic = np.zeros_like(weighted_var.flatten(), dtype = float)
    predicted_skewness = np.zeros_like(weighted_skewness.flatten(), dtype = float)
    predicted_skewness_sig3 = np.zeros_like(weighted_skewness.flatten(), dtype = float)
    predicted_x_bar[0] = weighted_x.flatten()[0]
    predicted_var_heuristic[0] = weighted_var.flatten()[0]
    predicted_skewness[0] = weighted_skewness.flatten()[0]
    predicted_skewness_sig3[0] = weighted_skewness.flatten()[0]*(weighted_var.flatten()[0])**(3/2)
    
    for i in range(1,Nt):
        x_curr = np.copy(predicted_x_bar[i-1])
        predicted_x_bar[i] = x_curr + dt*(U_x(x_curr) + 2*D*beta_of_x(x_curr)/func_x(x_curr))
        
        d_sigx_dt = (
            2*D 
            + U_x(x_curr)**2*func_x(x_curr)**2/(24*D) 
            - 4*U0*a0**2*predicted_var_heuristic[i-1]*beta_of_x(x_curr)/func_x(x_curr)**3 
            +(2*U0*a0**2*predicted_skewness[i-1]*predicted_var_heuristic[i-1]**(3/2)*(
             3*beta_of_x(x_curr)**2 - 4*func_x(x_curr)*gamma_of_x(x_curr))/(func_x(x_curr))**4 
         ) 
        )
        
        predicted_var_heuristic[i] = predicted_var_heuristic[i-1] + dt*d_sigx_dt
        predicted_skewness_sig3[i] = predicted_skewness_sig3[i-1] + dt*(
            3*a0**2*U0*predicted_skewness_sig3[i-1]*(-2*beta_of_x(x_curr)/(func_x(x_curr))**3)
            - 3/2*U0*a0**2*predicted_var_heuristic[i-1]**2*(6*beta_of_x(x_curr)**2/(func_x(x_curr))**4 - 2*gamma_of_x(x_curr)/(func_x(x_curr))**3)
            + U0**2*a0**4/(8*D)*predicted_var_heuristic[i-1]*(-2*beta_of_x(x_curr)/(func_x(x_curr))**3)
            
            - 3*D*predicted_var_heuristic[i-1]**2*(2*beta_of_x(x_curr)**3/(func_x(x_curr))**3 - 3*beta_of_x(x_curr)*gamma_of_x(x_curr)/(func_x(x_curr))**2 
                                                   + delta_of_x(x_curr)/func_x(x_curr))
            + U0**2*a0**4/(8*D)*1/2*predicted_skewness_sig3[i-1]*(6*beta_of_x(x_curr)**2/(func_x(x_curr))**4 - 2*gamma_of_x(x_curr)/(func_x(x_curr))**3)
            -1/2*U0*a0**2*predicted_var_heuristic[i-1]*predicted_skewness_sig3[i-1]*(
                -24*beta_of_x(x_curr)**3/(func_x(x_curr))**5
                +18*beta_of_x(x_curr)*gamma_of_x(x_curr)/(func_x(x_curr))**4
                -2*delta_of_x(x_curr)/(func_x(x_curr))**3
            )
        )
        predicted_skewness[i] = predicted_skewness_sig3[i]/(predicted_var_heuristic[i])**(3/2)
        #predicted_skewness[i] = predicted_skewness[i-1] + dt*(
        #    -3/2*predicted_skewness[i-1]/predicted_var_heuristic[i-1]*d_sigx_dt 
        #    -U0**2*a0**4/(4*D*np.sqrt(predicted_var_heuristic[i-1]))*beta_of_x(x_curr)/func_x(x_curr)**3
        #    #+U0**2*a0**4/(8*D)*predicted_skewness[i-1]*(3*beta_of_x(x_curr)**2 - 4*func_x(x_curr)*gamma_of_x(x_curr))/(func_x(x_curr))**4 
        #)
        
    result = {'x': x,
              'r': r,
              'theta': theta,
              'T': T,
              'weighted_x': weighted_x, 
              'weighted_var': weighted_var,
              'weighted_skewness': weighted_skewness,
              'weighted_kurtosis': weighted_kurtosis,
              'x_range': x_range,
              'x_mid': x_mid,
              'dx': dx,
              'a_x': a_x,
              'beta': beta,
              'gamma': gamma,
              'delta': delta,
              'predicted_x_bar': predicted_x_bar,
              'predicted_var_heuristic': predicted_var_heuristic,
              'predicted_skewness': predicted_skewness,
              'predicted_skewness_sig3': predicted_skewness_sig3,
              'a0': a0, 'Pe0': Pe0, 'D': D, 'U0': U0, 'dt': dt, 'Npts': Npts, 'Nt': Nt}
    return result



def simulation_var_cone_initialVar(Pe0, func_x, Nt0 = 500, seed = 0, sigx2_0 = 300, upper_bound = None):
    # define parameter
    U0 = 1
    a0 = 1
    A = 40
    B = A/Pe0
    dt = 1/B
    D = B/A
    Npts = 5000
    Nt = Nt0*A+1
    sig_s = np.sqrt(2*D*dt)
    np.random.seed(seed)

    # initialization
    r = np.zeros((Npts, Nt))
    theta = np.zeros((Npts, Nt))
    x = np.zeros((Npts, Nt))
    x[:, 0] = np.random.randn(Npts)*np.sqrt(sigx2_0)
    theta[:, 0] = (np.random.rand(Npts))*2*np.pi - np.pi
    r[:,0] = np.sqrt((np.random.rand(Npts))*func_x(x[:, 0])**2)
    
    if upper_bound:
        x_range_est = upper_bound
    else:
        x_range_est = Nt*dt*U0*8
    x_range_for_ur = np.linspace(-500, x_range_est, 10000+1)
    dx = (x_range_est+500)/10000
    a_x = func_x(x_range_for_ur)
    beta_for_ur = np.gradient(a_x, dx)
    beta_of_x_for_ur = interp1d(x_range_for_ur, beta_for_ur, kind='cubic')
    
    ux = lambda x, r: 2*(a0**2*U0/(func_x(x))**2)*(1 - r**2/(func_x(x))**2)
    ur = lambda x, r: 2*(a0**2*U0/(func_x(x))**2)*beta_of_x_for_ur(x)*(r/(func_x(x)) - r**3/(func_x(x))**3)
    rand = np.random.randn(Nt-1, 3*Npts)

    # simulation
    for i in range(1, Nt):
        x[:, i] = x[:, i-1] + ux(x[:, i-1], r[:, i-1])*dt + sig_s*rand[i-1, 0:Npts]
        r_temp = r[:, i-1] + ur(x[:, i-1], r[:, i-1])*dt
        x2 = r_temp*np.cos(theta[:, i-1]) + sig_s*rand[i-1, Npts:2*Npts]
        x3 = r_temp*np.sin(theta[:, i-1]) + sig_s*rand[i-1, 2*Npts:3*Npts]
        theta[:, i] = np.arctan2(x3, x2)
        r_new = np.sqrt(x2**2 + x3**2)
        loc_pos = (r_new > func_x(x[:, i]))
        r_new[loc_pos] = 2*func_x(x[loc_pos, i]) - r_new[loc_pos]
        loc_neg = (r_new < 0)
        r_new[loc_neg] = -r_new[loc_neg]
        theta[loc_neg, i] = theta[loc_neg, i] + np.pi
        loc_pos = (r_new > func_x(x[:, i]))
        r_new[loc_pos] = 2*func_x(x[loc_pos, i]) - r_new[loc_pos]
        loc_neg = (r_new < 0)
        r_new[loc_neg] = -r_new[loc_neg]
        theta[loc_neg, i] = theta[loc_neg, i] + np.pi
        r[:, i] = r_new

    # analysis
    T = np.arange(Nt)*dt
    weighted_x = weighted_mean_r_xfunc(x, func_x)
    weighted_var = weighted_variance_r_xfunc(x, func_x, weighted_x)
    weighted_skewness = weighted_skewness_r_xfunc(x, func_x, weighted_x, weighted_var)
    weighted_kurtosis = weighted_kurtosis_r_xfunc(x, func_x, weighted_x, weighted_var)

    x_min = 0
    x_max = np.max(weighted_x)*2
    n_diff = 5000
    x_range = np.linspace(x_min, x_max, n_diff+1)
    x_mid = (x_range[0:-1] + x_range[1:])/2
    dx = (x_max - x_min)/n_diff
    f_x_2 = func_x(x_range)**2
    f_x_2 = (f_x_2[0:-1] + f_x_2[1:])/2
    F_x = np.hstack([0, np.cumsum(f_x_2*dx)])/(U0*a0**2)
    F_inv = interp1d(F_x, x_range, kind='cubic')

    a_x = func_x(x_range)
    beta = np.gradient(a_x, dx)
    beta_of_x = interp1d(x_range, beta, kind='cubic')
    gamma = np.gradient(beta, dx)
    G_x = np.hstack([0, np.cumsum((f_x_2)/(1 + 1/6*((beta[0:-1]+beta[1:])/2)**2 + 1/12*func_x(x_mid)*(gamma[0:-1]+gamma[1:])/2)*dx)])/(U0*a0**2)
    G_inv = interp1d(G_x, x_range, kind='cubic')
    
    one_beta_agamma_o_aEN2 = (1 + 1/6*beta**2 + 1/12*a_x*gamma)/(a_x**2)
    diff_one_beta_agamma_o_aEN2 = np.gradient(one_beta_agamma_o_aEN2, dx)
    diff2_aEN2 = np.gradient(np.gradient(1/a_x**2, dx), dx)
    diff2_beta_o_a = np.gradient(np.gradient(beta/a_x, dx), dx)

    diff_one_beta_agamma_o_aEN2_of_x = interp1d(x_range, diff_one_beta_agamma_o_aEN2, kind='cubic')
    diff2_aEN2_of_x = interp1d(x_range, diff2_aEN2, kind = 'cubic')
    diff2_beta_o_a = interp1d(x_range, diff2_beta_o_a, kind = 'cubic')
    aEN2_of_x = interp1d(x_range, 1/a_x**2, kind = 'cubic')
    beta_o_a_of_x = interp1d(x_range, beta/a_x, kind = 'cubic')
    
    cdx = np.sum(1/func_x(x)**2, axis = 0)
    
    predicted_x_bar = G_inv(T)

    ln_cdx_0 = np.log(cdx[0])
    predicted_ln_cdx_heuristic = np.zeros_like(cdx, dtype = float)
    predicted_ln_cdx_heuristic[0] = ln_cdx_0
    dlnc_bar_dt = U0*a0**2*diff_one_beta_agamma_o_aEN2_of_x(predicted_x_bar) + D*(U0**2*a0**4/(48*D**2)*diff2_aEN2_of_x(predicted_x_bar) 
                                                                                  - 1/6*U0*a0**2/D*diff2_beta_o_a(predicted_x_bar))
    predicted_var = np.zeros_like(weighted_var, dtype = float)
    predicted_var_heuristic = np.zeros_like(weighted_var, dtype = float)
    predicted_var[0] = weighted_var[0]
    predicted_var_heuristic[0] = weighted_var[0]
    
    for i in range(1,Nt):
        predicted_ln_cdx_heuristic[i] = predicted_ln_cdx_heuristic[i-1] + dt*(
                U0*a0**2*diff_one_beta_agamma_o_aEN2_of_x(predicted_x_bar[i-1]) + D*(U0**2*a0**4/(48*D**2)*diff2_aEN2_of_x(predicted_x_bar[i-1]) 
                                                                                     - 1/6*U0*a0**2/D*diff2_beta_o_a(predicted_x_bar[i-1])))
        predicted_var[i] = predicted_var[i-1] + 2*D*dt*(1 + 1/48*U0**2*a0**4/D**2/func_x(predicted_x_bar[i-1])**2 
                                                        - 1/6*U0*a0**2/D*beta_of_x(predicted_x_bar[i-1])/func_x(predicted_x_bar[i-1]))
        predicted_var_heuristic[i] = predicted_var_heuristic[i-1] + dt*(
                -predicted_var_heuristic[i-1]*dlnc_bar_dt[i-1]
                +3*U0*a0**2*predicted_var_heuristic[i-1]*diff_one_beta_agamma_o_aEN2_of_x(predicted_x_bar[i-1])
                +2*D*(1 + 1/48*U0**2*a0**4/D**2*aEN2_of_x(predicted_x_bar[i-1]) - 1/6*U0*a0**2/D*beta_o_a_of_x(predicted_x_bar[i-1]))
                +D*predicted_var_heuristic[i-1]*(5/48*U0**2*a0**4/D**2*diff2_aEN2_of_x(predicted_x_bar[i-1])
                                                -5/6*U0*a0**2/D*diff2_beta_o_a(predicted_x_bar[i-1])))
    result = {'x': x,
              'r': r,
              'theta': theta,
              'T': T,
              'weighted_x': weighted_x, 
              'weighted_var': weighted_var,
              'weighted_skewness': weighted_skewness,
              'weighted_kurtosis': weighted_kurtosis,
              'x_range': x_range,
              'x_mid': x_min,
              'dx': dx,
              'F_x': F_x,
              'a_x': a_x,
              'beta': beta,
              'gamma': gamma,
              'G_x': G_x,
              'one_beta_agamma_o_aEN2': one_beta_agamma_o_aEN2,
              'diff_one_beta_agamma_o_aEN2': diff_one_beta_agamma_o_aEN2,
              'diff2_aEN2': diff2_aEN2,
              'diff2_beta_o_a': diff2_beta_o_a,
              'cdx': cdx,
              'predicted_x_bar': predicted_x_bar,
              'ln_cdx_0': ln_cdx_0,
              'predicted_ln_cdx_heuristic': predicted_ln_cdx_heuristic,
              'dlnc_bar_dt': dlnc_bar_dt,
              'predicted_var': predicted_var,
              'predicted_var_heuristic': predicted_var_heuristic,
              'a0': a0, 'Pe0': Pe0, 'D': D, 'U0': U0, 'dt': dt}
    return result

def simulation_var_cone(Pe0, func_x, Nt0 = 500, seed = 0, upper_bound = None):
    # define parameter
    U0 = 1
    a0 = 1
    A = 40
    B = A/Pe0
    dt = 1/B
    D = B/A
    Npts = 5000
    Nt = Nt0*A+1
    sig_s = np.sqrt(2*D*dt)
    np.random.seed(seed)

    # initialization
    r = np.zeros((Npts, Nt))
    theta = np.zeros((Npts, Nt))
    x = np.zeros((Npts, Nt))
    theta[:, 0] = (np.random.rand(Npts))*2*np.pi
    r[:,0] = np.sqrt((np.random.rand(Npts))*a0**2)
    
    if upper_bound:
        x_range_est = upper_bound
    else:
        x_range_est = Nt*dt*U0*8
    x_range_for_ur = np.linspace(-500, x_range_est, 10000+1)
    x_range_for_ur = np.linspace(-500, x_range_est, 10000+1)
    dx = (x_range_est+500)/10000
    a_x = func_x(x_range_for_ur)
    beta_for_ur = np.gradient(a_x, dx)
    beta_of_x_for_ur = interp1d(x_range_for_ur, beta_for_ur, kind='cubic')
    
    ux = lambda x, r: 2*(a0**2*U0/(func_x(x))**2)*(1 - r**2/(func_x(x))**2)
    ur = lambda x, r: 2*(a0**2*U0/(func_x(x))**2)*beta_of_x_for_ur(x)*(r/(func_x(x)) - r**3/(func_x(x))**3)
    rand = np.random.randn(Nt-1, 3*Npts)

    # simulation
    for i in range(1, Nt):
        x[:, i] = x[:, i-1] + ux(x[:, i-1], r[:, i-1])*dt + sig_s*rand[i-1, 0:Npts]
        r_temp = r[:, i-1] + ur(x[:, i-1], r[:, i-1])*dt
        x2 = r_temp*np.cos(theta[:, i-1]) + sig_s*rand[i-1, Npts:2*Npts]
        x3 = r_temp*np.sin(theta[:, i-1]) + sig_s*rand[i-1, 2*Npts:3*Npts]
        theta[:, i] = np.arctan2(x3, x2)
        r_new = np.sqrt(x2**2 + x3**2)
        loc_pos = (r_new > func_x(x[:, i]))
        r_new[loc_pos] = 2*func_x(x[loc_pos, i]) - r_new[loc_pos]
        loc_neg = (r_new < 0)
        r_new[loc_neg] = -r_new[loc_neg]
        theta[loc_neg, i] = theta[loc_neg, i] + np.pi
        loc_pos = (r_new > func_x(x[:, i]))
        r_new[loc_pos] = 2*func_x(x[loc_pos, i]) - r_new[loc_pos]
        loc_neg = (r_new < 0)
        r_new[loc_neg] = -r_new[loc_neg]
        theta[loc_neg, i] = theta[loc_neg, i] + np.pi
        r[:, i] = r_new

    # analysis
    T = np.arange(Nt)*dt
    weighted_x = weighted_mean_r_xfunc(x, func_x)
    weighted_var = weighted_variance_r_xfunc(x, func_x, weighted_x)
    weighted_skewness = weighted_skewness_r_xfunc(x, func_x, weighted_x, weighted_var)
    weighted_kurtosis = weighted_kurtosis_r_xfunc(x, func_x, weighted_x, weighted_var)

    x_min = 0
    x_max = np.max(weighted_x)*2
    n_diff = 5000
    x_range = np.linspace(x_min, x_max, n_diff+1)
    x_mid = (x_range[0:-1] + x_range[1:])/2
    dx = (x_max - x_min)/n_diff
    f_x_2 = func_x(x_range)**2
    f_x_2 = (f_x_2[0:-1] + f_x_2[1:])/2
    F_x = np.hstack([0, np.cumsum(f_x_2*dx)])/(U0*a0**2)
    F_inv = interp1d(F_x, x_range, kind='cubic')

    a_x = func_x(x_range)
    beta = np.gradient(a_x, dx)
    beta_of_x = interp1d(x_range, beta, kind='cubic')
    gamma = np.gradient(beta, dx)
    G_x = np.hstack([0, np.cumsum((f_x_2)/(1 + 1/6*((beta[0:-1]+beta[1:])/2)**2 + 1/12*func_x(x_mid)*(gamma[0:-1]+gamma[1:])/2)*dx)])/(U0*a0**2)
    G_inv = interp1d(G_x, x_range, kind='cubic')
    
    one_beta_agamma_o_aEN2 = (1 + 1/6*beta**2 + 1/12*a_x*gamma)/(a_x**2)
    diff_one_beta_agamma_o_aEN2 = np.gradient(one_beta_agamma_o_aEN2, dx)
    diff2_aEN2 = np.gradient(np.gradient(1/a_x**2, dx), dx)
    diff2_beta_o_a = np.gradient(np.gradient(beta/a_x, dx), dx)

    diff_one_beta_agamma_o_aEN2_of_x = interp1d(x_range, diff_one_beta_agamma_o_aEN2, kind='cubic')
    diff2_aEN2_of_x = interp1d(x_range, diff2_aEN2, kind = 'cubic')
    diff2_beta_o_a = interp1d(x_range, diff2_beta_o_a, kind = 'cubic')
    aEN2_of_x = interp1d(x_range, 1/a_x**2, kind = 'cubic')
    beta_o_a_of_x = interp1d(x_range, beta/a_x, kind = 'cubic')
    
    cdx = np.sum(1/func_x(x)**2, axis = 0)
    
    predicted_x_bar = G_inv(T)

    ln_cdx_0 = np.log(cdx[0])
    predicted_ln_cdx_heuristic = np.zeros_like(cdx, dtype = float)
    predicted_ln_cdx_heuristic[0] = ln_cdx_0
    dlnc_bar_dt = U0*a0**2*diff_one_beta_agamma_o_aEN2_of_x(predicted_x_bar) + D*(U0**2*a0**4/(48*D**2)*diff2_aEN2_of_x(predicted_x_bar) 
                                                                                  - 1/6*U0*a0**2/D*diff2_beta_o_a(predicted_x_bar))
    predicted_var = np.zeros_like(weighted_var, dtype = float)
    predicted_var_heuristic = np.zeros_like(weighted_var, dtype = float)
    
    for i in range(1,Nt):
        predicted_ln_cdx_heuristic[i] = predicted_ln_cdx_heuristic[i-1] + dt*(
                U0*a0**2*diff_one_beta_agamma_o_aEN2_of_x(predicted_x_bar[i-1]) + D*(U0**2*a0**4/(48*D**2)*diff2_aEN2_of_x(predicted_x_bar[i-1]) 
                                                                                     - 1/6*U0*a0**2/D*diff2_beta_o_a(predicted_x_bar[i-1])))
        predicted_var[i] = predicted_var[i-1] + 2*D*dt*(1 + 1/48*U0**2*a0**4/D**2/func_x(predicted_x_bar[i-1])**2 
                                                        - 1/6*U0*a0**2/D*beta_of_x(predicted_x_bar[i-1])/func_x(predicted_x_bar[i-1]))
        predicted_var_heuristic[i] = predicted_var_heuristic[i-1] + dt*(
                -predicted_var_heuristic[i-1]*dlnc_bar_dt[i-1]
                +3*U0*a0**2*predicted_var_heuristic[i-1]*diff_one_beta_agamma_o_aEN2_of_x(predicted_x_bar[i-1])
                +2*D*(1 + 1/48*U0**2*a0**4/D**2*aEN2_of_x(predicted_x_bar[i-1]) - 1/6*U0*a0**2/D*beta_o_a_of_x(predicted_x_bar[i-1]))
                +D*predicted_var_heuristic[i-1]*(5/48*U0**2*a0**4/D**2*diff2_aEN2_of_x(predicted_x_bar[i-1])
                                                -5/6*U0*a0**2/D*diff2_beta_o_a(predicted_x_bar[i-1])))
    result = {'x': x,
              'r': r,
              'theta': theta,
              'T': T,
              'weighted_x': weighted_x, 
              'weighted_var': weighted_var,
              'weighted_skewness': weighted_skewness,
              'weighted_kurtosis': weighted_kurtosis,
              'x_range': x_range,
              'x_mid': x_min,
              'dx': dx,
              'F_x': F_x,
              'a_x': a_x,
              'beta': beta,
              'gamma': gamma,
              'G_x': G_x,
              'one_beta_agamma_o_aEN2': one_beta_agamma_o_aEN2,
              'diff_one_beta_agamma_o_aEN2': diff_one_beta_agamma_o_aEN2,
              'diff2_aEN2': diff2_aEN2,
              'diff2_beta_o_a': diff2_beta_o_a,
              'cdx': cdx,
              'predicted_x_bar': predicted_x_bar,
              'ln_cdx_0': ln_cdx_0,
              'predicted_ln_cdx_heuristic': predicted_ln_cdx_heuristic,
              'dlnc_bar_dt': dlnc_bar_dt,
              'predicted_var': predicted_var,
              'predicted_var_heuristic': predicted_var_heuristic,
              'a0': a0, 'Pe0': Pe0, 'D': D, 'U0': U0, 'dt': dt}
    return result

def simulation_cone(Pe0, beta, Nt0 = 500, seed = 0):
    # define parameter
    U0 = 1
    a0 = 1
    A = 40
    B = A/Pe0
    dt = 1/B
    D = B/A
    Npts = 5000
    Nt = Nt0*A+1
    sig_s = np.sqrt(2*D*dt)
    np.random.seed(seed)

    # initialization
    r = np.zeros((Npts, Nt))
    theta = np.zeros((Npts, Nt))
    x = np.zeros((Npts, Nt))
    theta[:, 0] = (np.random.rand(Npts))*2*np.pi
    r[:,0] = np.sqrt((np.random.rand(Npts))*a0**2)
    ux = lambda x, r: 2*(a0**2*U0/(a0+beta*x)**2)*(1 - r**2/(a0+beta*x)**2)
    ur = lambda x, r: 2*(a0**2*U0/(a0+beta*x)**2)*beta*(r/(a0+beta*x) - r**3/(a0+beta*x)**3)
    rand = np.random.randn(Nt-1, 3*Npts)

    # simulation
    for i in range(1, Nt):
        x[:, i] = x[:, i-1] + ux(x[:, i-1], r[:, i-1])*dt + sig_s*rand[i-1, 0:Npts]
        r_temp = r[:, i-1] + ur(x[:, i-1], r[:, i-1])*dt
        x2 = r_temp*np.cos(theta[:, i-1]) + sig_s*rand[i-1, Npts:2*Npts]
        x3 = r_temp*np.sin(theta[:, i-1]) + sig_s*rand[i-1, 2*Npts:3*Npts]
        theta[:, i] = np.arctan2(x3, x2)
        r_new = np.sqrt(x2**2 + x3**2)
        loc_pos = (r_new > (a0 + beta*x[:, i]))
        r_new[loc_pos] = 2*(a0+beta*x[loc_pos, i]) - r_new[loc_pos]
        loc_neg = (r_new < 0)
        r_new[loc_neg] = -r_new[loc_neg]
        theta[loc_neg, i] = theta[loc_neg, i] + np.pi
        loc_pos = (r_new > (a0 + beta*x[:, i]))
        r_new[loc_pos] = 2*(a0+beta*x[loc_pos, i]) - r_new[loc_pos]
        loc_neg = (r_new < 0)
        r_new[loc_neg] = -r_new[loc_neg]
        theta[loc_neg, i] = theta[loc_neg, i] + np.pi
        r[:, i] = r_new

    # analysis
    T = np.arange(Nt)*dt
    weighted_x = weighted_mean_r(x, beta, a0)
    weighted_var = weighted_variance_r(x, beta, a0, weighted_x)
    weighted_skewness = weighted_skewness_r(x, beta, a0, weighted_x, weighted_var)
    weighted_kurtosis = weighted_kurtosis_r(x, beta, a0, weighted_x, weighted_var)

    cdx = np.sum(1/(a0 + beta*x)**2, axis = 0)
    
    predicted_x_bar = ((a0**3 + 3*beta*U0*a0**2*T*(1+1/6*beta**2))**(1/3) - a0)/beta

    ln_cdx_0 = np.log(cdx[0])
    predicted_ln_cdx_heuristic = np.zeros_like(cdx, dtype = float)
    predicted_ln_cdx_heuristic[0] = ln_cdx_0
    predicted_var = np.zeros_like(weighted_var, dtype = float)
    predicted_var_heuristic = np.zeros_like(weighted_var, dtype = float)
    dlnc_bar_dt = 1/8*U0**2*a0**4*beta**2/D*1/(a0+beta*predicted_x_bar)**4 - U0*a0**2*(1/3*beta**3 + 2*beta*(1+1/6*beta**2))*1/(
        a0+beta*predicted_x_bar)**3
    
    for i in range(1,Nt):
        predicted_ln_cdx_heuristic[i] = predicted_ln_cdx_heuristic[i-1] + dt*(
            1/8*U0**2*a0**4*beta**2/D*1/(a0+beta*predicted_x_bar[i-1])**4 - U0*a0**2*(1/3*beta**3 + 2*beta*(1+1/6*beta**2))*1/(
                a0+beta*predicted_x_bar[i-1])**3)
        predicted_var[i] = predicted_var[i-1] + 2*D*dt*(1 + 1/48*U0**2*a0**4/D**2/(a0+beta*predicted_x_bar[i-1])**2 
                                                        - 1/6*U0*a0**2/D*beta/(a0+beta*predicted_x_bar[i-1]))
        predicted_var_heuristic[i] = predicted_var_heuristic[i-1] + dt*(
            -predicted_var_heuristic[i-1]*dlnc_bar_dt[i-1]
            -6*beta*(1+1/6*beta**2)*U0*a0**2/(a0+beta*predicted_x_bar[i-1])**3*predicted_var_heuristic[i-1]
            +2*D*(1 + 1/48*U0**2*a0**4/D**2/(a0+beta*predicted_x_bar[i-1])**2 - 1/6*U0*a0**2*beta/D/(a0+beta*predicted_x_bar[i-1]))
            +D*predicted_var_heuristic[i-1]*(1/2*U0**2*a0**4*beta**2/D**2/(a0+beta*predicted_x_bar[i-1])**4 
                                            -4/3*U0*a0**2*beta**3/D/(a0+beta*predicted_x_bar[i-1])**3
                                            +1/8*U0**2*a0**4*beta**3/D**2/(a0+beta*predicted_x_bar[i-1])**4
                                            -1/3*U0*a0**2*beta**4/D/(a0+beta*predicted_x_bar[i-1])**3))

    result = {'x': x,
              'r': r,
              'theta': theta,
              'T': T,
              'weighted_x': weighted_x, 
              'weighted_var': weighted_var,
              'weighted_skewness': weighted_skewness,
              'weighted_kurtosis': weighted_kurtosis,
              'predicted_x_bar': predicted_x_bar,
              'predicted_ln_cdx_heuristic': predicted_ln_cdx_heuristic,
              'dlnc_bar_dt': dlnc_bar_dt,
              'predicted_var': predicted_var,
              'predicted_var_heuristic': predicted_var_heuristic,           
              'beta': beta, 'a0': a0, 'Pe0': Pe0, 'D': D, 'U0': U0, 'dt': dt,
              'cdx': cdx}
    return result