import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import sacc
import itertools
import yaml
import emcee


ALL_LABELS = {'A_lens':r'$A_{lens}$', 'r_tensor':'r', 'beta_d':r'$\beta_d$', 
            'epsilon_ds':r'$\epsilon_{ds}$', 'birefringence':r'$\theta$',
            'alpha_d_ee':r'$\alpha^{EE}_d$', 'amp_d_ee':r'$A^{EE}_d$', 
            'alpha_d_bb':r'$\alpha^{BB}_d$', 'amp_d_bb':r'$A^{BB}_d$', 
            'beta_s':r'$\beta_s$', 'alpha_s_ee':r'$\alpha^{EE}_s$', 
            'amp_s_ee':r'$A^{EE}_s$', 'alpha_s_bb':r'$\alpha^{BB}_s$', 
            'amp_s_bb':r'$A^{BB}_s$', 'amp_d_eb':r'$A^{EB}_d$', 
            'alpha_d_eb':r'$\alpha^{EB}_d$', 'amp_s_eb':r'$A^{EB}_s$', 
            'alpha_s_eb':r'$\alpha^{EB}_s$', 'decorr_amp_d':r'$\Delta_d$', 
            'decorr_amp_s':r'$\Delta_s$', 
            'gain_1':r'$G_1$', 'gain_2':r'$G_2$', 'gain_3':r'$G_3$', 
            'gain_4':r'$G_4$', 'gain_5':r'$G_5$', 'gain_6':r'$G_6$',
            'gain_7':r'$G_7$', 'gain_8':r'$G_8$', 'gain_9':r'$G_9$', 
            'gain_10':r'$G_{10}$', 'gain_11':'r$G_{11}$', 'gain_12':'r$G_{12}$', 
            'shift_1':r'$\Delta\nu_1$', 'shift_2':r'$\Delta\nu_2$', 'shift_3':r'$\Delta\nu_3$', 
            'shift_4':r'$\Delta\nu_4$', 'shift_5':r'$\Delta\nu_5$', 'shift_6':r'$\Delta\nu_6$', 
            'shift_7':r'$\Delta\nu_7$', 'shift_8':r'$\Delta\nu_8$', 'shift_9':r'$\Delta\nu_9$', 
            'shift_10':r'$\Delta\nu_{10}$', 'shift_11':r'$\Delta\nu_{11}$', 
            'shift_12':r'$\Delta\nu_{12}$', 
            'angle_1':r'$\phi_1$', 'angle_2':r'$\phi_2$', 'angle_3':r'$\phi_3$', 
            'angle_4':r'$\phi_4$', 'angle_5':r'$\phi_5$', 'angle_6':r'$\phi_6$', 
            'dphi1_1':r'$\Delta\phi_1$', 'dphi1_2':r'$\Delta\phi_2$', 
            'dphi1_3':r'$\Delta\phi_3$', 'dphi1_4':r'$\Delta\phi_4$', 
            'dphi1_5':r'$\Delta\phi_5$', 'dphi1_6':r'$\Delta\phi_6$', 
            'amp_d_beta': r'$B_d$', 'gamma_d_beta': r'$\gamma_d$', 
            'amp_s_beta': r'$B_s$', 'gamma_s_beta': r'$\gamma_s$'}


def save_cleaned_chains_e(fdir):
    reader = emcee.backends.HDFBackend(f'{fdir}emcee.npz.h5')
    x = np.load(f'{fdir}emcee.npz')
    # labels = [ALL_LABELS[k] for k in x['names']]

    try:
        tau = reader.get_autocorr_time(tol=50)
        taust = 'good tau'
    except:
        taust = 'POTENTIALLY BAD TAU'
        tau = reader.get_autocorr_time(tol=0)
        burnin = int(5. * np.max(tau))
        thin = int(0.5 * np.max(tau))
        samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
        N = samples.shape[0]

    return samples


def mean2yaml(mean, template, ofname, include_moment=False):
    """
    Convert a mean dictionary to a single point config file
    mean is a dictionary maps parameter to their mcmc means
    template is usually the yaml of the original emcee config file
    """
    cpsp = template['BBCompSep']
    
    cpsp['sampler'] = 'single_point'
    
    cpsp['cmb_model']['params']['r_tensor'][-1][1] = float(mean['r_tensor'])
    cpsp['cmb_model']['params']['A_lens'][-1][1] = float(mean['A_lens'])

    cpsp['fg_model']['component_1']['sed_parameters']['beta_d'][-1][0] = float(mean['beta_d'])
    cpsp['fg_model']['component_1']['cl_parameters']['BB']['amp_d_bb'][-1][1] = float(mean['amp_d_bb'])
    cpsp['fg_model']['component_1']['cl_parameters']['BB']['alpha_d_bb'][-1][1] = float(mean['alpha_d_bb'])
    cpsp['fg_model']['component_1']['cross']['epsilon_ds'][-1][1] = float(mean['epsilon_ds'])

    cpsp['fg_model']['component_2']['sed_parameters']['beta_s'][-1][0] = float(mean['beta_s'])
    cpsp['fg_model']['component_2']['cl_parameters']['BB']['amp_s_bb'][-1][1] = float(mean['amp_s_bb'])
    cpsp['fg_model']['component_2']['cl_parameters']['BB']['alpha_s_bb'][-1][1] = float(mean['alpha_s_bb'])

    if include_moment:
        cpsp['fg_model']['component_1']['moments']['gamma_d_beta'][-1][1] = float(mean['gamma_d_beta'])
        cpsp['fg_model']['component_1']['moments']['amp_d_beta'][-1][1] = float(mean['amp_d_beta'])
        cpsp['fg_model']['component_2']['moments']['gamma_s_beta'][-1][1] = float(mean['gamma_s_beta'])
        cpsp['fg_model']['component_2']['moments']['amp_s_beta'][-1][1] = float(mean['amp_s_beta'])
    # print(cpsp)
    template['BBCompSep'] = cpsp
    with open(ofname, 'w') as ostream:
        yaml.dump(template, ostream)


def plot_cls_best_fit(cl_coadd_path, covar_path, cl_emcee_path=None, output_dir=None, annotation='_emcee'):
    """
    Plot the best fit power spectra outputed by the modified single point.
    """
    cl_coadd = sacc.Sacc.load_fits(cl_coadd_path)
    if cl_emcee_path is not None:
        cl_emcee = np.load(cl_emcee_path, allow_pickle=True)['cls']
    covar = sacc.Sacc.load_fits(covar_path)

    fig, ax = plt.subplots(6, 6, sharex=True, dpi=300, figsize=(15, 12))
    for i, j in itertools.product(range(6), range(6)):
        ax[i, j].set_axis_off()
    for i, j in itertools.combinations_with_replacement(range(6), 2):
        e_l, dl= cl_coadd.get_ell_cl('cl_bb', f'band{i+1}', f'band{j+1}')
        msk = (e_l > 30) * (e_l < 300)
                
        cov_ind = covar.indices('cl_bb', (f'band{i+1}', f'band{j+1}'))
        var = covar.covariance.covmat[cov_ind][:, cov_ind].diagonal()
        
        ax[j, i].set_axis_on()
        ax[j, i].errorbar(e_l[msk], dl[msk], np.sqrt(var)[msk])
        if cl_emcee_path is not None:
            cl_emcee_single = cl_emcee[:, i, j]
            block_cov = covar.covariance.covmat[cov_ind[msk]][:, cov_ind[msk]]
            cl_diff = cl_emcee_single - dl[msk]
            block_chi2 = (cl_diff * np.linalg.solve(block_cov, cl_diff)).sum()
            ax[j, i].loglog(e_l[msk], cl_emcee_single, label=f'$\\chi^2=${block_chi2:.2f}\ndof = {msk.sum()}')
            ax[j, i].legend()
        ax[j, i].set_title(f'{i+1}x{j+1}')
        ax[j, i].set_xlim(28, 330)
        if j == 5:
            ax[j, i].set_xlabel(r"$\ell$")
        if i == 0:
            ax[j, i].set_ylabel(r"D$\ell^{BB}$ [$\mu$K$^2$]")
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(f"{output_dir}/cls_best_fit{annotation}.png", bbox_inches='tight')


def plot_cls_fit_bias(cl_coadd_path, cl_emcee_path, covar_path, output_dir=None, annotation='_emcee'):
    """
    Plot the bias of best-fit power spectra outputed by the modified single point. Measured in 
    terms of sigma.
    """
    cl_coadd = sacc.Sacc.load_fits(cl_coadd_path)
    cl_emcee = np.load(cl_emcee_path)['cls']
    covar = sacc.Sacc.load_fits(covar_path)

    fig, ax = plt.subplots(6, 6, sharex=True, dpi=300, figsize=(15, 12))
    for i, j in itertools.product(range(6), range(6)):
        ax[i, j].set_axis_off()
    for i, j in itertools.combinations_with_replacement(range(6), 2):
        e_l, dl= cl_coadd.get_ell_cl('cl_bb', f'band{i+1}', f'band{j+1}')
        msk = (e_l > 30) * (e_l < 300)
        cl_emcee_single = cl_emcee[:, i, j]
        
        cov_ind = covar.indices('cl_bb', (f'band{i+1}', f'band{j+1}'))
        var = covar.covariance.covmat[cov_ind][:, cov_ind].diagonal()
        
        ax[j, i].set_axis_on()
        ax[j, i].errorbar(e_l[msk], (cl_emcee_single - dl[msk]) / np.sqrt(var)[msk])
        ax[j, i].axhline(c='k', ls='--')
        ax[j, i].set_title(f'{i+1}x{j+1}')
        ax[j, i].set_xlim(28, 330)
        if j == 5:
            ax[j, i].set_xlabel(r"$\ell$")
        if i == 0:
            ax[j, i].set_ylabel(r"D$\ell^{BB}$ [$\mu$K$^2$]")
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(f"{output_dir}/cls_bias{annotation}.png", bbox_inches='tight')


def plot_covariance(covar_path, cov_labels=None, log_abs=True, output_dir=None, anotation=''):
    """
    Plot one or a list of 6-channel covariance given pathese to covariances.
    """
    if not isinstance(covar_path, list):
        covar_path_ls = [covar_path]
    fig, ax = plt.subplots(18, 7, sharex=True, dpi=300, figsize=(16, 30))
    for cov_no, path in enumerate(covar_path_ls):
        covar = sacc.Sacc.load_fits(path)
        e_l, _ = covar.get_ell_cl('cl_bb', f'band1', f'band1')
        for sub_ind, (t1, t2, t3, t4) in enumerate(itertools.combinations_with_replacement('123456', 4)):
            ind12 = covar.indices('cl_bb', (f'band{t1}', f'band{t2}'))
            ind34 = covar.indices('cl_bb', (f'band{t3}', f'band{t4}'))
            sim_cov = covar.covariance.covmat[ind12][:, ind34]
            sim_diag = np.diagonal(sim_cov).copy()
            if cov_labels is not None:
                lb = cov_labels[cov_no]
            else:
                lb = cov_no
            
            msk = (e_l > 30) * (e_l < 300)
            if log_abs:
                ax[sub_ind//7, sub_ind%7].loglog(e_l[msk], np.abs(sim_diag[msk]), label=f"{lb}")
            else:
                ax[sub_ind//7, sub_ind%7].plot(e_l[msk], sim_diag[msk], label=f"{lb}")

            ax[sub_ind//7, sub_ind%7].set_title(f'{t1}{t2}x{t3}{t4}')
    ax[0, 0].legend()
    if output_dir is not None:
        fig.savefig(f"{output_dir}/covariance{anotation}.png", bbox_inches='tight')