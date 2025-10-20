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
        burnin = int(5. * np.max(tau))
        thin = int(0.5 * np.max(tau))
        samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
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


def plot_cls_best_fit(cl_coadd_path, covar_path, loglog=True, cl_emcee_path=None, output_dir=None, annotation='_emcee'):
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
        if loglog:
            ax[j, i].loglog()
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(f"{output_dir}/cls_best_fit{annotation}.png", bbox_inches='tight')


def plot_cls_best_fit_v2(
    cl_coadd_path, 
    covar_path, 
    tracers,
    loglog=True,
    cl_emcee_path=None, 
    output_dir=None, 
    annotation='_emcee'
):
    """
    Plot the best fit power spectra outputed by the modified single point.
    """
    cl_coadd = sacc.Sacc.load_fits(cl_coadd_path)
    if cl_emcee_path is not None:
        cl_emcee = np.load(cl_emcee_path, allow_pickle=True)['cls']
    covar = sacc.Sacc.load_fits(covar_path)

    ntr = tracers.__len__()

    fig, ax = plt.subplots(ntr, ntr, sharex=True, dpi=500, figsize=(35, 28))
    for i, j in itertools.product(range(ntr), range(ntr)):
        ax[i, j].set_axis_off()
    for i, j in itertools.combinations_with_replacement(range(ntr), 2):
        e_l, dl= cl_coadd.get_ell_cl('cl_bb', tracers[i], tracers[j])
        msk = (e_l > 30) * (e_l < 300)
                
        cov_ind = covar.indices('cl_bb', (tracers[i], tracers[j]))
        var = covar.covariance.covmat[cov_ind][:, cov_ind].diagonal()
        cl2dl = e_l * (e_l - 1) / 2 / np.pi        
        ax[j, i].set_axis_on()
        ax[j, i].errorbar(e_l[msk], dl[msk] * cl2dl[msk], np.sqrt(var)[msk] * cl2dl[msk])
        if cl_emcee_path is not None:
            cl_emcee_single = cl_emcee[:, i, j]
            block_cov = covar.covariance.covmat[cov_ind[msk]][:, cov_ind[msk]]
            cl_diff = cl_emcee_single - dl[msk]
            block_chi2 = (cl_diff * np.linalg.solve(block_cov, cl_diff)).sum()
            ax[j, i].loglog(e_l[msk], cl_emcee_single, label=f'$\\chi^2=${block_chi2:.2f}\ndof = {msk.sum()}')
            ax[j, i].legend()
        ax[j, i].set_title(f'{tracers[i]}\n{tracers[j]}')
        ax[j, i].set_xlim(28, 330)
        if j == ntr - 1:
            ax[j, i].set_xlabel(r"$\ell$")
        if i == 0:
            ax[j, i].set_ylabel(r"D$\ell^{BB}$ [$\mu$K$^2$]")
        if loglog:
            ax[j, i].loglog()
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(f"{output_dir}/cls_best_fit{annotation}.png", bbox_inches='tight')


def _check_fg_cl_format(fg_ell, dust_cl, sync_cl, ntr, convert_cl2dl):
    """ Check the format of foreground Cls and return the cl2dl conversion factor and Cl mask """
    if fg_ell is not None:
        if dust_cl is not None:
            assert(dust_cl.shape[0] == ntr)
            assert(dust_cl.shape[1] == fg_ell.shape[0])
        if sync_cl is not None:
            assert(sync_cl.shape[0] == ntr)
            assert(sync_cl.shape[1] == fg_ell.shape[0])
        if dust_cl is None and sync_cl is None:
            raise ValueError("fg_ell is given but both dust_cl and sync_cl are None")
        if convert_cl2dl:
            fg_cl2dl = fg_ell * (fg_ell - 1) / 2 / np.pi
        else:
            fg_cl2dl = np.ones_like(fg_ell)
        return fg_cl2dl, (fg_ell > 1) * (fg_ell < 300)
    else:
        return None, None
    

def _check_cmb_cl_format(cmb_ell, cmbl_cl, r, cmb_r1_cl, convert_cl2dl):
    """ Check the format of CMB Cls and return the cl2dl conversion factor, Cl mask, r = 1 B-mode Cls """
    if cmb_ell is not None:
        assert(cmbl_cl is not None)
        assert(cmbl_cl.shape[0] == cmb_ell.shape[0])
        if r is not None:
            assert(cmb_r1_cl is not None)
            assert(cmb_r1_cl.shape[0] == cmb_ell.shape[0])
            cmblr_cl = cmbl_cl + r * (cmb_r1_cl - cmbl_cl)
            cmbr_cl = r * (cmb_r1_cl - cmbl_cl)
        else:
            cmblr_cl, cmbr_cl = None, None
        if convert_cl2dl:
            cmb_cl2dl = cmb_ell * (cmb_ell - 1) / 2 / np.pi
        else:
            cmb_cl2dl = np.ones_like(cmb_ell)
        return cmb_cl2dl, (cmb_ell > 1) * (cmb_ell < 300), cmblr_cl, cmbr_cl
    else:
        return None, None, None, None


def _plot_emcee_fit(cl_emcee_path, cl_emcee, covar, cov_ind, e_l, dl, msk, i, j):
    """ plot emcee fit on the current plt axis """
    if cl_emcee_path is not None:
        cl_emcee_single = cl_emcee[:, i, j]
        block_cov = covar.covariance.covmat[cov_ind[msk]][:, cov_ind[msk]]
        cl_diff = cl_emcee_single - dl[msk]
        block_chi2 = (cl_diff * np.linalg.solve(block_cov, cl_diff)).sum()
        plt.loglog(e_l[msk], cl_emcee_single, label=f'$\\chi^2=${block_chi2:.2f}\ndof = {msk.sum()}')


def _plot_fg_cmb(fg_ell, dust_cl, sync_cl, fg_cl2dl, cmb_ell, cmbl_cl, cmblr_cl, cmbr_cl, r, cmb_cl2dl, fgmsk, cmbmsk, i, j):
    """ Plot foreground and CMB Cls on the current plt axis """
    if cmb_ell is not None:
        plt.loglog(cmb_ell[cmbmsk], (cmbl_cl * cmb_cl2dl)[cmbmsk], ls='--', c='k', label='CMB Lensing')
        if r is not None:
            plt.loglog(cmb_ell[cmbmsk], (cmbr_cl * cmb_cl2dl)[cmbmsk], ls='-.', c='k', label=f'CMB r={r:.3f}')
            plt.loglog(cmb_ell[cmbmsk], (cmblr_cl * cmb_cl2dl)[cmbmsk], ls='-', c='k', label=f'CMB Lensing + r={r:.3f}')
    if dust_cl is not None:
        plt.loglog(fg_ell[fgmsk], (np.sqrt(dust_cl[i] * dust_cl[j]) * fg_cl2dl)[fgmsk], label='Dust', ls=':', alpha=.7)
    if sync_cl is not None:
        plt.loglog(fg_ell[fgmsk], (np.sqrt(sync_cl[i] * sync_cl[j]) * fg_cl2dl)[fgmsk], label='Sync', ls='-.', alpha=.7)
        if dust_cl is not None:
            plt.loglog(fg_ell[fgmsk], ((np.sqrt(sync_cl[i] * sync_cl[j]) + np.sqrt(dust_cl[i] * dust_cl[j]))* fg_cl2dl)[fgmsk], label='D+S', ls='-', alpha=.7)


def _plot_label_title(tracers, tracer_aliases, i, j, convert_cl2dl, loglog):
    """ Plot the label and title on the current plt axis """
    if tracer_aliases is not None:
        plt.title(f'{tracer_aliases[i]}\n{tracer_aliases[j]}')
    else:
        plt.title(f'{tracers[i]}\n{tracers[j]}')
    plt.xlim(1, 350)
    plt.xlabel(r"$\ell$")
    lpow = .6
    xticks = np.array((2, 10, 50, 100, 200, 300))
    xtick_labels = [f"{i}" for i in xticks]
    plt.xscale('function', functions=(lambda x: x**lpow, lambda x: x**(1/lpow)))
    plt.xticks(ticks=xticks, labels=xtick_labels)
    if convert_cl2dl:
        plt.ylabel(r"D$\ell^{BB}$ [$\mu$K$^2$]")
    else:
        plt.ylabel(r"$C_\ell^{BB}$ [$\mu$K$^2$]")
    if loglog:
        plt.loglog()
    plt.legend()


def _plot_savefig(output_dir, annotation, tracer_aliases, tracers, i, j):
    """ Save the current plt figure """
    if output_dir is not None:
        if tracer_aliases is not None:
            plt.savefig(f"{output_dir}/cls_{annotation}_{tracer_aliases[i]}_X_{tracer_aliases[j]}.png", bbox_inches='tight')
        else:
            plt.savefig(f"{output_dir}/cls_{annotation}_{tracers[i]}_X_{tracers[j]}.png", bbox_inches='tight')
            

def plot_cls_best_fit_individual_single(
    cl_coadd_path : str,
    covar_path : str,
    tracer1 : str,
    tracer2 : str,
    tracers : list[str],
    tracer_aliases : list[str] | None = None,
    loglog : bool = True,
    cl_emcee_path : str | None = None,
    output_dir : str | None = None,
    annotation : str = '_emcee',
    convert_cl2dl : bool = True,
    fg_ell : np.ndarray | None = None,
    dust_cl : np.ndarray | None = None,
    sync_cl : np.ndarray | None = None,
    cmb_ell : np.ndarray | None = None,
    cmbl_cl : np.ndarray | None = None,
    r : float | None = None,
    cmbr1_cl : np.ndarray | None = None,
    ipynb=False,
):
    """
    Plot the best fit power spectra outputed by the modified single point.
    """
    cl_coadd = sacc.Sacc.load_fits(cl_coadd_path)
    cl_emcee = None
    if cl_emcee_path is not None:
        cl_emcee = np.load(cl_emcee_path, allow_pickle=True)['cls']
    covar = sacc.Sacc.load_fits(covar_path)

    ntr = tracers.__len__()
    if tracer_aliases is not None:
        assert(len(tracer_aliases) == ntr)

    fg_cl2dl, fgmsk = _check_fg_cl_format(fg_ell, dust_cl, sync_cl, ntr, convert_cl2dl)
    cmb_cl2dl, cmbmsk, cmblr_cl, cmbr_cl = _check_cmb_cl_format(cmb_ell, cmbl_cl, r, cmbr1_cl, convert_cl2dl)
    
    i = tracers.index(tracer1)
    j = tracers.index(tracer2)
    # for i, j in itertools.combinations_with_replacement(range(ntr), 2):
    e_l, dl= cl_coadd.get_ell_cl('cl_bb', tracers[i], tracers[j])
    if convert_cl2dl:
        cl2dl = e_l * (e_l - 1) / 2 / np.pi
    else:
        cl2dl = np.ones_like(e_l)
    msk = (e_l > 30) * (e_l < 300)

    # using fig axes
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    cov_ind = covar.indices('cl_bb', (tracers[i], tracers[j]))
    var = covar.covariance.covmat[cov_ind][:, cov_ind].diagonal()
    plt.errorbar(e_l[msk], dl[msk] * cl2dl[msk], np.sqrt(var)[msk] * cl2dl[msk], \
        ls='', fmt='.', capsize=3, label='Data')

    _plot_emcee_fit(cl_emcee_path, cl_emcee, covar, cov_ind, e_l, dl, msk, i, j)
    _plot_fg_cmb(fg_ell, dust_cl, sync_cl, fg_cl2dl, cmb_ell, cmbl_cl, cmblr_cl, cmbr_cl, \
        r, cmb_cl2dl, fgmsk, cmbmsk, i, j)
    _plot_label_title(tracers, tracer_aliases, i, j, convert_cl2dl, loglog)
    _plot_savefig(output_dir, annotation, tracer_aliases, tracers, i, j)
    if ipynb:
        plt.show()

    plt.close()
    return fig, ax, f"{tracer_aliases[i]}X{tracer_aliases[j]}"


def plot_cls_best_fit_individual_all(
    cl_coadd_path : str,
    covar_path : str,
    tracers : list[str],
    tracer_aliases : list[str] | None = None,
    loglog : bool = True,
    cl_emcee_path : str | None = None,
    output_dir : str | None = None,
    annotation : str = '_emcee',
    convert_cl2dl : bool = True,
    fg_ell : np.ndarray | None = None,
    dust_cl : np.ndarray | None = None,
    sync_cl : np.ndarray | None = None,
    cmb_ell : np.ndarray | None = None,
    cmbl_cl : np.ndarray | None = None,
    r : float | None = None,
    cmbr1_cl : np.ndarray | None = None,
):
    """
    Plot the best fit power spectra outputed by the modified single point.
    """
    cl_coadd = sacc.Sacc.load_fits(cl_coadd_path)
    cl_emcee = None
    if cl_emcee_path is not None:
        cl_emcee = np.load(cl_emcee_path, allow_pickle=True)['cls']
    covar = sacc.Sacc.load_fits(covar_path)

    ntr = tracers.__len__()
    if tracer_aliases is not None:
        assert(len(tracer_aliases) == ntr)

    fg_cl2dl, fgmsk = _check_fg_cl_format(fg_ell, dust_cl, sync_cl, ntr, convert_cl2dl)
    cmb_cl2dl, cmbmsk, cmblr_cl, cmbr_cl = _check_cmb_cl_format(cmb_ell, cmbl_cl, r, cmbr1_cl, convert_cl2dl)

    for i, j in itertools.combinations_with_replacement(range(ntr), 2):
        e_l, dl= cl_coadd.get_ell_cl('cl_bb', tracers[i], tracers[j])
        if convert_cl2dl:
            cl2dl = e_l * (e_l - 1) / 2 / np.pi
        else:
            cl2dl = 1.0
        msk = (e_l > 30) * (e_l < 300)
        plt.figure(figsize=(7, 5), dpi=300)
        cov_ind = covar.indices('cl_bb', (tracers[i], tracers[j]))
        var = covar.covariance.covmat[cov_ind][:, cov_ind].diagonal()
        plt.errorbar(e_l[msk], dl[msk] * cl2dl[msk], np.sqrt(var)[msk] * cl2dl[msk], ls='', fmt='.', capsize=3)

        _plot_emcee_fit(cl_emcee_path, cl_emcee, covar, cov_ind, e_l, dl, msk, i, j)
        _plot_fg_cmb(fg_ell, dust_cl, sync_cl, fg_cl2dl, cmb_ell, cmbl_cl, cmblr_cl, cmbr_cl, \
                     r, cmb_cl2dl, fgmsk, cmbmsk, i, j)
        _plot_label_title(tracers, tracer_aliases, i, j, convert_cl2dl, loglog)
        _plot_savefig(output_dir, annotation, tracer_aliases, tracers, i, j)

        plt.close()


def calc_block_chi2(
    cl_coadd_path: str, 
    covar_path: str, 
    cl_emcee_path: str
):
    """
    Get the chi2 for each block of selected outputs.
    """
    cl_coadd = sacc.Sacc.load_fits(cl_coadd_path)
    if cl_emcee_path is not None:
        cl_emcee = np.load(cl_emcee_path, allow_pickle=True)['cls']
    covar = sacc.Sacc.load_fits(covar_path)

    block_chi2_arr = np.zeros(21)
    ind1_arr = np.zeros(21)
    ind2_arr = np.zeros(21)

    for idx, (i, j) in enumerate(itertools.combinations_with_replacement(range(6), 2)):
        e_l, dl= cl_coadd.get_ell_cl('cl_bb', f'band{i+1}', f'band{j+1}')
        msk = (e_l > 30) * (e_l < 300)
                
        cov_ind = covar.indices('cl_bb', (f'band{i+1}', f'band{j+1}'))
        
        cl_emcee_single = cl_emcee[:, i, j]
        block_cov = covar.covariance.covmat[cov_ind[msk]][:, cov_ind[msk]]
        cl_diff = cl_emcee_single - dl[msk]
        block_chi2 = (cl_diff * np.linalg.solve(block_cov, cl_diff)).sum()
        block_chi2_arr[idx] = block_chi2
        ind1_arr[idx] = i
        ind2_arr[idx] = j
    return block_chi2_arr, ind1_arr, ind2_arr


def calc_full_block_chi2(
    cl_coadd_path: str, 
    covar_path: str, 
    cl_emcee_path: str
):
    """
    Get the chi2 for each block of selected outputs.
    """
    cl_coadd = sacc.Sacc.load_fits(cl_coadd_path)
    if cl_emcee_path is not None:
        cl_emcee = np.load(cl_emcee_path, allow_pickle=True)['cls']
    covar = sacc.Sacc.load_fits(covar_path)

    e_l, _ = cl_coadd.get_ell_cl('cl_bb', f'band1', f'band1')
    msk = (e_l > 30) * (e_l < 300)
    covar_ind = []
    diff_dls = []
    for i, j in itertools.combinations_with_replacement(range(6), 2):
                
        cov_ind = covar.indices('cl_bb', (f'band{i+1}', f'band{j+1}'))
        covar_ind.append(cov_ind[msk])
        
        cl_emcee_single = cl_emcee[:, i, j]
        _, dl = cl_coadd.get_ell_cl('cl_bb', f'band{i+1}', f'band{j+1}')
        diff_dls.append(dl[msk] - cl_emcee_single)

    covar_ind_arr = np.concatenate(covar_ind)
    diff_dls_arr = np.concatenate(diff_dls)
    block_cov = covar.covariance.covmat[covar_ind_arr][:, covar_ind_arr]
    block_chi2 = diff_dls_arr * np.linalg.solve(block_cov, diff_dls_arr)
    inc = msk.sum()
    result_chi2 = np.zeros(21)
    for i in range(21):
        result_chi2[i] = block_chi2[i*inc:(i+1)*inc].sum()
    return result_chi2



def print_cls_chi2(cl_coadd_path, covar_path, cl_emcee_path):
    """ print chi2 contribution for each cross term """
    cl_coadd = sacc.Sacc.load_fits(cl_coadd_path)
    cl_emcee = np.load(cl_emcee_path, allow_pickle=True)['cls']
    covar = sacc.Sacc.load_fits(covar_path)
    inv_cov = np.linalg.solve(covar.covariance.covmat, np.identity(covar.covariance.covmat.shape[0]))
    counter = 0
    out_header = ""
    out_info = ""
    chi2_total = 0
    for t1, t2, t3, t4 in itertools.combinations_with_replacement('123456', 4):
        counter += 1
        header= f"{t1}{t2}x{t3}{t4}"
        out_header += f"{header:<15}"
        e_l, dl12= cl_coadd.get_ell_cl('cl_bb', f'band{t1}', f'band{t2}')
        e_l, dl34= cl_coadd.get_ell_cl('cl_bb', f'band{t3}', f'band{t4}')
        msk = (e_l > 30) * (e_l < 300)

        cov_ind12 = covar.indices('cl_bb', (f'band{t1}', f'band{t2}'))
        cov_ind34 = covar.indices('cl_bb', (f'band{t3}', f'band{t4}'))

        cl_emcee_single12 = cl_emcee[:, int(t1) - 1, int(t2) - 1]
        cl_emcee_single34 = cl_emcee[:, int(t3) - 1, int(t4) - 1]
        cl_diff12 = cl_emcee_single12 - dl12[msk]
        cl_diff34 = cl_emcee_single34 - dl34[msk]

        inv_cov_slice = inv_cov[cov_ind12[msk]][:, cov_ind34[msk]]
        info = f"{np.dot(cl_diff34, np.dot(inv_cov_slice, cl_diff12)):.2f}"
        if f"{t1}{t2}" == f"{t3}{t4}":
            chi2_total += np.dot(cl_diff34, np.dot(inv_cov_slice, cl_diff12))
        else:
            chi2_total += np.dot(cl_diff34, np.dot(inv_cov_slice, cl_diff12)) * 2
        out_info += f"{info:<15}"
        if counter == 7:
            counter = 0
            print(out_header)
            print(out_info)
            out_header = ""
            out_info = ""
    print(f"Chi2 : {chi2_total}")


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
    else:
        covar_path_ls = covar_path
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


def plot_covariance_v2(covar_path, cov_labels=None, log_abs=True, output_dir=None, anotation=''):
    """
    Plot one or a list of 6-channel covariance given pathese to covariances.
    """
    if not isinstance(covar_path, list):
        covar_path_ls = [covar_path]
    else:
        covar_path_ls = covar_path
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