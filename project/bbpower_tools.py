import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import sacc
import itertools


def plot_cls_best_fit(cl_coadd_path, cl_emcee_path, covar_path, output_dir=None, annotation='_emcee'):
    """
    Plot the best fit power spectra outputed by the modified single point.
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
        ax[j, i].errorbar(e_l[msk], dl[msk], np.sqrt(var)[msk])
        ax[j, i].loglog(e_l[msk], cl_emcee_single)
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