import numpy as np
import scipy as sci
import matplotlib as ml
import matplotlib.pyplot as plt
from astropy.io import fits 
import healpy as hp
from scipy.interpolate import splrep, splev
from scipy import ndimage
from scipy import stats
import datetime
import nawrapper as nw


def read_abs_map(mpath, nside_abs):
    # zack note: not exactly sure what is going on in this
    # don't use this directly
    npix = hp.nside2npix(nside_abs)
    qmap = np.zeros(npix)
    umap = np.zeros(npix)
    d = fits.open(mpath)[1].data
    ind_sc = d['PIXLIST']
    q = d['Q']/d['QQ']
    u = d['U']/d['UU']
    q_wt=d['QQ']
    u_wt=d['UU']
    print( "Q rms = %f, U rms = %f"%(np.std(q), np.std(u)))
    qmap[ind_sc] = q
    umap[ind_sc] = u
    return qmap, umap, q_wt, u_wt, ind_sc

def get_ABS_maps(data_dir, iuse_sc=True, nside_abs=256):
    
    # Units are mK thermodynamic
    file_abs  = f'{data_dir}/ABS_data_fielda.fits'
    file_abs_mask  = f'{data_dir}/ABS_fielda_mask.fits'
    
    abs=fits.open(file_abs)
    abs_hp_pix=abs[1].data.field(0)
    abs_q=abs[1].data.field(1)
    abs_u=abs[1].data.field(2)
    abs_q_wt=abs[1].data.field(3)  #1./uK^2
    abs_qu_wt=abs[1].data.field(4)  #1./uK^2
    abs_u_wt=abs[1].data.field(5)  #1./uK^2

    abs_mask = hp.read_map(file_abs_mask, nest=False, verbose=False)
    mask = abs_mask[abs_hp_pix]
    iwh_ge03 = np.where(mask >= 0.03) #The max of the mask is 0.069. This keeps 21479 pixels.

    # The map and the script to it (Q and U) are here: 
    # http://physics.princeton.edu/actdata/users/schoi/maps/ABS/
    # The default ABS map format has pixels, weighted maps, and weights: 
    # healpix indices, Q/sigma_Q^2, U/sigma_U^2, 1/sigma_Q^2, 1/sigma_U^2. 
    # - Steve
    # ---------------
    
    npix_abs = hp.nside2npix(nside_abs)
    q_sc, u_sc, q_wt_sc, u_wt_sc, ind_sc = read_abs_map(
        f'{data_dir}/data_abs_data_fielda.fits', nside_abs=256)
    
    abs_q_map = np.zeros(npix_abs)
    abs_u_map = np.zeros(npix_abs)
    abs_q_map[abs_hp_pix] = abs_q
    abs_u_map[abs_hp_pix] = abs_u
    
    
    abs_q_map_wt = np.zeros(npix_abs)
    abs_u_map_wt = np.zeros(npix_abs)
    abs_q_map_wt[abs_hp_pix] = abs_q_wt
    abs_u_map_wt[abs_hp_pix] = abs_u_wt

    iuse_sc = 1
    if (iuse_sc):
        abs_q_map = q_sc  #the full maps
        abs_u_map = u_sc
        abs_q_wt = q_wt_sc #just the nom pixels
        abs_u_wt = u_wt_sc
        
    return abs_q_map, abs_u_map, abs_mask, abs_q_map_wt, abs_u_map_wt


def to_nside(m, nside=256, rotate_gal_to_eq=False, lmax=1200):
    """change nside by applying map2alm and back"""
    
    # credit to mat/sigurd for useful reference
    alm   = hp.map2alm(m, lmax=lmax)
    # work around healpix bug
    alm   = alm.astype(np.complex128,copy=False)
    if rotate_gal_to_eq:
        euler = np.array([57.06793215,  62.87115487, -167.14056929])* np.pi/180
        hp.rotate_alm(alm, euler[0], euler[1], euler[2])
    return hp.alm2map(alm, nside=nside, lmax=lmax)

def get_ACT_maps(data_dir, nside=256, freq='090'):
    
    # File handling
    file_act = f'{data_dir}/act_s08_s18_cmb_f{freq}_daynight_map_healpix_512.fits'
    file_actdiv = f'{data_dir}/act_s08_s18_cmb_f{freq}_daynight_div_healpix_512.fits'

    act_t_n = hp.read_map(file_act, field=0, nest=True, verbose=False)
    act_q_n = hp.read_map(file_act, field=1, nest=True, verbose=False)
    act_u_n = hp.read_map(file_act, field=2, nest=True, verbose=False)
    act_t_nwt = hp.read_map(file_actdiv, field=0, nest=True, verbose=False) 
    act_q_nwt = hp.read_map(file_actdiv, field=1, nest=True, verbose=False)
    act_u_nwt = hp.read_map(file_actdiv, field=2, nest=True, verbose=False)

    # Convert to ring storage
    act_t_512 = hp.reorder(act_t_n,inp='NESTED',out='RING')
    act_q_512 = hp.reorder(act_q_n,inp='NESTED',out='RING')
    act_u_512 = hp.reorder(act_u_n,inp='NESTED',out='RING')
    act_t_wt_512 = hp.reorder(act_t_nwt,inp='NESTED',out='RING')
    act_q_wt_512 = hp.reorder(act_q_nwt,inp='NESTED',out='RING')
    act_u_wt_512 = hp.reorder(act_u_nwt,inp='NESTED',out='RING')
    
    # convert to target nside
    act_t_final = to_nside(act_t_512, nside)
    act_q_final = to_nside(act_q_512, nside)
    act_u_final = to_nside(act_u_512, nside)
    act_t_nwt_final = to_nside(act_t_wt_512, nside)
    act_q_nwt_final = to_nside(act_q_wt_512, nside)
    act_u_nwt_final = to_nside(act_u_wt_512, nside)
    
    return (act_t_final, act_q_final, act_u_final, 
            act_t_nwt_final, act_q_nwt_final, act_u_nwt_final)


def get_Planck_maps(map_dir, mask_dir, freq='143', nside=256, mfile_end='2048_R2.02_full.fits', flipq=1.0, flipu=1.0):
    
    mfile = 'rotated_planck_hm1.fits'
    print("Reading", mfile)
    
    map_I = hp.read_map(mfile, field=0, verbose=False) * 1e6
    map_Q = flipq * hp.read_map(mfile, field=1, verbose=False) * 1e6
    map_U = flipu * hp.read_map(mfile, field=2, verbose=False) * 1e6
    
    mask = hp.read_map(maskfile, verbose=False)
    
    print("CONVERTING")
    # Planck maps are in galactic coordinates
    map_I = to_nside(map_I, nside, rotate_gal_to_eq=True)
    map_Q = to_nside(map_Q, nside, rotate_gal_to_eq=True)
    map_U = to_nside(map_U, nside, rotate_gal_to_eq=True)
    # use ud_grade on the mask, don't want ringing
    mask = to_nside(mask, nside=2048, rotate_gal_to_eq=True) # rotate first
    mask = hp.pixelfunc.ud_grade(mask, nside_out=nside)
    
    return map_I, map_Q, map_U, mask
