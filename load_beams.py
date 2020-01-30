"""BEAMS: representative beams for Planck, ACT, ABS."""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import nawrapper as nw
from scipy.interpolate import interp1d
from astropy.table import Table
from pathlib import Path  # this is Python 3


def get_Planck_beam(beam_dir):
    """Planck beam for the 143 GHz half mission cross"""
    beam_Planck = nw.read_beam(
        beam_dir / 'beam_likelihood_143hm1x143hm2.dat')
    return beam_Planck

def get_ABS_beam(beam_dir):
    """ ABS (with transfer function for filter-and-bin maps)"""
    beam_ABS = nw.read_beam(beam_dir / 'abs_beam.txt')
    
    ABS_ell_transfer, ABS_EE_transfer, ABS_BB_transfer = np.genfromtxt(
        beam_dir / 'abs_xfer.txt', unpack=True)

    
    ABS_transfer_func = interp1d(
        [0] + ABS_ell_transfer.tolist() + [30000], 
        np.sqrt([ABS_EE_transfer[0]] + ABS_EE_transfer.tolist() + [1]), 
        kind='linear')
    
    beam_ABS *= ABS_transfer_func(np.arange(len(beam_ABS)))
    return beam_ABS

def get_ACT_beam(beam_dir):
    """ACT beam with transfer function for convergence in T
    
    We take mr3c s15 f150 night as representative of the instrument.
    We have (converged) mapmaker transfer functions from the ACT d56
    patch, which we apply.
    """

    # read in beam profile
    beam_t_act_angle = np.loadtxt(beam_dir / 'act-ish.txt') 
    beam_ACT = hp.sphtfunc.beam2bl(
        beam_t_act_angle[:,1], 
        theta=beam_t_act_angle[:,0]*np.pi/180.0, 
        lmax=21600)
    
    # multiply in transfer function for T that we computed from Planck
    ell_xfer_ACT, ell_TT_xfer_ACT = np.genfromtxt(beam_dir / 'act_xfer.txt')[:,0:]
    beam_ACT /= np.max(beam_ACT)
    beam_ACT_T = beam_ACT.copy()
    beam_ACT_E = beam_ACT.copy()
    beam_ACT_T[:len(ell_TT_xfer_ACT)] *= ell_TT_xfer_ACT
    
    # multiply in transfer function for E that we have from mapmaker sims
    ell_xfer, ACT_d56_xfer_T, ACT_d56_xfer_E =  np.genfromtxt(beam_dir / 'd56_xfer.txt', unpack=True)
    ACT_transfer_func_E = interp1d(
        [0] + ell_xfer.tolist() + [30000], 
        np.sqrt([0] + ( (ACT_d56_xfer_E) ).tolist() + [1]), 
        kind='linear')
    beam_ACT_E *= ACT_transfer_func_E(np.arange(len(beam_ACT)))
    
    return (beam_ACT_T, beam_ACT_E)