#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:41:10 2022

@author: dylan.owens
"""

# Import all the necessary modules
import numpy as np
import emcee
import corner
import argparse
import matplotlib.pyplot as plt
import astropy.units as u
from scipy import integrate
from astropy.io import ascii


# Define some constants (all in CGS)
h=6.626e-27
k=1.381e-16
c=2.99792e10


##### FUNCTIONS #############################################

# Returns dictionary of input parameters from filename (input, priors, and perturbs files)
def LoadFile(filename):
    names, values = ascii.read(filename)[0][:],ascii.read(filename)[1][:]
    output = dict(list(zip(names,values)))
    output['index'] = names
    output['values'] = values
    return output

# Returns array of data from spectrum file (spec or model file)
def LoadSpec(filename):
    return np.loadtxt(filename,skiprows = 1)

# Equation 1 from Jura 2003, returns a dust ring temp given a ring radius
def T_Ring(R):
    return ((2/(3*np.pi))**0.25) * ((WD_R/R)**0.75) * WD_T

# Returns dust ring radius for a given ring temperature, Eqn 1 from Jura 2003 inverted
def Tring_to_R(T,WD_T,WD_R):
    return ((2/(3*np.pi))**(1/3)) * ((WD_T/T)**(4/3)) * WD_R

# x substitution for intgration, returns x_out/in from a given T_in/out
def x_sub(nu,Tring):
    return (h*nu)/(k*Tring)

# Integrand in Equation 3 of Jura 2003
def integrand(x):
    return (x**(5/3))/(np.exp(x) - 1)
    
# MCMC FUNCTIONS #

### PROBABILITY FUNCTION ###
def lnprob(theta,x,y,yerr,weight):
    lp = lnprior(theta)
    if not lp == 0:
        return -np.inf
    return lp + lnlike(theta,x,y,yerr,weight)

### PRIOR FUNCTION ###
def lnprior(theta):
    inc, lnTin, lnTout = theta
    Tin,Tout = np.exp(lnTin), np.exp(lnTout)
    if (inc >= 0) & (inc < 90) & (Tin < Tsubl) & (Tin > Tout) & (Tout > Ttide):
        return 0.0
    else:
        return -np.inf

### LIKELIHOOD FUNCTION ###
# theta = (inclination, Tin, Tout)
# x     = observed nu array
# y     = observed flux array
# yerr  = error on observed flux
# weight= wavelength coverage at a given point
def lnlike(theta,x,y,yerr,weight):
    LnLike = (-1/2) * np.sum(weight*((y - model(theta,x))/yerr)**2)
    return LnLike

### MODEL FUNCTION ###
# theta = vector: (inclination, Tin, Tout)
# nu    = frequency array (Hz)
def model(theta,nu):
    # Vector of input parameters for model
    inc, lnTin, lnTout = theta
    Tin,Tout = np.exp(lnTin), np.exp(lnTout)
    # Array of x values calculated at each nu point (array)
    xin, xout = x_sub(nu,Tin), x_sub(nu,Tout)
    # Array we will fill with model values at each nu point
    model = np.zeros(len(nu))
    for n,pt in enumerate(nu):
        # Calculating model flux value at every nu point (pt)
        model[n] = ((12*(np.pi**(1/3))) * (((WD_R**2)*np.cos(np.radians(inc)))/(d**2)) *
                    (((2*k*WD_T)/(3*h*pt))**(8/3)) * ((h*(pt**3))/(c**2)) * integrate.quad(integrand,xin[n],xout[n])[0]) / 1e-29
    return model

### MAIN FUNCTION RUNNING MCMC ###
def main(p0, nwalkers, niters, ndim, lnprob, obs):
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=obs)

    print("Running MCMC...")
    pos, prob, state = sampler.run_mcmc(p0,niters,progress = True)
     
    return sampler, pos, prob, state

# RETURNS MEDIAN MODEL AND STD AT EACH POINT nu
## nsamples = number of MCMC steps to sample from chain
## flatchain = flattened chain across walker ensemble
## nu = frequency grid to calculate model over
def sample_walkers(nsamples,flatchain,nu):
    models = np.zeros((nsamples,len(nu)))
    draw = np.random.randint(0,len(flatchain),nsamples)
    thetas = flatchain[draw]
    for i,theta in enumerate(thetas):
        m = model(theta,nu)
        models[i,:] = m
    spread = np.std(models,axis=0)
    med_model = np.median(models,axis=0)
    return med_model, spread

#############################################################

parser = argparse.ArgumentParser()
parser.add_argument("inputfile", type=str,help='path to input file')
parser.add_argument('-newrun', action='store_true',help='Start a new MCMC run? Note: this will overwrite an existing run')
parser.add_argument('-plot', action='store_true',help='produce corner+excess SED plots of existing MCMC run')
parser.add_argument('-fit_R', action='store_true',help='put model in terms of disk radii instead of temperature')
parser.add_argument('-walkerplot',action='store_true',help='plot walkers vs steps for each model parameter')
args = parser.parse_args()

inpfile = args.inputfile

# Loading everything in the input file
inputs = LoadFile(inpfile)
index = np.array(inputs['index'])
values = np.array(inputs['values'])
nwalkers = np.int64(inputs['nwalkers'])
niters =np.int64(inputs['niters'])
nburn = np.int64(inputs['nburn'])
target = inputs['targname']

# defining the MCMC output filenames
chainfile = inputs['chainfile']
probfile = inputs['probfile']
accpfile = inputs['accpfile']
    
# Loading the prior and perturb files
priorfile = LoadFile(inputs['priorfile'])
perturbfile = LoadFile((inputs['perturbfile']))

# defining the WD physical parameters
WD_R = (inputs['WD_R']*u.Rsun).to(u.cm)  # Converting from Rsun to cm
d = (inputs['d']*u.pc).to(u.cm)  # Converting from pc to cm
WD_T = np.float64(inputs['WD_T'])
Tsubl = priorfile['Tsubl']
Rtide = (priorfile['Rtide']*u.Rsun).to(u.cm)
Ttide = T_Ring(Rtide.value)

# Loading observed spectral data
spec_obs = LoadSpec(inputs['specfile'])
wv_obs,f_obs,f_err,weights = spec_obs[:,0],spec_obs[:,1],spec_obs[:,2],spec_obs[:,3]

# Loading model spectral data
spec_mod = LoadSpec(inputs['modelfile'])
wv_mod,f_mod = spec_mod[:,0],spec_mod[:,1]

# Calculating excess flux 
f_exc = f_obs - np.interp(wv_obs,wv_mod,f_mod)

# Starting values for model params from input file
initial = np.array([inputs['inc'],np.log(float(inputs['Tin'])),np.log(float(inputs['Tout']))],dtype = float)
# Initial perturb values for MCMC run from perturbs file
perturbs = np.array([perturbfile['inc'],perturbfile['Tin'],perturbfile['Tout']])

# Observed data (nu observed, excess flux, flux errors, weights)
obs = (c/wv_obs*1e8, f_exc, f_err, weights/np.max(weights))
ndim = np.int64(len(initial))
p0 = [np.array(initial) + perturbs*np.random.randn(ndim) for i in range(nwalkers)]

if args.plot: #Plotting corner and excess SED
    emceechain = np.load(chainfile)
    emceeprob = np.load(probfile)
    emceeaccp = np.load(accpfile)
    # Flattened chain 
    samples=emceechain[:,nburn:,:].reshape((-1,ndim))
    samples[:,1],samples[:,2] = np.exp(samples[:,1]), np.exp(samples[:,2])

    ## Corner plot ##
    if args.fit_R: # Plot in terms of Rin/Rout
        labels = [r'i (deg)',r'$\mathrm{R_{in}}$ ($\mathrm{R_{\odot}}$)',r'$\mathrm{R_{out}}$ ($\mathrm{R_{\odot}}$)']
        if np.mean(samples[:,1])<Tsubl: # Checking if chain file has model in terms of temperature
            samples[:,1:] = Tring_to_R(samples[:,1:],WD_T,(WD_R*u.cm).to(u.Rsun).value)
    else:
        labels = [r'i (deg)',r'T$_{Rin}$ (K)',r'T$_{Rout}$ (K)']
    fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=False,
                quantiles=[0.32,0.5,0.68],range = [(0,90),1,1],smooth=0.5,bins = 40)
    plt.savefig(target+'_corner.png')
    plt.close(fig)
    if args.fit_R: # Converting back to Tin/Tout from radii
       samples[:,1:] = T_Ring((samples[:,1:]*u.Rsun).to(u.cm))
    
    ## SED plot ##
    samples[:,1],samples[:,2] = np.log(samples[:,1]), np.log(samples[:,2])
    # Wavelength grid to for plotting model (microns!!)
    wvplot = np.linspace(wv_obs[0]*0.5/1e4,wv_obs[-1]*1.15/1e4,1000)
    # Finding the highest probability set of model params
    best_theta = samples[np.argmax(samples[np.argmax(emceeprob[:,nburn:])])]
    # Sampling MCMC chain to estimate median/std of model
    med_model, spread = sample_walkers(100,samples,c/wvplot*1e4)
    # Calculating highest probability model along wvplot
    bestmodel = model(best_theta,c/wvplot*1e4)
    # Plotting it all
    plt.figure(figsize = (12,6))
    plt.errorbar(wv_obs/1e4,f_exc,f_err,fmt='.',label='IR-excess photometry',mfc='none',markersize=15,capsize=0,elinewidth=2,markeredgewidth=2)
    plt.plot(wvplot,bestmodel,label = 'Jura Model')
    plt.fill_between(wvplot,bestmodel-spread, bestmodel+spread,label=r'1-$\mathrm{\sigma}$ error',color='grey',alpha=0.3)
    plt.xlabel(r'Wavelength ($\mathrm{\mu}$m)',fontsize=20)
    plt.ylabel(r'F$_\mathrm{\nu}$ ($\mu$Jy)',fontsize=20)
    plt.legend(fontsize=12,loc='lower right',frameon=True)
    plt.title(target,fontsize = 20)
    plt.savefig(target+'_SED.pdf')
    plt.close()
    
    # Saving highest likelihood model and std to txt file (columns: [wv,model flux,model error])
    savearray = np.zeros((len(wvplot),3))
    savearray[:,0],savearray[:,1],savearray[:,2] = wvplot, bestmodel, spread
    np.savetxt(target+'-bestmodel.txt',savearray,'%.5f')
    
elif args.walkerplot:
    emceechain = np.load(chainfile)
    niters = (emceechain.shape)[1]
    emceechain[:,:,1],emceechain[:,:,2] = np.exp(emceechain[:,:,1]),np.exp(emceechain[:,:,2])
    
    if args.fit_R: # Plot in terms of R instead of T
        mosaic = [[r'i (deg)'],
                  [r'$\mathrm{R_{in}}$ ($\mathrm{R_{\odot}}$)'],
                  [r'$\mathrm{R_{out}}$ ($\mathrm{R_{\odot}}$)']]
        if np.mean(emceechain[:,:,1])<Tsubl: # Checking if chain file has model in terms of temperature
            emceechain[:,:,1:] = Tring_to_R(emceechain[:,:,1:],WD_T,(WD_R*u.cm).to(u.Rsun).value)
    else:
        mosaic = [[r'i (deg)'],
                  [r'T$_{Rin}$ (K)'],
                  [r'T$_{Rout}$ (K)']]
    fig,axes = plt.subplot_mosaic(mosaic,figsize=(18,15))
    xaxis = np.arange(0,niters,1)    
    for i in range(len(mosaic)):
        ax = mosaic[i][0]
        for j,walker in enumerate(emceechain[:,:,i]):
            axes[ax].plot(xaxis,walker,'k',alpha = niters**(-0.34))
        axes[ax].set_ylabel(ax,fontsize=12)
    axes[ax].set_xlabel('steps',fontsize=12)
    plt.savefig(target+'-walkerplot.png')
    plt.close()

else:
    oldrun=False
    if not args.newrun:
        try: # Try to load an old run
            oldchain = np.load(chainfile)
            oldprob = np.load(probfile)
            oldaccp = np.load(accpfile)
            oldsteps = (oldchain.shape)[1] # Number of steps in old run
            p0 = oldchain[:,-1,:] # Making the starting position the last step of the existing run
            oldrun = True
        except IOError: # If no old run, start a new one
            print('Starting new run')
    # Running MCMC:
    sampler, pos, prob, state = main(p0,nwalkers, niters, ndim, lnprob, obs)
    
    emceechain = sampler.chain
    emceeprob  = sampler.lnprobability
    emceeaccp  = sampler.acceptance_fraction
    
    if args.fit_R: # Save model params in terms of Rin/Rout
        emceechain[:,:,1:] = Tring_to_R(emceechain[:,:,1:],WD_T,(WD_R*u.cm).to(u.Rsun).value)
        
    # Saving emceechain, emceeprob, emceeaccp to npy files
    if oldrun: # If we continued an older run
        chainout = np.append(oldchain,emceechain,axis=1) # Combining new and existing chain
        probout = np.append(oldprob,emceeprob,axis=1)
        oldaccp*=oldsteps # Converting acceptance fraction to nsteps accepted
        emceeaccp*=niters # Doing same as above
        accpout = (oldaccp+emceeaccp)/(oldsteps+niters) # Converting back to acceptance fraction
        np.save(chainfile,chainout)
        np.save(probfile,probout)
        np.save(accpfile,accpout)
    else: # If this was a new run (or overwriting an old run)
        np.save(chainfile,emceechain)
        np.save(probfile,emceeprob)
        np.save(accpfile,emceeaccp)





