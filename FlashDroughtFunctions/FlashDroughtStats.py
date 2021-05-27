#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 00:08:54 2020

@author: stuartedris
"""

#%%
# cell 1
#####################################
### Import some libraries ###########
#####################################


import os, sys, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colorbar as mcolorbar
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
from scipy import stats
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from matplotlib.ticker import MultipleLocator

#%%
# cell 2
######################################
### Beginning of Functions ###########
######################################

#%%
# cell 3

##########################
### CorrCoeff Function ###
##########################

# Calculate the correlation coefficient and 95% significance level

def CorrCoef(x, y, N = 5000):
    '''
    
    '''
    print('Calculating r')
    # Get the data sizes. Note that the interpolated NARR should have the same sizes as the GFS
    J, I, T = x.shape
    
        
    # Reshape the data into space x time arrays.
    x2d = x.reshape(J*I, T, order = 'F')
    y2d = y.reshape(J*I, T, order = 'F')
    
    xAnom = np.ones((I*J, T)) * np.nan
    yAnom = np.ones((I*J, T)) * np.nan
    
    for ij in range(I*J):
        xAnom[ij,:] = x2d[ij,:] - np.nanmean(x2d[ij,:], axis = -1)
        yAnom[ij,:] = y2d[ij,:] - np.nanmean(y2d[ij,:], axis = -1)
    
    r_space = np.ones((I*J)) * np.nan
    
    # Calculate the spatial correlation coefficient
    r_space = np.nanmean(xAnom[:,:] * yAnom[:,:], axis = -1)/(np.nanstd(x2d[:,:], axis = -1) * np.nanstd(y2d[:,:], axis = -1))
    # for ij in range(I*J):
    #     r_space[:] = np.nanmean(xAnom[ij,:] * yAnom[ij,:])/(np.nanstd(x2d[ij,:], axis = -1) * np.nanstd(y2d[ij,:], axis = -1))
    
    # Calculate the temporal correlation coefficient
    # r_time = np.dot(GFS_2d.T, NARR_2d)/(I*J * np.nanstd(GFS_2d, axis = 0) * np.nanstd(NARR_2d, axis = 0))
    
    # Reorder the correlation coefficients to lon x lat (space) and time (temporal)
    #   Note also that the diagonals are the correlation between grid 1 and grid 1, grid 2 and grid 2,... 
    #   which are the desired correlations values.
    r_space = r_space.reshape(J, I, order = 'F')
    
    # r_time  = np.diag(r_time)
    
    # Perform Monte-Carlo testing (for N iterations) for statistical significance.
    print('Calculating significance of r')
    
    # Initialize the random index
    space_ind = np.random.randint(0, T, (N, T))
    
    # time_ind = np.random.randint(0, T, (N, T))
    
    r_spaceCorr = np.ones((I*J, N)) * np.nan
    # r_timeCorr = np.ones((T, N)) * np.nan
    
    # Calcualte the spatial correlation N times
    for n, t in enumerate(space_ind):
        print('Currently %4.2f %% done.' %(n/N*100))
        r_spaceCorr[:,n] = np.nanmean(xAnom[:,:] * yAnom[:,t], axis = -1)/(np.nanstd(x2d[:,:], axis = -1) * np.nanstd(y2d[:,:], axis = -1))
        # for ij in range(I*J):
        #     r_spaceCorr[ij,n] = np.nanmean(xAnom[ij,:] * yAnom[ij,t])/(np.nanstd(x2d[ij,:], axis = -1) * np.nanstd(y2d[ij,:], axis = -1))
        
    # Calculate the temporal correlation N times
    # for n, t in enumerate(time_ind):
    #     r_timeCorr[:,n] = np.dot(GFS_2d[:,t].T, NARR_2d)/(I*J * np.nanstd(GFS_2d, axis = 0) * np.nanstd(NARR_2d, axis = 0))
    
    
    # Collect the p-value from the the random samples
    space_op = r_space.reshape(I*J, order = 'F')
    # time_op = r_time

    pval_space = np.array([stats.percentileofscore(r_spaceCorr[ij,:], space_op[ij])/100 for ij in range(space_op.size)])
    # pval_time = np.array([stats.percentileofscore(r_timeCorr[t,:], time_op[t])/100 for t in range(time_op.size)])
    
    pval_space = pval_space.reshape(J, I, order = 'F')

    
    print('Done \n')
    return r_space, pval_space

#%%
# cell 4

####################################
### CompositeDifference Function ###
####################################

# Calcualte the composite mean difference and 95% significance levels.

def CompositeDifference(x, y, N = 5000):
    '''
    '''
    print('Calculating the composite mean difference')
    # Get the data sizes. Note that the interpolated NARR should have the same sizes as the GFS
    J, I, T = x.shape
    
    # Reshape the data into space x time arrays. For simplicity, focus on the mean to collapse the time dimension
    xComp = np.nanmean(x.reshape(I*J, T, order = 'F'), axis = -1)
    yComp = np.nanmean(y.reshape(I*J, T, order = 'F'), axis = -1)
    
    CompDiff = yComp - xComp
    
    # Next perform Monty - Carlo testing (N iterations) to determine statistical significance
    print('Calculating the significance of the composite mean difference')
    
    # Initialize the index
    ind = np.random.randint(0, I*J, (N, I*J))
    
    CompDiffmc = np.ones((I*J, N)) * np.nan
    
    for n, ij in enumerate(ind):
        CompDiffmc[:,n] = yComp - xComp[ij]
        
    pval = np.array([stats.percentileofscore(CompDiffmc[ij,:], CompDiff[ij])/100 for ij in range(CompDiff.size)])
    
    # Reorder the desired variables into lon x lat
    
    CompDiff = CompDiff.reshape(J, I, order = 'F')
    pval = pval.reshape(J, I, order = 'F')
    
    print('Done \n')
    return CompDiff, pval

#%%
# cell 5

############################
### PlotStatMap Function ###
############################

# Map a single statistic for purposes of testing, overall climatology, etc.
    
def PlotStatMap(x, lon, lat, title = 'Title', y = 0.68, pval = None, alpha = 0.05,
            cmin = -1.0, cmax = 1.0, cint = 0.1, RoundDigit = 1, savename = 'tmp.png',
            OutPath = './Figures/'):
    '''
    
    '''
    
    # Lonitude and latitude tick information
    lat_int = 10
    lon_int = 15

    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)

    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()

    # Colorbar information
    clevs = np.arange(cmin, np.round(cmax+cint, RoundDigit), cint) # The np.round removes floating point error in the adition (which is then ceiled in np.arange)
    nlevs = len(clevs) - 1
    cmap = plt.get_cmap(name = 'RdBu_r', lut = nlevs)
    
    # Get the normalized color values
    norm = mcolors.Normalize(vmin = cmin, vmax = cmax)
    # # Generate the colors from the orginal color map
    colors = cmap(np.linspace(0, 1, cmap.N))
    colors[int(nlevs/2-1):int(nlevs/2+1),:] = np.array([1., 1., 1., 1.]) # Change the value of 0 to white

    # Create a new colorbar cut from the original colors with the white inserted in the middle
    cmap = mcolors.LinearSegmentedColormap.from_list('cut_RdBu_r', colors)

    # Projection informatino
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()
    
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    ShapeName = 'admin_0_countries'
    CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)
    
    CountriesReader = shpreader.Reader(CountriesSHP)
    
    USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
    NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']

    # Create the figure
    fig = plt.figure(figsize = [16, 18], frameon = True)
    fig.suptitle(title, y = y, size = 20)

    # Set the first part of the figure
    ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

    # ax.coastlines()
    # ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'none', zorder = 2)
    ax.add_feature(cfeature.STATES)
    ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 2)
    ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'none', zorder = 2)
    
    ax.set_xticks(LonLabel, crs = fig_proj)
    ax.set_yticks(LatLabel, crs = fig_proj)
    
    ax.set_yticklabels(LatLabel, fontsize = 18)
    ax.set_xticklabels(LonLabel, fontsize = 18)
    
    ax.xaxis.set_major_formatter(LonFormatter)
    ax.yaxis.set_major_formatter(LatFormatter)
    
    cs = ax.pcolormesh(lon, lat, x[:,:], vmin = cmin, vmax = cmax,
                     cmap = cmap, transform = data_proj, zorder = 1)
    # cmap._lut[nlevs/2-1:nlevs/2+1] = [1., 1., 1., 1.]
    if pval is None:
        pass
    else:
        stipple = (pval < alpha/2) | (pval > (1-alpha/2))
        # stipple = pval < alpha
        ax.plot(lon[stipple][::3], lat[stipple][::3], 'o', color = 'Gold', markersize = 1.5, zorder = 1)

    ax.set_extent([-129, -65, 25-1.5, 50-1.5])
    cbax = fig.add_axes([0.12, 0.29, 0.78, 0.015])
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')
    
    for i in cbar.ax.get_xticklabels():
        i.set_size(18)
    # cbar.ax.set_xticks(clevs[::2])
    # cbar.ax.set_xticklabels(np.round(clevs[::2], RoundDigit), fontsize = 18)
    
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)