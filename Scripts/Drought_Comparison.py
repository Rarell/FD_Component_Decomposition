#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:03:59 2020

@author: stuartedris

A script designed to compare the rasterized USDM data with the drought precentiles already calculated.
This is used to help evaluate how well SESR represents drought compared to the USDm.
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

# Add a new path to look for additional, custom libraries and functions
sys.path.append('./FlashDroughtFunctions/')
from FlashDroughtFunctions.FlashDroughtMapping import *
from FlashDroughtFunctions.FlashDroughtCaseStudies import *
from FlashDroughtFunctions.FlashDroughtStats import *

#%%
# cell 2

# Create a function to import the nc files
def LoadNC(SName, filename, path = './Data/pentad_NARR_grid/'):
    '''
    This function loads .nc files. This function takes in the
    name of the data, and short name of the variable to load the .nc file. 
    
    Inputs:
    - SName: The short name of the variable being loaded. I.e., the name used
             to call the variable in the .nc file.
    - filename: The name of the .nc file.
    - path: The path from the current directory to the directory the .nc file is in.
    
    Outputs:
    - X: A directory containing all the data loaded from the .nc file. The 
         entry 'lat' contains latitude (space dimensions), 'lon' contains longitude
         (space dimensions), 'date' contains the dates in a datetime variable
         (time dimension), 'month' 'day' are the numerical month
         and day value for the given time (time dimension), 'ymd' contains full
         datetime values, and 'SName' contains the variable (space and time demsions).
    '''
    
    # Initialize the directory to contain the data
    X = {}
    DateFormat = '%Y-%m-%d %H:%M:%S'
    
    with Dataset(path + filename, 'r') as nc:
        # Load the grid
        lat = nc.variables['lat'][:,:]
        lon = nc.variables['lon'][:,:]
#        lat = nc.variables['lat'][:]
#        lon = nc.variables['lon'][:]
#        
#        lon, lat = np.meshgrid(lon, lat)
#        
        X['lat'] = lat
        X['lon'] = lon
        
        # Collect the time information
        time = nc.variables['date'][:]
        dates = np.asarray([datetime.strptime(time[d], DateFormat) for d in range(len(time))])
        
        X['date'] = dates
        X['year']  = np.asarray([d.year for d in dates])
        X['month'] = np.asarray([d.month for d in dates])
        X['day']   = np.asarray([d.day for d in dates])
        X['ymd']   = np.asarray([datetime(d.year, d.month, d.day) for d in dates])
        
        # Collect the data itself
        X[str(SName)] = nc.variables[str(SName)][:,:,:]
        
    return X

#%%
# cell 3

# Create a function to write nc files
def WriteNC(var, lat, lon, dates, filename = 'tmp.nc', VarName = 'tmp', VarSName = 'tmp'):
    '''
    This function is deisgned write data, and additional information such as
    latitude and longitude and timestamps to a .nc file.
    
    Inputs:
    - var: The variable being written (lat x lon x time format).
    - lat: The latitude data with the same spatial grid as var.
    - lon: The longitude data with the same spatial grid as var.
    - dates: The timestamp for each pentad in var in a %Y-%m-%d format, same time grid as var.
    - filename: The filename of the .nc file being written.
    - VarName: The full name of the variable being written (for the nc description).
    - VarSName: The short name of the variable being written. I.e., the name used
                to call the variable in the .nc file.
                
    Outputs:
    - None. Data is written to a .nc file.
    '''
    
    # Define the path
    path = './Data/MISC_Data/'
    
     # Determine the spatial and temporal lengths
    I, J, T = var.shape
    T = len(dates)
    
    with Dataset(path + filename, 'w', format = 'NETCDF4') as nc:
        # Write a description for the .nc file
        nc.description = 'This file contains SESR percentile data from 2010 - 2019, ' +\
                         'on the USDM time scale. ' +\
                         'Variable: ' + str(VarSName) + ' (unitless). This is the ' +\
                         'main variable for this file. It is in the format ' +\
                         'lat x lon x time.\n' +\
                         'lat: The latitude (matrix form).\n' +\
                         'lon: The longitude (matrix form).\n' +\
                         'date: List of dates starting ' +\
                         '01-05-2010 to 12-31-2019 (%Y-%m-%d format). '

        
        # Create the spatial and temporal dimensions
        nc.createDimension('lat', size = I)
        nc.createDimension('lon', size = J)
        nc.createDimension('time', size = T)
        
        # Create the lat and lon variables
        nc.createVariable('lat', lat.dtype, ('lat', 'lon'))
        nc.createVariable('lon', lon.dtype, ('lat', 'lon'))
        
        nc.variables['lat'][:,:] = lat[:,:]
        nc.variables['lon'][:,:] = lon[:,:]
        
        # Create the date variable
        nc.createVariable('date', str, ('time', ))
        for n in range(len(dates)):
            nc.variables['date'][n] = np.str(dates[n])
            
        # Create the main variable
        nc.createVariable(VarSName, var.dtype, ('lat', 'lon', 'time'))
        nc.variables[str(VarSName)][:,:,:] = var[:,:,:]
#%%
# cell 4
# Load the data

# Define some names
USDMpath = './Data/Misc_Data/'
USDMfn = 'USDM_grid_all_years.nc'
USDMname = 'USDM'

# DPfn = 'DroughtPercentile.nc'
# DPname = 'DP'

# Try with daily data instead of pentad
DPpath = './Data/MISC_data/'
DPfn = 'sesr_percentiles_all_years_daily_conus.nc'
DPname = 'SP'

# Load the data
USDM = LoadNC(USDMname, USDMfn, path = USDMpath)
DP = LoadNC(DPname, DPfn, path = DPpath)

#%%
# cell 5
# Convert the DP to the USDM time scale

# Note that the timestamp on the USDM is the end of the time period it is for.
# The pentads are at the beginning of the time period they cover.

# Initialize some values
I, J, T = USDM['USDM'].shape
DP_usdm = np.zeros((I, J, T)) * np.nan

for n, date in enumerate(USDM['ymd']):
    # Leap days will have 8 days in that week, since the day is ignored in the dataset.
    ind = np.where( (DP['ymd'] >= (date - timedelta(days = 6))) & (DP['ymd'] < date) )[0]
    
    # Calculate the drought percentiles on the USDM time scale by taking an average.
    DP_usdm[:,:,n] = np.nanmean(DP['SP'][:,:,ind], axis = -1)

# Note this loop converts DP on the pentad time scale to the USDM time scale
# for n, date in enumerate(USDM['ymd']):
#     # Leap days will have 8 days in that week, since the day is ignored in the dataset.
#     if (np.mod(date.year, 4) == 0) & (date.month == 3) & (date.day < 6):
#         TotalDays = 8
#     else:
#         TotalDays = 7
    
#     ind = np.where( (DP['ymd'] >= (date - timedelta(days = TotalDays+3))) & (DP['ymd'] < date) )[0]
#     # The timedelta is total days + 4 because USDM date is the end of the week (-TotalDays to encompass full week) 
#     # and because a pentad may be be included in the week (e.g., last day of pentad may be in the week range). 
#     # Maximum up to 4 additional days need to be searched (3 as the subtraction excludes date itself.
#     weights = np.zeros((ind.size)) * np.nan
#     for m, i in enumerate(ind):
#         if i == ind[0]:
#             # One day is added here because the subtraction excludes the last day (date itself) from the subtraction.
#             weights[m] = (5 - ((date - timedelta(days = TotalDays-1)) - DP['ymd'][i]).days)/TotalDays
#         elif i == ind[-1]:
#             # One day is added here because the subtraction excludes the last day (date itself) from the subtraction.
#             weights[m] = ((date + timedelta(days = 1)) - DP['ymd'][i]).days/TotalDays
#         else:
#             weights[m] = 5/TotalDays
    
#     # Calculate the drought percentiles on the USDM time scale by taking a weighted average.
#     # However many days out of 7 (out of 8 on leap days) the pentad has in the week of the USDM date composes the weight for that pentad.
#     DP_usdm[:,:,n] = np.sum(DP['DP'][:,:,ind]*weights, axis = -1)/np.nansum(weights)
#     print(n, np.nansum(weights))
    
    
#%%
# cell 6
# Write the precentile data to a file for future use/consideration.
WriteNC(DP_usdm, DP['lat'], DP['lon'], USDM['ymd'], filename = 'SESRPercentiles_USDM_timescale.nc', VarName = 'SESR Percentiles', VarSName = 'DP')

#%%
# cell 7
# Calculate the categorized drought

# Determine the offset between the MatLab and Python methods of calculating percentiles
#   (for comparison with the results of Christian et al. 2019)
#Offset = 0
Offset = 100/(2*len(np.unique(USDM['year'])))

# Determine the size of the data nd initialize some value
I, J, T = DP_usdm.shape

DC = np.zeros((I, J, T)) * np.nan

DP_usdm = DP_usdm.reshape(I*J, T, order = 'F')
DC = DC.reshape(I*J, T, order = 'F')

# Determine the drought intensity based on the USDM table for D1 - D4 based
#  on percentiles using SESR percentiles
for ij in range(I*J):
    print('Currently %4.2f %% done.' %(ij/(I*J) * 100))
    
    for t, date in enumerate(USDM['ymd']):
        
        ### Determine drought severity ###
        if (DP_usdm[ij,t]-Offset <= 20) & (DP_usdm[ij,t]-Offset > 10):
            DC[ij,t] = 1
        elif (DP_usdm[ij,t]-Offset <= 10) & (DP_usdm[ij,t]-Offset > 5):
            DC[ij,t] = 2
        elif (DP_usdm[ij,t]-Offset <= 5) & (DP_usdm[ij,t]-Offset > 2):
            DC[ij,t] = 3
        elif (DP_usdm[ij,t]-Offset <= 2):
            DC[ij,t] = 4
        else:
            DC[ij,t] = 0

# Reshape the data back to 3D data
DP_usdm = DP_usdm.reshape(I, J, T, order = 'F')
DC = DC.reshape(I, J, T, order = 'F')
#%%
# cell 8
# Create some maps to ensure everything went well.

# Determine the date of the drought
PlotDate = datetime(2017, 8, 29)

ind = np.where(PlotDate == USDM['ymd'])[0]

# Create a color map
cmap = mcolors.LinearSegmentedColormap.from_list('DMcmap', ['white', '#FCD37F', '#FFAA00', '#E60000', '#730000'], N = 5)    

# Plot the USDM map
fig = plt.figure(figsize = [16, 18])
ax  = fig.add_subplot(1, 1, 1, projection = ccrs.PlateCarree())

# Create title
ax.set_title('USDM Map for ' + PlotDate.strftime('%Y/%m/%d'), size = 22)

# Add the state outlines and set the extent so that only the U.S. is shown
ax.add_feature(cfeature.STATES, zorder = 6)

# Set the map extent
ax.set_extent([-129, -65, 25-1.5, 50-1.5])

# Plot the data
cs = ax.pcolormesh(USDM['lon'], USDM['lat'], USDM['USDM'][:,:,ind[0]], vmin = 0, vmax = 4, cmap = cmap, transform = ccrs.PlateCarree())

# Set the location and size of the colorbar
cbax = fig.add_axes([0.125, 0.32, 0.775, 0.015])

# Create the colorbar
cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

cbar.set_ticks(np.arange(0+2/5, 4, 4/5))
cbar.ax.set_xticklabels(['None', 'D1', 'D2', 'D3', 'D4'], size = 20)

plt.show(block = False) 


# Plot the Drought Component map
fig = plt.figure(figsize = [16, 18])
ax  = fig.add_subplot(1, 1, 1, projection = ccrs.PlateCarree())

# Create title
ax.set_title('Drought Component Map for ' + PlotDate.strftime('%Y/%m/%d'), size = 22)

# Add the state outlines and set the extent so that only the U.S. is shown
ax.add_feature(cfeature.STATES, zorder = 6)

# Set the map extent
ax.set_extent([-129, -65, 25-1.5, 50-1.5])

# Plot the data
cs = ax.pcolormesh(USDM['lon'], USDM['lat'], DC[:,:,ind[0]], vmin = 0, vmax = 4, cmap = cmap, transform = ccrs.PlateCarree())

# Set the  location and size of the colorbar
cbax = fig.add_axes([0.125, 0.32, 0.775, 0.015])

# Create the colorbar
cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

cbar.set_ticks(np.arange(0+2/5, 4, 4/5))
cbar.ax.set_xticklabels(['None', 'D1', 'D2', 'D3', 'D4'], size = 20)

plt.show(block = False) 

#%%
# cell 9
# Begin statistical comparison, starting with comparison with intensity

# Focus on the growing season
growInd = np.where( (USDM['month'] >= 4) & (USDM['month'] <= 10) )[0]

YearsGrow = USDM['year'][growInd]

USDMgrow = USDM['USDM'][:,:,growInd]
DCgrow   = DC[:,:,growInd]

# Perform the statistical calculations
IntR, IntRPVal = CorrCoef(USDMgrow[:,:,:], DCgrow[:,:,:])

IntCompDiff, IntCompDiffPVal = CompositeDifference(USDMgrow[:,:,:], DCgrow[:,:,:])


# Create a map of the correlation coefficient and composite difference

PlotStatMap(IntR, USDM['lon'], USDM['lat'], title = 'Correlation Coefficient between USDM and SESR Drought Component for 2010 - 2019', pval = IntRPVal, 
            savename = 'DroughtCompCorrCoef.png')

PlotStatMap(IntCompDiff, USDM['lon'], USDM['lat'], title = 'Composite Mean Difference between USDM and SESR Drought Component for 2010 - 2019', pval = IntCompDiffPVal,
            cmin = -0.10, cmax = 0.10, cint = 0.01, RoundDigit = 2, savename = 'DroughtCompCompMeanDiff.png')

#%%
# cell 10
# Statistical calculations for spatial coverage only.
I, J, T = DCgrow.shape

DCcov = np.zeros((I, J, T))
USDMcov = np.zeros((I, J, T))

# To focus on only coverage, turn all drought values (value >= 1) to 1
DCcov[DCgrow >= 1] = 1
USDMcov[USDMgrow >= 1] = 1

# Perform the statistical calculations
CovR, CovRPVal = CorrCoef(USDMcov[:,:,:], DCcov[:,:,:])

CovCompDiff, CovCompDiffPVal = CompositeDifference(USDMcov[:,:,:], DCcov[:,:,:])


# Create a map of the correlation coefficient and composite difference

PlotStatMap(CovR, USDM['lon'], USDM['lat'], title = 'Correlation Coefficient between USDM and SESR Drought Component for 2010 - 2019', pval = CovRPVal, 
            savename = 'DroughtCoverageCorrCoef.png')

PlotStatMap(CovCompDiff, USDM['lon'], USDM['lat'], title = 'Composite Mean Difference between USDM and SESR Drought Component for 2010 - 2019', pval = CovCompDiffPVal,
            cmin = -0.10, cmax = 0.10, cint = 0.01, RoundDigit = 2, savename = 'DroughtCoverageCompMeanDiff.png')

#%%
# cell 11
# Create the contingency table maps

# Initialize some variables
I, J, T = DCcov.shape

YoN = np.zeros((I, J, T)) * np.nan # YoN stands for Yes or No, that is correct identification or false positive/negative.

FPoFN = np.zeros((I, J, T)) * np.nan # NPoFN stands for false negative for false positive.

FP = np.zeros((I, J, T)) * np.nan # NP stands for false positive.

FN = np.zeros((I, J, T)) * np.nan # FN stands for false negative. 

DCcov = DCcov.reshape(I*J, T, order = 'F')
USDMcov = USDMcov.reshape(I*J, T, order = 'F')

YoN = YoN.reshape(I*J, T, order = 'F')
FPoFN = FPoFN.reshape(I*J, T, order = 'F')

FP = FP.reshape(I*J, T, order = 'F')
FN = FN.reshape(I*J, T, order = 'F')

for ij in range(I*J):
    for t in range(T):
        # Find the spots where the DC and USDM agree there is or is not drought.
        if ((DCcov[ij,t] == 1) & (USDMcov[ij,t] == 1)) | ((USDMcov[ij,t] == 0) & (USDMcov[ij,t] == 0)):
            YoN[ij,t] = 1
        # For the remaining points (where they do not agree) set to 0.
        else:
            YoN[ij,t] = 0
            
        # Find specifically false positives (false alarm)
        if (DCcov[ij,t] == 1) & (USDMcov[ij,t] == 0):
            FPoFN[ij,t] = 1
        # Find the false negatives (misses)
        elif (DCcov[ij,t] == 0) & (USDMcov[ij,t] == 1):
            FPoFN[ij,t] = -1
        # Set remaining values to nan
        else:
            FPoFN[ij,t] = 0
            
        # Find specifically false positives (false alarms)
        if (DCcov[ij,t] == 1) & (USDMcov[ij,t] == 0):
            FP[ij,t] = 1
        # Set remaining values to nan
        else:
            FP[ij,t] = 0
            
        # Find the false negatives (misses)
        if (DCcov[ij,t] == 0) & (USDMcov[ij,t] == 1):
            FN[ij,t] = 1
        # Set remaining values to nan
        else:
            FN[ij,t] = 0

# Reshape the data back into 3D data
DCcov = DCcov.reshape(I, J, T, order = 'F')
USDMcov = USDMcov.reshape(I, J, T, order = 'F')

YoN = YoN.reshape(I, J, T, order = 'F')
FPoFN = FPoFN.reshape(I, J, T, order = 'F')

FP = FP.reshape(I, J, T, order = 'F')
FN = FN.reshape(I, J, T, order = 'F')

# Reapply the land-sea mask

YoN[maskSub[:,:,0] == 0] = 0
FPoFN[maskSub[:,:,0] == 0] = 0
# YoN[DP['mask'] == 0] = 0
# FPoFN[DP['mask'] == 0] = 0

FP[maskSub[:,:,0] == 0] = 0
FN[maskSub[:,:,0] == 0] = 0
# FP[DP['mask'] == 0] = 0
# FN[DP['mask'] == 0] = 0


#%%
# cell 12
# Create two four panel plot with the derived correlation and composite mean diference data.
# Top left, Cor. Coef./Composite Diff. for drought intensity.
# Top right, statistical significance for Cor. Coef./Composite Diff. for drought intensity.
# Bottom left, Cor. Coef./Composite Diff. for drought coverage.
# Bottom right, statistical significance for Cor. Coef./Composite Diff. for drought coverage.

alpha = 0.05 # Statistical significance level

# Lat/Lon tick information
lat_int = 10
lon_int = 20

LatLabel = np.arange(-90, 90, lat_int)
LonLabel = np.arange(-180, 180, lon_int)

LonFormatter = cticker.LongitudeFormatter()
LatFormatter = cticker.LatitudeFormatter()

# Colorbar information
cmax = 1; cmin = -1; cint = 0.1
clevs = np.arange(cmin, np.round(cmax+cint, 2), cint) # The np.round removes floating point error in the adition (which is then ceiled in np.arange)
nlevs = len(clevs) - 1
cmap = mcolors.LinearSegmentedColormap.from_list("BuGrRd", ["navy", "cornflowerblue", "gainsboro", "darksalmon", "maroon"], nlevs)

# Code from https://stackoverflow.com/questions/19199359/modify-discrete-linearsegmentedcolormap
# Turn the segments near 0 into grey ([0.5, 0.8627451, 0.8627451 ])
colors_i = np.concatenate((np.linspace(0,1,nlevs), (0., 0., 0., 0.)))
colors_rgba = cmap(colors_i) # Collect all the colors in the color map
indices = np.linspace(0., 1., nlevs+1)
# Make the dictionary of color values
cdict = {}
for k, key in enumerate(('red', 'green', 'blue')):
    cdict[key] = [ (indices[i], colors_rgba[i-1,k], colors_rgba[i,k]) for i in range(nlevs+1) ]
    
# Now grey out the middle bands
N_middle_bands = 2 - (nlevs % 2)
middle_band_start = (nlevs - N_middle_bands)//2 # Integer division for indexing and to round the value

for middle_band_ind in range(middle_band_start, middle_band_start + N_middle_bands):
    for key in cdict.keys():
        old = cdict[key][middle_band_ind]
        cdict[key][middle_band_ind] = old[:2] + (0.8627451,)
        old = cdict[key][middle_band_ind+1]
        cdict[key][middle_band_ind+1] = old[:1] + (0.8627451,) + old[2:]
        
cmap = mcolors.LinearSegmentedColormap('BuGrRd', cdict, nlevs)


cmin_stats = 0; cmax_stats = 1; cint_stats = 1
clevs_stats = np.arange(cmin_stats, cmax_stats + cint_stats, cint_stats)
nlevs_stats = len(clevs_stats)
cmap_stats = mcolors.LinearSegmentedColormap.from_list("", ["white", "red"], 2)



# Additional shapefiles for removing non-US countries
ShapeName = 'admin_0_countries'
CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

CountriesReader = shpreader.Reader(CountriesSHP)

USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']



# Create the first figure
fig, axes = plt.subplots(figsize = [12, 20], nrows = 2, ncols = 2, 
                             subplot_kw = {'projection': ccrs.PlateCarree()})

# Set some figure parameters
plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.05, hspace = -0.83)

ax11 = axes[0,0]; ax12 = axes[0,1]
ax21 = axes[1,0]; ax22 = axes[1,1]

# Create the main title
fig.suptitle('SESR Drought Component and USDM Correlation Coefficient' + '\n' + 'for April - October 2010 - 2019', fontsize = 22, y = 0.675)

# Top left plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax11.add_feature(cfeature.STATES)
ax11.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax11.add_geometries(USGeom, crs = ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black', zorder = 3)
ax11.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set labels and local title
ax11.set_title('Correlation Coefficient (unitless)', fontsize = 18)
ax11.set_ylabel('Drought Intensity', fontsize = 16)

# Set the tick information
ax11.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax11.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax11.set_yticklabels(LatLabel, fontsize = 16)
ax11.set_xticklabels(LonLabel, fontsize = 16)

ax11.xaxis.set_major_formatter(LonFormatter)
ax11.yaxis.set_major_formatter(LatFormatter)

ax11.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = False, labeltop = True,
                labelleft = True, labelright = False)

# Plot the data
cs = ax11.pcolormesh(USDM['lon'], USDM['lat'], IntR[:,:], vmin = -1, vmax = 1, cmap = cmap, transform = ccrs.PlateCarree(), zorder = 0)

# Set the map extent over the U.S.
ax11.set_extent([-129, -65, 25-1.5, 50-1.5])



# Bottom left plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax21.add_feature(cfeature.STATES)
ax21.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax21.add_geometries(USGeom, crs = ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black', zorder = 3)
ax21.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set labels
ax21.set_ylabel('Drought Coverage', fontsize = 16)

# Set the tick information
ax21.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax21.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax21.set_yticklabels(LatLabel, fontsize = 16)
ax21.set_xticklabels(LonLabel, fontsize = 16)

ax21.xaxis.set_major_formatter(LonFormatter)
ax21.yaxis.set_major_formatter(LatFormatter)

ax21.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = True, labeltop = False,
                labelleft = True, labelright = False)

# Plot the data
cs = ax21.pcolormesh(USDM['lon'], USDM['lat'], CovR[:,:], vmin = -1, vmax = 1, cmap = cmap, transform = ccrs.PlateCarree())

# Set the location and size of the colorbar
cbax = fig.add_axes([0.11, 0.35, 0.38, 0.015])

# Create the colorbar
cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

# Set the colorbar ticks
cticks = np.round(np.arange(-1, 1.5, 0.5), 1)
cbar.set_ticks(np.round(np.arange(-1, 1.5, 0.5), 1))
cbar.ax.set_xticklabels(cticks, fontsize = 18)

# Set the map extent over the U.S.
ax21.set_extent([-129, -65, 25-1.5, 50-1.5])




# Top right plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax12.add_feature(cfeature.STATES)
ax12.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax12.add_geometries(USGeom, crs = ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black', zorder = 3)
ax12.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set local title
ax12.set_title('Statistical Significance', fontsize = 18)

# Set the tick information
ax12.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax12.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax12.set_yticklabels(LatLabel, fontsize = 16)
ax12.set_xticklabels(LonLabel, fontsize = 16)

ax12.xaxis.set_major_formatter(LonFormatter)
ax12.yaxis.set_major_formatter(LatFormatter)

ax12.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = False, labeltop = True,
                labelleft = False, labelright = True)

# Plot the data
stipple = (IntRPVal < alpha/2) | (IntRPVal > (1-alpha/2))
ax12.pcolormesh(USDM['lon'], USDM['lat'], stipple, vmin = 0, vmax = 1, cmap = cmap_stats, transform = ccrs.PlateCarree(), zorder = 1)

# Set the map extent over the U.S.
ax12.set_extent([-129, -65, 25-1.5, 50-1.5])



# Bottom right plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax22.add_feature(cfeature.STATES)
ax22.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax22.add_geometries(USGeom, crs = ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black', zorder = 3)
ax22.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set the tick information
ax22.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax22.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax22.set_yticklabels(LatLabel, fontsize = 16)
ax22.set_xticklabels(LonLabel, fontsize = 16)

ax22.xaxis.set_major_formatter(LonFormatter)
ax22.yaxis.set_major_formatter(LatFormatter)

ax22.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = True, labeltop = False,
                labelleft = False, labelright = True)

# Plot the data
stipple = (CovRPVal < alpha/2) | (CovRPVal > (1-alpha/2))  
ax22.pcolormesh(USDM['lon'], USDM['lat'], stipple, vmin = 0, vmax = 1, cmap = cmap_stats, transform = ccrs.PlateCarree(), zorder = 1)

# Set the map extent over the U.S.
ax22.set_extent([-129, -65, 25-1.5, 50-1.5])

# Save the first plot
plt.savefig('./Figures/USDM_DC_stats_r_all_years_SESR.png', bbox_inches = 'tight')
plt.show(block = False)




# Reapply the land-sea mask
IntCompDiff[maskSub[:,:,0] == 0] = np.nan
CovCompDiff[maskSub[:,:,0] == 0] = np.nan

### Next plot
# Create the second plot
fig, axes = plt.subplots(figsize = [12, 20], nrows = 2, ncols = 2, 
                             subplot_kw = {'projection': ccrs.PlateCarree()})

# Set some figure parameters
plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.05, hspace = -0.83)

ax11 = axes[0,0]; ax12 = axes[0,1]
ax21 = axes[1,0]; ax22 = axes[1,1]

# Create the main title
fig.suptitle('Composite Mean Difference between SESR Drought Component and USDM' + '\n' + 'for April - October 2010 - 2019', fontsize = 22, y = 0.675)

# Top left plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
# ax11.add_feature(cfeature.OCEAN, facecolor = 'white', zorder = 6)
ax11.add_feature(cfeature.STATES)
ax11.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax11.add_geometries(USGeom, crs = ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black', zorder = 3)
ax11.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set labels and local title
ax11.set_title('Composite Mean Difference (unitless)', fontsize = 18)
ax11.set_ylabel('Drought Intensity', fontsize = 16)

# Set the tick information
ax11.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax11.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax11.set_yticklabels(LatLabel, fontsize = 16)
ax11.set_xticklabels(LonLabel, fontsize = 16)

ax11.xaxis.set_major_formatter(LonFormatter)
ax11.yaxis.set_major_formatter(LatFormatter)

ax11.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = False, labeltop = True,
                labelleft = True, labelright = False)

# Plot the data
cs = ax11.pcolormesh(USDM['lon'], USDM['lat'], IntCompDiff[:,:], vmin = -0.5, vmax = 0.5, cmap = cmap, transform = ccrs.PlateCarree())

# Set the map extent over the U.S.
ax11.set_extent([-129, -65, 25-1.5, 50-1.5])



# Bottom left plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
# ax21.add_feature(cfeature.OCEAN, facecolor = 'white', zorder = 6)
ax21.add_feature(cfeature.STATES)
ax21.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax21.add_geometries(USGeom, crs = ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black', zorder = 3)
ax21.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set labels
ax21.set_ylabel('Drought Coverage', fontsize = 16)

# Set the tick information
ax21.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax21.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax21.set_yticklabels(LatLabel, fontsize = 16)
ax21.set_xticklabels(LonLabel, fontsize = 16)

ax21.xaxis.set_major_formatter(LonFormatter)
ax21.yaxis.set_major_formatter(LatFormatter)

ax21.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = True, labeltop = False,
                labelleft = True, labelright = False)

# Plot the data
cs = ax21.pcolormesh(USDM['lon'], USDM['lat'], CovCompDiff[:,:], vmin = -0.5, vmax = 0.5, cmap = cmap, transform = ccrs.PlateCarree())

# Set the map extent over the U.S.
ax21.set_extent([-129, -65, 25-1.5, 50-1.5])

# Set the location and size of the colorbar
cbax = fig.add_axes([0.11, 0.35, 0.38, 0.015])

# Create the colorbar
cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

# Set the colorbar ticks
cticks = np.round(np.arange(-0.5, 0.75, 0.25), 2)
cbar.set_ticks(np.round(np.arange(-0.5, 0.75, 0.25), 2))
cbar.ax.set_xticklabels(cticks, fontsize = 18)



# Top right plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax12.add_feature(cfeature.STATES)
ax12.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax12.add_geometries(USGeom, crs = ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black', zorder = 3)
ax12.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set local title
ax12.set_title('Statistical Significance', fontsize = 18)

# Set the tick information
ax12.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax12.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax12.set_yticklabels(LatLabel, fontsize = 16)
ax12.set_xticklabels(LonLabel, fontsize = 16)

ax12.xaxis.set_major_formatter(LonFormatter)
ax12.yaxis.set_major_formatter(LatFormatter)

ax12.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = False, labeltop = True,
                labelleft = False, labelright = True)

# Plot the data
stipple = (IntCompDiffPVal < alpha/2) | (IntCompDiffPVal > (1-alpha/2))  
ax12.pcolormesh(USDM['lon'], USDM['lat'], stipple, vmin = 0, vmax = 1, cmap = cmap_stats, transform = ccrs.PlateCarree(), zorder = 1)

# Set the map extent over the U.S.
ax12.set_extent([-129, -65, 25-1.5, 50-1.5])



# Bottom right plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax22.add_feature(cfeature.STATES)
ax22.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax22.add_geometries(USGeom, crs = ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black', zorder = 3)
ax22.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set the tick information
ax22.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax22.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax22.set_yticklabels(LatLabel, fontsize = 16)
ax22.set_xticklabels(LonLabel, fontsize = 16)

ax22.xaxis.set_major_formatter(LonFormatter)
ax22.yaxis.set_major_formatter(LatFormatter)

ax22.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = True, labeltop = False,
                labelleft = False, labelright = True)

# Plot the data
stipple = (CovCompDiffPVal < alpha/2) | (CovCompDiffPVal > (1-alpha/2))  
ax22.pcolormesh(USDM['lon'], USDM['lat'], stipple, vmin = 0, vmax = 1, cmap = cmap_stats, transform = ccrs.PlateCarree(), zorder = 1)

# Set the map extent over the U.S.
ax22.set_extent([-129, -65, 25-1.5, 50-1.5])

# Save the second plot
plt.savefig('./Figures/USDM_DC_stats_CD_all_years_SESR.png', bbox_inches = 'tight')
plt.show(block = False)


#%%
# cell 13
# Next make the contingency table/confusion matrix maps

# Lat/Lon tick information
lat_int = 10
lon_int = 20

LatLabel = np.arange(-90, 90, lat_int)
LonLabel = np.arange(-180, 180, lon_int)

LonFormatter = cticker.LongitudeFormatter()
LatFormatter = cticker.LatitudeFormatter()

# Colorbar information
cmax = 1; cmin = -1; cint = 0.1
clevs = np.arange(cmin, np.round(cmax+cint, 2), cint) # The np.round removes floating point error in the adition (which is then ceiled in np.arange)
nlevs = len(clevs) - 1
cmap = plt.get_cmap(name = 'RdBu_r', lut = nlevs)

# Get the normalized color values
norm = mcolors.Normalize(vmin = cmin, vmax = cmax)
# # Generate the colors from the orginal color map
colors = cmap(np.linspace(0, 1, cmap.N))
colors[int(nlevs/2-1):int(nlevs/2+1),:] = np.array([1., 1., 1., 1.]) # Change the value of 0 to white

# Create a new colorbar cut from the original colors with the white inserted in the middle
cmap = mcolors.LinearSegmentedColormap.from_list('cut_RdBu_r', colors)

# Additional shapefiles for removing non-US countries
ShapeName = 'admin_0_countries'
CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

CountriesReader = shpreader.Reader(CountriesSHP)

USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']



# Create the plot
fig, axes = plt.subplots(figsize = [12, 20], nrows = 1, ncols = 2, 
                             subplot_kw = {'projection': ccrs.PlateCarree()})

# Adjust some figure parameters
plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.25, hspace = -0.83)

ax1 = axes[0]; ax2 = axes[1]

# Set the main figure title
fig.suptitle('USDM and SESR Drought Component Truth Table for 2010 - 2019', fontsize = 22, y = 0.605)


# Left plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax1.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax1.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax1.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set a local title
ax1.set_title('Correct Identification', fontsize = 18)

# Set the tick information
ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax1.set_yticklabels(LatLabel, fontsize = 16)
ax1.set_xticklabels(LonLabel, fontsize = 16)

ax1.xaxis.set_major_formatter(LonFormatter)
ax1.yaxis.set_major_formatter(LatFormatter)

ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = False, labeltop = True,
                labelleft = True, labelright = False)

# Plot the data
cs = ax1.pcolormesh(USDM['lon'], USDM['lat'], np.nanmean(YoN[:,:,:], axis = -1), vmin = -1, vmax = 1, cmap = cmap, transform = ccrs.PlateCarree())

# Set the map extent over the U.S.
ax1.set_extent([-129, -65, 25-1.5, 50-1.5])

# Set the colorbar location and size
cbax = fig.add_axes([0.10, 0.42, 0.36, 0.015])

# Create the colorbar
cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

# Set the colorbar ticks and labels
cbar.set_ticks(np.round(np.arange(-1, 1+1, 2)))
cbar.ax.set_xticklabels(['Incorrect', 'Correct'], fontsize = 18)

# Top left plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax2.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax2.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax2.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set a local title
ax2.set_title('Type of Error', fontsize = 18)

# Set the tick information
ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax2.set_yticklabels(LatLabel, fontsize = 16)
ax2.set_xticklabels(LonLabel, fontsize = 16)

ax2.xaxis.set_major_formatter(LonFormatter)
ax2.yaxis.set_major_formatter(LatFormatter)

ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = False, labeltop = True,
                labelleft = True, labelright = False)

# Plot the data
cs = ax2.pcolormesh(USDM['lon'], USDM['lat'], np.nanmean(FPoFN[:,:,:], axis = -1), vmin = -1, vmax = 1, cmap = cmap, transform = ccrs.PlateCarree())

# Set the map extent over the U.S.
ax2.set_extent([-129, -65, 25-1.5, 50-1.5])

# Set the colorbar location and size
cbax = fig.add_axes([0.55, 0.42, 0.36, 0.015])

# Create the colorbar
cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

# Set the colorbar ticks and labels
cbar.set_ticks(np.round(np.arange(-1, 1+1, 2)))
cbar.ax.set_xticklabels(['Miss', 'False Alarm'], fontsize = 18)

# Save the figure
plt.savefig('./Figures/USDM_DC_truth_table_all_years.png', bbox_inches = 'tight')
plt.show(block = False)


#%%
# cell 14
# Next make a change contingency matrix plot based on suggestions

# Lat/Lon tick information
lat_int = 10
lon_int = 20

LatLabel = np.arange(-90, 90, lat_int)
LonLabel = np.arange(-180, 180, lon_int)

LonFormatter = cticker.LongitudeFormatter()
LatFormatter = cticker.LatitudeFormatter()

# Colorbar information
cmax = 100; cmin = 0; cint = 5
clevs = np.arange(cmin, np.round(cmax+cint, 2), cint) # The np.round removes floating point error in the adition (which is then ceiled in np.arange)
nlevs = len(clevs) - 1
cmap = plt.get_cmap(name = 'Reds', lut = nlevs)

# Additional shapefiles for removing non-US countries
ShapeName = 'admin_0_countries'
CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

CountriesReader = shpreader.Reader(CountriesSHP)

USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']



# Create the plot
fig, axes = plt.subplots(figsize = [12, 20], nrows = 1, ncols = 3, 
                             subplot_kw = {'projection': ccrs.PlateCarree()})

# Adjust some figure parameters
plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.10, hspace = -0.73)

ax1 = axes[0]; ax2 = axes[1]; ax3 = axes[2]

# Set the main figure title
fig.suptitle('SESR Drought Component and USDM Confusion Matrix ' + '\n' + 'for April - October 2010 - 2019', fontsize = 22, y = 0.580)



# Left plot; Correct identification frequency

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax1.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax1.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax1.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set a local title
ax1.set_title('Frequency of Agreement', fontsize = 16)

# Set the tick information
ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax1.set_yticklabels(LatLabel, fontsize = 16)
ax1.set_xticklabels(LonLabel, fontsize = 16)

ax1.xaxis.set_major_formatter(LonFormatter)
ax1.yaxis.set_major_formatter(LatFormatter)

ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = True, labeltop = False,
                labelleft = True, labelright = False)

# Plot the data
cs = ax1.pcolormesh(USDM['lon'], USDM['lat'], np.nanmean(YoN[:,:,:], axis = -1)*100, vmin = 0, vmax = 100, cmap = cmap, transform = ccrs.PlateCarree())

# Set the map extent over the U.S.
ax1.set_extent([-129, -65, 25-1.5, 50-1.5])




# Center plot: False positive frequency

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax2.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax2.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax2.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set a local title
ax2.set_title('Frequency of False Positives', fontsize = 16)

# Set the tick information
ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax2.set_yticklabels(LatLabel, fontsize = 16)
ax2.set_xticklabels(LonLabel, fontsize = 16)

ax2.xaxis.set_major_formatter(LonFormatter)
ax2.yaxis.set_major_formatter(LatFormatter)

ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = True, labeltop = False,
                labelleft = False, labelright = False)

# Plot the data
cs = ax2.pcolormesh(USDM['lon'], USDM['lat'], np.nanmean(FP[:,:,:], axis = -1)*100, vmin = 0, vmax = 100, cmap = cmap, transform = ccrs.PlateCarree())

# Set the map extent over the U.S.
ax2.set_extent([-129, -65, 25-1.5, 50-1.5])





# Right plot: False negative frequency

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax3.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax3.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax3.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set a local title
ax3.set_title('Frequency of False Negatives', fontsize = 16)

# Set the tick information
ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax3.set_yticklabels(LatLabel, fontsize = 16)
ax3.set_xticklabels(LonLabel, fontsize = 16)

ax3.xaxis.set_major_formatter(LonFormatter)
ax3.yaxis.set_major_formatter(LatFormatter)

ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = True, labeltop = False,
                labelleft = False, labelright = True)

# Plot the data
cs = ax3.pcolormesh(USDM['lon'], USDM['lat'], np.nanmean(FN[:,:,:], axis = -1)*100, vmin = 0, vmax = 100, cmap = cmap, transform = ccrs.PlateCarree())

# Set the map extent over the U.S.
ax3.set_extent([-129, -65, 25-1.5, 50-1.5])

# Set the colorbar location and size
cbax = fig.add_axes([0.10, 0.435, 0.80, 0.012])

# Create the colorbar
cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

# Set the colorbar ticks and labels
cbar.set_ticks(np.round(np.arange(0, 100+20, 20)))
cbar.ax.set_xticklabels(np.round(np.arange(0, 100+20, 20)), fontsize = 18)

cbar.ax.set_xlabel('Frequency of Agreement/Error Type (%)', fontsize = 18)

# Save the figure
plt.savefig('./Figures/USDM_DC_truth_table_all_years_version2.png', bbox_inches = 'tight')
plt.show(block = False)

#%%
# Make an empty plot

# Lonitude and latitude tick information
lat_int = 10
lon_int = 15

LatLabel = np.arange(-90, 90, lat_int)
LonLabel = np.arange(-180, 180, lon_int)

LonFormatter = cticker.LongitudeFormatter()
LatFormatter = cticker.LatitudeFormatter()

# Projection informatino
data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Collect shapefile information for the U.S. and other countries
# ShapeName = 'Admin_1_states_provinces_lakes_shp'
ShapeName = 'admin_0_countries'
CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

CountriesReader = shpreader.Reader(CountriesSHP)

USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']

# Create the figure
fig = plt.figure(figsize = [16, 18], frameon = True)

# Set the first part of the figure
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
ax.add_feature(cfeature.STATES)
ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'none', zorder = 2)

# Adjust the ticks
ax.set_xticks(LonLabel, crs = fig_proj)
ax.set_yticks(LatLabel, crs = fig_proj)

ax.set_yticklabels(LatLabel, fontsize = 18)
ax.set_xticklabels(LonLabel, fontsize = 18)

ax.xaxis.set_major_formatter(LonFormatter)
ax.yaxis.set_major_formatter(LatFormatter)

ax.set_extent([-129, -65, 25-1.5, 50-1.5])

# Save the figure.
plt.savefig('./Figures/empty_map.png', bbox_inches = 'tight')
plt.show(block = False)
#%%
# cell 15
# Repeat the above figures for a given year (an addition to case studies).
Year = 2012

# Find the values corresponding to the given year.
YearInd = np.where(Year == YearsGrow)[0]

USDMYear = USDMgrow[:,:,YearInd]
DCYear = DCgrow[:,:,YearInd]
YoNYear = YoN[:,:,YearInd]
FPoFNYear = FPoFN[:,:,YearInd]

# Rerun the statistics calculations for the given year.
I, J, T = DCYear.shape

DCcovYear = np.zeros((I, J, T))
USDMcovYear = np.zeros((I, J, T))

DCcovYear[DCYear > 0] = 1
USDMcovYear[USDMYear > 0] = 1

# Statistics calculations for drought intensity and coverage
IntRYear, IntRPValYear = CorrCoef(USDMYear, DCYear)
IntCompDiffYear, IntCompDiffPValYear = CompositeDifference(USDMYear, DCYear)

# Staistics calculations for drought coverage only
CovRYear, CovRPValYear = CorrCoef(USDMcovYear, DCcovYear)
CovCompDiffYear, CovCompDiffPValYear = CompositeDifference(USDMcovYear, DCcovYear)


# Most to all values that are exactly 0 should be nan only grids.
IntCompDiffYear[IntCompDiffYear == 0] = np.nan
CovCompDiffYear[CovCompDiffYear == 0] = np.nan

IntCompDiffYear[maskSub[:,:,0] == 0] = np.nan
CovCompDiffYear[maskSub[:,:,0] == 0] = np.nan



# Create the plots
alpha = 0.05 # Statistical significance level

# Colorbar information
cmax = 1; cmin = -1; cint = 0.1
clevs = np.arange(cmin, np.round(cmax+cint, 2), cint) # The np.round removes floating point error in the adition (which is then ceiled in np.arange)
nlevs = len(clevs) - 1
cmap = mcolors.LinearSegmentedColormap.from_list("BuGrRd", ["navy", "cornflowerblue", "gainsboro", "darksalmon", "maroon"], nlevs)

# Code from https://stackoverflow.com/questions/19199359/modify-discrete-linearsegmentedcolormap
# Turn the segments near 0 into grey ([0.5, 0.8627451, 0.8627451 ])
colors_i = np.concatenate((np.linspace(0,1,nlevs), (0., 0., 0., 0.)))
colors_rgba = cmap(colors_i) # Collect all the colors in the color map
indices = np.linspace(0., 1., nlevs+1)
# Make the dictionary of color values
cdict = {}
for k, key in enumerate(('red', 'green', 'blue')):
    cdict[key] = [ (indices[i], colors_rgba[i-1,k], colors_rgba[i,k]) for i in range(nlevs+1) ]
    
# Now grey out the middle bands
N_middle_bands = 2 - (nlevs % 2)
middle_band_start = (nlevs - N_middle_bands)//2 # Integer division for indexing and to round the value

for middle_band_ind in range(middle_band_start, middle_band_start + N_middle_bands):
    for key in cdict.keys():
        old = cdict[key][middle_band_ind]
        cdict[key][middle_band_ind] = old[:2] + (0.8627451,)
        old = cdict[key][middle_band_ind+1]
        cdict[key][middle_band_ind+1] = old[:1] + (0.8627451,) + old[2:]
        
cmap = mcolors.LinearSegmentedColormap('BuGrRd', cdict, nlevs)

cmin_stats = 0; cmax_stats = 1; cint_stats = 1
clevs_stats = np.arange(cmin_stats, cmax_stats + cint_stats, cint_stats)
nlevs_stats = len(clevs_stats)
cmap_stats = mcolors.LinearSegmentedColormap.from_list("", ["white", "red"], 2)



# Additional shapefiles for removing non-US countries
ShapeName = 'admin_0_countries'
CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

CountriesReader = shpreader.Reader(CountriesSHP)

USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']

# Create the first plot
fig, axes = plt.subplots(figsize = [12, 20], nrows = 2, ncols = 2, 
                             subplot_kw = {'projection': ccrs.PlateCarree()})

# Adjust some figure parameters
if Year == 2011:
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.10, hspace = -0.78)
    
    y = 0.700
    
    lat_int = 5
    lon_int = 10
    
    extent = [-110, -78, 25, 42]
    
    fig_location = [0.095, 0.33, 0.39, 0.015]
    
elif Year == 2012:
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.05, hspace = -0.61)
    
    y = 0.780
    
    lat_int = 5
    lon_int = 10
    
    extent = [-105, -82, 30, 50]
    
    fig_location = [0.10, 0.25, 0.39, 0.015]
    
else:
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.05, hspace = -0.83)
    
    y = 0.675
    
    lat_int = 10
    lon_int = 20
    
    extent = [-129, -65, 25-1.5, 50-1.5]
    
    fig_location = [0.11, 0.35, 0.38, 0.015]
    
    
# Lat/Lon tick information
LatLabel = np.arange(-90, 90, lat_int)
LonLabel = np.arange(-180, 180, lon_int)

LonFormatter = cticker.LongitudeFormatter()
LatFormatter = cticker.LatitudeFormatter()


ax11 = axes[0,0]; ax12 = axes[0,1]
ax21 = axes[1,0]; ax22 = axes[1,1]

# Set the main figure title
fig.suptitle('SESR Drought Component and USDM Correlation Coefficient' + '\n'+ 'for April - October ' + str(Year), fontsize = 22, y = y)

# Top left plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax11.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax11.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax11.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set labels and local title
ax11.set_title('Correlation Coefficient (unitless)', fontsize = 18)
ax11.set_ylabel('Drought Intensity', fontsize = 16)

# Set the tick information
ax11.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax11.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax11.set_yticklabels(LatLabel, fontsize = 16)
ax11.set_xticklabels(LonLabel, fontsize = 16)

ax11.xaxis.set_major_formatter(LonFormatter)
ax11.yaxis.set_major_formatter(LatFormatter)

ax11.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = False, labeltop = True,
                labelleft = True, labelright = False)

# Plot the data
cs = ax11.pcolormesh(USDM['lon'], USDM['lat'], IntRYear[:,:], vmin = -1, vmax = 1, cmap = cmap, transform = ccrs.PlateCarree(), zorder = 0)

# Set the map extent over the U.S.
ax11.set_extent(extent)



# Bottom left plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax21.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax21.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax21.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set labels
ax21.set_ylabel('Drought Coverage', fontsize = 16)

# Set the tick information
ax21.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax21.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax21.set_yticklabels(LatLabel, fontsize = 16)
ax21.set_xticklabels(LonLabel, fontsize = 16)

ax21.xaxis.set_major_formatter(LonFormatter)
ax21.yaxis.set_major_formatter(LatFormatter)

ax21.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = True, labeltop = False,
                labelleft = True, labelright = False)

# Plot the data
cs = ax21.pcolormesh(USDM['lon'], USDM['lat'], CovRYear[:,:], vmin = -1, vmax = 1, cmap = cmap, transform = ccrs.PlateCarree())

# Set the colorbar location and size
cbax = fig.add_axes(fig_location)

# Create the colorbar
cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

# Set the colorbar ticks and labels
cticks = np.round(np.arange(-1, 1.5, 0.5), 1)
cbar.set_ticks(np.round(np.arange(-1, 1.5, 0.5), 1))
cbar.ax.set_xticklabels(cticks, fontsize = 18)

# Set the map extent over the U.S.
ax21.set_extent(extent)




# Top right plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax12.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax12.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax12.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set local title
ax12.set_title('Statistical Significance', fontsize = 18)

# Set the tick information
ax12.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax12.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax12.set_yticklabels(LatLabel, fontsize = 16)
ax12.set_xticklabels(LonLabel, fontsize = 16)

ax12.xaxis.set_major_formatter(LonFormatter)
ax12.yaxis.set_major_formatter(LatFormatter)

ax12.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = False, labeltop = True,
                labelleft = False, labelright = True)

# Plot the data
stipple = (IntRPValYear < alpha/2) | (IntRPValYear > (1-alpha/2))
ax12.pcolormesh(USDM['lon'], USDM['lat'], stipple, vmin = 0, vmax = 1, cmap = cmap_stats, transform = ccrs.PlateCarree(), zorder = 1)

# Set the map extent over the U.S.
ax12.set_extent(extent)



# Bottom right plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax22.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax22.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax22.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set the tick information
ax22.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax22.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax22.set_yticklabels(LatLabel, fontsize = 16)
ax22.set_xticklabels(LonLabel, fontsize = 16)

ax22.xaxis.set_major_formatter(LonFormatter)
ax22.yaxis.set_major_formatter(LatFormatter)

ax22.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = True, labeltop = False,
                labelleft = False, labelright = True)

# Plot the data
stipple = (CovRPValYear < alpha/2) | (CovRPValYear > (1-alpha/2))  
ax22.pcolormesh(USDM['lon'], USDM['lat'], stipple, vmin = 0, vmax = 1, cmap = cmap_stats, transform = ccrs.PlateCarree(), zorder = 1)

# Set the map extent over the U.S.
ax22.set_extent(extent)

# Save the first figure
plt.savefig('./Figures/USDM_DC_stats_r_' + str(Year) + '.png', bbox_inches = 'tight')
plt.show(block = False)






### Next plot
# Create the second plot
fig, axes = plt.subplots(figsize = [12, 20], nrows = 2, ncols = 2, 
                             subplot_kw = {'projection': ccrs.PlateCarree()})

# Adjust some figure parameters
if Year == 2011:
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.10, hspace = -0.78)
    
    y = 0.700
    
    extent = [-110, -78, 25, 42]
    
    fig_location = [0.095, 0.33, 0.39, 0.015]
    
elif Year == 2012:
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.05, hspace = -0.61)
    
    y = 0.780
    
    extent = [-105, -82, 30, 50]
    
    fig_location = [0.10, 0.25, 0.39, 0.015]
    
else:
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.05, hspace = -0.83)
    
    y = 0.675
    
    extent = [-129, -65, 25-1.5, 50-1.5]
    
    fig_location = [0.11, 0.35, 0.38, 0.015]

ax11 = axes[0,0]; ax12 = axes[0,1]
ax21 = axes[1,0]; ax22 = axes[1,1]

# Set the main figure title
fig.suptitle('Composite Mean Difference between SESR Drought Component and USDM' + '\n' + 'for April - October ' + str(Year), fontsize = 22, y = y)

# Top left plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
#ax11.add_feature(cfeature.OCEAN, facecolor = 'white', zorder = 6)
ax11.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax11.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax11.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set labels and local title
ax11.set_title('Composite Mean Difference (unitless)', fontsize = 18)
ax11.set_ylabel('Drought Intensity', fontsize = 16)

# Set the tick information
ax11.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax11.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax11.set_yticklabels(LatLabel, fontsize = 16)
ax11.set_xticklabels(LonLabel, fontsize = 16)

ax11.xaxis.set_major_formatter(LonFormatter)
ax11.yaxis.set_major_formatter(LatFormatter)

ax11.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = False, labeltop = True,
                labelleft = True, labelright = False)

# Plot the data
cs = ax11.pcolormesh(USDM['lon'], USDM['lat'], IntCompDiffYear[:,:], vmin = -0.5, vmax = 0.5, cmap = cmap, transform = ccrs.PlateCarree())

# Set the map extent over the U.S.
ax11.set_extent(extent)



# Bottom left plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
#ax21.add_feature(cfeature.OCEAN, facecolor = 'white', zorder = 6)
ax21.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax21.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax21.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set labels
ax21.set_ylabel('Drought Coverage', fontsize = 16)

# Set the tick information
ax21.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax21.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax21.set_yticklabels(LatLabel, fontsize = 16)
ax21.set_xticklabels(LonLabel, fontsize = 16)

ax21.xaxis.set_major_formatter(LonFormatter)
ax21.yaxis.set_major_formatter(LatFormatter)

ax21.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = True, labeltop = False,
                labelleft = True, labelright = False)

# Plot the data
cs = ax21.pcolormesh(USDM['lon'], USDM['lat'], CovCompDiffYear[:,:], vmin = -0.5, vmax = 0.5, cmap = cmap, transform = ccrs.PlateCarree())

# Set the map extent over the U.S.
ax21.set_extent(extent)

# Set the colorbar location and size
cbax = fig.add_axes(fig_location)

# Create the colorbar
cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

# Set the colorbar ticks and labels
cticks = np.round(np.arange(-0.5, 0.75, 0.25), 2)
cbar.set_ticks(np.round(np.arange(-0.5, 0.75, 0.25), 2))
cbar.ax.set_xticklabels(cticks, fontsize = 18)



# Top right plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax12.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax12.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax12.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set local title
ax12.set_title('Statistical Significance', fontsize = 18)

# Set the tick information
ax12.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax12.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax12.set_yticklabels(LatLabel, fontsize = 16)
ax12.set_xticklabels(LonLabel, fontsize = 16)

ax12.xaxis.set_major_formatter(LonFormatter)
ax12.yaxis.set_major_formatter(LatFormatter)

ax12.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = False, labeltop = True,
                labelleft = False, labelright = True)

# Plot the data
stipple = (IntCompDiffPValYear < alpha/2) | (IntCompDiffPValYear > (1-alpha/2))  
ax12.pcolormesh(USDM['lon'], USDM['lat'], stipple, vmin = 0, vmax = 1, cmap = cmap_stats, transform = ccrs.PlateCarree(), zorder = 1)

# Set the map extent over the U.S.
ax12.set_extent(extent)



# Bottom right plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax22.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax22.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax22.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set the tick information
ax22.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax22.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax22.set_yticklabels(LatLabel, fontsize = 16)
ax22.set_xticklabels(LonLabel, fontsize = 16)

ax22.xaxis.set_major_formatter(LonFormatter)
ax22.yaxis.set_major_formatter(LatFormatter)

ax22.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = True, labeltop = False,
                labelleft = False, labelright = True)

# Plot the data
stipple = (CovCompDiffPValYear < alpha/2) | (CovCompDiffPValYear > (1-alpha/2))  
ax22.pcolormesh(USDM['lon'], USDM['lat'], stipple, vmin = 0, vmax = 1, cmap = cmap_stats, transform = ccrs.PlateCarree(), zorder = 1)

# Set the map extent over the U.S.
ax22.set_extent(extent)

# Save the second figure
plt.savefig('./Figures/USDM_DC_stats_CD_' + str(Year) + '.png', bbox_inches = 'tight')
plt.show(block = False)





# Next make the truth table maps
# Reuse the colorbar, lat/lon tick, and shapefile information.

# Create the third plot
fig, axes = plt.subplots(figsize = [12, 20], nrows = 1, ncols = 2, 
                             subplot_kw = {'projection': ccrs.PlateCarree()})

# Adjust some figure parameters
plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.25, hspace = -0.83)

ax1 = axes[0]; ax2 = axes[1]

# Set the main figure title
fig.suptitle('USDM and SESR Drought Component Truth Table for ' + str(Year), fontsize = 22, y = 0.605)

# Left plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax1.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax1.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax1.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set local title
ax1.set_title('Correct Identification', fontsize = 18)

# Set the tick information
ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax1.set_yticklabels(LatLabel, fontsize = 16)
ax1.set_xticklabels(LonLabel, fontsize = 16)

ax1.xaxis.set_major_formatter(LonFormatter)
ax1.yaxis.set_major_formatter(LatFormatter)

ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = False, labeltop = True,
                labelleft = True, labelright = False)

# Plot the data
cs = ax1.pcolormesh(USDM['lon'], USDM['lat'], np.nanmean(YoNYear[:,:,:], axis = -1), vmin = -1, vmax = 1, cmap = cmap, transform = ccrs.PlateCarree())

# Set the map extent over the U.S.
ax1.set_extent([-129, -65, 25-1.5, 50-1.5])

# Set the colorbar location and size
cbax = fig.add_axes([0.10, 0.42, 0.36, 0.015])

# Create the colorbar
cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

# Set the colorbar ticks and labels
cbar.set_ticks(np.round(np.arange(-1, 1+1, 2)))
cbar.ax.set_xticklabels(['Incorrect', 'Correct'], fontsize = 18)



# Top left plot

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax2.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax2.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax2.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set local title
ax2.set_title('Type of Error', fontsize = 18)

# Set the tick information
ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax2.set_yticklabels(LatLabel, fontsize = 16)
ax2.set_xticklabels(LonLabel, fontsize = 16)

ax2.xaxis.set_major_formatter(LonFormatter)
ax2.yaxis.set_major_formatter(LatFormatter)

ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = False, labeltop = True,
                labelleft = True, labelright = False)

# Plot the data
cs = ax2.pcolormesh(USDM['lon'], USDM['lat'], np.nanmean(FPoFNYear[:,:,:], axis = -1), vmin = -1, vmax = 1, cmap = cmap, transform = ccrs.PlateCarree())

# Set the map extent over the U.S.
ax2.set_extent([-129, -65, 25-1.5, 50-1.5])

# Set the colorbar location and size
cbax = fig.add_axes([0.55, 0.42, 0.36, 0.015])

# Create the colorbar
cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

# Set the colorbar ticks and labels
cbar.set_ticks(np.round(np.arange(-1, 1+1, 2)))
cbar.ax.set_xticklabels(['Miss', 'False Alarm'], fontsize = 18)

# Save the third figure
plt.savefig('./Figures/USDM_DC_truth_table_' + str(Year) + '.png', bbox_inches = 'tight')
plt.show(block = False)