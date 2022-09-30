#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 16:06:51 2021

@author: stuartedris


This is a test script to create and test a map.
"""


#%%
import os, sys, warnings
import numpy as np
import multiprocessing as mp
import pathos.multiprocessing as pmp
from joblib import parallel_backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colorbar as mcolorbar
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
from scipy import stats
from scipy import interpolate
from scipy import signal
from scipy.special import gamma
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sklearn import tree
from sklearn import neural_network
from sklearn import ensemble
from sklearn import svm
from sklearn import metrics

#%%
def SubsetData(X, Lat, Lon, LatMin, LatMax, LonMin, LonMax):
    '''
    This function is designed to subset data for any gridded dataset, including
    the non-simple grid used in the NARR dataset, where the size of the subsetted
    data is unknown. Note this function only makes square subsets with a maximum 
    and minimum latitude/longitude.
    
    Inputs:
    - X: The variable to be subsetted.
    - Lat: The gridded latitude data corresponding to X.
    - Lon: The gridded Longitude data corresponding to X.
    - LatMax: The maximum latitude of the subsetted data.
    - LatMin: The minimum latitude of the subsetted data.
    - LonMax: The maximum longitude of the subsetted data.
    - LonMin: The minimum longitude of the subsetted data.
    
    Outputs:
    - XSub: The subsetted data.
    - LatSub: Gridded, subsetted latitudes.
    - LonSub: Gridded, subsetted longitudes.
    '''
    
    # Collect the original sizes of the data/lat/lon
    I, J, T = X.shape
    
    # Reshape the data into a 2D array and lat/lon to a 1D array for easier referencing.
    X2D   = X.reshape(I*J, T, order = 'F')
    Lat1D = Lat.reshape(I*J, order = 'F')
    Lon1D = Lon.reshape(I*J, order = 'F')
    
    # Find the indices in which to make the subset.
    LatInd = np.where( (Lat1D >= LatMin) & (Lat1D <= LatMax) )[0]
    LonInd = np.where( (Lon1D >= LonMin) & (Lon1D <= LonMax) )[0]
    
    # Find the points where the lat and lon subset overlap. This comprises the subsetted grid.
    SubInd = np.intersect1d(LatInd, LonInd)
    
    # Next find, the I and J dimensions of subsetted grid.
    Start = 0 # The starting point of the column counting.
    Count = 1 # Row count starts at 1
    Isub  = 0 # Start by assuming subsetted column size is 0.
    
    for n in range(len(SubInd[:-1])): # Exclude the last value to prevent indexing errors.
        IndDiff = SubInd[n+1] - SubInd[n] # Obtain difference between this index and the next.
        if (n+2) == len(SubInd): # At the last value, everything needs to be increased by 2 to account for the missing indice at the end.
            Isub = np.nanmax([Isub, n+2 - Start]) # Note since this is the last indice, and this row is counted, there is no Count += 1.
        elif ( (IndDiff > 1) |              # If the difference is greater than 1, or if
             (np.mod(SubInd[n]+1,I) == 0) ):# SubInd is divisible by I, then a new row 
                                            # is started in the gridded array.
            Isub = np.nanmax([Isub, n+1 - Start]) # Determine the highest column count (may not be the same from row to row)
            Start = n+1 # Start the counting anew.
            Count = Count + 1 # Increment the row count by 1 as the next row is entered.
        else:
            pass
        
    # At the end, Count has the total number of rows in the subset.
    Jsub = Count
    
    # Next, the column size may not be the same from row to row. The rows with
    # with columns less than Isub need to be filled in. 
    # Start by finding how many placeholders are needed.
    PH = Isub * Jsub - len(SubInd) # Total number of needed points - number in the subset
    
    # Initialize the variable that will hold the needed indices.
    PlaceHolder = np.ones((PH)) * np.nan
    
    # Fill the placeholder values with the indices needed to complete a Isub x Jsub matrix
    Start = 0
    m = 0
    
    for n in range(len(SubInd[:-1])):
        # Identify when row changes occur.
        IndDiff = SubInd[n+1] - SubInd[n]
        if (n+2) == len(SubInd): # For the end of last row, an n+2 is needed to account for the missing index (SubInd[:-1] was used)
            ColNum = n+2-Start
            PlaceHolder[m:m+Isub-ColNum] = SubInd[n+1] + np.arange(1, 1+Isub-ColNum)
            # Note this is the last value, so nothing else needs to be incremented up.
        elif ( (IndDiff > 1) | (np.mod(SubInd[n]+1,I) == 0) ):
            # Determine how many columns this row has.
            ColNum = n+1-Start
            
            # Fill the placeholder with the next index(ices) when the row has less than
            # the maximum number of columns (Isub)
            PlaceHolder[m:m+Isub-ColNum] = SubInd[n] + np.arange(1, 1+Isub-ColNum)
            
            # Increment the placeholder index by the number of entries filled.
            m = m + Isub - ColNum
            Start = n+1
            
        
        else:
            pass
    
    # Next, convert the placeholders to integer indices.
    PlaceHolderInt = PlaceHolder.astype(int)
    
    # Add and sort the placeholders to the indices.
    SubIndTotal = np.sort(np.concatenate((SubInd, PlaceHolderInt), axis = 0))
    
    # The placeholder indices are technically outside of the desired subset. So
    # turn those values to NaN so they do not effect calculations.
    # (In theory, X2D is not the same variable as X, so the original dataset 
    #  should remain untouched.)
    X2D[PlaceHolderInt,:] = np.nan
    
    # Collect the subset of the data, lat, and lon
    XSub = X2D[SubIndTotal,:]
    LatSub = Lat1D[SubIndTotal]
    LonSub = Lon1D[SubIndTotal]
    
    # Reorder the data back into a 3D array, and lat and lon into gridded 2D arrays
    XSub = XSub.reshape(Isub, Jsub, T, order = 'F')
    LatSub = LatSub.reshape(Isub, Jsub, order = 'F')
    LonSub = LonSub.reshape(Isub, Jsub, order = 'F')
    
    # Return the the subsetted data
    return XSub, LatSub, LonSub


def LoadNC(SName, filename, path = './Data/Indices/'):
    '''
    This function loads .nc files, specifically raw NARR files. This function takes in the
    name of the data, and short name of the variable to load the .nc file. 
    
    Inputs:
    - SName: The short name of the variable being loaded. I.e., the name used
             to call the variable in the .nc file.
    - filename: The name of the .nc file.
    - sm: Boolean determining if soil moisture is being loaded (an extra variable and dimension, level,
          needs to be loaded).
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

        X['lat'] = lat
        X['lon'] = lon
        
        # Collect the time information
        time = nc.variables['date'][:]
        X['time'] = time
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
# Load and subset some data
path = './Data/Indices/'

sesr  = LoadNC('sesr', 'sesr.NARR.CONUS.pentad.nc', path = path)

sesrNW, latNW, lonNW = SubsetData(sesr['sesr'], sesr['lat'], sesr['lon'], 42, 50, -130, -111)
sesrWNC, latWNC, lonWNC = SubsetData(sesr['sesr'], sesr['lat'], sesr['lon'], 42, 50, -111, -94)
sesrENC, latENC, lonENC = SubsetData(sesr['sesr'], sesr['lat'], sesr['lon'], 38, 50, -94, -75.5)
sesrNE, latNE, lonNE = SubsetData(sesr['sesr'], sesr['lat'], sesr['lon'], 38, 50, -75.5, -65)

sesrW, latW, lonW = SubsetData(sesr['sesr'], sesr['lat'], sesr['lon'], 25, 42, -130, -114)
sesrSW, latSW, lonSW = SubsetData(sesr['sesr'], sesr['lat'], sesr['lon'], 25, 42, -114, -105)
sesrS, latS, lonS = SubsetData(sesr['sesr'], sesr['lat'], sesr['lon'], 25, 42, -105, -94)
sesrSE, latSE, lonSE = SubsetData(sesr['sesr'], sesr['lat'], sesr['lon'], 25, 38, -94, -65)

for t in range(sesr['sesr'].shape[-1]):
    sesrNW[:,:,t]  = np.where( ((latNW > 50) | (latNW < 42)) | ((lonNW > -111) | (lonNW < -130)), np.nan, sesrNW[:,:,t])
    sesrWNC[:,:,t] = np.where( ((latWNC > 50) | (latWNC < 42)) | ((lonWNC > -94) | (lonWNC < -111)), np.nan, sesrWNC[:,:,t])
    sesrENC[:,:,t] = np.where( ((latENC > 50) | (latENC < 38)) | ((lonENC > -75.5) | (lonENC < -94)), np.nan, sesrENC[:,:,t])
    sesrNE[:,:,t]  = np.where( ((latNE > 50) | (latNE < 38)) | ((lonNE > -65) | (lonNE < -75.5)), np.nan, sesrNE[:,:,t])
    sesrW[:,:,t]  = np.where( ((latW > 42) | (latW < 25)) | ((lonW > -114) | (lonW < -130)), np.nan, sesrW[:,:,t])
    sesrSW[:,:,t] = np.where( ((latSW > 42) | (latSW < 25)) | ((lonSW > -105) | (lonSW < -114)), np.nan, sesrSW[:,:,t])
    sesrS[:,:,t]  = np.where( ((latS > 42) | (latS < 25)) | ((lonS > -94) | (lonS < -105)), np.nan, sesrS[:,:,t])
    sesrSE[:,:,t] = np.where( ((latSE > 38) | (latSE < 25)) | ((lonSE > -65) | (lonSE < -94)), np.nan, sesrSE[:,:,t])


# NW  = np.ones((sesrNW.shape[:2])) * 1
# WNC = np.ones((sesrWNC.shape[:2])) * 2
# ENC = np.ones((sesrENC.shape[:2])) * 3
# NE  = np.ones((sesrNE.shape[:2])) * 4
# W   = np.ones((sesrW.shape[:2])) * 5
# SW  = np.ones((sesrSW.shape[:2])) * 6
# S   = np.ones((sesrS.shape[:2])) * 7
# SE  = np.ones((sesrSE.shape[:2])) * 8

# NW  = np.where( ((latNW > 50) | (latNW < 42)) | ((lonNW > -111) | (lonNW < -130)), np.nan, NW)
# WNC = np.where( ((latWNC > 50) | (latWNC < 42)) | ((lonWNC > -94) | (lonWNC < -111)), np.nan, WNC)
# ENC = np.where( ((latENC > 50) | (latENC < 38)) | ((lonENC > -75.5) | (lonENC < -94)), np.nan, ENC)
# NE  = np.where( ((latNE > 50) | (latNE < 38)) | ((lonNE > -65) | (lonNE < -75.5)), np.nan, NE)
# W  = np.where( ((latW > 42) | (latW < 25)) | ((lonW > -114) | (lonW < -130)), np.nan, W)
# SW = np.where( ((latSW > 42) | (latSW < 25)) | ((lonSW > -105) | (lonSW < -114)), np.nan, SW)
# S  = np.where( ((latS > 42) | (latS < 25)) | ((lonS > -94) | (lonS < -105)), np.nan, S)
# SE = np.where( ((latSE > 38) | (latSE < 25)) | ((lonSE > -65) | (lonSE < -94)), np.nan, SE)

#%%
# Create a figure

# Shapefile information
ShapeName = 'admin_0_countries'
CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)
    
CountriesReader = shpreader.Reader(CountriesSHP)
    
USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
    
# Lonitude and latitude tick information
lat_int = 10
lon_int = 20

LatLabel = np.arange(-90, 90, lat_int)
LonLabel = np.arange(-180, 180, lon_int)

LonFormatter = cticker.LongitudeFormatter()
LatFormatter = cticker.LatitudeFormatter()

# Projection information
data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Create the figure
fig = plt.figure(figsize = [18, 15])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Set the title
ax.set_title('SESR Histograms', size = 18)

# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
ax.add_feature(cfeature.STATES, zorder = 2)
ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Adjust the ticks
ax.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax.set_yticklabels(LatLabel, fontsize = 18)
ax.set_xticklabels(LonLabel, fontsize = 18)

ax.xaxis.set_major_formatter(LonFormatter)
ax.yaxis.set_major_formatter(LatFormatter)

# Plot the flash drought data
ax.add_patch(patches.Rectangle(xy = [-130, 42], width = 19, height = 8, color = '#40E0D0', alpha = 1.0, transform = fig_proj, zorder = 1))
ax.add_patch(patches.Rectangle(xy = [-111, 42], width = 17, height = 8, color = '#FF7F50', alpha = 1.0, transform = fig_proj, zorder = 1))
ax.add_patch(patches.Rectangle(xy = [-94, 38], width = 18.5, height = 12, color = '#9ACD32', alpha = 1.0, transform = fig_proj, zorder = 1))
ax.add_patch(patches.Rectangle(xy = [-75.5, 38], width = 10.5, height = 12, color = '#7BC8F6', alpha = 1.0, transform = fig_proj, zorder = 1))

ax.add_patch(patches.Rectangle(xy = [-130, 25], width = 16, height = 17, color = '#ADD8E6', alpha = 1.0, transform = fig_proj, zorder = 1))
ax.add_patch(patches.Rectangle(xy = [-114, 25], width = 9, height = 17, color = '#DAA520', alpha = 1.0, transform = fig_proj, zorder = 1))
ax.add_patch(patches.Rectangle(xy = [-105, 25], width = 11, height = 17, color = '#FBDD7E', alpha = 1.0, transform = fig_proj, zorder = 1))
ax.add_patch(patches.Rectangle(xy = [-94, 25], width = 29, height = 13, color = '#FAC205', alpha = 1.0, transform = fig_proj, zorder = 1))


# Set the map extent to the U.S.
ax.set_extent([-130, -65, 23.5, 48.5])

# NW inset
axnw = inset_axes(ax, width = 1.3, height = 1.3, bbox_to_anchor = (0.25, 0.9), bbox_transform = ax.transAxes)
axnw.hist(np.nanmean(np.nanmean(sesrNW, axis = 0), axis = 0), color = '#40E0D0')

# axnw.set_xlabel('FPR', fontsize = 16)
# axnw.set_ylabel('TPR', fontsize = 16)

for i in axnw.xaxis.get_ticklabels() + axnw.yaxis.get_ticklabels():
    i.set_size(14)
    

# WNC inset
axwnc = inset_axes(ax, width = 1.3, height = 1.3, bbox_to_anchor = (0.475, 0.9), bbox_transform = ax.transAxes)
axwnc.hist(np.nanmean(np.nanmean(sesrWNC, axis = 0), axis = 0), color = '#FF7F50')

# axwnc.set_xlabel('FPR', fontsize = 16)
# axwnc.set_ylabel('TPR', fontsize = 16)

for i in axwnc.xaxis.get_ticklabels() + axwnc.yaxis.get_ticklabels():
    i.set_size(14)
    
    
# NC inset
axnc = inset_axes(ax, width = 1.3, height = 1.3, bbox_to_anchor = (0.75, 0.795), bbox_transform = ax.transAxes)
axnc.hist(np.nanmean(np.nanmean(sesrENC, axis = 0), axis = 0), color = '#9ACD32')

# axnc.set_xlabel('FPR', fontsize = 16)
# axnc.set_ylabel('TPR', fontsize = 16)

for i in axnc.xaxis.get_ticklabels() + axnc.yaxis.get_ticklabels():
    i.set_size(14)
    

# NE inset
axne = inset_axes(ax, width = 1.3, height = 1.3, bbox_to_anchor = (0.98, 0.9), bbox_transform = ax.transAxes)
axne.hist(np.nanmean(np.nanmean(sesrNE, axis = 0), axis = 0), color = '#7BC8F6')

# axne.set_xlabel('FPR', fontsize = 16)
# axne.set_ylabel('TPR', fontsize = 16)

for i in axne.xaxis.get_ticklabels() + axne.yaxis.get_ticklabels():
    i.set_size(14)
    
    
# W inset
axw = inset_axes(ax, width = 1.3, height = 1.3, bbox_to_anchor = (0.23, 0.55), bbox_transform = ax.transAxes)
axw.hist(np.nanmean(np.nanmean(sesrW, axis = 0), axis = 0), color = '#ADD8E6')

# axw.set_xlabel('FPR', fontsize = 16)
# axw.set_ylabel('TPR', fontsize = 16)

for i in axw.xaxis.get_ticklabels() + axw.yaxis.get_ticklabels():
    i.set_size(14)
    
    
# SW inset
axsw = inset_axes(ax, width = 1.3, height = 1.3, bbox_to_anchor = (0.38, 0.55), bbox_transform = ax.transAxes)
axsw.hist(np.nanmean(np.nanmean(sesrSW, axis = 0), axis = 0), color = '#DAA520')

# axsw.set_xlabel('FPR', fontsize = 16)
# axsw.set_ylabel('TPR', fontsize = 16)

for i in axsw.xaxis.get_ticklabels() + axsw.yaxis.get_ticklabels():
    i.set_size(14)
    
    
# S inset
axs = inset_axes(ax, width = 1.3, height = 1.3, bbox_to_anchor = (0.53, 0.53), bbox_transform = ax.transAxes)
axs.hist(np.nanmean(np.nanmean(sesrS, axis = 0), axis = 0), color = '#FBDD7E')

# axs.set_xlabel('FPR', fontsize = 16)
# axs.set_ylabel('TPR', fontsize = 16)

for i in axs.xaxis.get_ticklabels() + axs.yaxis.get_ticklabels():
    i.set_size(16)
    
    
# SE inset
axse = inset_axes(ax, width = 1.3, height = 1.3, bbox_to_anchor = (0.76, 0.48), bbox_transform = ax.transAxes)
axse.hist(np.nanmean(np.nanmean(sesrSE, axis = 0), axis = 0), color = '#FAC205')

# axse.set_xlabel('FPR', fontsize = 16)
# axse.set_ylabel('TPR', fontsize = 16)

for i in axse.xaxis.get_ticklabels() + axse.yaxis.get_ticklabels():
    i.set_size(16)

plt.show(block = False)




