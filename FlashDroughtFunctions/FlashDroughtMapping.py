#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:30:52 2020

@author: stuartedris

The following set of functions were designed to create some of the early maps
of the flash drought component decomposition project. The maps provide a broad
overview of the full 38 year dataset, showing annual means across all years and
the 38 year mean across each month, season, growing season, etc.

Note because these maps are early versions, they do not contain 2017, 2018, or 2019
as the data for those years were added later in the project.
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
############################
### MapAllYears Function ###
############################

# Creates a map of four criteria plus flash drought for all years in the first dataset (1979 - 2016)

def MapAllYears(c1, c2, c3, c4, FD, lon, lat, years, cmin, cmax, cint, title = 'Title', 
                clabel = 'Criteria Label', savename = 'tmp.txt', OutPath = './Figures/'):
    '''
    This function creates a large map showing the annual average value for
    each criteria in Christian et al. (2019) plus flash drought identified
    for all years in the dataset. Note the map this function makes is a bit
    of a troll map, made in a (non-serious) bet to fit as many maps in 1
    figure as possible. The next two functions (SixYearPlot and EightYearPlot)
    provides more readible versions of this figure. Note, this function assumes 
    the entered data has already been averaged to the annual averaged data.
    
    Inputs:
    - c1: The gridded criteria 1 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c2: The gridded criteria 2 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c3: The gridded criteria 3 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c4: The gridded criteria 4 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - FD: The gridded flash drought data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - lon: The gridded longitude data associated with c1, c2, c3, c4, and FD.
    - lat: The gridded latitude data associated with c1, c2, c3, c4, and FD.
    - years: A list containing all the years in the datasets (i.e., 1979, 1980,..., 2016).
    - cmin, cmax, cint: The minimum and maximum values for the color bars in this figure,
                        as well as the interval between each value in the colorbar (cint).
    - title: The main title of the figure.
    - clabel: The label of colorbar for the figure.
    - savename: The filename the figure will be saved as.
    - OutPath: The path from the current directory to the directory the figure will
               be saved in.
               
    Outputs:
    - None. A figure is produced and will be saved.
    '''
    
    # Define the number of years in the dataset
    NumYears = 38
    
    # Set colorbar information
    cmin = cmin; cmax = cmax; cint = cint
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)
    #cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "yellow", "orange", "red", "black"], nlevs)
    
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
    
    # Create the oversized plot
    fig, axes = plt.subplots(figsize = [12, 18], nrows = NumYears, ncols = 5, 
                             subplot_kw = {'projection': fig_proj})
    
    # Adjust the figure parameters
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.8, hspace = 0.1)
    fig.suptitle(title, y = 0.94, size = 14)
    for y in range(NumYears):
        ax1 = axes[y,0]; ax2 = axes[y,1]; ax3 = axes[y,2]; ax4 = axes[y,3]; ax5 = axes[y,4]
        
        # Criteria 1 plot
        
        # Add features (e.g., country/state borders)
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax1.set_yticklabels(LatLabel, fontsize = 6)
        ax1.set_xticklabels(LonLabel, fontsize = 6)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax1.pcolormesh(lon, lat, c1[:,:,y], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax1.set_extent([-130, -65, 25, 50])
        
        # Add labels to the left most plot (what year the data is for.)
        ax1.set_ylabel(years[y], size = 8, labelpad = 15.0, rotation = 0)
        
        
        
        # Criteria 2 plot
        
        # Add features (e.g., country/state borders)
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax2.set_yticklabels(LatLabel, fontsize = 6)
        ax2.set_xticklabels(LonLabel, fontsize = 6)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax2.pcolormesh(lon, lat, c2[:,:,y], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax2.set_extent([-130, -65, 25, 50])
        
        
        
        # Criteria 3 plot
        
        # Add features (e.g., country/state borders)
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax3.set_yticklabels(LatLabel, fontsize = 6)
        ax3.set_xticklabels(LonLabel, fontsize = 6)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax3.pcolormesh(lon, lat, c3[:,:,y], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax3.set_extent([-130, -65, 25, 50])
        
        
        
        # Criteria 4 plot
        
        # Add features (e.g., country/state borders)
        ax4.coastlines()
        ax4.add_feature(cfeature.BORDERS)
        ax4.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax4.set_yticklabels(LatLabel, fontsize = 6)
        ax4.set_xticklabels(LonLabel, fontsize = 6)
        
        ax4.xaxis.set_major_formatter(LonFormatter)
        ax4.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax4.pcolormesh(lon, lat, c4[:,:,y], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax4.set_extent([-130, -65, 25, 50])
        
        
        
        # Flash Drought plot
        
        # Add features (e.g., country/state borders)
        ax5.coastlines()
        ax5.add_feature(cfeature.BORDERS)
        ax5.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax5.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax5.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax5.set_yticklabels(LatLabel, fontsize = 6)
        ax5.set_xticklabels(LonLabel, fontsize = 6)
        
        ax5.xaxis.set_major_formatter(LonFormatter)
        ax5.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax5.pcolormesh(lon, lat, FD[:,:,y], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax5.set_extent([-130, -65, 25, 50])
        
        if y == 0:
            # For the top row, set the titles
            ax1.set_title('Criteria 1')
            ax2.set_title('Criteria 2')
            ax3.set_title('Criteria 3')
            ax4.set_title('Criteria 4')
            ax5.set_title('Flash Drought')
            
            # Set some extra tick parameters
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 6, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 6, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 6, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 6, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 6, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = True)
            
        elif y == NumYears-1:
            # Set some extra tick parameters for the last iteration
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 6, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 6, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 6, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 6, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 6, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = True)
            
        else:
            # Set some extra tick parameters for the remaining iterations
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 6, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 6, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 6, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 6, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 6, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = True)
        
    # Create the axis location and size for the colorbar
    cbax = fig.add_axes([0.75, 0.12, 0.015, 0.78])
    
    # Create the colorbar
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
    cbar.ax.set_ylabel(clabel)
    #fig.tight_layout()
    
    # Save the figure
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    
#%%
# cell 4
###############################################
### SixYearPlot and EightYearPlot Functions ###
###############################################
    
# Creates plots of all four criteria plus flash drought for six year and eight year intervals (more readable than
# the MappAllYears plot)
    
def SixYearPlot(c1, c2, c3, c4, FD, lon, lat, years, cmin, cmax, cint, FDmin, FDmax, title = 'Title', 
                clabel = 'Criteria Label', FDlabel = 'Flash Drought Label', savename = 'tmp.txt', 
                OutPath = './Figures/',  shift = 0):
    '''
    This function creates a set of map showing the annual average value for
    each criteria in Christian et al. (2019) plus flash drought identified
    for six years in the dataset. Note, this function assumes the entered 
    data has already been averaged to the annual averaged data.
    
    Inputs:
    - c1: The gridded criteria 1 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c2: The gridded criteria 2 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c3: The gridded criteria 3 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c4: The gridded criteria 4 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - FD: The gridded flash drought data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - lon: The gridded longitude data associated with c1, c2, c3, c4, and FD.
    - lat: The gridded latitude data associated with c1, c2, c3, c4, and FD.
    - years: A list containing the six years to be plotted (e.g., 1979, 1980, 1981, 1982,
             1983, 1984).
    - cmin, cmax, cint: The minimum and maximum values for the c2 and c3 color bars in this 
                        figure, as well as the interval between each value in the colorbar (cint).
    - FDmin, FDmax: The minimum and maximum values for the c1, c4, and FD color bars in this figure.
    - title: The main title of the figure.
    - clabel: The label of c2 and c3 colorbar.
    - FDlabel: The label of c1, c4, and FD colorbar.
    - savename: The filename the figure will be saved as.
    - OutPath: The path from the current directory to the directory the figure will
               be saved in.
    - shift: The number of years the first value in years is from the first year in the
             dataset (i.e., years[0] - 1979).
               
    Outputs:
    - None. A figure is produced and will be saved.
    '''
    
    # Define the number of years in the figure
    NumYears = 6
    
    # Set colorbar information
    cmin = cmin; cmax = cmax; cint = cint
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)
    #cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "yellow", "orange", "red", "black"], nlevs)
    
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
    
    # Create the plot
    fig, axes = plt.subplots(figsize = [12, 18], nrows = NumYears, ncols = 5, 
                             subplot_kw = {'projection': fig_proj})
    
    # Adjust the figure parameters
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.87)
    fig.suptitle(title, y = 0.72, size = 18)
    for y in range(NumYears):
        ax1 = axes[y,0]; ax2 = axes[y,1]; ax3 = axes[y,2]; ax4 = axes[y,3]; ax5 = axes[y,4]
        
        # Criteria 1 plot
        
        # Add features (e.g., country/state borders)
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax1.pcolormesh(lon, lat, c1[:,:,y + shift], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax1.set_extent([-130, -65, 25, 50])
        
        # Add labels to the left most plot (what year the data is for.)
        ax1.set_ylabel(years[y + shift], size = 14, labelpad = 20.0, rotation = 0)
        
        
        
        # Criteria 2 plot
        
        # Add features (e.g., country/state borders)
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax2.pcolormesh(lon, lat, c2[:,:,y + shift], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        # Modify the map extent to focus on the U.S.
        ax2.set_extent([-130, -65, 25, 50])
        
        
        
        # Criteria 3 plot
        
        # Add features (e.g., country/state borders)
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax3.pcolormesh(lon, lat, c3[:,:,y + shift], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax3.set_extent([-130, -65, 25, 50])
        
        
        
        # Criteria 4 plot
        
        # Add features (e.g., country/state borders)
        ax4.coastlines()
        ax4.add_feature(cfeature.BORDERS)
        ax4.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax4.set_yticklabels(LatLabel, fontsize = 10)
        ax4.set_xticklabels(LonLabel, fontsize = 10)
        
        ax4.xaxis.set_major_formatter(LonFormatter)
        ax4.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax4.pcolormesh(lon, lat, c4[:,:,y + shift], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        # Modify the map extent to focus on the U.S.
        ax4.set_extent([-130, -65, 25, 50])
        
        
        
        # Flash Drought plot
        
        # Add features (e.g., country/state borders)
        ax5.coastlines()
        ax5.add_feature(cfeature.BORDERS)
        ax5.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax5.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax5.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax5.set_yticklabels(LatLabel, fontsize = 10)
        ax5.set_xticklabels(LonLabel, fontsize = 10)
        
        ax5.xaxis.set_major_formatter(LonFormatter)
        ax5.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cfd = ax5.pcolormesh(lon, lat, FD[:,:,y + shift], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax5.set_extent([-130, -65, 25, 50])
        
        if y == 0:
            # For the top row, set the titles
            ax1.set_title('Criteria 1')
            ax2.set_title('Criteria 2')
            ax3.set_title('Criteria 3')
            ax4.set_title('Criteria 4')
            ax5.set_title('Flash Drought')
            
            # Set some extra tick parameters
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = True)
            
        elif y == NumYears-1:
            # Set some extra tick parameters for the last iteration
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = True)
            
        else:
            # Set some extra tick parameters for the remaining iterations
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = True)
        
    # Create the axis location and size for the colorbar
    cbax = fig.add_axes([0.95, 0.32, 0.015, 0.36])
    
    # Create the c2/c3 colorbar
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
    cbar.ax.set_ylabel(clabel, fontsize = 14)
    
    # Create the axis location and size for the colorbar
    cbaxfd = fig.add_axes([0.07, 0.28, 0.85, 0.015])
    
    # Create the c1/c4/FD colorbar
    cbarfd = fig.colorbar(cfd, cax = cbaxfd, orientation = 'horizontal')
    cbarfd.ax.set_xlabel(FDlabel, fontsize = 14)
    #fig.tight_layout()
    
    # Save the figure
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    

def EightYearPlot(c1, c2, c3, c4, FD, lon, lat, years, cmin, cmax, cint, FDmin, FDmax, title = 'Title', 
                  clabel = 'Criteria Label', FDlabel = 'Flash Drought Label', savename = 'tmp.txt', 
                  OutPath = './Figures/',  shift = 0):
    '''
    This function creates a set of map showing the annual average value for
    each criteria in Christian et al. (2019) plus flash drought identified
    for eight years in the dataset. Note, this function assumes the entered 
    data has already been averaged to the annual averaged data.
    
    Inputs:
    - c1: The gridded criteria 1 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c2: The gridded criteria 2 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c3: The gridded criteria 3 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c4: The gridded criteria 4 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - FD: The gridded flash drought data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - lon: The gridded longitude data associated with c1, c2, c3, c4, and FD.
    - lat: The gridded latitude data associated with c1, c2, c3, c4, and FD.
    - years: A list containing the six years to be plotted (e.g., 2009, 2010, 2011, 2012,
             2013, 2014. 2015, 2016).
    - cmin, cmax, cint: The minimum and maximum values for the c2 and c3 color bars in this 
                        figure, as well as the interval between each value in the colorbar (cint).
    - FDmin, FDmax: The minimum and maximum values for the c1, c4, and FD color bars in this figure.
    - title: The main title of the figure.
    - clabel: The label of c2 and c3 colorbar.
    - FDlabel: The label of c1, c4, and FD colorbar.
    - savename: The filename the figure will be saved as.
    - OutPath: The path from the current directory to the directory the figure will
               be saved in.
    - shift: The number of years the first value in years is from the first year in the
             dataset (i.e., years[0] - 1979).
               
    Outputs:
    - None. A figure is produced and will be saved.
    '''
    
    # Set colorbar information
    cmin = cmin; cmax = cmax; cint = cint
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)
    #cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "yellow", "orange", "red", "black"], nlevs)
    
    # Define the number of years in the figure
    NumYears = 8
    
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
    
    # Create the plot
    fig, axes = plt.subplots(figsize = [12, 18], nrows = NumYears, ncols = 5, 
                             subplot_kw = {'projection': fig_proj})
    
    # Adjust the figure parameters
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.87)
    fig.suptitle(title, y = 0.76, size = 18)
    for y in range(NumYears):
        ax1 = axes[y,0]; ax2 = axes[y,1]; ax3 = axes[y,2]; ax4 = axes[y,3]; ax5 = axes[y,4]
        
        # Criteria 1 plot
        
        # Add features (e.g., country/state borders)
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax1.pcolormesh(lon, lat, c1[:,:,y + shift], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax1.set_extent([-130, -65, 25, 50])
        
        # Add labels to the left most plot (what year the data is for.)
        ax1.set_ylabel(years[y + shift], size = 14, labelpad = 20.0, rotation = 0)
        
        
        
        # Criteria 2 plot
        
        # Add features (e.g., country/state borders)
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax2.pcolormesh(lon, lat, c2[:,:,y + shift], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax2.set_extent([-130, -65, 25, 50])
        
        
        
        # Criteria 3 plot
        
        # Add features (e.g., country/state borders)
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax3.pcolormesh(lon, lat, c3[:,:,y + shift], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax3.set_extent([-130, -65, 25, 50])
        
        
        
        # Criteria 4 plot
        
        # Add features (e.g., country/state borders)
        ax4.coastlines()
        ax4.add_feature(cfeature.BORDERS)
        ax4.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax4.set_yticklabels(LatLabel, fontsize = 10)
        ax4.set_xticklabels(LonLabel, fontsize = 10)
        
        ax4.xaxis.set_major_formatter(LonFormatter)
        ax4.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax4.pcolormesh(lon, lat, c4[:,:,y + shift], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax4.set_extent([-130, -65, 25, 50])
        
        
        
        # Flash Drought plot
        
        # Add features (e.g., country/state borders)
        ax5.coastlines()
        ax5.add_feature(cfeature.BORDERS)
        ax5.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax5.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax5.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax5.set_yticklabels(LatLabel, fontsize = 10)
        ax5.set_xticklabels(LonLabel, fontsize = 10)
        
        ax5.xaxis.set_major_formatter(LonFormatter)
        ax5.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cfd = ax5.pcolormesh(lon, lat, FD[:,:,y + shift], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax5.set_extent([-130, -65, 25, 50])
        
        if y == 0:
            # For the top row, set the titles
            ax1.set_title('Criteria 1')
            ax2.set_title('Criteria 2')
            ax3.set_title('Criteria 3')
            ax4.set_title('Criteria 4')
            ax5.set_title('Flash Drought')
            
            # Set some extra tick parameters
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = True)
            
        elif y == NumYears-1:
            # Set some extra tick parameters for the last iteration
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = True)
            
        else:
            # Set some extra tick parameters for the remaining iterations
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = True)
        
    # Create the axis location and size for the colorbar
    cbax = fig.add_axes([0.95, 0.28, 0.015, 0.44])
    
    # Create the c2/c3 colorbar
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
    cbar.ax.set_ylabel(clabel, fontsize = 14)
    
    # Create the axis location and size for the colorbar
    cbaxfd = fig.add_axes([0.07, 0.25, 0.85, 0.015])
    
    # Create the c1/c4/FD colorbar
    cbarfd = fig.colorbar(cfd, cax = cbaxfd, orientation = 'horizontal')
    cbarfd.ax.set_xlabel(FDlabel, fontsize = 14)
    #fig.tight_layout()
    
    # Save the figure
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    
    
    

#%%
# cell 5
#############################
### MonthlyMaps Functions ###
#############################
    
# Creates a map for all four criteria plus flash drought for all months in the year.
    
def MonthlyMaps(c1, c2, c3, c4, FD, lon, lat, months, cmin, cmax, cint, FDmin, FDmax, title = 'Title', 
                clabel = 'Criteria Label', FDlabel = 'Flash Drought Label', savename = 'tmp.txt', 
                OutPath = './Figures/'):
    '''
    This function creates a set of map showing the average value for
    each criteria in Christian et al. (2019) plus flash drought identified
    for each month. The average is taken over all years in the dataset. 
    Note, this function assumes the entered data has already been averaged 
    to the monthly averaged data.
    
    Inputs:
    - c1: The gridded criteria 1 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c2: The gridded criteria 2 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c3: The gridded criteria 3 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c4: The gridded criteria 4 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - FD: The gridded flash drought data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - lon: The gridded longitude data associated with c1, c2, c3, c4, and FD.
    - lat: The gridded latitude data associated with c1, c2, c3, c4, and FD.
    - years: A list containing the six years to be plotted (e.g., 2009, 2010, 2011, 2012,
             2013, 2014. 2015, 2016).
    - cmin, cmax, cint: The minimum and maximum values for the c2 and c3 color bars in this 
                        figure, as well as the interval between each value in the colorbar (cint).
    - FDmin, FDmax: The minimum and maximum values for the c1, c4, and FD color bars in this figure.
    - title: The main title of the figure.
    - clabel: The label of c2 and c3 colorbar.
    - FDlabel: The label of c1, c4, and FD colorbar.
    - savename: The filename the figure will be saved as.
    - OutPath: The path from the current directory to the directory the figure will
               be saved in.
               
    Outputs:
    - None. A figure is produced and will be saved.
    '''
    
    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Set colorbar information
    cmin = cmin; cmax = cmax; cint = cint
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)
    #cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "yellow", "orange", "red", "black"], nlevs)
    
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
    
    # Create the plot
    fig, axes = plt.subplots(figsize = [12, 18], nrows = len(month_names), ncols = 5, 
                             subplot_kw = {'projection': fig_proj})
    
    # Adjust the figure parameters
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.65)
    
    fig.suptitle(title, y = 0.88, size = 18)
    
    for m in range(len(month_names)):
        ax1 = axes[m,0]; ax2 = axes[m,1]; ax3 = axes[m,2]; ax4 = axes[m,3]; ax5 = axes[m,4]
        
        # Criteria 1 plot
        
        # Add features (e.g., country/state borders)
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax1.pcolormesh(lon, lat, c1[:,:,m], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax1.set_extent([-130, -65, 25, 50])
        
        # Add labels to the left most plot (what month the data is for.)
        ax1.set_ylabel(month_names[m], size = 14, labelpad = 20.0, rotation = 0)
        
        
        
        # Criteria 2 plot
        
        # Add features (e.g., country/state borders)
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax2.pcolormesh(lon, lat, c2[:,:,m], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax2.set_extent([-130, -65, 25, 50])
        
        
        
        # Criteria 3 plot
        
        # Add features (e.g., country/state borders)
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax3.pcolormesh(lon, lat, c3[:,:,m], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax3.set_extent([-130, -65, 25, 50])
        
        
        
        # Criteria 4 plot
        
        # Add features (e.g., country/state borders)
        ax4.coastlines()
        ax4.add_feature(cfeature.BORDERS)
        ax4.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax4.set_yticklabels(LatLabel, fontsize = 10)
        ax4.set_xticklabels(LonLabel, fontsize = 10)
        
        ax4.xaxis.set_major_formatter(LonFormatter)
        ax4.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax4.pcolormesh(lon, lat, c4[:,:,m], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax4.set_extent([-130, -65, 25, 50])
        
        
        
        # Flash Drought plot
        
        # Add features (e.g., country/state borders)
        ax5.coastlines()
        ax5.add_feature(cfeature.BORDERS)
        ax5.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax5.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax5.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax5.set_yticklabels(LatLabel, fontsize = 10)
        ax5.set_xticklabels(LonLabel, fontsize = 10)
        
        ax5.xaxis.set_major_formatter(LonFormatter)
        ax5.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cfd = ax5.pcolormesh(lon, lat, FD[:,:,m], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax5.set_extent([-130, -65, 25, 50])
        
        if m == 0:
            # For the top row, set the titles
            ax1.set_title('Criteria 1')
            ax2.set_title('Criteria 2')
            ax3.set_title('Criteria 3')
            ax4.set_title('Criteria 4')
            ax5.set_title('Flash Drought')
            
            # Set some extra tick parameters
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = True)
            
        elif m == len(month_names)-1:
            # Set some extra tick parameters for the last iteration
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = True)
            
        else:
            # Set some extra tick parameters for the remaining iterations
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = True)
        
    # Create the axis location and size for the colorbar
    cbax = fig.add_axes([0.95, 0.15, 0.015, 0.68])
    
    # Create the c2/c3 colorbar
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
    cbar.ax.set_ylabel(clabel, fontsize = 14)
    
    # Create the axis location and size for the colorbar
    cbaxfd = fig.add_axes([0.07, 0.12, 0.85, 0.015])
    
    # Create the c1/c4/FD colorbar
    cbarfd = fig.colorbar(cfd, cax = cbaxfd, orientation = 'horizontal')
    cbarfd.ax.set_xlabel(FDlabel, fontsize = 14)
    #fig.tight_layout()
    
    # Save the figure
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    
    
######################
### MonhlyGrowMaps ###
######################
    
# Same map as above, but for only the warm/growing season months.
    
def MonthlyGrowMaps(c1, c2, c3, c4, FD, lon, lat, months, cmin, cmax, cint, FDmin, FDmax, title = 'Title', 
                 clabel = 'Criteria Label', FDlabel = 'Flash Drought Label', savename = 'tmp.txt', 
                 OutPath = './Figures/'):
    '''
    This function creates a set of map showing the average value for
    each criteria in Christian et al. (2019) plus flash drought identified
    for each month in the growing season (AMJJASO). The average is taken 
    over all years in the dataset. Note, this function assumes the entered 
    data has already been averaged to the monthly averaged data.
    
    Inputs:
    - c1: The gridded criteria 1 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c2: The gridded criteria 2 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c3: The gridded criteria 3 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c4: The gridded criteria 4 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - FD: The gridded flash drought data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - lon: The gridded longitude data associated with c1, c2, c3, c4, and FD.
    - lat: The gridded latitude data associated with c1, c2, c3, c4, and FD.
    - years: A list containing the six years to be plotted (e.g., 2009, 2010, 2011, 2012,
             2013, 2014. 2015, 2016).
    - cmin, cmax, cint: The minimum and maximum values for the c2 and c3 color bars in this 
                        figure, as well as the interval between each value in the colorbar (cint).
    - FDmin, FDmax: The minimum and maximum values for the c1, c4, and FD color bars in this figure.
    - title: The main title of the figure.
    - clabel: The label of c2 and c3 colorbar.
    - FDlabel: The label of c1, c4, and FD colorbar.
    - savename: The filename the figure will be saved as.
    - OutPath: The path from the current directory to the directory the figure will
               be saved in.
               
    Outputs:
    - None. A figure is produced and will be saved.
    '''
    
    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Set colorbar information
    cmin = cmin; cmax = cmax; cint = cint
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)
    #cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "yellow", "orange", "red", "black"], nlevs)
    
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
    
    # Create the plot
    fig, axes = plt.subplots(figsize = [12, 18], nrows = len(month_names)-5, ncols = 5, 
                             subplot_kw = {'projection': fig_proj})
    
    # Adjust the figure parameters
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.85)
    
    fig.suptitle(title, y = 0.76, size = 18)
    
    for m in range(len(month_names)-5):
        ax1 = axes[m,0]; ax2 = axes[m,1]; ax3 = axes[m,2]; ax4 = axes[m,3]; ax5 = axes[m,4]
        
        # Criteria 1 plot
        
        # Add features (e.g., country/state borders)
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cfd = ax1.pcolormesh(lon, lat, c1[:,:,m+3], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax1.set_extent([-130, -65, 25, 50])
        
        # Add labels to the left most plot (what month the data is for.)
        ax1.set_ylabel(month_names[m+3], size = 14, labelpad = 20.0, rotation = 0)
        
        
        
        # Criteria 2 plot
        
        # Add features (e.g., country/state borders)
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax2.pcolormesh(lon, lat, c2[:,:,m+3], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax2.set_extent([-130, -65, 25, 50])
        
        
        
        # Criteria 3 plot
        
        # Add features (e.g., country/state borders)
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax3.pcolormesh(lon, lat, c3[:,:,m+3], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax3.set_extent([-130, -65, 25, 50])
        
        
        
        # Criteria 4 plot
        
        # Add features (e.g., country/state borders)
        ax4.coastlines()
        ax4.add_feature(cfeature.BORDERS)
        ax4.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax4.set_yticklabels(LatLabel, fontsize = 10)
        ax4.set_xticklabels(LonLabel, fontsize = 10)
        
        ax4.xaxis.set_major_formatter(LonFormatter)
        ax4.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cfd = ax4.pcolormesh(lon, lat, c4[:,:,m+3], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax4.set_extent([-130, -65, 25, 50])
        
        
        
        # Flash Drought plot
        
        # Add features (e.g., country/state borders)
        ax5.coastlines()
        ax5.add_feature(cfeature.BORDERS)
        ax5.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax5.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax5.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax5.set_yticklabels(LatLabel, fontsize = 10)
        ax5.set_xticklabels(LonLabel, fontsize = 10)
        
        ax5.xaxis.set_major_formatter(LonFormatter)
        ax5.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cfd = ax5.pcolormesh(lon, lat, FD[:,:,m+3], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax5.set_extent([-130, -65, 25, 50])
        
        if m == 0:
            # For the top row, set the titles
            ax1.set_title('Criteria 1')
            ax2.set_title('Criteria 2')
            ax3.set_title('Criteria 3')
            ax4.set_title('Criteria 4')
            ax5.set_title('Flash Drought')
            
            # Set some extra tick parameters
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = True)
            
        elif m == len(month_names)-5-1:
            # Set some extra tick parameters for the last iteration
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = True)
            
        else:
            # Set some extra tick parameters for the remaining iterations
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = True)
        
    # Create the axis location and size for the colorbar
    cbax = fig.add_axes([0.95, 0.28, 0.015, 0.43])
    
    # Create the c2/c3 colorbar
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
    cbar.ax.set_ylabel(clabel, fontsize = 14)
    
    # Create the axis location and size for the colorbar
    cbaxfd = fig.add_axes([0.07, 0.25, 0.85, 0.015])
    
    # Create the c1/c4/FD colorbar
    cbarfd = fig.colorbar(cfd, cax = cbaxfd, orientation = 'horizontal')
    cbarfd.ax.set_xlabel(FDlabel, fontsize = 14)
    #fig.tight_layout()
    
    # Save the figure
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
#%%
# cell 6
############################
### SeasonMaps Functions ###
############################
    
# Creates a map for all four criteria plus flash drougth for each season.
    
def SeasonMaps(c1, c2, c3, c4, FD, lon, lat, cmin, cmax, cint, FDmin, FDmax, title = 'Title', 
               clabel = 'Criteria Label', FDlabel = 'Flash Drought Label', savename = 'tmp.txt', 
               OutPath = './Figures/'):
    '''
    This function creates a set of map showing the average value for
    each criteria in Christian et al. (2019) plus flash drought identified
    for each season; Spring (MAM), June (JJA), Autumn (SON), and Winter (DJF). 
    The average is taken over all years in the dataset. Note, this function
    assumes the entered data has already been averaged to the seasonal data.
    
    Inputs:
    - c1: The gridded criteria 1 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c2: The gridded criteria 2 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c3: The gridded criteria 3 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c4: The gridded criteria 4 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - FD: The gridded flash drought data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - lon: The gridded longitude data associated with c1, c2, c3, c4, and FD.
    - lat: The gridded latitude data associated with c1, c2, c3, c4, and FD.
    - years: A list containing the six years to be plotted (e.g., 2009, 2010, 2011, 2012,
             2013, 2014. 2015, 2016).
    - cmin, cmax, cint: The minimum and maximum values for the c2 and c3 color bars in this 
                        figure, as well as the interval between each value in the colorbar (cint).
    - FDmin, FDmax: The minimum and maximum values for the c1, c4, and FD color bars in this figure.
    - title: The main title of the figure.
    - clabel: The label of c2 and c3 colorbar.
    - FDlabel: The label of c1, c4, and FD colorbar.
    - savename: The filename the figure will be saved as.
    - OutPath: The path from the current directory to the directory the figure will
               be saved in.
               
    Outputs:
    - None. A figure is produced and will be saved.
    '''
    
    # Season months
    seasons = ['MAM', 'JJA', 'SON', 'DJF']
    
    # Set colorbar information
    cmin = cmin; cmax = cmax; cint = cint
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)
    #cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "yellow", "orange", "red", "black"], nlevs)
    
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
    
    # Create the plot
    fig, axes = plt.subplots(figsize = [12, 18], nrows = len(seasons), ncols = 5, 
                             subplot_kw = {'projection': fig_proj})
    
    # Adjust the figure parameters
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.87)
    
    fig.suptitle(title, y = 0.68, size = 18)
    
    for s in range(len(seasons)):
        ax1 = axes[s,0]; ax2 = axes[s,1]; ax3 = axes[s,2]; ax4 = axes[s,3]; ax5 = axes[s,4]
        
        # Criteria 1 plot
        
        # Add features (e.g., country/state borders)
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax1.pcolormesh(lon, lat, c1[:,:,s], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax1.set_extent([-130, -65, 25, 50])
        
        # Add labels to the left most plot (what season the data is for.)
        ax1.set_ylabel(seasons[s], size = 14, labelpad = 20.0, rotation = 0)
        
        
        
        # Criteria 2 plot
        
        # Add features (e.g., country/state borders)
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax2.pcolormesh(lon, lat, c2[:,:,s], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax2.set_extent([-130, -65, 25, 50])
        
        
        
        # Criteria 3 plot
        
        # Add features (e.g., country/state borders)
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax3.pcolormesh(lon, lat, c3[:,:,s], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax3.set_extent([-130, -65, 25, 50])
        
        
        
        # Criteria 4 plot
        
        # Add features (e.g., country/state borders)
        ax4.coastlines()
        ax4.add_feature(cfeature.BORDERS)
        ax4.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax4.set_yticklabels(LatLabel, fontsize = 10)
        ax4.set_xticklabels(LonLabel, fontsize = 10)
        
        ax4.xaxis.set_major_formatter(LonFormatter)
        ax4.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax4.pcolormesh(lon, lat, c4[:,:,s], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax4.set_extent([-130, -65, 25, 50])
        
        
        
        # Flash Drought plot
        
        # Add features (e.g., country/state borders)
        ax5.coastlines()
        ax5.add_feature(cfeature.BORDERS)
        ax5.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax5.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax5.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax5.set_yticklabels(LatLabel, fontsize = 10)
        ax5.set_xticklabels(LonLabel, fontsize = 10)
        
        ax5.xaxis.set_major_formatter(LonFormatter)
        ax5.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cfd = ax5.pcolormesh(lon, lat, FD[:,:,s], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax5.set_extent([-130, -65, 25, 50])
        
        if s == 0:
            # For the top row, set the titles
            ax1.set_title('Criteria 1')
            ax2.set_title('Criteria 2')
            ax3.set_title('Criteria 3')
            ax4.set_title('Criteria 4')
            ax5.set_title('Flash Drought')
            
            # Set some extra tick parameters
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = True)
            
        elif s == len(seasons)-1:
            # Set some extra tick parameters for the last iteration
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = True)
            
        else:
            # Set some extra tick parameters for the remaining iterations
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = True)
        
    # Create the axis location and size for the colorbar
    cbax = fig.add_axes([0.95, 0.35, 0.015, 0.29])
    
    # Create the c2/c3 colorbar
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
    cbar.ax.set_ylabel(clabel, fontsize = 14)
    
    # Create the axis location and size for the colorbar
    cbaxfd = fig.add_axes([0.07, 0.33, 0.85, 0.015])
    
    # Create the c1/c4/FD colorbar
    cbarfd = fig.colorbar(cfd, cax = cbaxfd, orientation = 'horizontal')
    cbarfd.ax.set_xlabel(FDlabel, fontsize = 14)
    #fig.tight_layout()
    
    # Save the figure
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    
#%%
# cell 7
################################
### GrowSeasonMaps Functions ###
################################
    
# Creates a map for each of the four criteria plus flash drought for the growing and non-growing season
    
def GrowSeasonMaps(c1, c2, c3, c4, FD, lon, lat, cmin, cmax, cint, FDmin, FDmax, title = 'Title', 
                   clabel = 'Criteria Label', FDlabel = 'Flash Drought Label', savename = 'tmp.txt', 
                   OutPath = './Figures/'):
    '''
    This function creates a set of map showing the average value for
    each criteria in Christian et al. (2019) plus flash drought identified
    for the growing season (AMJJASO) and non-growing season (NDJFM). 
    The average is taken over all years in the dataset. Note, this function
    assumes the entered data has already been averaged to the growing and
    non-growing seasons.
    
    Inputs:
    - c1: The gridded criteria 1 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c2: The gridded criteria 2 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c3: The gridded criteria 3 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - c4: The gridded criteria 4 data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - FD: The gridded flash drought data. Note the criteria and flash drought data
          contains 1 (the criteria/flash drought occurred) or 0 (the criteria/flash 
          drought did not occur) for every grid point and time step in the data.
    - lon: The gridded longitude data associated with c1, c2, c3, c4, and FD.
    - lat: The gridded latitude data associated with c1, c2, c3, c4, and FD.
    - years: A list containing the six years to be plotted (e.g., 2009, 2010, 2011, 2012,
             2013, 2014. 2015, 2016).
    - cmin, cmax, cint: The minimum and maximum values for the c2 and c3 color bars in this 
                        figure, as well as the interval between each value in the colorbar (cint).
    - FDmin, FDmax: The minimum and maximum values for the c1, c4, and FD color bars in this figure.
    - title: The main title of the figure.
    - clabel: The label of c2 and c3 colorbar.
    - FDlabel: The label of c1, c4, and FD colorbar.
    - savename: The filename the figure will be saved as.
    - OutPath: The path from the current directory to the directory the figure will
               be saved in.
               
    Outputs:
    - None. A figure is produced and will be saved.
    '''
    
    # Growing season months
    GrowSeasons = ['AMJJASO', 'NDJFM']
    
    # Set colorbar information
    cmin = cmin; cmax = cmax; cint = cint
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)
    #cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "yellow", "orange", "red", "black"], nlevs)
    
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
    
    # Create the plot
    fig, axes = plt.subplots(figsize = [12, 18], nrows = len(GrowSeasons), ncols = 5, 
                             subplot_kw = {'projection': fig_proj})
    
    # Adjust the figure parameters
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.87)
    
    fig.suptitle(title, y = 0.61, size = 18)
    
    for g in range(len(GrowSeasons)):
        ax1 = axes[g,0]; ax2 = axes[g,1]; ax3 = axes[g,2]; ax4 = axes[g,3]; ax5 = axes[g,4]
        
        # Criteria 1 plot
        
        # Add features (e.g., country/state borders)
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax1.pcolormesh(lon, lat, c1[:,:,g], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax1.set_extent([-130, -65, 25, 50])
        
        # Add labels to the left most plot (whether the data is for growing season or not.)
        ax1.set_ylabel(GrowSeasons[g], size = 14, labelpad = 30.0, rotation = 0)
        
        
        
        # Criteria 2 plot
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax2.pcolormesh(lon, lat, c2[:,:,g], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax2.set_extent([-130, -65, 25, 50])
        
        
        
        # Criteria 3 plot
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax3.pcolormesh(lon, lat, c3[:,:,g], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax3.set_extent([-130, -65, 25, 50])
        
        
        
        # Criteria 4 plot
        ax4.coastlines()
        ax4.add_feature(cfeature.BORDERS)
        ax4.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax4.set_yticklabels(LatLabel, fontsize = 10)
        ax4.set_xticklabels(LonLabel, fontsize = 10)
        
        ax4.xaxis.set_major_formatter(LonFormatter)
        ax4.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax4.pcolormesh(lon, lat, c4[:,:,g], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax4.set_extent([-130, -65, 25, 50])
        
        
        
        # Flash Drought plot
        ax5.coastlines()
        ax5.add_feature(cfeature.BORDERS)
        ax5.add_feature(cfeature.STATES)
        
        # Adjust the ticks
        ax5.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax5.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax5.set_yticklabels(LatLabel, fontsize = 10)
        ax5.set_xticklabels(LonLabel, fontsize = 10)
        
        ax5.xaxis.set_major_formatter(LonFormatter)
        ax5.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cfd = ax5.pcolormesh(lon, lat, FD[:,:,g], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Modify the map extent to focus on the U.S.
        ax5.set_extent([-130, -65, 25, 50])
        
        if g == 0:
            # For the top row, set the titles
            ax1.set_title('Criteria 1')
            ax2.set_title('Criteria 2')
            ax3.set_title('Criteria 3')
            ax4.set_title('Criteria 4')
            ax5.set_title('Flash Drought')
            
            # Set some extra tick parameters
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = False, labelright = True)
            
        elif g == len(GrowSeasons)-1:
            # Set some extra tick parameters for the last iteration
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = True, labeltop = False,
                            labelleft = False, labelright = True)
            
        else:
            # Set some extra tick parameters for the remaining iterations
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax4.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = False)
            ax5.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = False,
                            labelleft = False, labelright = True)
        
    # Create the axis location and size for the colorbar
    cbax = fig.add_axes([0.07, 0.40, 0.85, 0.015])
    
    # Create the c2/c3 colorbar
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')
    cbar.ax.set_xlabel(clabel, fontsize = 14)
    
    # Create the axis location and size for the colorbar
    cbaxfd = fig.add_axes([0.07, 0.34, 0.85, 0.015])
    
    # Create the c1/c4/FD colorbar
    cbarfd = fig.colorbar(cfd, cax = cbaxfd, orientation = 'horizontal')
    cbarfd.ax.set_xlabel(FDlabel, fontsize = 14)
    #fig.tight_layout()
    
    # Save the figure
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    