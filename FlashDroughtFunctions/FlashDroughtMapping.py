#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:30:52 2020

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
############################
### MapAllYears Function ###
############################

# Creates a map of four criteria plus flash drought for all years in the first dataset (1979 - 2016)

def MapAllYears(c1, c2, c3, c4, FD, lon, lat, years, cmin, cmax, cint, title = 'Title', 
                clabel = 'Criteria Label', savename = 'tmp.txt', OutPath = './Figures/'):
    '''
    
    '''
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
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.8, hspace = 0.1)
    fig.suptitle(title, y = 0.94, size = 14)
    for y in range(NumYears):
        ax1 = axes[y,0]; ax2 = axes[y,1]; ax3 = axes[y,2]; ax4 = axes[y,3]; ax5 = axes[y,4]
        
        # Criteria 1 plot
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        
        ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax1.set_yticklabels(LatLabel, fontsize = 6)
        ax1.set_xticklabels(LonLabel, fontsize = 6)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax1.pcolormesh(lon, lat, c1[:,:,y], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax1.set_extent([-130, -65, 25, 50])
        
        ax1.set_ylabel(years[y], size = 8, labelpad = 15.0, rotation = 0)
        
        # Criteria 2 plot
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        
        ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax2.set_yticklabels(LatLabel, fontsize = 6)
        ax2.set_xticklabels(LonLabel, fontsize = 6)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax2.pcolormesh(lon, lat, c2[:,:,y], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax2.set_extent([-130, -65, 25, 50])
        
        # Criteria 3 plot
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.STATES)
        
        ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax3.set_yticklabels(LatLabel, fontsize = 6)
        ax3.set_xticklabels(LonLabel, fontsize = 6)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax3.pcolormesh(lon, lat, c3[:,:,y], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax3.set_extent([-130, -65, 25, 50])
        
        # Criteria 4 plot
        ax4.coastlines()
        ax4.add_feature(cfeature.BORDERS)
        ax4.add_feature(cfeature.STATES)
        
        ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax4.set_yticklabels(LatLabel, fontsize = 6)
        ax4.set_xticklabels(LonLabel, fontsize = 6)
        
        ax4.xaxis.set_major_formatter(LonFormatter)
        ax4.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax4.pcolormesh(lon, lat, c4[:,:,y], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax4.set_extent([-130, -65, 25, 50])
        
        # Flash Drought plot
        ax5.coastlines()
        ax5.add_feature(cfeature.BORDERS)
        ax5.add_feature(cfeature.STATES)
        
        ax5.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax5.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax5.set_yticklabels(LatLabel, fontsize = 6)
        ax5.set_xticklabels(LonLabel, fontsize = 6)
        
        ax5.xaxis.set_major_formatter(LonFormatter)
        ax5.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax5.pcolormesh(lon, lat, FD[:,:,y], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
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
        
    
    cbax = fig.add_axes([0.75, 0.12, 0.015, 0.78])
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
    cbar.ax.set_ylabel(clabel)
    #fig.tight_layout()
    
    # This will be saved as a pdf so it can be zoomed in without treamendous loss in detail
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
    
    '''
    
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
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.87)
    fig.suptitle(title, y = 0.72, size = 18)
    for y in range(NumYears):
        ax1 = axes[y,0]; ax2 = axes[y,1]; ax3 = axes[y,2]; ax4 = axes[y,3]; ax5 = axes[y,4]
        
        # Criteria 1 plot
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        
        ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax1.pcolormesh(lon, lat, c1[:,:,y + shift], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        ax1.set_extent([-130, -65, 25, 50])
        
        ax1.set_ylabel(years[y + shift], size = 14, labelpad = 20.0, rotation = 0)
        
        # Criteria 2 plot
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        
        ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax2.pcolormesh(lon, lat, c2[:,:,y + shift], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax2.set_extent([-130, -65, 25, 50])
        
        # Criteria 3 plot
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.STATES)
        
        ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax3.pcolormesh(lon, lat, c3[:,:,y + shift], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax3.set_extent([-130, -65, 25, 50])
        
        # Criteria 4 plot
        ax4.coastlines()
        ax4.add_feature(cfeature.BORDERS)
        ax4.add_feature(cfeature.STATES)
        
        ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax4.set_yticklabels(LatLabel, fontsize = 10)
        ax4.set_xticklabels(LonLabel, fontsize = 10)
        
        ax4.xaxis.set_major_formatter(LonFormatter)
        ax4.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax4.pcolormesh(lon, lat, c4[:,:,y + shift], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        ax4.set_extent([-130, -65, 25, 50])
        
        # Flash Drought plot
        ax5.coastlines()
        ax5.add_feature(cfeature.BORDERS)
        ax5.add_feature(cfeature.STATES)
        
        ax5.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax5.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax5.set_yticklabels(LatLabel, fontsize = 10)
        ax5.set_xticklabels(LonLabel, fontsize = 10)
        
        ax5.xaxis.set_major_formatter(LonFormatter)
        ax5.yaxis.set_major_formatter(LatFormatter)
        
        cfd = ax5.pcolormesh(lon, lat, FD[:,:,y + shift], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
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
        
    
    cbax = fig.add_axes([0.95, 0.32, 0.015, 0.36])
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
    cbar.ax.set_ylabel(clabel, fontsize = 14)
    
    cbaxfd = fig.add_axes([0.07, 0.28, 0.85, 0.015])
    cbarfd = fig.colorbar(cfd, cax = cbaxfd, orientation = 'horizontal')
    cbarfd.ax.set_xlabel(FDlabel, fontsize = 14)
    #fig.tight_layout()
    
    # This will be saved as a pdf so it can be zoomed in without treamendous loss in detail
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    

def EightYearPlot(c1, c2, c3, c4, FD, lon, lat, years, cmin, cmax, cint, FDmin, FDmax, title = 'Title', 
                  clabel = 'Criteria Label', FDlabel = 'Flash Drought Label', savename = 'tmp.txt', 
                  OutPath = './Figures/',  shift = 0):
    '''
    
    '''
    
    # Set colorbar information
    cmin = cmin; cmax = cmax; cint = cint
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)
    #cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "yellow", "orange", "red", "black"], nlevs)
    
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
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.87)
    fig.suptitle(title, y = 0.76, size = 18)
    for y in range(NumYears):
        ax1 = axes[y,0]; ax2 = axes[y,1]; ax3 = axes[y,2]; ax4 = axes[y,3]; ax5 = axes[y,4]
        
        # Criteria 1 plot
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        
        ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax1.pcolormesh(lon, lat, c1[:,:,y + shift], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        ax1.set_extent([-130, -65, 25, 50])
        
        ax1.set_ylabel(years[y + shift], size = 14, labelpad = 20.0, rotation = 0)
        
        # Criteria 2 plot
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        
        ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax2.pcolormesh(lon, lat, c2[:,:,y + shift], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax2.set_extent([-130, -65, 25, 50])
        
        # Criteria 3 plot
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.STATES)
        
        ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax3.pcolormesh(lon, lat, c3[:,:,y + shift], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax3.set_extent([-130, -65, 25, 50])
        
        # Criteria 4 plot
        ax4.coastlines()
        ax4.add_feature(cfeature.BORDERS)
        ax4.add_feature(cfeature.STATES)
        
        ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax4.set_yticklabels(LatLabel, fontsize = 10)
        ax4.set_xticklabels(LonLabel, fontsize = 10)
        
        ax4.xaxis.set_major_formatter(LonFormatter)
        ax4.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax4.pcolormesh(lon, lat, c4[:,:,y + shift], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        ax4.set_extent([-130, -65, 25, 50])
        
        # Flash Drought plot
        ax5.coastlines()
        ax5.add_feature(cfeature.BORDERS)
        ax5.add_feature(cfeature.STATES)
        
        ax5.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax5.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax5.set_yticklabels(LatLabel, fontsize = 10)
        ax5.set_xticklabels(LonLabel, fontsize = 10)
        
        ax5.xaxis.set_major_formatter(LonFormatter)
        ax5.yaxis.set_major_formatter(LatFormatter)
        
        cfd = ax5.pcolormesh(lon, lat, FD[:,:,y + shift], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
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
        
    
    cbax = fig.add_axes([0.95, 0.28, 0.015, 0.44])
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
    cbar.ax.set_ylabel(clabel, fontsize = 14)
    
    cbaxfd = fig.add_axes([0.07, 0.25, 0.85, 0.015])
    cbarfd = fig.colorbar(cfd, cax = cbaxfd, orientation = 'horizontal')
    cbarfd.ax.set_xlabel(FDlabel, fontsize = 14)
    #fig.tight_layout()
    
    # This will be saved as a pdf so it can be zoomed in without treamendous loss in detail
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
    
    # Create the first plot (1979 - 1986)
    fig, axes = plt.subplots(figsize = [12, 18], nrows = len(month_names), ncols = 5, 
                             subplot_kw = {'projection': fig_proj})
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.65)
    
    fig.suptitle(title, y = 0.88, size = 18)
    
    for m in range(len(month_names)):
        ax1 = axes[m,0]; ax2 = axes[m,1]; ax3 = axes[m,2]; ax4 = axes[m,3]; ax5 = axes[m,4]
        
        # Criteria 1 plot
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        
        ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax1.pcolormesh(lon, lat, c1[:,:,m], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        ax1.set_extent([-130, -65, 25, 50])
        
        ax1.set_ylabel(month_names[m], size = 14, labelpad = 20.0, rotation = 0)
        
        # Criteria 2 plot
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        
        ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax2.pcolormesh(lon, lat, c2[:,:,m], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax2.set_extent([-130, -65, 25, 50])
        
        # Criteria 3 plot
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.STATES)
        
        ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax3.pcolormesh(lon, lat, c3[:,:,m], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax3.set_extent([-130, -65, 25, 50])
        
        # Criteria 4 plot
        ax4.coastlines()
        ax4.add_feature(cfeature.BORDERS)
        ax4.add_feature(cfeature.STATES)
        
        ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax4.set_yticklabels(LatLabel, fontsize = 10)
        ax4.set_xticklabels(LonLabel, fontsize = 10)
        
        ax4.xaxis.set_major_formatter(LonFormatter)
        ax4.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax4.pcolormesh(lon, lat, c4[:,:,m], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        ax4.set_extent([-130, -65, 25, 50])
        
        # Flash Drought plot
        ax5.coastlines()
        ax5.add_feature(cfeature.BORDERS)
        ax5.add_feature(cfeature.STATES)
        
        ax5.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax5.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax5.set_yticklabels(LatLabel, fontsize = 10)
        ax5.set_xticklabels(LonLabel, fontsize = 10)
        
        ax5.xaxis.set_major_formatter(LonFormatter)
        ax5.yaxis.set_major_formatter(LatFormatter)
        
        cfd = ax5.pcolormesh(lon, lat, FD[:,:,m], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
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
        
    
    cbax = fig.add_axes([0.95, 0.15, 0.015, 0.68])
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
    cbar.ax.set_ylabel(clabel, fontsize = 14)
    
    cbaxfd = fig.add_axes([0.07, 0.12, 0.85, 0.015])
    cbarfd = fig.colorbar(cfd, cax = cbaxfd, orientation = 'horizontal')
    cbarfd.ax.set_xlabel(FDlabel, fontsize = 14)
    #fig.tight_layout()
    
    # This will be saved as a pdf so it can be zoomed in without treamendous loss in detail
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
    
    # Create the first plot (1979 - 1986)
    fig, axes = plt.subplots(figsize = [12, 18], nrows = len(month_names)-5, ncols = 5, 
                             subplot_kw = {'projection': fig_proj})
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.85)
    
    fig.suptitle(title, y = 0.76, size = 18)
    
    for m in range(len(month_names)-5):
        ax1 = axes[m,0]; ax2 = axes[m,1]; ax3 = axes[m,2]; ax4 = axes[m,3]; ax5 = axes[m,4]
        
        # Criteria 1 plot
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        
        ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        cfd = ax1.pcolormesh(lon, lat, c1[:,:,m+3], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        ax1.set_extent([-130, -65, 25, 50])
        
        ax1.set_ylabel(month_names[m+3], size = 14, labelpad = 20.0, rotation = 0)
        
        # Criteria 2 plot
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        
        ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax2.pcolormesh(lon, lat, c2[:,:,m+3], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax2.set_extent([-130, -65, 25, 50])
        
        # Criteria 3 plot
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.STATES)
        
        ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax3.pcolormesh(lon, lat, c3[:,:,m+3], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax3.set_extent([-130, -65, 25, 50])
        
        # Criteria 4 plot
        ax4.coastlines()
        ax4.add_feature(cfeature.BORDERS)
        ax4.add_feature(cfeature.STATES)
        
        ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax4.set_yticklabels(LatLabel, fontsize = 10)
        ax4.set_xticklabels(LonLabel, fontsize = 10)
        
        ax4.xaxis.set_major_formatter(LonFormatter)
        ax4.yaxis.set_major_formatter(LatFormatter)
        
        cfd = ax4.pcolormesh(lon, lat, c4[:,:,m+3], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        ax4.set_extent([-130, -65, 25, 50])
        
        # Flash Drought plot
        ax5.coastlines()
        ax5.add_feature(cfeature.BORDERS)
        ax5.add_feature(cfeature.STATES)
        
        ax5.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax5.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax5.set_yticklabels(LatLabel, fontsize = 10)
        ax5.set_xticklabels(LonLabel, fontsize = 10)
        
        ax5.xaxis.set_major_formatter(LonFormatter)
        ax5.yaxis.set_major_formatter(LatFormatter)
        
        cfd = ax5.pcolormesh(lon, lat, FD[:,:,m+3], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
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
        
    
    cbax = fig.add_axes([0.95, 0.28, 0.015, 0.43])
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
    cbar.ax.set_ylabel(clabel, fontsize = 14)
    
    cbaxfd = fig.add_axes([0.07, 0.25, 0.85, 0.015])
    cbarfd = fig.colorbar(cfd, cax = cbaxfd, orientation = 'horizontal')
    cbarfd.ax.set_xlabel(FDlabel, fontsize = 14)
    #fig.tight_layout()
    
    # This will be saved as a pdf so it can be zoomed in without treamendous loss in detail
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
    
    # Create the first plot (1979 - 1986)
    fig, axes = plt.subplots(figsize = [12, 18], nrows = len(seasons), ncols = 5, 
                             subplot_kw = {'projection': fig_proj})
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.87)
    
    fig.suptitle(title, y = 0.68, size = 18)
    
    for s in range(len(seasons)):
        ax1 = axes[s,0]; ax2 = axes[s,1]; ax3 = axes[s,2]; ax4 = axes[s,3]; ax5 = axes[s,4]
        
        # Criteria 1 plot
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        
        ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax1.pcolormesh(lon, lat, c1[:,:,s], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        ax1.set_extent([-130, -65, 25, 50])
        
        ax1.set_ylabel(seasons[s], size = 14, labelpad = 20.0, rotation = 0)
        
        # Criteria 2 plot
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        
        ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax2.pcolormesh(lon, lat, c2[:,:,s], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax2.set_extent([-130, -65, 25, 50])
        
        # Criteria 3 plot
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.STATES)
        
        ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax3.pcolormesh(lon, lat, c3[:,:,s], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax3.set_extent([-130, -65, 25, 50])
        
        # Criteria 4 plot
        ax4.coastlines()
        ax4.add_feature(cfeature.BORDERS)
        ax4.add_feature(cfeature.STATES)
        
        ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax4.set_yticklabels(LatLabel, fontsize = 10)
        ax4.set_xticklabels(LonLabel, fontsize = 10)
        
        ax4.xaxis.set_major_formatter(LonFormatter)
        ax4.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax4.pcolormesh(lon, lat, c4[:,:,s], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        ax4.set_extent([-130, -65, 25, 50])
        
        # Flash Drought plot
        ax5.coastlines()
        ax5.add_feature(cfeature.BORDERS)
        ax5.add_feature(cfeature.STATES)
        
        ax5.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax5.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax5.set_yticklabels(LatLabel, fontsize = 10)
        ax5.set_xticklabels(LonLabel, fontsize = 10)
        
        ax5.xaxis.set_major_formatter(LonFormatter)
        ax5.yaxis.set_major_formatter(LatFormatter)
        
        cfd = ax5.pcolormesh(lon, lat, FD[:,:,s], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
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
        
    
    cbax = fig.add_axes([0.95, 0.35, 0.015, 0.29])
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
    cbar.ax.set_ylabel(clabel, fontsize = 14)
    
    cbaxfd = fig.add_axes([0.07, 0.33, 0.85, 0.015])
    cbarfd = fig.colorbar(cfd, cax = cbaxfd, orientation = 'horizontal')
    cbarfd.ax.set_xlabel(FDlabel, fontsize = 14)
    #fig.tight_layout()
    
    # This will be saved as a pdf so it can be zoomed in without treamendous loss in detail
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
    
    # Create the first plot (1979 - 1986)
    fig, axes = plt.subplots(figsize = [12, 18], nrows = len(GrowSeasons), ncols = 5, 
                             subplot_kw = {'projection': fig_proj})
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.87)
    
    fig.suptitle(title, y = 0.61, size = 18)
    
    for g in range(len(GrowSeasons)):
        ax1 = axes[g,0]; ax2 = axes[g,1]; ax3 = axes[g,2]; ax4 = axes[g,3]; ax5 = axes[g,4]
        
        # Criteria 1 plot
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        
        ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax1.pcolormesh(lon, lat, c1[:,:,g], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        ax1.set_extent([-130, -65, 25, 50])
        
        ax1.set_ylabel(GrowSeasons[g], size = 14, labelpad = 30.0, rotation = 0)
        
        # Criteria 2 plot
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        
        ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax2.pcolormesh(lon, lat, c2[:,:,g], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax2.set_extent([-130, -65, 25, 50])
        
        # Criteria 3 plot
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.STATES)
        
        ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax3.pcolormesh(lon, lat, c3[:,:,g], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        ax3.set_extent([-130, -65, 25, 50])
        
        # Criteria 4 plot
        ax4.coastlines()
        ax4.add_feature(cfeature.BORDERS)
        ax4.add_feature(cfeature.STATES)
        
        ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax4.set_yticklabels(LatLabel, fontsize = 10)
        ax4.set_xticklabels(LonLabel, fontsize = 10)
        
        ax4.xaxis.set_major_formatter(LonFormatter)
        ax4.yaxis.set_major_formatter(LatFormatter)
        
        cs = ax4.pcolormesh(lon, lat, c4[:,:,g], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        ax4.set_extent([-130, -65, 25, 50])
        
        # Flash Drought plot
        ax5.coastlines()
        ax5.add_feature(cfeature.BORDERS)
        ax5.add_feature(cfeature.STATES)
        
        ax5.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax5.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax5.set_yticklabels(LatLabel, fontsize = 10)
        ax5.set_xticklabels(LonLabel, fontsize = 10)
        
        ax5.xaxis.set_major_formatter(LonFormatter)
        ax5.yaxis.set_major_formatter(LatFormatter)
        
        cfd = ax5.pcolormesh(lon, lat, FD[:,:,g], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
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
        
    
    cbax = fig.add_axes([0.07, 0.40, 0.85, 0.015])
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')
    cbar.ax.set_xlabel(clabel, fontsize = 14)
    
    cbaxfd = fig.add_axes([0.07, 0.34, 0.85, 0.015])
    cbarfd = fig.colorbar(cfd, cax = cbaxfd, orientation = 'horizontal')
    cbarfd.ax.set_xlabel(FDlabel, fontsize = 14)
    #fig.tight_layout()
    
    # This will be saved as a pdf so it can be zoomed in without treamendous loss in detail
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    