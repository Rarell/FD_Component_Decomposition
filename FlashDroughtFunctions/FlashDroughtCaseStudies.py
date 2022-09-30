#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:03:09 2020

@author: stuartedris

This is a set of functions designed to create various case study figures for the
"Decomposing the Critical Components of Flash Drought" study.
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
from FlashDroughtFunctions.FlashDroughtStats import *

#%%
# cell 2
######################################
### Beginning of Functions ###########
######################################

#%%
# cell 3
###########################
### SubsetData Function ###
###########################

# Creates a subset of any gridded dataset using longitude and latitude

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
            # Determine how man columns this row has.
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

#%%
# cell 4
############################
### Case Study Functions ###
############################
    
# Functions used to create year by year case study figures. List of figures includes:
# - Monthly Persistence map of drought and flash components and flash drought
# - Cumulative FC/FD maps for each month
# - Drought severity and persistence maps
# - Fractional areal coverage (compared to the total area) time series of FC, DC, and FD (includes the cumultative areal coverage)
# - Statistical maps (correlation coefficient and composite mean difference)
# - Offical case study maps (used in paper) with USDM, SESR identified drought, and FC and FD identified
    
# Monthly maps are for the warm season (MJJAS)


# Calculate monthly means
def MonthAverage(x, year, month, AllYears, AllMonths):
    '''
    This function is designed to intake data on the submonthly timescale (e.g., weekly, pentad, daily, etc.)
    and average it down to a monthly time scale for a given year for every grid point in the data entered.
    
    Inputs:
    - x: The submonthly gridded data. Data is assumed to have the same I, J, T or lat x lon (or lon x lat) x time.
    - year: The year over which the data is being averaged.
    - month: The month over which the data is averaged.
    - AllYears: A list of all year values equal in size to the timescale of x.
    - AllMonths: A list all month values equal in size to the timescale of x.
    
    Outputs:
    - xMean: The monthly averaged grid of the data x.
    '''
    
    # Initialize the temporal mean variable
    I, J, T = x.shape
    xMean   = np.ones((I, J)) * np.nan
    
    # Find all data points corresponding to the desired month
    ind = np.where( (AllYears == year) & (AllMonths == month) )[0]
    
    # Calcualte the monthly mean
    xMean = np.nanmean(x[:,:,ind], axis = -1)
    
    return xMean


# Calculate the cumulative values
def CalculateCumulative(x):
    '''
    This function calculates the cumulative coverage of gridded data x. That is,
    this function takes a gridded system of true/false or numerical values and
    determines whether a given grid point experienced a true/non-zero value for
    a given timestep and previous timesteps. This is used to determine the total
    coverage over a given time range.
    
    Inputs:
    - x: The gridded input data. Data is assumed to have the same I, J, T or lat x lon (or lon x lat) x time.
    
    Outputs:
    - xCumul: The cumulative results of gridded data x.
    '''
    
    # Initialize the cumulative variable
    I, J, T = x.shape
    
    xCumul = np.ones((I, J ,T)) * np.nan
    
    # Reshape for easier loops
    x2d = x.reshape(I*J, T, order = 'F')
    xCumul = xCumul.reshape(I*J, T, order = 'F')
    
    # Assign the initial value
    xCumul[:,0] = x2d[:,0]
    
    # Calculate the cumulative value
    for ij in range(I*J):
        for t in range(1, T):
            if (x2d[ij,t] > 0) & (xCumul[ij,t-1] == 0): # Assign a unique value if this is the first time a non-zero value is found
                xCumul[ij,t] = 2
            elif xCumul[ij,t-1] > 0:
                xCumul[ij,t] = 1
            else:
                xCumul[ij,t] = 0
    
    # Reshape the cumulative value back in to a 3D array
    xCumul = xCumul.reshape(I, J, T, order = 'F')
    
    return xCumul


# Calculate the percent areal coverage
def CalculatePerArea(x, mask):
    '''
    This function takes in a gridded dataset and counts the number of non-zero valued
    grid points. Assuming a 32 km x 32 km grid size, the area coverage is calculated 
    and compared to the total area of the land mass (determined by an identical method
    using the associated land-sea mask [land = 1, sea = 0]). This calculation is 
    performed for every time step in the inputted data.
    
    Inputs:
    - x: The gridded dataset for which the area is calculated. Data is assumed 
         to have the same I, J, T or lat x lon (or lon x lat) x time.
    - mask: The land-sea mask associated with the input data x.
    
    Outputs:
    - perArea: The relative area coverage (percentage) of non-zero values in x compared
               to the total land area. Same size as the time dimension of x.
    '''
    
    # Calculate the total area using the land - sea mask
    LandTot = np.nansum(mask)
    AreaTot = LandTot * 32 * 32 # km^2; NARR grid is on a 32 km x 32 km grid
    
    # Calculate the areal coverage of x (assuming binary values)
    xtmp = np.nansum(x[:,:,:], axis = 0)
    xTot = np.nansum(xtmp[:,:], axis = 0)
    xArea = xTot * 32 * 32
    
    # Calcualte the percent coverage
    perArea = xArea/AreaTot * 100
    
    return perArea


# Plots for the monthly persistence maps
def CaseMonthlyMaps(c2, c4, FD, lon, lat, LonMin, LonMax, LatMin, LatMax, 
                subset, year, cmin, cmax, cint, FDmin, FDmax, title = 'Title', 
                DClabel = 'Drought Component', FClabel = 'Flash Component', FDlabel = 'Flash Drought', savename = 'tmp.png', 
                OutPath = './Figures/'):
    '''
    This function creates some of the early monthly case study maps. This takes in monthly averaged drought
    (c2) data, cumulative rapid intensification (C4) data, and cumulative flash drought data to create monthly
    maps for the growing season (AMJJASO). The resulting figure is saved under a designated name to a designated
    directory.
    
    Inputs:
    - c2: Monthly averaged, gridded drought data.
    - c4: Cumulative, monthly, gridded rapid intensification data.
    - FD: Cumulative, monthly, gridded flash drought data.
    - lon: Associated longitude grid for c2, c4, and FD.
    - lat: Associated latitude grid for c2, c4, and FD.
    - LonMin, LonMax, LatMin, LatMax: Minimum and maximum longitude/latitude 
                              displayed on the maps (i.e., the monthly maps will
                              be displayed on the domain of LonMin, LonMax, LatMin,
                              LatMax).
    - subset: True/False value to determine if the data is for a subset of the U.S. (subset = True)
              or not. This is used to determine specific dimensions and spacing of the maps.
    - year: The year of the given case study (the year the monthly maps are in).
    - cmin, cmax, cint: Minimum and maximum value limits for the drought data and interval 
                        of the displayed data (cint)
    - FDmin, FDmax, FDint: Minimum and maximum value limits for the flash drought data and interval 
                           of the displayed data (FDint)
    - title: Main title of the figure.
    - DClabel, FClabel, FDlabel: The label of the drought, rapid intensification, and flash drought
                                 colorbar labels respectively.
    - savename: The file name of the created figure will be saved as.
    - OutPath: The path from the current directory to the directory the figure will be saved in.
    
    Outputs:
    - None. A figure displaying monthly results will be created and saved.
    '''
    
    # Month names
    # month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_names = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    
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
    
    # Collect shapefile information for the U.S. and other countries
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    ShapeName = 'admin_0_countries'
    CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)
    
    CountriesReader = shpreader.Reader(CountriesSHP)
    
    USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
    NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
    
    # Create the figure
    fig, axes = plt.subplots(figsize = [12, 18], nrows = len(month_names), ncols = 3, 
                             subplot_kw = {'projection': fig_proj})
    
    # Adjust the figure parameters based whether a subset is considered and the year
    if subset is True:
        if year == 1988:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2000:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2002:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.5, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2003:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.45, hspace = 0.1)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2007:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2008:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2011:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2012:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.5, hspace = 0.15)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2016:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.5, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
    else:
        plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.65)
        fig.suptitle(title, y = 0.85, size = 18)
    
    
    for m in range(len(month_names)):
        ax1 = axes[m,0]; ax2 = axes[m,1]; ax3 = axes[m,2]
        
        # Create the first plot (c2; drought)
        
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        # ax1.coastlines()
        # ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 3)
        ax1.add_feature(cfeature.STATES)
        ax1.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 2)
        ax1.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 3)
        
        # Adjust the ticks
        ax1.set_xticks(LonLabel, crs = fig_proj)
        ax1.set_yticks(LatLabel, crs = fig_proj)
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cdc = ax1.pcolormesh(lon, lat, c2[:,:,m], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj, zorder = 1)
        
        # Set the map extent
        ax1.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        # Add month labels for the leftmost plot
        ax1.set_ylabel(month_names[m], size = 14, labelpad = 20.0, rotation = 0)
        
        
        # Create the second plot (c4; rapid intensification)
        
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        # ax2.coastlines()
        # ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'none', zorder = 2)
        ax2.add_feature(cfeature.STATES)
        ax2.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
        ax2.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'none', zorder = 2)
        
        # Adjust the ticks
        ax2.set_xticks(LonLabel, crs = fig_proj)
        ax2.set_yticks(LatLabel, crs = fig_proj)
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cfc = ax2.pcolormesh(lon, lat, c4[:,:,m], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj, zorder = 1)
        
        # Set the map extent
        ax2.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        
        # Create the third plot (flash drought)
        
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        # ax3.coastlines()
        # ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'none', zorder = 2)
        ax3.add_feature(cfeature.STATES)
        ax3.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 2)
        ax3.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'none', zorder = 2)
        
        # Adjust the ticks
        ax3.set_xticks(LonLabel, crs = fig_proj)
        ax3.set_yticks(LatLabel, crs = fig_proj)
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cfd = ax3.pcolormesh(lon, lat, FD[:,:,m], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj, zorder = 1)
        
        # Set the map extent
        ax3.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        if m == 0:
            # For the top row, set the titles
            ax1.set_title('Drought Component (C2)')
            ax2.set_title('Flash Component (C4)')
            ax3.set_title('Flash Drought')
            
            
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
                            labelleft = False, labelright = True)
    
    #######
    #### These need to be toyed with and modified! ###
    ######
    # Set the colorbar parameters (size, location, etc.) based on whether a
    #   subset is considered and year.
    if subset is True:
        if year == 1988:
            cbaxdc = fig.add_axes([0.095, 0.15, 0.25, 0.015])
        
            cbaxfc = fig.add_axes([0.375, 0.15, 0.25, 0.015])
            
            cbaxfd = fig.add_axes([0.655, 0.15, 0.25, 0.015])
        elif year == 2000:
            cbax = fig.add_axes([0.95, 0.09, 0.015, 0.83])
            
            cbaxfd = fig.add_axes([0.09, 0.07, 0.80, 0.015])
        elif year == 2002:
            cbax = fig.add_axes([0.88, 0.095, 0.015, 0.82])
            
            cbaxfd = fig.add_axes([0.170, 0.08, 0.65, 0.015])
        elif year == 2003:
            cbaxdc = fig.add_axes([0.190, 0.07, 0.18, 0.015])
        
            cbaxfc = fig.add_axes([0.410, 0.07, 0.18, 0.015])
            
            cbaxfd = fig.add_axes([0.625, 0.07, 0.18, 0.015])
        elif year == 2007:
            cbax = fig.add_axes([0.95, 0.09, 0.015, 0.83])
            
            cbaxfd = fig.add_axes([0.09, 0.07, 0.80, 0.015])
        elif year == 2008:
            cbax = fig.add_axes([0.95, 0.09, 0.015, 0.83])
            
            cbaxfd = fig.add_axes([0.09, 0.07, 0.80, 0.015])
        elif year == 2011:
            cbaxdc = fig.add_axes([0.095, 0.06, 0.25, 0.015])
            
            cbaxfc = fig.add_axes([0.375, 0.06, 0.25, 0.015])
            
            cbaxfd = fig.add_axes([0.655, 0.06, 0.25, 0.015])
        elif year == 2012:
            cbaxdc = fig.add_axes([0.220, 0.07, 0.16, 0.015])
        
            cbaxfc = fig.add_axes([0.420, 0.07, 0.16, 0.015])
            
            cbaxfd = fig.add_axes([0.620, 0.07, 0.16, 0.015])
        elif year == 2016:
            cbax = fig.add_axes([0.875, 0.09, 0.015, 0.81])
            
            cbaxfd = fig.add_axes([0.15, 0.07, 0.70, 0.015])
    else:
        cbaxdc = fig.add_axes([0.095, 0.15, 0.25, 0.015])
        
        cbaxfc = fig.add_axes([0.375, 0.15, 0.25, 0.015])
        
        cbaxfd = fig.add_axes([0.655, 0.15, 0.25, 0.015])
        
    # Create the colorbar and set the labels
    cbardc = fig.colorbar(cdc, cax = cbaxdc, orientation = 'horizontal')
    cbardc.ax.set_xlabel(DClabel, fontsize = 14)
    
    cbarfc = fig.colorbar(cfc, cax = cbaxfc, orientation = 'horizontal')
    cbarfc.ax.set_xlabel(FClabel, fontsize = 14)
    
    cbarfd = fig.colorbar(cfd, cax = cbaxfd, orientation = 'horizontal')
    cbarfd.ax.set_xlabel(FDlabel, fontsize = 14)
    
    #fig.tight_layout()
    
    # Save the figure
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    
    ################
    ###### Subset == True parameters are untested and may not come out well.
    #####################


# Create the cumulative maps
def CumulativeMaps(FC, FD, lon, lat, LonMin, LonMax, LatMin, LatMax, 
                subset, year, title = 'Title', 
                savename = 'tmp.txt', OutPath = './Figures/'):
    '''
    This function creates monthly cumulative maps for rapid intensification (flash component) and flash drought. 
    This takes in cumulative rapid intensification (C4) data, and cumulative flash drought data to create monthly
    cumulative maps for the growing season (AMJJASO). The resulting figure is saved under a designated name to a designated
    directory.
    
    Inputs:
    - FC: Cumulative, monthly, gridded rapid intensification (flash component) data.
    - FD: Cumulative, monthly, gridded flash drought data.
    - lon: Associated longitude grid for FC, and FD.
    - lat: Associated latitude grid for FC, and FD.
    - LonMin, LonMax, LatMin, LatMax: Minimum and maximum longitude/latitude 
                              displayed on the maps (i.e., the monthly maps will
                              be displayed on the domain of LonMin, LonMax, LatMin,
                              LatMax).
    - subset: True/False value to determine if the data is for a subset of the U.S. (subset = True)
              or not. This is used to determine specific dimensions and spacing of the maps.
    - year: The year of the given case study (the year the monthly maps are in).
    - title: Main title of the figure.
    - savename: The file name of the created figure will be saved as.
    - OutPath: The path from the current directory to the directory the figure will be saved in.
    
    Outputs:
    - None. A figure displaying monthly results will be created and saved.
    '''
    
    # Month names
    # month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_names = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    
    # Color information
    cmin = 0; cmax = 2; cint = 0.5
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "black", "red"], 3)
    
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
    
    # Collect shapefile information for the U.S. and other countries
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    ShapeName = 'admin_0_countries'
    CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)
    
    CountriesReader = shpreader.Reader(CountriesSHP)
    
    USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
    NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
    
    # Create the figure
    fig, axes = plt.subplots(figsize = [12, 18], nrows = len(month_names), ncols = 2, 
                             subplot_kw = {'projection': fig_proj})
    
    # Adjust the figure parameters based whether a subset is considered and the year
    if subset is True:
        if year == 1988:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2000:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2002:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.5, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2003:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.7, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2007:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2008:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2011:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.3, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2012:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.7, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2016:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.5, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
    else:
        plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.15, hspace = 0.15)
        fig.suptitle(title, y = 0.95, size = 18)
    
    
    for m in range(len(month_names)):
        ax1 = axes[m,0]; ax2 = axes[m,1]
        
        # Flash Component plot
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        # ax1.coastlines()
        # ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'none', zorder = 2)
        ax1.add_feature(cfeature.STATES)
        ax1.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 2)
        ax1.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'none', zorder = 2)
        
        # Adjust the ticks
        ax1.set_xticks(LonLabel, crs = fig_proj)
        ax1.set_yticks(LatLabel, crs = fig_proj)
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the cumulative data
        cfc = ax1.pcolormesh(lon, lat, FC[:,:,m], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj, zorder = 1)
        
        # Set the map extent
        ax1.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        # Add month labels for the leftmost plot
        ax1.set_ylabel(month_names[m], size = 14, labelpad = 20.0, rotation = 0)
        
        
        # Flash Drought plot
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        # ax2.coastlines()
        # ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'none', zorder = 2)
        ax2.add_feature(cfeature.STATES)
        ax2.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 2)
        ax2.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'none', zorder = 2)
        
        # Adjust the ticks
        ax2.set_xticks(LonLabel, crs = fig_proj)
        ax2.set_yticks(LatLabel, crs = fig_proj)
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the cumulative data
        cfd = ax2.pcolormesh(lon, lat, FD[:,:,m], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj, zorder = 1)
        
        # Set the map extent
        ax2.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        if m == 0:
            # For the top row, set the titles
            ax1.set_title('Flash Component (C4)')
            ax2.set_title('Flash Drought')
            
            
            # Set some extra tick parameters
            ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                            labelsize = 10, bottom = True, top = True, left = True,
                            right = True, labelbottom = False, labeltop = True,
                            labelleft = True, labelright = False)
            ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
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
                            labelleft = False, labelright = True)
    
    
    #fig.tight_layout()
    
    # Save the figure
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    
    ################
    ###### Some Subset == True parameters are untested and may not come out well.
    #####################
    
    
# Plot the drought severity maps
def DroughtIntensityMaps(DC, DP, DI, lon, lat, LonMin, LonMax, LatMin, LatMax, 
                subset, year, cmin, cmax, cint, title = 'Title', 
                DClabel = 'Drought Component', DPlabel = 'Drought Percentiles', 
                DIlabel = 'Drought Intensity', savename = 'tmp.png', 
                OutPath = './Figures/'):
    '''
    This function creates monthly average drought maps, displaying drought persistence (average c2; 
    relative frequency a grid point was in drought for a given month), drought percentiles (a 
    measure of drought strength), and drought intensity (based USDM categories for percentiles). 
    This takes in monthly averaged drought (drought component) data, monthly averaged drought
    percentiles, and monthly averaged drought intensity based on SESR percentiles and USDM categories, 
    to create monthly maps for the growing season (AMJJASO). The resulting figure is saved under a 
    designated name to a designated directory.
    
    Inputs:
    - DC: Monthly averaged, gridded drought component (criteria 2) data.
    - DP: Monthly averaged, gridded drought percentiles (i.e., percentiles of SESR).
    - DI: Monthly averaged drought intensity data.
    - lon: Associated longitude grid for DC, DP, and DI.
    - lat: Associated latitude grid for DC, DP, and DI.
    - LonMin, LonMax, LatMin, LatMax: Minimum and maximum longitude/latitude 
                              displayed on the maps (i.e., the monthly maps will
                              be displayed on the domain of LonMin, LonMax, LatMin,
                              LatMax).
    - subset: True/False value to determine if the data is for a subset of the U.S. (subset = True)
              or not. This is used to determine specific dimensions and spacing of the maps.
    - year: The year of the given case study (the year the monthly maps are in).
    - cmin, cmax, cint: Minimum and maximum value limits for the DC data and interval 
                        of the displayed data (cint)
    - title: Main title of the figure.
    - DClabel, DPlabel, DIlabel: The label of the drought persistence, drought percentiles, and drought intensity
                                 colorbar labels respectively.
    - savename: The file name of the created figure will be saved as.
    - OutPath: The path from the current directory to the directory the figure will be saved in.
    
    Outputs:
    - None. A figure displaying monthly results will be created and saved.
    '''
    
    # Month names
    # month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_names = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    
    # Set colorbar information
    cmin = cmin; cmax = cmax; cint = cint
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)
    
    DPmin = 0; DPmax = 20; DPint = 1
    DPlevs = np.arange(DPmin, DPmax + DPint, DPint)
    DPnlevs = len(DPlevs)
    DPcmap = plt.get_cmap(name = 'hot', lut = DPnlevs)
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
    
    # Collect shapefile information for the U.S. and other countries
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    ShapeName = 'admin_0_countries'
    CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)
    
    CountriesReader = shpreader.Reader(CountriesSHP)
    
    USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
    NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
    
    # Create the figure
    fig, axes = plt.subplots(figsize = [12, 18], nrows = len(month_names), ncols = 3, 
                             subplot_kw = {'projection': fig_proj})
    
    # Adjust the figure parameters based whether a subset is considered and the year
    if subset is True:
        if year == 1988:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2000:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2002:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.5, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2003:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.45, hspace = 0.1)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2007:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2008:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2011:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2012:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.5, hspace = 0.15)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2016:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.5, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
    else:
        plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.65)
        fig.suptitle(title, y = 0.85, size = 18)
    
    
    for m in range(len(month_names)):
        ax1 = axes[m,0]; ax2 = axes[m,1]; ax3 = axes[m,2]
        
        # Drought Persistence plot
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        # ax1.coastlines()
        # ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'none', zorder = 2)
        ax1.add_feature(cfeature.STATES)
        ax1.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 2)
        ax1.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'none', zorder = 2)
        
        # Adjust the ticks
        ax1.set_xticks(LonLabel, crs = fig_proj)
        ax1.set_yticks(LatLabel, crs = fig_proj)
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cdc = ax1.pcolormesh(lon, lat, DC[:,:,m], vmin = 0, vmax = 1,
                          cmap = cmap, transform = data_proj, zorder = 1)
        
        # Set the extent
        ax1.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        # Add month labels for the leftmost plot
        ax1.set_ylabel(month_names[m], size = 14, labelpad = 20.0, rotation = 0)
        
        
        # Drought Percentile plot
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        # ax2.coastlines()
        # ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'none', zorder = 2)
        ax2.add_feature(cfeature.STATES)
        ax2.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 2)
        ax2.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'none', zorder = 2)
        
        # Adjust the ticks
        ax2.set_xticks(LonLabel, crs = fig_proj)
        ax2.set_yticks(LatLabel, crs = fig_proj)
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cdp = ax2.pcolormesh(lon, lat, DP[:,:,m], vmin = DPmin, vmax = DPmax,
                          cmap = DPcmap, transform = data_proj, zorder = 1)
        
        # Set the extent
        ax2.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        
        # Drought Intensity plot
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        # ax3.coastlines()
        # ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'none', zorder = 2)
        ax3.add_feature(cfeature.STATES)
        ax3.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 2)
        ax3.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'none', zorder = 2)
        
        # Adjust the ticks
        ax3.set_xticks(LonLabel, crs = fig_proj)
        ax3.set_yticks(LatLabel, crs = fig_proj)
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cdi = ax3.pcolormesh(lon, lat, DI[:,:,m], vmin = 0, vmax = 5,
                          cmap = cmap, transform = data_proj, zorder = 1)
        
        # Set the extent
        ax3.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        if m == 0:
            # For the top row, set the titles
            ax1.set_title('Drought Component (C2)')
            ax2.set_title('Drought Percentiles')
            ax3.set_title('Drought Intensity')
            
            
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
                            labelleft = False, labelright = True)
    
    #######
    #### These need to be toyed with and modified! ###
    ######
    # Set the axes and size for the color bars
    if subset is True:
        if year == 1988:
            cbaxdc = fig.add_axes([0.095, 0.15, 0.25, 0.012])
        
            cbaxdp = fig.add_axes([0.375, 0.15, 0.25, 0.012])
            
            cbaxdi = fig.add_axes([0.655, 0.15, 0.25, 0.012])
        elif year == 2000:
            cbax = fig.add_axes([0.95, 0.09, 0.015, 0.83])
            
            cbaxfd = fig.add_axes([0.09, 0.07, 0.80, 0.015])
        elif year == 2002:
            cbax = fig.add_axes([0.88, 0.095, 0.015, 0.82])
            
            cbaxfd = fig.add_axes([0.17, 0.07, 0.65, 0.015])
        elif year == 2003:
            cbaxdc = fig.add_axes([0.190, 0.07, 0.18, 0.015])
        
            cbaxdp = fig.add_axes([0.410, 0.07, 0.18, 0.015])
            
            cbaxdi = fig.add_axes([0.625, 0.07, 0.18, 0.015])
        elif year == 2007:
            cbax = fig.add_axes([0.95, 0.09, 0.015, 0.83])
            
            cbaxfd = fig.add_axes([0.09, 0.07, 0.80, 0.015])
        elif year == 2008:
            cbax = fig.add_axes([0.95, 0.09, 0.015, 0.83])
            
            cbaxfd = fig.add_axes([0.09, 0.07, 0.80, 0.015])
        elif year == 2011:
            cbaxdc = fig.add_axes([0.095, 0.06, 0.25, 0.012])
        
            cbaxdp = fig.add_axes([0.375, 0.06, 0.25, 0.012])
            
            cbaxdi = fig.add_axes([0.655, 0.06, 0.25, 0.012])
        elif year == 2012:
            cbaxdc = fig.add_axes([0.220, 0.07, 0.16, 0.015])
        
            cbaxdp = fig.add_axes([0.420, 0.07, 0.16, 0.015])
            
            cbaxdi = fig.add_axes([0.620, 0.07, 0.16, 0.015])
        elif year == 2016:
            cbaxdc = fig.add_axes([0.875, 0.09, 0.015, 0.81])
            
            cbaxdi = fig.add_axes([0.15, 0.07, 0.70, 0.015])
    else:
        cbaxdc = fig.add_axes([0.095, 0.15, 0.25, 0.012])
        
        cbaxdp = fig.add_axes([0.375, 0.15, 0.25, 0.012])
        
        cbaxdi = fig.add_axes([0.655, 0.15, 0.25, 0.012])
        
    # Drought Duration colorbar
    cbardc = fig.colorbar(cdc, cax = cbaxdc, orientation = 'horizontal')
    for i in cbardc.ax.xaxis.get_ticklabels():
        i.set_size(16)
    cbardc.ax.set_xlabel(DClabel, fontsize = 16)
    
    
    # Drought Percentiles colorbar
    cbardp = fig.colorbar(cdp, cax = cbaxdp, orientation = 'horizontal')
    for i in cbardp.ax.xaxis.get_ticklabels():
        i.set_size(16)
    cbardp.ax.set_xlabel(DPlabel, fontsize = 16)
    
    
    # Drought Intensity colorbar
    cbardi = fig.colorbar(cdi, cax = cbaxdi, orientation = 'horizontal')
    
    cbardi.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
    cbardi.ax.set_xticklabels(['Nope', 'D1', 'D2', 'D3', 'D4'], size = 16)
    cbardi.ax.set_xlabel(DIlabel, fontsize = 16)
    
    #fig.tight_layout()
    
    # Save the figure
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    
    ################
    ###### Some Subset == True parameters are untested and may not come out well.
    #####################
    
    
# Plot the percent areal coverage
def PlotPerArea(perDC, perDCcumul, perFC, perFCcumul, perFD, perFDcumul, AllDates, year, cumulative = True):
    '''
    This function creates a set of time series plot of the percent of area covered
    in drought, rapid intensification (flash component), and flash drought for a given year.
    If desired, this function will also plot the cumulative area covered by the given
    variables. At the end the figure is saved to a specified directory.
    
    Inputs:
    - perDC: The percent of area covered by drought (drought component) time series.
    - perDCcumul: The cumulative The percent of area covered by drought (drought component) time series.
    - perFC: The percent of area covered by rapid intensification (flash component) time series.
    - perFCcumul: The cumulative percent of area covered by rapid intensification (flash component) time series.
    - perFD: The percent of area covered by flash drought time series.
    - perFDcumul: The cumulative percent of area covered by flash drought time series.
    - AllDates: A 1D array of datetimes containing associated time stamps for each
                time step in the above variables.
    - year: The year of case study (the year this data occurred in).
    - cumulative: A boolean True/False value indicating whether to plot the cumulative time series.
    
    Outputs:
    - None. A figure displaying the time series will be created and saved.
    '''
    
    # Find all the indices for the growing season only
    ind = np.where( (AllDates >= datetime(year, 4, 1)) & (AllDates <= datetime(year, 10, 31)) )[0]
    
    # Format for date labels (x-axis tick labels)
    DateFMT = DateFormatter('%b')
    
    # Create the figure
    fig = plt.figure(figsize = [14, 10])
    ax1 = fig.add_subplot(2, 1, 1)
    
    # Set the title
    ax1.set_title('Percentage of Flash Drought Component Coverage for ' + str(year), size = 18)
    
    # Plot the drought time series
    ax1.plot(AllDates[ind], perDC[ind], 'r-', label = 'Drought (C2)')
    
    # Plot the cumulative time series if applicable. Note this requires the
    #  the limits to be adjusted.
    if cumulative is True:
        ax1.plot(AllDates[ind], perDCcumul[ind], 'r--', label = 'Cumulative Drought (C2)')
        
        ax1.legend(loc = 'upper right', fontsize = 16)
        
        ax1.set_ylim([0, np.nanmax(np.ceil(perDCcumul[ind]))])
    else:
        ax1.set_ylim([0.0, np.ceil(np.nanmax(perDC[ind]))])
    
    # Set the axis drought labels
    ax1.set_xlabel('Time', size = 16)
    ax1.set_ylabel('Percent Areal Coverage (%)', size = 16)
    
    # Adjust the ticks
    ax1.xaxis.set_major_formatter(DateFMT)
    # ax1.set_ylim([0.0, 80.0])
    
    for i in ax1.xaxis.get_ticklabels() + ax1.yaxis.get_ticklabels():
        i.set_size(16)
    
    # Create the FC/FD time series plot
    ax2 = fig.add_subplot(2, 1, 2)
    
    # Plot the FC/FC time series
    # And plot the cumulative time series if applicable. Note this requires the
    #  the limits to be adjusted.
    ax2.plot(AllDates[ind], perFC[ind], 'b-', label = 'Rapid Drying (C4)')
    if cumulative is True:
        ax2.plot(AllDates[ind], perFCcumul[ind], 'b--', label = 'Cumulative Rapid Drying (C4)')
    else:
        pass
    
    ax2.plot(AllDates[ind], perFD[ind], 'k-', label = 'Flash Drought')
    if cumulative is True:
        ax2.plot(AllDates[ind], perFDcumul[ind], 'k--', label = 'Cumulative Flash Drought')
        
        ax2.set_ylim([0, np.ceil(np.nanmax(perFCcumul[ind]))])
    else:
        ax2.set_ylim([0, np.ceil(np.nanmax(perFC[ind]))])
    
    # Add a legend for the FD/FC
    ax2.legend(loc = 'upper right', fontsize = 16)
    
    # Set the axis labels
    ax2.set_xlabel('Time', size = 16)
    ax2.set_ylabel('Percent Areal Coverage (%)', size = 16)
    
    # Adjust the ticks
    ax2.xaxis.set_major_formatter(DateFMT)
    # ax2.set_ylim([0, 0.8])
    
    for i in ax2.xaxis.get_ticklabels() + ax2.yaxis.get_ticklabels():
        i.set_size(16)
    
    # Save the figure
    plt.savefig('./Figures/' + 'PerAreaCoverage-' + str(year) + '.png', bbox_inches = 'tight')
    plt.show(block = False)


# Plot the case study maps
def CaseStudyMaps(USDMDates, DI, FCFD, lon, lat, LonMin, LonMax, LatMin, LatMax, 
                subset, year, title = 'Title', 
                savename = 'tmp.png', OutPath = './Figures/'):
    '''
    This function creates the final case study maps used in paper. This takes in USDM data, monthly averaged drought
    (c2) data, rapid intensification (C4) data, and flash drought data to create monthly
    maps for the warm season (MJJAS). Note the USDM map plotted is the last map the USDM produced for the given
    month. The resulting figure is saved under a designated name to a designated directory.
    
    NOTE: These figures were modified while the paper was being edited to include the USDM. As a result, 
    this function will not run for years where there is no USDM data (prior to 2010, 2003 excepting). This 
    will need to be reverted to an earlier form (showing DC in column 1, FC in column 2, and FD in column 
    3) to run for years without USDM data. Also note the individual USDM data files are located in a seperate
    directory and repository from this project, titled "USDM_Data_Collection".
    
    Inputs:
    - USDMDates: List of dates associated with the USDM map to be plotted. List should be in
                 the form %Y%m%d.
    - DI: Monthly averaged, gridded drought intensity data based on SESR percentiles
          and the USDM classification system.
    - FCFD: Gridded, combined rapid intensification (flash component) and flash drought data.
            Note flash drought occurs where there is rapid intensification. Rapid intensification
            without flash drought is given a unique value.
    - lon: Associated longitude grid for DI and FCFD.
    - lat: Associated latitude grid for DI and FCFD.
    - LonMin, LonMax, LatMin, LatMax: Minimum and maximum longitude/latitude 
                              displayed on the maps (i.e., the monthly maps will
                              be displayed on the domain of LonMin, LonMax, LatMin,
                              LatMax).
    - subset: True/False value to determine if the data is for a subset of the U.S. (subset = True)
              or not. This is used to determine specific dimensions and spacing of the maps.
    - year: The year of the given case study (the year the monthly maps are in).
    - title: Main title of the figure.
    - savename: The file name of the created figure will be saved as.
    - OutPath: The path from the current directory to the directory the figure will be saved in.
    
    Outputs:
    - None. A figure displaying monthly results will be created and saved.
    '''
    
    # Month names
    # month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_names = ['May', 'Jun', 'Jul', 'Aug', 'Sep']
    
    # Set colorbar information
    
    # USDM color information
    USDMmin = -1; USDMmax = 5; USDMint = 1
    USDMlevs = np.arange(USDMmin, USDMmax + USDMint, USDMint)
    USDMnlevs = len(USDMlevs)
    cmap_list = ['white', '#FFFF00', '#FCD37F', '#FFAA00', '#E60000', '#730000']
    USDMnorm = mcolors.Normalize(vmin = USDMmin, vmax = USDMmax)
    USDMcmap = mcolors.LinearSegmentedColormap.from_list('USDMcmap', cmap_list, N = 6)
    
    # Drought component color information
    DCmin = 0; DCmax = 4; DCint = 1
    DClevs = np.arange(DCmin, DCmax + DCint, DCint)
    DCnlevs = len(DClevs)
    DCcmap = mcolors.LinearSegmentedColormap.from_list('DMcmap', ['white', '#FCD37F', '#FFAA00', '#E60000', '#730000'], N = 5)    
    
    # Rapid intensification color information
    FCcmin = 0; FCcmax = 2; FCcint = 0.5
    FCclevs = np.arange(FCcmin, FCcmax + FCcint, FCcint)
    FCnlevs = len(FCclevs)
    FCcmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "black", "red"], 3)
    
    # Projection information
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
    fig, axes = plt.subplots(figsize = [12, 18], nrows = len(month_names), ncols = 3, 
                             subplot_kw = {'projection': fig_proj})
    
    # Adjust the figure parameters based whether a subset is considered and the year
    if subset is True:
        if year == 1988:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
            lat_int = 10
            lon_int = 20
            
        elif year == 2000:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
            lat_int = 10
            lon_int = 20
            
        elif year == 2002:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.5, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
            lat_int = 10
            lon_int = 20
            
        elif year == 2003:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.15, hspace = 0.1)
            fig.suptitle(title, y = 0.945, size = 18)
            
            lat_int = 5
            lon_int = 10
            
        elif year == 2007:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
            lat_int = 10
            lon_int = 20
            
        elif year == 2008:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
            lat_int = 10
            lon_int = 20
            
        elif year == 2011:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.7)
            fig.suptitle(title, y = 0.810, size = 18)
            
            lat_int = 5
            lon_int = 10
            
        elif year == 2012:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.15)
            fig.suptitle(title, y = 0.945, size = 18)
            
            lat_int = 5
            lon_int = 10
            
        elif year == 2016:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.5, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
            lat_int = 10
            lon_int = 20
            
    else:
        plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.80)
        fig.suptitle(title, y = 0.765, size = 18)
        
        lat_int = 10
        lon_int = 20
        
    # Lonitude and latitude tick information
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    
    
    for m in range(len(month_names)):
        ax1 = axes[m,0]; ax2 = axes[m,1]; ax3 = axes[m,2]
        
        # USDM plot
        # ax1.coastlines()
        # ax1.add_feature(cfeature.BORDERS)
        
        # Collect the USDM polygon data
        path = '../USDM_Data_Collection/USDM_Data/' + str(year) + '_USDM_M/'
        DirectoryName = 'USDM_' + str(USDMDates[m+1]) + '_M/'
        SHPname = 'USDM_' + str(USDMDates[m+1])
        
        # Plot the USDM polygons
        csf = shpreader.Reader(path + DirectoryName + SHPname)
        
        recs = [rec for rec in csf.records()]
        geoms = [geom for geom in csf.geometries()]
        
        
        for (rec, geo) in zip(recs, geoms):
            DM = rec.attributes['DM']#.decode('latin-1')
            
            if DM == 0:
                #facecolor = (1, 1, 0, 0)
                facecolor = '#FFFF00' 
                order = 1
            elif DM == 1:
                #facecolor = (0.988, 0.827, 0.498, 0.012)
                facecolor = '#FCD37F'
                order = 2
            elif DM == 2:
                #facecolor = (1, 0.667, 0, 0)
                facecolor = '#FFAA00'
                order = 3
            elif DM == 3:
                #facecolor = (0.913, 0, 0, 0.098)
                facecolor = '#E60000'
                order = 4
            elif DM == 4:
                #facecolor = (0.451, 0, 0, 0.549)
                facecolor = '#730000'
                order = 5
            cusdm = ax1.add_geometries([geo], ccrs.PlateCarree(), facecolor = facecolor, edgecolor = facecolor, zorder = 1)
        
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        ax1.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
        ax1.add_feature(cfeature.STATES)
        ax1.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
        ax1.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
        
        # Adjust the ticks
        ax1.set_xticks(LonLabel, crs = fig_proj)
        ax1.set_yticks(LatLabel, crs = fig_proj)
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        # Set the map extent
        ax1.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        ax1.set_ylabel(month_names[m], size = 14, labelpad = 20.0, rotation = 0)
        
        # Drought Persistence plot
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        # ax2.coastlines()
        # ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
        ax2.add_feature(cfeature.STATES)
        ax2.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
        ax2.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
        
        # Adjust the ticks
        ax2.set_xticks(LonLabel, crs = fig_proj)
        ax2.set_yticks(LatLabel, crs = fig_proj)
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cdi = ax2.pcolormesh(lon, lat, DI[:,:,m+1], vmin = 0, vmax = 5,
                          cmap = DCcmap, transform = data_proj, zorder = 1)
        
        # Set the map extent
        ax2.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        
        # Drought Percentile plot
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        # ax3.coastlines()
        # ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
        ax3.add_feature(cfeature.STATES)
        ax3.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
        ax3.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
        
        # Adjust the ticks
        ax3.set_xticks(LonLabel, crs = fig_proj)
        ax3.set_yticks(LatLabel, crs = fig_proj)
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cfc = ax3.pcolormesh(lon, lat, FCFD[:,:,m+1], vmin = 0, vmax = 2,
                          cmap = FCcmap, transform = data_proj, zorder = 1)
        
        # Set the map extent
        ax3.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        
        if m == 0:
            # For the top row, set the titles
            ax1.set_title('USDM')
            ax2.set_title('Drought Intensity')
            ax3.set_title('RI (C4) and FD')
            
            
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
                            labelleft = False, labelright = True)
    
    #######
    #### These need to be toyed with and modified! ###
    ######
    # Set the axes and size for the color bars
    if subset is True:
        if year == 1988:
            cbax = fig.add_axes([0.095, 0.25, 0.25, 0.015])
        elif year == 2000:
            cbax = fig.add_axes([0.095, 0.25, 0.25, 0.015])
        elif year == 2002:
            cbax = fig.add_axes([0.095, 0.25, 0.25, 0.015])
        elif year == 2003:
            cbax = fig.add_axes([0.095, 0.07, 0.25, 0.015])
            
            cbax2 = fig.add_axes([0.3765, 0.07, 0.25, 0.015])
            
            cbax3 = fig.add_axes([0.6540, 0.07, 0.25, 0.015])
        elif year == 2007:
            cbax = fig.add_axes([0.095, 0.25, 0.25, 0.015])
        elif year == 2008:
            cbax = fig.add_axes([0.095, 0.25, 0.25, 0.015])
        elif year == 2011:
            cbax = fig.add_axes([0.095, 0.195, 0.25, 0.015])
            
            cbax2 = fig.add_axes([0.375, 0.195, 0.25, 0.015])
            
            cbax3 = fig.add_axes([0.650, 0.195, 0.25, 0.015])
        elif year == 2012:
            cbax = fig.add_axes([0.100, 0.07, 0.25, 0.015])
            
            cbax2 = fig.add_axes([0.375, 0.07, 0.25, 0.015])
            
            cbax3 = fig.add_axes([0.650, 0.07, 0.25, 0.015])
        elif year == 2016:
            cbax = fig.add_axes([0.095, 0.25, 0.25, 0.015])
    else:
        cbax = fig.add_axes([0.095, 0.25, 0.25, 0.015])
        
        cbax2 = fig.add_axes([0.375, 0.25, 0.25, 0.015])
        
        cbax3 = fig.add_axes([0.650, 0.25, 0.25, 0.015])
    
    # USDM colorbar 
    cbarusdm = mcolorbar.ColorbarBase(ax = cbax, cmap = USDMcmap, norm = USDMnorm, boundaries = USDMlevs, orientation = 'horizontal')
    # print(cmap_list)
    # print(USDMcmap._segmentdata)
    
    cbarusdm.set_ticks([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], update_ticks = True)
    cbarusdm.ax.set_xticklabels(['ND', 'D0', 'D1', 'D2', 'D3', 'D4'])
    cbarusdm.ax.tick_params(labelsize = 16)
    # cbarusdm.update_ticks()
    
    # Drought intensity colorbar
    cbardi = fig.colorbar(cdi, cax = cbax2, orientation = 'horizontal')
    
    cbardi.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
    cbardi.ax.set_xticklabels(['ND', 'D1', 'D2', 'D3', 'D4'], fontsize = 16)
    
    # FC and FD colorbar
    cbarfcfd = fig.colorbar(cfc, cax = cbax3, orientation = 'horizontal')
    
    cbarfcfd.set_ticks([2/3+1/3, 4/3+1/3])
    cbarfcfd.ax.set_xticklabels(['RI', 'FD'], fontsize = 16)
    
    #fig.tight_layout()
    
    # Save the figure
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    
    ################
    ###### Some Subset == True parameters are untested and may not come out well.
    #####################


#########################
### CaseStudyFunction ###
#########################
    
# Function that calls calculations and creates the maps

def CaseStudy(DC, FC, FD, DI, DP, year, lat, lon, AllDates, AllYears, AllMonths, USDMDates, USDMYears, USDMMonths, mask, subset = False):
    '''
    This is the main function used to produce the case studies in the Flash Drought
    Decomposition paper. This function takes in the criteria data (criteria 2 and 4)
    from Christian et al. (2019), identified flash drought, drought intensity, and SESR
    percentiles and performs calculations for a given year and produces several figures
    to examine to provide examples and illustrations and information for a given year.
    Figures produced include a drought intensity map to examine the evolution of drought
    during the growing season of a given year, cumulative rapid intensification and flash
    drought maps across the growing season, drought, flash drought, and rapid intensification
    areal coverage across the growing season, statistical correlation and composite mean
    difference between rapid intensification and flash drought for the given year, and
    USDM with drought intensifity, and rapid intensification and flash drought across
    the growing season.
    
    Inputs:
    - DC: The gridded drought component (criteria 2).
    - FC: The gridded rapid intensification or flash component (criteria 4).
    - FD: The gridded flash drought identified data.
    - DI: The gridded drought intensity based on SESR percentiles and USDM classification.
    - DP: The gridded drought percentiles (i.e., SESR percentiles).
    - year: The year the case study focuses on.
    - lat: The gridded latitude data of the above variables.
    - lon: The gridded longitude data of the above variables.
    - AllDates: A 1D array of datetimes give time stamps for all the time steps in the
                above variables.
    - AllYears: A 1D array giving the year for all time steps in the above variables.
    - AllMonths: A 1D array giving the month for all time steps in the above variables.
    - USDMDates: A 1D array giving the date USDM maps were released.
    - USDMYears: A 1D array giving the year for each time step in USDMDates.
    - USDMMonths: A 1D array giving the month for each time step in USDMDates.
    - mask: The land-sea mask associated with the main variables (DC, FC, FD, DI, and DP).
    - subset: A boolean True/False value indicating whether to examine the entire contigous 
              U.S. or a subset of it.
              
    Outputs:
    - None. Numerous figures are made and saved.
    '''
    
    # Determine the max/min lat/lon for a flash drought area for a given year
    if subset is True:
        if year == 1988:
            LonMin = -115
            LonMax = -80
            LatMin = 30
            LatMax = 50
            
        elif year == 2000:
            LonMin = -105
            LonMax = -70
            LatMin = 28
            LatMax = 45
            
        elif year == 2002:
            LonMin = -115
            LonMax = -85
            LatMin = 30
            LatMax = 50
            
        elif year == 2003:
            LonMin = -107
            LonMax = -83
            LatMin = 30
            LatMax = 50
            
        elif year == 2007:
            LonMin = -100
            LonMax = -70
            LatMin = 27
            LatMax = 45
            
        elif year == 2008:
            LonMin = -100
            LonMax = -70
            LatMin = 27
            LatMax = 45
            
        elif year == 2011:
            LonMin = -110
            LonMax = -78
            LatMin = 25
            LatMax = 42
            
        elif year == 2012:
            LonMin = -105
            LonMax = -82
            LatMin = 30
            LatMax = 50
            
        elif year == 2016:
            LonMin = -90
            LonMax = -67
            LatMin = 30
            LatMax = 47
        
        # Use default (full CONUS) if no major drought year is specified    
        else:
            LonMin = -130
            LonMax = -65
            LatMin = 25
            LatMax = 50
        
    else:
        LonMin = -130
        LonMax = -65
        LatMin = 25
        LatMax = 50
        
    # Subset the data to the flash drought area
    DCsub, LatSub, LonSub = SubsetData(DC, lat, lon, LatMin, LatMax, LonMin-3, LonMax)
    FCsub, LatSub, LonSub = SubsetData(FC, lat, lon, LatMin, LatMax, LonMin-3, LonMax)
    FDsub, LatSub, LonSub = SubsetData(FD, lat, lon, LatMin, LatMax, LonMin-3, LonMax)
    
    DIsub, LatSub, LonSub = SubsetData(DI, lat, lon, LatMin, LatMax, LonMin-3, LonMax)
    DPsub, LatSub, LonSub = SubsetData(DP, lat, lon, LatMin, LatMax, LonMin-3, LonMax)
    
    MaskSub, LatSub, LonSub = SubsetData(mask, lat, lon, LatMin , LatMax, LonMin-3, LonMax) 
    
    # With everything subsetted, obtain the 2D mask.
    # This is used for masking purely 2D data (3D would incur a size error, but a 2D subset for a 3D variable will also incur a size error)
    Mask2D = MaskSub[:,:,0]
    
    # find the array size for later use.
    I, J, T = FDsub.shape
    
    # Find the data corresponding to the specified year and for the warm season only
    ind = np.where( (AllYears == year) & (AllMonths >= 4) & (AllMonths <= 10) )[0]
    YearDates = AllDates[ind]
    
    months = np.unique(AllMonths)
    monT = len(months)-5 # -5 because of the exclusion of non-growing months
    
    
    # Initialize some vairables
    DCyear = DCsub[:,:,ind]
    FCyear = FCsub[:,:,ind]
    FDyear = FDsub[:,:,ind]
    
    MonDC = np.ones((I, J, monT)) * np.nan
    MonFC = np.ones((I, J, monT)) * np.nan
    MonFD = np.ones((I, J, monT)) * np.nan
    
    MonUSDM =['tmp']*monT
    
    FCFDMon = np.ones((I, J, monT)) * np.nan
    
    # Calculate the monthly means (persistence values)
    for m in range(len(months)-5):
        MonDC[:,:,m] = MonthAverage(DCsub, year, m+4, AllYears, AllMonths) # m+4 starts in the growing months (April to October)
        MonFC[:,:,m] = MonthAverage(FCsub, year, m+4, AllYears, AllMonths)
        MonFD[:,:,m] = MonthAverage(FDsub, year, m+4, AllYears, AllMonths)
        
        # Identify the last day in the month with a USDM map
        usdmInd = np.where( (USDMYears == year) & (USDMMonths == m+4) )[0]
        
        MonUSDM[m] = USDMDates[usdmInd[-1]].strftime('%Y%m%d')
    
    # Calculate the cumulative values for all time points
    DCcumul = CalculateCumulative(DCyear)
    FCcumul = CalculateCumulative(FCyear)
    FDcumul = CalculateCumulative(FDyear)
    
    # Calculate the monthly cumulative values
    DCMon = np.where(MonDC == 0, MonDC, 1) # Turns non-zero values to 1 (component is true in that month)
    FCMon = np.where(MonFC == 0, MonFC, 1)
    FDMon = np.where(MonFD == 0, MonFD, 1)
    
    FCFDMon = FCMon
    FCFDMon = np.where( (FDMon == 0) | (np.isnan(FDMon)), FCFDMon, 2)
    FCFDMon = np.where(MaskSub == 1, FCFDMon, np.nan) # Reapply the mask
    
    DCcumulMon = CalculateCumulative(DCMon)
    FCcumulMon = CalculateCumulative(FCMon)
    FDcumulMon = CalculateCumulative(FDMon)
    
    # The above commands removes masked values. Re-add them
    DCcumul = np.where(MaskSub == 1, DCcumul, np.nan)
    FCcumul = np.where(MaskSub == 1, FCcumul, np.nan)
    FDcumul = np.where(MaskSub == 1, FDcumul, np.nan)
    
    DCMon = np.where(MaskSub == 1, DCMon, np.nan)
    FCMon = np.where(MaskSub == 1, FCMon, np.nan)
    FDMon = np.where(MaskSub == 1, FDMon, np.nan)
    
    DCcumulMon = np.where(MaskSub == 1, DCcumulMon, np.nan)
    FCcumulMon = np.where(MaskSub == 1, FCcumulMon, np.nan)
    FDcumulMon = np.where(MaskSub == 1, FDcumulMon, np.nan)
    
    # Calculate the monthly drought intensity
    MonDI = np.ones((I, J, monT)) * np.nan
    MonDP = np.ones((I, J, monT)) * np.nan
    
    for m in range(len(months)-5):
        MonDI[:,:,m] = MonthAverage(DIsub, year, m+4, AllYears, AllMonths)
        MonDP[:,:,m] = MonthAverage(DPsub, year, m+4, AllYears, AllMonths)
    
    
    # Calculate relative areal coverage of drought, flash component, and flash drought
    perDC = CalculatePerArea(DCyear, Mask2D)
    perFC = CalculatePerArea(FCyear, Mask2D)
    perFD = CalculatePerArea(FDyear, Mask2D)
    
    # Make all true values (1 and 2 alike) 1 for area calculation
    DCcumul = np.where( (DCcumul == 0) | (np.isnan(DCcumul)), DCcumul, 1) 
    FCcumul = np.where( (FCcumul == 0) | (np.isnan(FCcumul)), FCcumul, 1)
    FDcumul = np.where( (FDcumul == 0) | (np.isnan(FDcumul)), FDcumul, 1)
    
    # Calculate relative areal coverage of drought, flash component, and flash drought with cumulative data
    perDCcumul = CalculatePerArea(DCcumul, Mask2D)
    perFCcumul = CalculatePerArea(FCcumul, Mask2D)
    perFDcumul = CalculatePerArea(FDcumul, Mask2D)
    
    
    # Calculate the correlation coefficient and statistical significance for FC and FD for the year.
    # Useful for identifying areas where they may be flash, but no drought
    r, rPval = CorrCoef(FDyear, FCyear)
    
    # Calculate the composite mean difference and statistical significance for FC and FD for the year.
    # Useful for identifying areas where they may be flash, but no drought
    CompDiff, CompDiffPval = CompositeDifference(FDyear, FCyear)
    
    # Remove points to p-values that may have been assigned to the oceans (percentile of score is 0 over oceans due to comparison with a time serise of NaNs)
    rPval = np.where(Mask2D == 1, rPval, np.nan)
    CompDiffPval = np.where(Mask2D == 1, CompDiffPval, np.nan)
    
    # Plot the monthly means (persistence maps)
    CaseMonthlyMaps(MonDC, MonFC, MonFD, LonSub, LatSub, LonMin, LonMax, LatMin-1, LatMax-1, subset = subset, year = year, 
                    cmin = 0, cmax = 1, cint = 0.1, FDmin = 0.0, FDmax = 0.2, title = 'Average Flash and Drought Components for ' + str(year), 
                    DClabel = 'Drought Component', FClabel = 'Flash Component', FDlabel = 'Flash Drought',
                    savename = 'FD_Component_panel_map_' + str(year) + '_pentad.png')
    
    # Plot the cumulative FC/FD maps
    CumulativeMaps(FCcumulMon, FDcumulMon, LonSub, LatSub, LonMin, LonMax, LatMin-1, LatMax-1, subset = subset, year = year,
                   title = 'Cumulative Flash Component and Flash Drought for ' + str(year), savename = 'cumulative_FD_for_' + str(year) + '.png')
    
    # Plot the drought intensity maps
    DroughtIntensityMaps(MonDC, MonDP, MonDI, LonSub, LatSub, LonMin, LonMax, LatMin-1, LatMax-1, subset = subset, year = year, 
                         cmin = 0, cmax = 5, cint = 1, title = 'Drought Severity for ' + str(year), 
                         DClabel = 'Drought Component', DIlabel = 'Drought Intensity', savename = 'Drought-Severity-' + str(year) + '.png')
    
    # Plot the percentage area coverage maps
    if subset is True:
        PlotPerArea(perDC, perDCcumul, perFC, perFCcumul, perFD, perFDcumul, YearDates, year, cumulative = True)
    else:
        PlotPerArea(perDC, perDCcumul, perFC, perFCcumul, perFD, perFDcumul, YearDates, year, cumulative = False)
    
    # Plot the statistical maps
    PlotStatMap(r, LonSub, LatSub, title = 'Correlation between Flash Drought and Flash Component for ' + str(year), pval = rPval,
                savename = 'CorrCoef-' + str(year) + '.png')

    PlotStatMap(CompDiff, LonSub, LatSub, title = 'Composite Mean Difference between Flash Component and Flash Drought for ' + str(year), pval = CompDiffPval,
                cmin = -0.05, cmax = 0.05, cint = 0.01, RoundDigit = 2, savename = 'CompMeanDiff-' + str(year) + '.png')
    
    # Plot the case study maps used in the paper figures
    # CaseStudyMaps(MonDI, FCMon, FDMon, LonSub, LatSub, LonMin, LonMax, LatMin-1, LatMax-1, subset = subset, year = year, 
    #                      title = 'Case Study for May - September ' + str(year), savename = 'CaseStudy-' + str(year) + '.png')
    CaseStudyMaps(MonUSDM, MonDI, FCFDMon, LonSub, LatSub, LonMin, LonMax, LatMin-1, LatMax-1, subset = subset, year = year, 
                         title = 'Case Study for May - September ' + str(year), savename = 'CaseStudy-' + str(year) + '.png')
    
    # print('r')
    # print(r)
    # print('r p-value')
    # print(rPval)
    # print('Composite difference')
    # print(CompDiff)
    # print('Composite difference p-values')
    # print(CompDiffPval)
    # print('C2 cumulative areal percentage')
    # print(perDCcumul)
    # print('C4 cumulative areal percentage')
    # print(perFCcumul)
    # print('FD cumulative areal percentage')
    # print(perFDcumul)
#%%
# cell 5
######################
### DIMap Function ###
######################
    
# Create a map for drought intensity for comparison with Drought Monitor maps
def DIMap(DI, DMDate, lat, lon, AllDates, savename = 'tmp.png', OutPath = './Figures/'):
    '''
    This function creates a single map using of the drought intensity (found from
    SESR percentiles), based on a 2 week average calculated in this function, for 
    quick comparisons to the USDM. The created figure is then saved.
    
    Inputs:
    - DI: The gridded drought intensity data.
    - DMDate: The date for which the DI is plotted. Note the date should be a date
              that the USDM released a drought map.
    - lat: The gridded latitude data corresponding to DI.
    - lon: The gridded longitude data corresponding to DI.
    - AllDates: An array of datetimes containing time stamps for each time step in DI.
    - savename: The filename the figure will be saved as.
    - OutPath: The path from the current directory to the directory the figure will
               be saved in.
               
    Outputs:
    - None. The created figure is saved.
    '''
    
    # Determine the end date (use specified) and start date
    DeltaDays = timedelta(days = 14)
    
    StartDate = DMDate - DeltaDays
    
    # Find all pentads between start and end dates
    ind = np.where( (AllDates >= StartDate) & (AllDates <= DMDate) )[0]
    
    # Determine the mean drought intensity within the date range
    DImean = np.nanmean(DI[:,:,ind], axis = -1)
    
    # Define the date format
    DateFormat = '%Y-%m-%d'
    
    
    ### Create the map ###
    # Lonitude and latitude tick information
    lat_int = 10
    lon_int = 10
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    
    # Color information
    cmin = 0; cmax = 5; cint = 1
    clevs = np.arange(cmin, cmax, cint)
    nlevs = len(clevs)
    cmap = plt.get_cmap(name = 'hot_r', lut = nlevs)
    
    # Projection information
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
    fig.suptitle('Drought Map for ' + StartDate.strftime(DateFormat) + ' to ' + DMDate.strftime(DateFormat), y = 0.68, size = 28)
    
    ax = fig.add_subplot(1, 1, 1, projection = fig_proj)
    
    # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
    # ax.coastlines()
    # ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
    ax.add_feature(cfeature.STATES)
    ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
    ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
    
    # Adjust the ticks
    ax.set_xticks(LonLabel, crs = fig_proj)
    ax.set_yticks(LatLabel, crs = fig_proj)
    
    ax.set_yticklabels(LatLabel, fontsize = 22)
    ax.set_xticklabels(LonLabel, fontsize = 22)
    
    ax.xaxis.set_major_formatter(LonFormatter)
    ax.yaxis.set_major_formatter(LatFormatter)
    
    # Plot the data
    cs = ax.pcolormesh(lon, lat, DImean, vmin = cmin, vmax = cmax,
                     cmap = cmap, transform = data_proj, zorder = 1)
    
    # Set the map extent
    ax.set_extent([-129, -65, 25-1.5, 50-1.5])
    
    # Set the colorbar location and size
    cbax = fig.add_axes([0.12, 0.29, 0.78, 0.015])
    
    # Create the colorbar
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')
    
    # Adjsut the colorbar ticks and ticklabels
    cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
    cbar.ax.set_xticklabels(['No Drought', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], size = 18)
        
    # Save the figure
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
#%%
# cell 6
##################################
### Early Case Study Functions ###
##################################
    
# Early case studies (when Criteria 1 and 4 were used for the flash component). For a given year,
# Create maps of the flash component, drought component, and flash drought, and time series of areal coverage.

# Note this is an early version that is no longer used. Comments may be light as a result.

########
### Note much of the codework is unrefined and could use some tweeking and cleaning
###########
    
# First calculate the early version of FC and DC
def CalculateEarlyFCandDC(c1, c2, c4):
    '''
    This function determines the flash and drought components using the criteria
    from Christian et al. (2019). Note this function was made before criteria 1
    was incorporated into criteria 4 as part of the algorithm (which rendered this
    function unnecessary).
    
    Inputs:
    - c1: The criteria 1 data.
    - c2: The criteria 2 data.
    - c4: The criteria 4 data.
    
    Outputs:
    - FC: The flash component of flash drought (here defined as when criteria 1 and 4 are true).
    - DC: The drought component of flash drought (here defined as criterion 2).
    '''
    
    # Determine the size of the data and initialize variables
    I, J, T = c2.shape
    
    FC = np.ones((I, J, T)) * np.nan
    DC = np.ones((I, J, T)) * np.nan
    
    # Define the drought component (criteria 2) and the flash component (intersection of criteria 1 and 4)
    for i in range(I):
        print(i/I)
        for j in range(J):
            for t in range(T):
                if (c1[i,j,t] == 1) & (c4[i,j,t] == 1):
                    FC[i,j,t] = 1
                else:
                    FC[i,j,t] = 0
                    
    DC = c2
    
    return FC, DC

# Next calculate the areal coverage
def GetArea(x, year, AllYears, AllMonths):
    '''
    This function calculates the area of a gridded dataset assuming a boolean style
    data (1 where the phenomena has occurred and 0 where it has not) and assuming
    a NARR size grid (32 km x 32 km). The area is in km^2, and in km^2/10^5. These
    calculations are made for each time step in a specified year in the gridded dataset.
    
    Inputs:
    - x: Boolean, gridded dataset. The total area of all True gridpoints for each 
         time step will be calculated.
    - year: The year the calculation is made for.
    - AllYears: A 1D array containing the year for all time steps in x.
    - AllMonths: A 1D array containing the month for all time steps in x.
    
    Outputs:
    - Area: The total area of true grid points in x in km^2/10^5. Note the reduction in
            magnitude allows for easier representation of Area in figures.
    - AreaMon: The monthly average total area of all the true grid points in x for a 
               given year in km^2/10^5. Note the reduction in magnitude allows for 
               easier representation of Area in figures.
    '''
    
    # Initialize the area variable
    YearLen = 365
    Area = np.ones((YearLen)) * np.nan
    
    # Find the year overwhich the calculations will be performed
    ind = np.where(AllYears == year)[0]
    
    # Calcualte total number of grids over which the criteria/component is true
    tmp = np.nansum(x[:,:,ind], axis = 0)
    TotPoints = np.nansum(tmp[:,:], axis = 0)
    
    # Calculate the total area (the number of successful grid points times the area of the grid)
    # The NARR grid is 32 km x 32 km.
    Area = TotPoints * 32 * 32
    
    # Reduce the area by a magnitude of 10^5 for easier representation on figures
    Area = Area/(10**5)
    
    # Calculate the monthly average area
    NumMonths = 12
    AreaMon = np.ones((NumMonths)) * np.nan
    
    Months = np.arange(1, 12+1, 1)
    
    for m in Months:
        ind = np.where( (AllYears == year) & (AllMonths == m) )[0]
        tmp = np.nansum(x[:,:,ind], axis = -1)
        tmp = np.where(tmp == 0, tmp, 1) # This changes nonzero values to 1. I.e., flase drought happened.
        AreaMon[m-1] = 32 * 32 * np.nansum(tmp)
        
    AreaMon = AreaMon/(10**5)
    
    return Area, AreaMon


# Next plot the areal coverage
def EarlyPlotArea(DCArea, FCArea, FDArea, time, year, LonMin = -130, LonMax = -65, LatMin = 25, LatMax = 50,
             savename = 'tmp.png'):
    '''
    This funcion plots the total areal coverage time series created in GetArea function.
    The plot is saved once made.
    
    Inputs:
    - DCArea: The areal coverage of the drought component (criteria 2)
    - FCArea: The areal coverage of the flash component/rapid intensification (criteria 4 and criteria 1)
    - FDArea: The areal coverage of the flash drought
    - time: The 1D array containing datetime timestamps for each time step in DCArea, FCArea, and FDArea
    - year: The year for which the areal coverage data is being plotted
    - LonMin, LonMax, LatMin, LatMax: The maximum and minimum longitude and latitude over which the areal
                                      coverage was calculated.
    - savename: The name of the file the figure will be saved as
    
    Outputs:
    - None. A figure is created and saved.
    '''
    
    # Define the path to the directory the figure will be saved in
    OutPath = './Figures/'
    
    # Define the date format used in the tick labels
    DateFMT = DateFormatter('%b')
    
    # Create the figure
    fig = plt.figure(figsize = [12, 14])
    ax  = fig.add_subplot(1, 1, 1)
    
    # Use a separate axis to plot the flash component and flash drought
    axx = ax.twinx()
    
    # Create the title
    ax.set_title('Total Area Covered by Flash Drought and Flash Drought Components for ' + str(year) + '\n' +\
                 ' from ' + str(LatMin) + 'N, ' + str(LonMin) + 'E to ' + str(LatMax) + 'N, ' + str(LonMax) + 'E',
                 size = 18)
    
    # Plot the data
    ax.plot(time, DCArea, 'r-', label = 'Drought (C2)')
    axx.plot(time, FCArea, 'b-.', label = r'Rapid Drying $(C1 \cap C4)$')
    axx.plot(time, FDArea, 'k--', label = 'Flash Drought')
    
    # Generate legends
    ax.legend(loc = 'upper right', fontsize = 16)
    axx.legend(loc = 'upper left', fontsize = 16)
    
    # Set the labels
    ax.set_xlabel('Time', size = 16)
    ax.set_ylabel(r'Area Covered in Drought $(\times 10^5 km^2)$', size = 16)
    axx.set_ylabel(r'Area Covered in Flash Drought and Rapid Drying $(\times 10^5 km^2)$', size = 16)
    
    # Format ticks
    ax.xaxis.set_major_formatter(DateFMT)
    axx.xaxis.set_major_formatter(DateFMT)
    
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(16)
        
    for i in axx.yaxis.get_ticklabels():
        i.set_size(16)
    
    # Save the plot
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    
    
# Next plot the monthly maps
def EarlyCaseMonthlyMaps(c1, c4, FC, DC, FD, lon, lat, LonMin, LonMax, LatMin, LatMax, 
                subset, year, cmin, cmax, cint, FDmin, FDmax, title = 'Title', 
                clabel = 'Criteria Label', FDlabel = 'Flash Drought Label', savename = 'tmp.txt', 
                OutPath = './Figures/'):
    '''
    This function creates one the early case study maps. That is, it displaysthe monthly average
    criterion 1, criterion 4, the flash component, the drought component, and flash drought for each
    month in the thet designated year. The figure is then saved after being created.
    
    Inputs:
    - c1: Gridded, monthly averaged criterion 1 data
    - c4: Gridded, monthly averaged criterion 4 data
    - FC: Gridded, monthly averaged flash component data
    - DC: Gridded, monthly averaged drought component data
    - FD: Gridded, monthly averaged flash drought data
    - lon: Gridded longitude data associated with the above datasets
    - lat: Gridded latitude data associated with the above datasets
    - LonMin, LonMax, LatMin, LatMax: The minimum and maximum longitude and latitude
                                      of the map extends
    - subset: A boolean value indicating whether to use a subset of the U.S. or not
    - year: The year the case study is focusd on
    - cmin, cmax, cint: The minimum and maximum values used in the drought component
                        colorbar as well as the intervals in the colorbar (cint)
    - FDmin, FDmax: The minimum and maximum values used in the c1, c4, FC, and FD
                    colorbars
    - title: The main title of the figure
    - clabel: The label of the drought component colorbar
    - FDlabel: The label of the c1/c4/FC/FD colorbar
    - savename: The name of the file the figure will be saved as
    - OutPath: The path from the current directory to the directory the figure will be saved in
    
    Outputs:
    - None. A figure will be created and saved.
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
    
    # Create the figure
    fig, axes = plt.subplots(figsize = [12, 18], nrows = len(month_names), ncols = 5, 
                             subplot_kw = {'projection': fig_proj})
    
    # Adjust some figure parameters based on whether there is a subset and the year
    if subset is True:
        if year == 1988:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2000:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2002:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.5, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2003:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.7, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2007:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2008:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2011:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2012:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.7, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
            
        elif year == 2016:
            plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = -0.5, hspace = 0.2)
            fig.suptitle(title, y = 0.945, size = 18)
    else:
        plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.65)
        fig.suptitle(title, y = 0.88, size = 18)
    
    
    for m in range(len(month_names)):
        ax1 = axes[m,0]; ax2 = axes[m,1]; ax3 = axes[m,2]; ax4 = axes[m,3]; ax5 = axes[m,4]
        
        # Criteria 1 plot
        
        # Create some features (e.g., coastlines, borders, etc.)
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS)
        ax1.add_feature(cfeature.STATES)
        
        # Set the ticks
        ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax1.set_yticklabels(LatLabel, fontsize = 10)
        ax1.set_xticklabels(LonLabel, fontsize = 10)
        
        ax1.xaxis.set_major_formatter(LonFormatter)
        ax1.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax1.pcolormesh(lon, lat, c1[:,:,m], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Set the map extend
        ax1.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        # Add labels to the left most maps
        ax1.set_ylabel(month_names[m], size = 14, labelpad = 20.0, rotation = 0)
        
        # Criteria 4 plot
        
        # Create some features (e.g., coastlines, borders, etc.)
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS)
        ax2.add_feature(cfeature.STATES)
        
        # Set the ticks
        ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax2.set_yticklabels(LatLabel, fontsize = 10)
        ax2.set_xticklabels(LonLabel, fontsize = 10)
        
        ax2.xaxis.set_major_formatter(LonFormatter)
        ax2.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax2.pcolormesh(lon, lat, c4[:,:,m], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        ax2.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        # Flash Component plot
        
        # Create some features (e.g., coastlines, borders, etc.)
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS)
        ax3.add_feature(cfeature.STATES)
        
        # Set the ticks
        ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax3.set_yticklabels(LatLabel, fontsize = 10)
        ax3.set_xticklabels(LonLabel, fontsize = 10)
        
        ax3.xaxis.set_major_formatter(LonFormatter)
        ax3.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax3.pcolormesh(lon, lat, FC[:,:,m], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Set the map extend
        ax3.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        # Drought Component plot
        
        # Create some features (e.g., coastlines, borders, etc.)
        ax4.coastlines()
        ax4.add_feature(cfeature.BORDERS)
        ax4.add_feature(cfeature.STATES)
        
        # Set the ticks
        ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax4.set_yticklabels(LatLabel, fontsize = 10)
        ax4.set_xticklabels(LonLabel, fontsize = 10)
        
        ax4.xaxis.set_major_formatter(LonFormatter)
        ax4.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax4.pcolormesh(lon, lat, DC[:,:,m], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj)
        
        # Set the map extend
        ax4.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        # Flash Drought plot
        
        # Create some features (e.g., coastlines, borders, etc.)
        ax5.coastlines()
        ax5.add_feature(cfeature.BORDERS)
        ax5.add_feature(cfeature.STATES)
        
        # Set the ticks
        ax5.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax5.set_yticks(LatLabel, crs = ccrs.PlateCarree())
        
        ax5.set_yticklabels(LatLabel, fontsize = 10)
        ax5.set_xticklabels(LonLabel, fontsize = 10)
        
        ax5.xaxis.set_major_formatter(LonFormatter)
        ax5.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cfd = ax5.pcolormesh(lon, lat, FD[:,:,m], vmin = FDmin, vmax = FDmax,
                          cmap = cmap, transform = data_proj)
        
        # Set the map extend
        ax5.set_extent([LonMin, LonMax, LatMin, LatMax])
        
        if m == 0:
            # For the top row, set the titles
            ax1.set_title('C1')
            ax2.set_title('C4')
            ax3.set_title(r'FC $(C1 \cap C4)$')
            ax4.set_title('DC (C2)')
            ax5.set_title('FD')
            
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
    
    # Set the colorbar location and size based on whether a subset is used (and year)
    if subset is True:
        if year == 1988:
            cbax = fig.add_axes([0.95, 0.09, 0.015, 0.83])
            
            cbaxfd = fig.add_axes([0.09, 0.07, 0.80, 0.015])
        elif year == 2000:
            cbax = fig.add_axes([0.95, 0.09, 0.015, 0.83])
            
            cbaxfd = fig.add_axes([0.09, 0.07, 0.80, 0.015])
        elif year == 2002:
            cbax = fig.add_axes([0.88, 0.095, 0.015, 0.82])
            
            cbaxfd = fig.add_axes([0.17, 0.07, 0.65, 0.015])
        elif year == 2003:
            cbax = fig.add_axes([0.80, 0.12, 0.015, 0.78])
            
            cbaxfd = fig.add_axes([0.17, 0.07, 0.65, 0.015])
        elif year == 2007:
            cbax = fig.add_axes([0.95, 0.09, 0.015, 0.83])
            
            cbaxfd = fig.add_axes([0.09, 0.07, 0.80, 0.015])
        elif year == 2008:
            cbax = fig.add_axes([0.95, 0.09, 0.015, 0.83])
            
            cbaxfd = fig.add_axes([0.09, 0.07, 0.80, 0.015])
        elif year == 2011:
            cbax = fig.add_axes([0.95, 0.09, 0.015, 0.83])
            
            cbaxfd = fig.add_axes([0.09, 0.07, 0.80, 0.015])
        elif year == 2012:
            cbax = fig.add_axes([0.80, 0.12, 0.015, 0.78])
            
            cbaxfd = fig.add_axes([0.17, 0.07, 0.65, 0.015])
        elif year == 2016:
            cbax = fig.add_axes([0.875, 0.09, 0.015, 0.81])
            
            cbaxfd = fig.add_axes([0.15, 0.07, 0.70, 0.015])
    else:
        cbax = fig.add_axes([0.95, 0.15, 0.015, 0.68])
        
        cbaxfd = fig.add_axes([0.07, 0.12, 0.85, 0.015])
        
    # Create the drought component colorbar
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
    cbar.ax.set_ylabel(clabel, fontsize = 14)
    
    # Create the c1/c4/FC/FD colorbar
    cbarfd = fig.colorbar(cfd, cax = cbaxfd, orientation = 'horizontal')
    cbarfd.ax.set_xlabel(FDlabel, fontsize = 14)
    
    #fig.tight_layout()
    
    # Save the figure
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    
    

##############################
### EarlyCaseStudyFunction ###
##############################
    
# Function that calls calculations and creates the maps
def EarlyCaseStudy(c1, c2, c4, FD, lon, lat, AllMonths, AllYears, AllDates, 
              year = 2012, subset = False):
    '''
    This function performs some of the early case studies for the flash drought
    decomposition project. It takes in criteria 1, 2, and 4 and flash drought
    identified (using the method is Christian et al. 2019) and determines the
    flash and drought components, calculates their monthly averages for the
    case study year, and plots the monthly average maps and areal coverage
    time series.
    
    Inputs:
    - c1: Gridded criterion 1 dataset
    - c2: Gridded criterion 2 dataset
    - c4: Gridded criterion 4 dataset
    - FD: Gridded flash drought dataset
    - lon: The gridded longitude dataset associated with the above data
    - lat: The gridded latitude dataset associated with the above data
    - AllMonths: 1D array containing the month for all time steps in the above data
    - AllYears: 1D array containing the year for all time steps in the above data
    - AllDates: 1D array of datetimes containing the date of all the time steps in the above data
    - year: The year the case study focuses on
    - subset: A boolean value indicating whether to focus the case study on a specific
              region (or subset of the U.S.)
              
    - None. Two figures (areal coverage time series and monthly average maps) area
      created and saved.
    '''

    I, J, T = FD.shape
    
    # Determine the max/min lat/lon for a flash drought area for a given year
    if year == 1988:
        LonMin = -115
        LonMax = -80
        LatMin = 30
        LatMax = 50
        
    elif year == 2000:
        LonMin = -105
        LonMax = -70
        LatMin = 28
        LatMax = 45
        
    elif year == 2002:
        LonMin = -115
        LonMax = -85
        LatMin = 30
        LatMax = 50
        
    elif year == 2003:
        LonMin = -105
        LonMax = -85
        LatMin = 30
        LatMax = 50
        
    elif year == 2007:
        LonMin = -100
        LonMax = -70
        LatMin = 27
        LatMax = 45
        
    elif year == 2008:
        LonMin = -100
        LonMax = -70
        LatMin = 27
        LatMax = 45
        
    elif year == 2011:
        LonMin = -110
        LonMax = -80
        LatMin = 25
        LatMax = 42
        
    elif year == 2012:
        LonMin = -105
        LonMax = -85
        LatMin = 30
        LatMax = 50
        
    elif year == 2016:
        LonMin = -90
        LonMax = -67
        LatMin = 30
        LatMax = 47
    
    # Use default (full CONUS) if no major year is specified    
    else:
        LonMin = -130
        LonMax = -65
        LatMin = 25
        LatMax = 50
        
    # Subset the data to the flash drought area
    ######### Needs to be edited for the NARR grid
    if subset is False:
        LonMin = -130
        LonMax = -65
        LatMin = 25
        LatMax = 50
        
        subset_str = 'NA'
        
    else:        
        subset_str = 'subset'
        
        
        
    # Calculate the flash and drought components
    FC, DC = CalculateEarlyFCandDC(c1, c2, c4)
    
    # Subset the data        
    DCsub, LatSub, LonSub = SubsetData(DC, lat, lon, LatMin, LatMax, LonMin, LonMax)
    FCsub, LatSub, LonSub = SubsetData(FC, lat, lon, LatMin, LatMax, LonMin, LonMax)
    FDsub, LatSub, LonSub = SubsetData(FD, lat, lon, LatMin, LatMax, LonMin, LonMax)
    
    # Define the figure names
    TSname = 'Drought_Area_Coverage_' + str(year) + '_' + subset_str + '_pentad.png'
    TSMonname = 'Monthly_Drought_Area_Coverage_' + str(year) + '_' + subset_str + '_pentad.png'
    Mapname = 'FD_panel_map_' + str(year) + '_' + subset_str + '_pentad.png'
    
    # Declare some time variables
    ind = np.where(AllYears == year)[0]
    YearDates = AllDates[ind]
    
    months = np.unique(AllMonths)
    
    MonthDates = np.asarray([datetime(year, m, 1) for m in months])
    
    # Initialize variables for monthly means
    MonC1 = np.ones((I, J, T)) * np.nan
    MonC4 = np.ones((I, J, T)) * np.nan
    MonFC = np.ones((I, J, T)) * np.nan
    MonDC = np.ones((I, J, T)) * np.nan
    MonFD = np.ones((I, J, T)) * np.nan
    
    # Calculate the monthly means
    for m in range(len(months)):
        MonC1[:,:,m] = MonthAverage(c1, year, m+1, AllYears, AllMonths)
        MonC4[:,:,m] = MonthAverage(c4, year, m+1, AllYears, AllMonths)
        MonFC[:,:,m] = MonthAverage(FC, year, m+1, AllYears, AllMonths)
        MonDC[:,:,m] = MonthAverage(DC, year, m+1, AllYears, AllMonths)
        MonFD[:,:,m] = MonthAverage(FD, year, m+1, AllYears, AllMonths)
    
    # Calculate the areal coverage time series
    DCarea, DCareaMon = GetArea(DCsub, year, AllYears, AllMonths)
    FCarea, FCareaMon = GetArea(FCsub, year, AllYears, AllMonths)
    FDarea, FDareaMon = GetArea(FDsub, year, AllYears, AllMonths)
    
    # Plot the areal coverage time series
    EarlyPlotArea(DCarea, FCarea, FDarea, YearDates, year, LonMin, LonMax, LatMin, LatMax, TSname)
    EarlyPlotArea(DCareaMon, FCareaMon, FDareaMon, MonthDates, year, LonMin, LonMax, LatMin, LatMax, TSMonname)
    
    # Plot monthly maps
    EarlyCaseMonthlyMaps(MonC1, MonC4, MonFC, MonDC, MonFD, lon, lat, LonMin, LonMax, LatMin, LatMax,
                    subset, year, cmin = 0, cmax = 1, cint = 0.1, FDmin = 0, FDmax = 0.1,
                    title = 'Average Criteria Value for Each Month for ' + str(year),
                    clabel = 'Average Criteria Value for C4 and DC',
                    FDlabel = 'Average Criteria Value for C1, FC, and FD',
                    savename = Mapname)
