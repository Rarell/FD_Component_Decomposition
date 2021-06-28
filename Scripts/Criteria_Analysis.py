#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:37:18 2020

@author: stuartedris

This scripts performs the main analyses and produces the majority of the figures
for the Decomposing the Critical Components of Flash Drought Using the Standardized 
Evaporative Stress Ratio paper. The script uses the criteria, percentile, and
flash drought data produced by the Flash_Drought_Criteria_Analysis script and
performs the analysis calculations for the case studies and rapid intensification
or flash component. Final results are saved in the form of figures. The SESR drought
and USDM comparison is performed on the Drought_Comparison script.
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
import matplotlib as mpl

# Add a new path to look for additional, custom libraries and functions
sys.path.append('./FlashDroughtFunctions/')
from FlashDroughtFunctions.FlashDroughtMapping import *
from FlashDroughtFunctions.FlashDroughtCaseStudies import *
from FlashDroughtFunctions.FlashDroughtStats import *

#%%
# cell 2
# Name some user defined variables

crit1FN = 'criteria1.nc'
crit2FN = 'criteria2.nc'
crit3FN = 'criteria3.nc'
crit4FN = 'criteria4.nc'
FDFN    = 'FlashDrought.nc'
DIFN    = 'DroughtIntensity.nc'
DPFN    = 'DroughtPercentile.nc'
dsesrPFN    = 'DeltaSesrPercentiles.nc'
msesrPFN    = 'MeanDeltaSesrPercentiles.nc'

crit1SName = 'c1'
crit2SName = 'c2'
crit3SName = 'c3'
crit4SName = 'c4'
FDSName    = 'FD'
DISName    = 'DI'
DPSName    = 'DP'
dsesrPName = 'dsesrp'
msesrPName = 'mdsesrp'

OutPath = './Figures/'

#%%
# cell 3
# Examine the .nc files

# Examine climatology values and make sure they are present
print(Dataset('./Data/pentad_NARR_grid/criteria1.nc', 'r'))
print(Dataset('./Data/pentad_NARR_grid/criteria2.nc', 'r'))
print(Dataset('./Data/pentad_NARR_grid/criteria3.nc', 'r'))
print(Dataset('./Data/pentad_NARR_grid/criteria4.nc', 'r'))
print(Dataset('./Data/pentad_NARR_grid/FlashDrought.nc', 'r'))
print(Dataset('./Data/pentad_NARR_grid/DroughtIntensity.nc', 'r'))

#%%
# cell 4
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
# cell 5
# Load the data
    
path = './Data/pentad_NARR_grid/'

c1 = LoadNC(crit1SName, crit1FN)
c2 = LoadNC(crit2SName, crit2FN)
c3 = LoadNC(crit3SName, crit3FN)
c4 = LoadNC(crit4SName, crit4FN)

FD = LoadNC(FDSName, FDFN)
DI = LoadNC(DISName, DIFN)
DP = LoadNC(DPSName, DPFN)

dsesrp  = LoadNC(dsesrPName, dsesrPFN)
mdsesrp = LoadNC(msesrPName, msesrPFN)

#%%
# cell 6
# Load USDM Data

# Define some names
USDMpath = './Data/Misc_Data/'
USDMfn = 'USDM_grid_all_years.nc'
USDMname = 'USDM'


USDM = LoadNC(USDMname, USDMfn, path = USDMpath)

#%%
# cell 7
# Find the anual and monthly totals of each criteria and FD

# Initialize variables
I, J, T = FD['FD'].shape
years  = np.unique(FD['year'])
months = np.unique(FD['month'])

AnnC1 = np.ones((I, J, years.size)) * np.nan
AnnC2 = np.ones((I, J, years.size)) * np.nan
AnnC3 = np.ones((I, J, years.size)) * np.nan
AnnC4 = np.ones((I, J, years.size)) * np.nan

AnnFD = np.ones((I, J, years.size)) * np.nan

MonC1 = np.ones((I, J, months.size)) * np.nan
MonC2 = np.ones((I, J, months.size)) * np.nan
MonC3 = np.ones((I, J, months.size)) * np.nan
MonC4 = np.ones((I, J, months.size)) * np.nan

MonFD = np.ones((I, J, months.size)) * np.nan

DailyC1 = np.ones((I, J, 365)) * np.nan
DailyC2 = np.ones((I, J, 365)) * np.nan
DailyC3 = np.ones((I, J, 365)) * np.nan
DailyC4 = np.ones((I, J, 365)) * np.nan

DailyFD = np.ones((I, J, 365)) * np.nan

YearDates = np.asarray([datetime(2000, d.month, d.day) for d in FD['ymd'][:365]])

AllMonths = np.asarray([d.month for d in YearDates])

# Count the total number of identified criterion/flash drought for the year for each year
for y in range(years.size):
    yInd = np.where(years[y] == FD['year'])[0]
    
    AnnC1[:,:,y] = np.nansum(c1['c1'][:,:,yInd], axis = -1)
    AnnC2[:,:,y] = np.nansum(c2['c2'][:,:,yInd], axis = -1)
    AnnC3[:,:,y] = np.nansum(c3['c3'][:,:,yInd], axis = -1)
    AnnC4[:,:,y] = np.nansum(c4['c4'][:,:,yInd], axis = -1)
    
    AnnFD[:,:,y] = np.nansum(FD['FD'][:,:,yInd], axis = -1)

# Determine the daily average value for each criterion/flash drought for each day in the calendar year
for d in range(YearDates.size):
    dInd = np.where( (YearDates[d].month == FD['month']) & (YearDates[d].day == FD['day']) )[0]
    
    DailyC1[:,:,d] = np.nanmean(c1['c1'][:,:,dInd], axis = -1)
    DailyC2[:,:,d] = np.nanmean(c2['c2'][:,:,dInd], axis = -1)
    DailyC3[:,:,d] = np.nanmean(c3['c3'][:,:,dInd], axis = -1)
    DailyC4[:,:,d] = np.nanmean(c4['c4'][:,:,dInd], axis = -1)
    
    DailyFD[:,:,d] = np.nanmean(FD['FD'][:,:,dInd], axis = -1)

# Determine the total number of identified criterion/flash drought for the month for each month
for m in range(months.size):
    mInd = np.where(months[m] == AllMonths)[0]
    
    MonC1[:,:,m] = np.nansum(DailyC1[:,:,mInd], axis = -1)
    MonC2[:,:,m] = np.nansum(DailyC2[:,:,mInd], axis = -1)
    MonC3[:,:,m] = np.nansum(DailyC3[:,:,mInd], axis = -1)
    MonC4[:,:,m] = np.nansum(DailyC4[:,:,mInd], axis = -1)
    
    MonFD[:,:,m] = np.nansum(DailyFD[:,:,mInd], axis = -1)

#%%
# cell 8
# Create the panel plots for each year
    
# print(np.nanmax(AnnC1), np.nanmax(AnnC2), np.nanmax(AnnC3), np.nanmax(AnnC4), np.nanmax(AnnFD))
    
# From the above, commented command, 365 is the highest value
    
MapAllYears(AnnC1, AnnC2, AnnC3, AnnC4, AnnFD, FD['lon'], FD['lat'], years,
            cmin = 0, cmax = 365, cint = 5, title = 'Number of Successful Criteria in a Given Year',
            clabel = 'Number of Days of Successful Criteria/Flash Drought Identified',
            savename = 'AnnTotal_Crit_FD_all-years_pentads.png')

#%%
# cell 9
# After the memeing is done, recreate the above plot 6 times (each 1/6 the size
#   of the original) to make it more readable.

# Create the first plot (1979 - 1986)
EightYearPlot(AnnC1, AnnC2, AnnC3, AnnC4, AnnFD, FD['lon'], FD['lat'], years, cmin = 0, cmax = 365, cint = 5, 
              FDmin = 0, FDmax = 50, title = 'Number of Successful Criteria in a Given Year', 
              clabel = 'Number of Days of Successful Criteria',
              FDlabel = 'Number of Flash Droughts Identified/Successful Criteria 1', 
              savename = 'AnnTotal_Crit_FD_all_1979-1986_pentads.png', shift = 0)

# Create the second plot (1987 - 1992)
SixYearPlot(AnnC1, AnnC2, AnnC3, AnnC4, AnnFD, FD['lon'], FD['lat'], years, cmin = 0, cmax = 365, cint = 5, 
            FDmin = 0, FDmax = 50, title = 'Number of Successful Criteria in a Given Year', 
            clabel = 'Number of Days of Successful Criteria',
            FDlabel = 'Number of Flash Droughts Identified/Successful Criteria 1', 
            savename = 'AnnTotal_Crit_FD_all_1987-1992_pentads.png', shift = 8)

# Create the third plot (1993 - 1998)
SixYearPlot(AnnC1, AnnC2, AnnC3, AnnC4, AnnFD, FD['lon'], FD['lat'], years, cmin = 0, cmax = 365, cint = 5, 
            FDmin = 0, FDmax = 50, title = 'Number of Successful Criteria in a Given Year', 
            clabel = 'Number of Days of Successful Criteria',
            FDlabel = 'Number of Flash Droughts Identified/Successful Criteria 1', 
            savename = 'AnnTotal_Crit_FD_all_1993-1998_pentads.png', shift = 14)

# Create the fourth plot (1999 - 2004)
SixYearPlot(AnnC1, AnnC2, AnnC3, AnnC4, AnnFD, FD['lon'], FD['lat'], years, cmin = 0, cmax = 365, cint = 5, 
            FDmin = 0, FDmax = 50, title = 'Number of Successful Criteria in a Given Year', 
            clabel = 'Number of Days of Successful Criteria',
            FDlabel = 'Number of Flash Droughts Identified/Successful Criteria 1', 
            savename = 'AnnTotal_Crit_FD_all_1999-2004_pentads.png', shift = 20)

# Create the fifth plot (2005 - 2010)
SixYearPlot(AnnC1, AnnC2, AnnC3, AnnC4, AnnFD, FD['lon'], FD['lat'], years, cmin = 0, cmax = 365, cint = 5, 
            FDmin = 0, FDmax = 50, title = 'Number of Successful Criteria in a Given Year', 
            clabel = 'Number of Days of Successful Criteria',
            FDlabel = 'Number of Flash Droughts Identified/Successful Criteria 1', 
            savename = 'AnnTotal_Crit_FD_all_2005-2010_pentads.png', shift = 26)


# Create the sixth plot (2011 - 2016)
SixYearPlot(AnnC1, AnnC2, AnnC3, AnnC4, AnnFD, FD['lon'], FD['lat'], years, cmin = 0, cmax = 365, cint = 5, 
            FDmin = 0, FDmax = 50, title = 'Number of Successful Criteria in a Given Year', 
            clabel = 'Number of Days of Successful Criteria',
            FDlabel = 'Number of Flash Droughts Identified/Successful Criteria 1', 
            savename = 'AnnTotal_Crit_FD_all_2011-2016_pentads.png', shift = 32)


#%%
# cell 10
# Plot criteria total for each month

MonthlyMaps(MonC1, MonC2, MonC3, MonC4, MonFD, FD['lon'], FD['lat'], months, cmin = 0, cmax = 30, cint = 2,
            FDmin = 0, FDmax = 2, title = 'Average Number of Successful Criteria in a Given Month',
            clabel = 'Number of Days of Successful Criteria', FDlabel = 'Number of Flash Droughts Identified/Successful Criteria 1',
            savename = 'MonTotal_Crit_FD_all-months_pentads.png')

#%%
# cell 11
# Find the anual and monthly meanss of each criteria and FD

# Initialize variabels
I, J, T = FD['FD'].shape
years  = np.unique(FD['year'])
months = np.unique(FD['month'])

AnnC1 = np.ones((I, J, years.size)) * np.nan
AnnC2 = np.ones((I, J, years.size)) * np.nan
AnnC3 = np.ones((I, J, years.size)) * np.nan
AnnC4 = np.ones((I, J, years.size)) * np.nan

AnnFD = np.ones((I, J, years.size)) * np.nan

MonC1 = np.ones((I, J, months.size)) * np.nan
MonC2 = np.ones((I, J, months.size)) * np.nan
MonC3 = np.ones((I, J, months.size)) * np.nan
MonC4 = np.ones((I, J, months.size)) * np.nan

MonFD = np.ones((I, J, months.size)) * np.nan

# Determine the annual average value of each criterion/flash drought for each year
for y in range(years.size):
    yInd = np.where(years[y] == FD['year'])[0]
    
    AnnC1[:,:,y] = np.nanmean(c1['c1'][:,:,yInd], axis = -1)
    AnnC2[:,:,y] = np.nanmean(c2['c2'][:,:,yInd], axis = -1)
    AnnC3[:,:,y] = np.nanmean(c3['c3'][:,:,yInd], axis = -1)
    AnnC4[:,:,y] = np.nanmean(c4['c4'][:,:,yInd], axis = -1)
    
    AnnFD[:,:,y] = np.nanmean(FD['FD'][:,:,yInd], axis = -1)

# Determine the monthly average value of each criterion/flash drought for each month
for m in range(months.size):
    mInd = np.where(months[m] == FD['month'])[0]
    
    MonC1[:,:,m] = np.nanmean(c1['c1'][:,:,mInd], axis = -1)
    MonC2[:,:,m] = np.nanmean(c2['c2'][:,:,mInd], axis = -1)
    MonC3[:,:,m] = np.nanmean(c3['c3'][:,:,mInd], axis = -1)
    MonC4[:,:,m] = np.nanmean(c4['c4'][:,:,mInd], axis = -1)
    
    MonFD[:,:,m] = np.nanmean(FD['FD'][:,:,mInd], axis = -1)


#%%
# cell 12
# Meme more. Plot the annual mean for all years
MapAllYears(AnnC1, AnnC2, AnnC3, AnnC4, AnnFD, FD['lon'], FD['lat'], years,
            cmin = 0, cmax = 1, cint = 0.1, title = 'Annual Mean Criteria in a Given Year',
            clabel = 'Average Criteria Value/Average Flash Drought Value',
            savename = 'AnnMean_Crit_FD_all-years_pentads.png')

#%%
# cell 13
# Memeing is over. Plot readable maps of the annual means

# Create the first plot (1979 - 1986)
EightYearPlot(AnnC1, AnnC2, AnnC3, AnnC4, AnnFD, FD['lon'], FD['lat'], years, cmin = 0, cmax = 1, cint = 0.1, 
              FDmin = 0, FDmax = 0.2, title = 'Annual Mean Criteria in a Given Year', 
              clabel = 'Average Criteria Value',
              FDlabel = 'Average Flash Drought/Criteria 1 Value', 
              savename = 'AnnMean_Crit_FD_all_1979-1986_pentads.png', shift = 0)

# Create the second plot (1987 - 1992)
SixYearPlot(AnnC1, AnnC2, AnnC3, AnnC4, AnnFD, FD['lon'], FD['lat'], years, cmin = 0, cmax = 1, cint = 0.1, 
            FDmin = 0, FDmax = 0.2, title = 'Annual Mean Criteria in a Given Year', 
            clabel = 'Average Criteria Value',
            FDlabel = 'Average Flash Drought/Criteria 1 Value', 
            savename = 'AnnMean_Crit_FD_all_1987-1992_pentads.png', shift = 8)

# Create the third plot (1993 - 1998)
SixYearPlot(AnnC1, AnnC2, AnnC3, AnnC4, AnnFD, FD['lon'], FD['lat'], years, cmin = 0, cmax = 1, cint = 0.1, 
            FDmin = 0, FDmax = 0.2, title = 'Annual Mean Criteria in a Given Year', 
            clabel = 'Average Criteria Value',
            FDlabel = 'Average Flash Drought/Criteria 1 Value', 
            savename = 'AnnMean_Crit_FD_all_1993-1998_pentads.png', shift = 14)

# Create the fourth plot (1999 - 2004)
SixYearPlot(AnnC1, AnnC2, AnnC3, AnnC4, AnnFD, FD['lon'], FD['lat'], years, cmin = 0, cmax = 1, cint = 0.1, 
            FDmin = 0, FDmax = 0.2, title = 'Annual Mean Criteria in a Given Year', 
            clabel = 'Average Criteria Value',
            FDlabel = 'Average Flash Drought/Criteria 1 Value', 
            savename = 'AnnMean_Crit_FD_all_1999-2004_pentads.png', shift = 20)

# Create the fifth plot (2005 - 2010)
SixYearPlot(AnnC1, AnnC2, AnnC3, AnnC4, AnnFD, FD['lon'], FD['lat'], years, cmin = 0, cmax = 1, cint = 0.1, 
            FDmin = 0, FDmax = 0.2, title = 'Annual Mean Criteria in a Given Year', 
            clabel = 'Average Criteria Value',
            FDlabel = 'Average Flash Drought/Criteria 1 Value', 
            savename = 'AnnMean_Crit_FD_all_2005-2010_pentads.png', shift = 26)


# Create the sixth plot (2011 - 2016)
SixYearPlot(AnnC1, AnnC2, AnnC3, AnnC4, AnnFD, FD['lon'], FD['lat'], years, cmin = 0, cmax = 1, cint = 0.1, 
            FDmin = 0, FDmax = 0.2, title = 'Annual Mean Criteria in a Given Year', 
            clabel = 'Average Criteria Value',
            FDlabel = 'Average Flash Drought/Criteria 1 Value', 
            savename = 'AnnMean_Crit_FD_all_2011-2016_pentads.png', shift = 32)

#%%
# cell 14
# Plot criteria mean for each month

MonthlyMaps(MonC1, MonC2, MonC3, MonC4, MonFD, FD['lon'], FD['lat'], months, cmin = 0, cmax = 1, cint = 0.1,
            FDmin = 0, FDmax = 0.06, title = 'Average Number of Successful Criteria in a Given Month',
            clabel = 'Average Criteria Value', FDlabel = 'Average Flash Drought/Criteria 1 Value',
            savename = 'MonMean_Crit_FD_all-months_pentads.png')

#%%
# cell 15
# Plot criteria mean for each month

MonthlyGrowMaps(MonC1, MonC2, MonC3, MonC4, MonFD, FD['lon'], FD['lat'], months, cmin = 0, cmax = 1, cint = 0.1,
             FDmin = 0, FDmax = 0.05, title = 'Average Number of Successful Criteria in a Given Month',
             clabel = 'Average Drought Component Value', FDlabel = 'Average Flash Components/Flash Drought Value',
             savename = 'MonMean_Crit_FD_growing-seasons_pentads.png')


#%%
# cell 16
# Find the average criteria for each season

# Initialize some variables
I, J, T = FD['FD'].shape

SeaC1 = np.ones((I, J, 4)) * np.nan
SeaC2 = np.ones((I, J, 4)) * np.nan
SeaC3 = np.ones((I, J, 4)) * np.nan
SeaC4 = np.ones((I, J, 4)) * np.nan

SeaFD = np.ones((I, J, 4)) * np.nan

# Find the spring means
ind = np.where( (FD['month'] >= 3) & (FD['month'] <= 5) )[0]

SeaC1[:,:,0] = np.nanmean(c1['c1'][:,:,ind], axis = -1)
SeaC2[:,:,0] = np.nanmean(c2['c2'][:,:,ind], axis = -1)
SeaC3[:,:,0] = np.nanmean(c3['c3'][:,:,ind], axis = -1)
SeaC4[:,:,0] = np.nanmean(c4['c4'][:,:,ind], axis = -1)

SeaFD[:,:,0] = np.nanmean(FD['FD'][:,:,ind], axis = -1)

# Find the summer means
ind = np.where( (FD['month'] >= 6) & (FD['month'] <= 8) )[0]

SeaC1[:,:,1] = np.nanmean(c1['c1'][:,:,ind], axis = -1)
SeaC2[:,:,1] = np.nanmean(c2['c2'][:,:,ind], axis = -1)
SeaC3[:,:,1] = np.nanmean(c3['c3'][:,:,ind], axis = -1)
SeaC4[:,:,1] = np.nanmean(c4['c4'][:,:,ind], axis = -1)

SeaFD[:,:,1] = np.nanmean(FD['FD'][:,:,ind], axis = -1)

# Find the fall means
ind = np.where( (FD['month'] >= 9) & (FD['month'] <= 11) )[0]

SeaC1[:,:,2] = np.nanmean(c1['c1'][:,:,ind], axis = -1)
SeaC2[:,:,2] = np.nanmean(c2['c2'][:,:,ind], axis = -1)
SeaC3[:,:,2] = np.nanmean(c3['c3'][:,:,ind], axis = -1)
SeaC4[:,:,2] = np.nanmean(c4['c4'][:,:,ind], axis = -1)

SeaFD[:,:,2] = np.nanmean(FD['FD'][:,:,ind], axis = -1)

# Find the winter means
ind = np.where( (FD['month'] >= 12) | (FD['month'] <= 2) )[0]

SeaC1[:,:,3] = np.nanmean(c1['c1'][:,:,ind], axis = -1)
SeaC2[:,:,3] = np.nanmean(c2['c2'][:,:,ind], axis = -1)
SeaC3[:,:,3] = np.nanmean(c3['c3'][:,:,ind], axis = -1)
SeaC4[:,:,3] = np.nanmean(c4['c4'][:,:,ind], axis = -1)

SeaFD[:,:,3] = np.nanmean(FD['FD'][:,:,ind], axis = -1)

#%%
# cell 17
# Plot the searsonal means

SeasonMaps(SeaC1, SeaC2, SeaC3, SeaC4, SeaFD, FD['lon'], FD['lat'], cmin = 0, cmax = 1, cint = 0.1,
           FDmin = 0, FDmax = 0.04, title = 'Average Criteria Value for Each Season',
           clabel = 'Average Criteria Value', FDlabel = 'Average Flash Drought/Criteria 1 Value',
           savename = 'SeaMean_Crit_FD_pentads.png')


#%%
# cell 18
# Next, find the growing season and non growing season means

# Initialize some variables
I, J, T = FD['FD'].shape

GrowC1 = np.ones((I, J, 2)) * np.nan
GrowC2 = np.ones((I, J, 2)) * np.nan
GrowC3 = np.ones((I, J, 2)) * np.nan
GrowC4 = np.ones((I, J, 2)) * np.nan

GrowFD = np.ones((I, J, 2)) * np.nan

# Find the growing season (months defined in Christian et al. 2019) means
ind = np.where( (FD['month'] >= 4) & (FD['month'] <= 10) )[0]

GrowC1[:,:,0] = np.nanmean(c1['c1'][:,:,ind], axis = -1)
GrowC2[:,:,0] = np.nanmean(c2['c2'][:,:,ind], axis = -1)
GrowC3[:,:,0] = np.nanmean(c3['c3'][:,:,ind], axis = -1)
GrowC4[:,:,0] = np.nanmean(c4['c4'][:,:,ind], axis = -1)

GrowFD[:,:,0] = np.nanmean(FD['FD'][:,:,ind], axis = -1)

# Find the nongrowing season means
ind = np.where( (FD['month'] >= 11) | (FD['month'] <= 3) )[0]

GrowC1[:,:,1] = np.nanmean(c1['c1'][:,:,ind], axis = -1)
GrowC2[:,:,1] = np.nanmean(c2['c2'][:,:,ind], axis = -1)
GrowC3[:,:,1] = np.nanmean(c3['c3'][:,:,ind], axis = -1)
GrowC4[:,:,1] = np.nanmean(c4['c4'][:,:,ind], axis = -1)

GrowFD[:,:,1] = np.nanmean(FD['FD'][:,:,ind], axis = -1)

#%%
# cell 19
# Plot the growing season versus non growing season means

GrowSeasonMaps(GrowC1, GrowC2, GrowC3, GrowC4, GrowFD, FD['lon'], FD['lat'],
               cmin = 0, cmax = 1, cint = 0.1, FDmin = 0, FDmax = 0.02,
               title = 'Average Criteria Value for Growing/Non-Growing Seasons',
               clabel = 'Average Criteria Value', FDlabel = 'Average Flash Drought/Criteria 1 Value',
               savename = 'GrowMean_Crit_FD_pentads.png')


#%%
# cell 20
# Load and subset the mask dataset

#### Remove this cell when the .nc files have the mask set included in them (this cell will then become unnecessary)

# As an extra, create a function to generate a range of datetimes
def DateRange(StartDate, EndDate):
    '''
    This function takes in two dates and outputs all the dates inbetween
    those two dates.
    
    Inputs:
        StartDate - A datetime. The starting date of the interval.
        EndDate - A datetime. The ending date of the interval.
        
    Outputs:
        All dates between StartDate and EndDate (inclusive)
    '''
    for n in range(int((EndDate - StartDate).days) + 1):
        yield StartDate + timedelta(n) 

# Create a function to load 2D data
def load2Dnc(filename, SName, path = './Data/'):
    '''
    This function loads 2 dimensional .nc files (e.g., the lat or lon files/
    only spatial files). Function is simple as these files only contain the raw data.
    
    Inputs:
    - filename: The filename of the .nc file to be loaded.
    - SName: The short name of the variable in the .nc file (i.e., the name to
             call when loading the data)
    - Path: The path from the present direction to the directory the file is in.
    
    Outputs:
    - var: The main variable in the .nc file.
    '''
    
    with Dataset(path + filename, 'r') as nc:
        var = nc.variables[SName][:,:]
        
    return var


# Load the mask data
mask = load2Dnc('land.nc', 'land')
lat = load2Dnc('lat_narr.nc', 'lat') # Dataset is lat x lon
lon = load2Dnc('lon_narr.nc', 'lon') # Dataset is lat x lon

# Turn positive lon values into negative
for i in range(len(lon[:,0])):
    ind = np.where( lon[i,:] > 0 )[0]
    lon[i,ind] = -1*lon[i,ind]

# Turn mask from time x lat x lon into lat x lon x time
T, I, J = mask.shape

maskNew = np.ones((I, J, T)) * np.nan
maskNew[:,:,0] = mask[0,:,:] # No loop is needed since the time dimension has length 1

# Subset the data to the same subset as the criteria data
LatMin = 25
LatMax = 50
LonMin = -130
LonMax = -65
maskSub, LatSub, LonSub = SubsetData(maskNew, lat, lon, LatMin = LatMin, LatMax = LatMax,
                                     LonMin = LonMin, LonMax = LonMax) 

 
#%%
# cell 21
######################
###### Beginning of Case Studies #########
#######################

# Early case studies
year = 2016
subset = True   
    
EarlyCaseStudy(c1['c1'], c2['c2'], c4['c4'], FD['FD'], FD['lon'], FD['lat'], FD['month'],
          FD['year'], FD['ymd'], year = year, subset = subset)

#%%
# cell 22
# Make the 2003 USDM time series so that a 2003 case study can be performed

# Create a function to generate a series of datetimes where each time step is 1 week
def WeekDateRange(StartDate, EndDate):
    '''
    This function takes in two dates and outputs all the dates inbetween
    those two dates.
    
    Inputs:
        StartDate - A datetime. The starting date of the interval.
        EndDate - A datetime. The ending date of the interval.
        
    Outputs:
        All dates between StartDate and EndDate (inclusive)
    '''
    for n in range(0, int((EndDate - StartDate).days) + 1, 7):
        yield StartDate + timedelta(n)

# Generate the datetimes corresponding to the date USDM maps were released in 2003
DateGen = WeekDateRange(datetime(2003, 1, 7), datetime(2003, 12, 30))
usdm03dateslist = ['tmp']*52
usdm03years  = np.zeros((52))
usdm03months = np.zeros((52))

for n, date in enumerate(DateGen):
    usdm03dateslist[n] = date.strftime('%Y-%m-%d')
    usdm03years[n]  = date.year
    usdm03months[n] = date.month
    
usdm03dates = np.asarray([datetime.strptime(date, '%Y-%m-%d') for date in usdm03dateslist])

#%%
# cell 23
# New and current case studies
year = 2019
subset = False

CaseStudy(c2['c2'], c4['c4'], FD['FD'], DI['DI'], DP['DP'], year, FD['lat'], FD['lon'], FD['ymd'], FD['year'], FD['month'], USDM['ymd'], USDM['year'], USDM['month'], maskSub, subset)
# CaseStudy(c2['c2'], c4['c4'], FD['FD'], DI['DI'], DP['DP'], year, FD['lat'], FD['lon'], FD['ymd'], FD['year'], FD['month'], usdm03dates, usdm03years, usdm03months, maskSub, subset)

#%%
# cell 24
# Plot various drought intensity maps for comparison with the drought monitor

# Define the date to plot the map for (the map will be for this date to 2 week prior)
DMDate = datetime(1988, 8, 7)

# Perform the calculations and plot the map
DIMap(DI['DI'], DMDate, DI['lat'], DI['lon'], DI['ymd'], savename = 'DI_map-' + DMDate.strftime('%Y-%m-%d') + '.png')
 
    
#%%
# cell 25
# Perform the correlation and composite difference analysis between FC and FD for all years
    
# Remove winter months
ind = np.where( (FD['month'] >= 4) & (FD['month'] <= 10) )[0]
DCgrow = c2['c2'][:,:,ind]
FCgrow = c4['c4'][:,:,ind]
FDgrow = FD['FD'][:,:,ind]

# Perform the statistical calculations
r, rPVal = CorrCoef(FDgrow, FCgrow)

CompDiff, CompDiffPVal = CompositeDifference(FDgrow, FCgrow)


#%%
# cell 26
# Create a map of the correlation coefficient and composite difference

PlotStatMap(r, FD['lon'], FD['lat'], title = 'Correlation Coefficient between Flash Drought and Flash Component for 1979 - 2019', pval = rPVal, 
            savename = 'CorrCoef.png')

PlotStatMap(CompDiff, FD['lon'], FD['lat'], title = 'Composite Mean Difference between Flash Component and Flash Drought for 1979 - 2019', pval = CompDiffPVal,
            cmin = -0.10, cmax = 0.10, cint = 0.01, RoundDigit = 2, savename = 'CompMeanDiff.png')


#%%
# cell 27
# Calculate the average percent area coverage for all years

# For the mask, 1 is land, 0 or nan is ocean. Count the total number of land points by summing.
LandTot = np.nansum(maskSub)
AreaTot = LandTot * 32 * 32 # km^2

# Count the total number of points that experience flash component, drought component, and flash drought in time
DCtmp = np.nansum(c2['c2'][:,:,:], axis = 0)
FCtmp = np.nansum(c4['c4'][:,:,:], axis = 0)
FDtmp = np.nansum(FD['FD'][:,:,:], axis = 0)

DCtot = np.nansum(DCtmp, axis = 0)
FCtot = np.nansum(FCtmp, axis = 0)
FDtot = np.nansum(FDtmp, axis = 0)

DCareaFull = DCtot * 32 * 32 # km^2
FCareaFull = FCtot * 32 * 32 # km^2
FDareaFull = FDtot * 32 * 32 # km^2

# Calculate the mean in growing season of the flash drought and component coverage
tmpind = np.where( (FD['ymd'] >= datetime(2012, 4, 1)) & (FD['ymd'] <= datetime(2012, 10, 31)) )[0]
T = len(tmpind)

DCarea = np.ones((T)) * np.nan
FCarea = np.ones((T)) * np.nan
FDarea = np.ones((T)) * np.nan

DCstd = np.ones((T)) * np.nan
FCstd = np.ones((T)) * np.nan
FDstd = np.ones((T)) * np.nan

for t in range(len(tmpind)):
    ind = np.where( (FD['month'][tmpind[t]] == FD['month']) & (FD['day'][tmpind[t]] == FD['day']) )[0]
    
    DCarea[t] = np.nanmean(DCareaFull[ind])
    FCarea[t] = np.nanmean(FCareaFull[ind])
    FDarea[t] = np.nanmean(FDareaFull[ind])
    
    DCstd[t] = np.nanstd(DCareaFull[ind]/AreaTot * 100)
    FCstd[t] = np.nanstd(FCareaFull[ind]/AreaTot * 100)
    FDstd[t] = np.nanstd(FDareaFull[ind]/AreaTot * 100)

# Calculate the percentage of flash drought and component coverage
perDC = DCarea/AreaTot * 100
perFC = FCarea/AreaTot * 100
perFD = FDarea/AreaTot * 100
#%%
# cell 28
# Plot the the percentage coverage for all years

# Date format for the tick labels
DateFMT = DateFormatter('%b')

# Create the figure
fig = plt.figure(figsize = [14, 10])
ax1 = fig.add_subplot(2, 1, 1)

# Set the title
ax1.set_title('Percentage of Flash Drought Component Coverage', size = 18)

# Plot the drought data
ax1.plot(FD['ymd'][tmpind], perDC, 'r-', label = 'Drought (C2)')

# Set the labels
ax1.set_xlabel('Time', size = 16)
ax1.set_ylabel('Percent Areal Coverage (%)', size = 16)

# Set the tick and tick labels
ax1.xaxis.set_major_formatter(DateFMT)
ax1.set_ylim([21.9, 22.0])
#ax1.set_ylim([19, 20])

for i in ax1.xaxis.get_ticklabels() + ax1.yaxis.get_ticklabels():
    i.set_size(16)

# Add the second plot below the first for the flash component and flash drought
ax2 = fig.add_subplot(2, 1, 2)

# Plot the flash component and flash drought data
ax2.plot(FD['ymd'][tmpind], perFC, 'b-.', label = 'Rapid Drying (C4)')
ax2.plot(FD['ymd'][tmpind], perFD, 'k--', label = 'Flash Drought')

# Add a legend
ax2.legend(loc = 'upper right', fontsize = 16)

# Set axis labels
ax2.set_xlabel('Time', size = 16)
ax2.set_ylabel('Percent Areal Coverage (%)', size = 16)

# Set the tick and tick labels
ax2.xaxis.set_major_formatter(DateFMT)
ax2.set_ylim([0, 2.2])

for i in ax2.xaxis.get_ticklabels() + ax2.yaxis.get_ticklabels():
    i.set_size(16)

plt.show(block = False)

#%%
# cell 29
# Repeat the above calculations and plots, but only east of the Rocky Mountains

# 'East' of the Rocky Mountains is considered east of the Colorado Piedmont (105 W)
LonMin = - 105

# Subset the data to 'east' of the Rockies
maskSubEast, LatSub, LonSub = SubsetData(maskNew, lat, lon, LatMin, LatMax, LonMin, LonMax)
DCSub, LatSub, LonSub   = SubsetData(c2['c2'], c2['lat'], c2['lon'], LatMin, LatMax, LonMin, LonMax)
FCSub, LatSub, LonSub   = SubsetData(c4['c4'], c4['lat'], c4['lon'], LatMin, LatMax, LonMin, LonMax)
FDSub, LatSub, LonSub   = SubsetData(FD['FD'], FD['lat'], FD['lon'], LatMin, LatMax, LonMin, LonMax)

# For the mask, 1 is land, 0 or nan is ocean. Count the total number of land points by summing.
LandTot = np.nansum(maskSubEast)
AreaTot = LandTot * 32 * 32 # km^2

# Count the total number of points that experience flash component, drought component, and flash drought in time
DCtmp = np.nansum(DCSub[:,:,:], axis = 0)
FCtmp = np.nansum(FCSub[:,:,:], axis = 0)
FDtmp = np.nansum(FDSub[:,:,:], axis = 0)

DCtot = np.nansum(DCtmp, axis = 0)
FCtot = np.nansum(FCtmp, axis = 0)
FDtot = np.nansum(FDtmp, axis = 0)

DCareaFull = DCtot * 32 * 32 # km^2
FCareaFull = FCtot * 32 * 32 # km^2
FDareaFull = FDtot * 32 * 32 # km^2

# Calculate the mean in growing season of the flash drought and component coverage
tmpind = np.where( (FD['ymd'] >= datetime(2012, 4, 1)) & (FD['ymd'] <= datetime(2012, 10, 31)) )[0]
T = len(tmpind)

DCarea = np.ones((T)) * np.nan
FCarea = np.ones((T)) * np.nan
FDarea = np.ones((T)) * np.nan

DCstdeast = np.ones((T)) * np.nan
FCstdeast = np.ones((T)) * np.nan
FDstdeast = np.ones((T)) * np.nan

for t in range(len(tmpind)):
    ind = np.where( (FD['month'][tmpind[t]] == FD['month']) & (FD['day'][tmpind[t]] == FD['day']) )[0]
    
    DCarea[t] = np.nanmean(DCareaFull[ind])
    FCarea[t] = np.nanmean(FCareaFull[ind])
    FDarea[t] = np.nanmean(FDareaFull[ind])
    
    DCstdeast[t] = np.nanstd(DCareaFull[ind]/AreaTot * 100)
    FCstdeast[t] = np.nanstd(FCareaFull[ind]/AreaTot * 100)
    FDstdeast[t] = np.nanstd(FDareaFull[ind]/AreaTot * 100)

# Calculate the percentage of flash drought and component coverage
perDCeast = DCarea/AreaTot * 100
perFCeast = FCarea/AreaTot * 100
perFDeast = FDarea/AreaTot * 100
#%%
# cell 30
# Plot the the percentage coverage 'east' of the Rockies

# Date format for the tick labels
DateFMT = DateFormatter('%b')

# Create the figure   
fig = plt.figure(figsize = [14, 10])
ax1 = fig.add_subplot(2, 1, 1)

# Set the title
ax1.set_title('Percentage of Flash Drought Component Coverage "East" of Rocky Mountains', size = 18)

# Plot the drought data
ax1.plot(FD['ymd'][tmpind], perDCeast, 'r-', label = 'Drought (C2)')

# Plot the drought data
ax1.set_xlabel('Time', size = 16)
ax1.set_ylabel('Percent Areal Coverage (%)', size = 16)

# Set the tick and tick labels
ax1.xaxis.set_major_formatter(DateFMT)
ax1.set_ylim([21, 22])

for i in ax1.xaxis.get_ticklabels() + ax1.yaxis.get_ticklabels():
    i.set_size(16)

# Add the second plot below the first for the flash component and flash drought
ax2 = fig.add_subplot(2, 1, 2)

# Plot the flash component and flash drought data
ax2.plot(FD['ymd'][tmpind], perFCeast, 'b-.', label = 'Rapid Drying (C4)')
ax2.plot(FD['ymd'][tmpind], perFDeast, 'k--', label = 'Flash Drought')

# Add a legend
ax2.legend(loc = 'upper right', fontsize = 16)

# Set axis labels
ax2.set_xlabel('Time', size = 16)
ax2.set_ylabel('Percent Areal Coverage (%)', size = 16)

# Set the tick and tick labels
ax2.xaxis.set_major_formatter(DateFMT)
ax2.set_ylim([0, 2.1])

for i in ax2.xaxis.get_ticklabels() + ax2.yaxis.get_ticklabels():
    i.set_size(16)

plt.show(block = False)

#%%
# cell 31
# Perform some quick calculations to get the FD and FC climatology before plotting them

# Initialize variables
I, J, T = FD['FD'].shape
years  = np.unique(FD['year'])

AnnC4 = np.ones((I, J, years.size)) * np.nan
AnnFD = np.ones((I, J, years.size)) * np.nan

# Calculate the average number of rapid intensifications and flash droughts in a year
for y in range(years.size):
    yInd = np.where( (years[y] == FD['year']) & ((FD['month'] >= 4) & (FD['month'] <=10)) )[0] # Second set of conditions ensures only growing season values
    
    # Calculate the mean number of rapid intensification and flash drought for each year
    AnnC4[:,:,y] = np.nanmean(c4['c4'][:,:,yInd], axis = -1)
    
    AnnFD[:,:,y] = np.nanmean(FD['FD'][:,:,yInd], axis = -1)
    
    # Turn nonzero values to nan (each year gets 1 count to the total)
    AnnC4[:,:,y] = np.where(( (AnnC4[:,:,y] == 0) | (np.isnan(AnnC4[:,:,y])) ), 
                            AnnC4[:,:,y], 1) # This changes nonzero  and nan (sea) values to 1.
    
    AnnFD[:,:,y] = np.where(( (AnnFD[:,:,y] == 0) | (np.isnan(AnnFD[:,:,y])) ), 
                            AnnFD[:,:,y], 1) # This changes nonzero  and nan (sea) values to 1.
    
    #### This part needs to be commented out if this code is run without additional help, as the land-sea mask was read in seperately from this
    #AnnC4[:,:,y] = np.where(np.isnan(maskSub[:,:,0]), AnnC4[:,:,y], np.nan)

# Calculate the percentage number of years with rapid intensifications and flash droughts
PerAnnC4 = np.nansum(AnnC4[:,:,:], axis = -1)/years.size
PerAnnFD = np.nansum(AnnFD[:,:,:], axis = -1)/years.size
#PerAnnC4 = np.nanmean(c4['c4'][:,:,:], axis = -1)

# Turn 0 values into nan
PerAnnC4 = np.where(PerAnnC4 != 0, PerAnnC4, np.nan)
PerAnnFD = np.where(PerAnnFD != 0, PerAnnFD, np.nan)


#%%
# cell 32
# Create a the first compound climatology figure for the paper
# This two panel plot has the total FD and FC climatology.

# Set colorbar information
cmin = -20; cmax = 80; cint = 1
clevs = np.arange(-20, cmax + cint, cint)
nlevs = len(clevs)
cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)


# Get the normalized color values
norm = mcolors.Normalize(vmin = 0, vmax = cmax)

# Generate the colors from the orginal color map in range from [0, cmax]
colors = cmap(np.linspace(1 - (cmax - 0)/(cmax - cmin), 1, cmap.N))  ### Note, in the event cmin and cmax share the same sign, 1 - (cmax - cmin)/cmax should be used
colors[:4,:] = np.array([1., 1., 1., 1.]) # Change the value of 0 to white

# Create a new colorbar cut from the colors in range [0, cmax.]
ColorMap = mcolors.LinearSegmentedColormap.from_list('cut_hot_r', colors)

colorsNew = cmap(np.linspace(0, 1, cmap.N))
colorsNew[abs(cmin)-1:abs(cmin)+1, :] = np.array([1., 1., 1., 1.]) # Change the value of 0 in the plotted colormap to white
cmap = mcolors.LinearSegmentedColormap.from_list('hot_r', colorsNew)

# Shapefile information
# ShapeName = 'Admin_1_states_provinces_lakes_shp'
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



# Create the plots
fig = plt.figure(figsize = [12, 10])


# Flash Drought plot
ax1 = fig.add_subplot(2, 1, 1, projection = fig_proj)

# Set the flash drought title
ax1.set_title('Percent of Years from 1979 - 2019 with Flash Drought', size = 18)

# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS)
ax1.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
ax1.add_feature(cfeature.STATES)
ax1.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
ax1.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Adjust the ticks
ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax1.set_yticklabels(LatLabel, fontsize = 18)
ax1.set_xticklabels(LonLabel, fontsize = 18)

ax1.xaxis.set_major_formatter(LonFormatter)
ax1.yaxis.set_major_formatter(LatFormatter)

# Plot the flash drought data
cs = ax1.pcolormesh(FD['lon'], FD['lat'], PerAnnFD*100, vmin = cmin, vmax = cmax,
                  cmap = cmap, transform = data_proj, zorder = 1)

# Set the map extent to the U.S.
ax1.set_extent([-130, -65, 23.5, 48.5])



# Rapid Intensification plot
ax2 = fig.add_subplot(2, 1, 2, projection = fig_proj)

# Set the rapid intensification title
ax2.set_title('Percent of Years from 1979 - 2019 with Rapid Intensification', size = 18)

# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS)
ax2.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
ax2.add_feature(cfeature.STATES)
ax2.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
ax2.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Adjust the ticks
ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax2.set_yticklabels(LatLabel, fontsize = 18)
ax2.set_xticklabels(LonLabel, fontsize = 18)

ax2.xaxis.set_major_formatter(LonFormatter)
ax2.yaxis.set_major_formatter(LatFormatter)

# Plot the rapid intensification data
cs = ax2.pcolormesh(FD['lon'], FD['lat'], PerAnnC4*100, vmin = cmin, vmax = cmax,
                  cmap = cmap, transform = data_proj, zorder = 1)

# Set the map extent to the U.S.
ax2.set_extent([-130, -65, 23.5, 48.5])


# Set the colorbar size and location
cbax2 = fig.add_axes([0.85, 0.13, 0.025, 0.75])

# Create the colorbar
cbar2 = mcolorbar.ColorbarBase(cbax2, cmap = ColorMap, norm = norm, orientation = 'vertical')

# Set the colorbar label
cbar2.ax.set_ylabel('% of years with Flash Drought / Rapid Intensification', fontsize = 18)

# Set the colorbar ticks
cbar2.set_ticks(np.arange(0, 90, 10))
cbar2.ax.set_yticklabels(np.arange(0, 90, 10), fontsize = 16)

# Save the figure
plt.savefig('./Figures/FD_FC_Climatologies.png', bbox_inches = 'tight')
plt.show(block = False)


#%%
# cell 33
# Create the second compount climatology figure for the paper.
# This 4 panel plot has correlation coefficient, composite mean difference, and the statisical significance (right plots)

# The statistical significance levels
alpha = 0.05


# Set colorbar information
cmax = 1; cmin = -1; cint = 0.1
clevs = np.arange(cmin, np.round(cmax+cint, 2), cint) # The np.round removes floating point error in the adition (which is then ceiled in np.arange)
nlevs = len(clevs) - 1
cmap = mcolors.LinearSegmentedColormap.from_list("BuGrRd", ["navy", "cornflowerblue", "gainsboro", "darksalmon", "maroon"], nlevs)


cmin_stats = 0; cmax_stats = 1; cint_stats = 1
clevs_stats = np.arange(cmin_stats, cmax_stats + cint_stats, cint_stats)
nlevs_stats = len(clevs_stats)
cmap_stats = mcolors.LinearSegmentedColormap.from_list("", ["white", "red"], 2)


# Shapefile information
# ShapeName = 'Admin_1_states_provinces_lakes_shp'
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
fig = plt.figure(figsize = [12, 10])

# Adjust some of the figure parameters
plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.95, wspace = 0.55, hspace = -0.55)


# Correlation Plot
ax1 = fig.add_subplot(2, 2, 1, projection = fig_proj)

# Set the left title
ax1.set_title('Statistical Comparison' + '\n' + 'between FC and FD', fontsize = 18)

# Set the label
ax1.set_ylabel('Correlation' + '\n' + 'Coefficient (r)', fontsize = 18)

# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax1.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
ax1.add_feature(cfeature.STATES)
ax1.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
ax1.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Adjust the ticks
ax1.set_xticks(LonLabel, crs = fig_proj)
ax1.set_yticks(LatLabel, crs = fig_proj)

ax1.set_yticklabels(LatLabel, fontsize = 18)
ax1.set_xticklabels(LonLabel, fontsize = 18)

ax1.xaxis.set_major_formatter(LonFormatter)
ax1.yaxis.set_major_formatter(LatFormatter)

# Plot the correlation data
cs = ax1.pcolormesh(FD['lon'], FD['lat'], r[:,:], vmin = cmin, vmax = cmax,
                 cmap = cmap, transform = data_proj, zorder = 1)

# Set the map extent to the U.S.
ax1.set_extent([-129, -65, 25-1.5, 50-1.5])

# Set the correlation colorbar size and location
cbax1 = fig.add_axes([0.42, 0.540, 0.025, 0.20])

# Create the correlation colorbar
cbar1 = fig.colorbar(cs, cax = cbax1, orientation = 'vertical')

# Set the correlation colorbar ticks and label
cbar1.ax.yaxis.set_ticks(np.round(np.arange(-1, 1.5, 0.5), 1))
cbar1.ax.set_xlabel('r', fontsize = 18)

for i in cbar1.ax.get_yticklabels():
    i.set_size(18)




# Composite Mean Difference Plot
ax2 = fig.add_subplot(2, 2, 3, projection = fig_proj)

# Set the axis label
ax2.set_ylabel('Composite Mean' + '\n' + 'Difference (CD)', fontsize = 18)

# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax2.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
ax2.add_feature(cfeature.STATES)
ax2.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
ax2.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Adjust the ticks
ax2.set_xticks(LonLabel, crs = fig_proj)
ax2.set_yticks(LatLabel, crs = fig_proj)

ax2.set_yticklabels(LatLabel, fontsize = 18)
ax2.set_xticklabels(LonLabel, fontsize = 18)

ax2.xaxis.set_major_formatter(LonFormatter)
ax2.yaxis.set_major_formatter(LatFormatter)

# Plot the composite difference data
cs = ax2.pcolormesh(FD['lon'], FD['lat'], CompDiff[:,:], vmin = -0.1, vmax = 0.1,
                 cmap = cmap, transform = data_proj, zorder = 1)

# Set the map extent to the U.S.
ax2.set_extent([-129, -65, 25-1.5, 50-1.5])

# Set the composite difference colorbar size and location
cbax2 = fig.add_axes([0.42, 0.260, 0.025, 0.20])

# Create the composite difference colorbar
cbar2 = fig.colorbar(cs, cax = cbax2, orientation = 'vertical')

# Set the composite difference colorbar ticks and label
cbar2.ax.yaxis.set_ticks(np.round(np.arange(-0.1, 0.15, 0.05), 2))
cbar2.ax.set_xlabel('CD', fontsize = 18)

for i in cbar2.ax.get_yticklabels():
    i.set_size(18)




# Correlation Significance
ax3 = fig.add_subplot(2, 2, 2, projection = fig_proj)

# Set the right title
ax3.set_title('Statistical Significance', fontsize = 18)

# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax3.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
ax3.add_feature(cfeature.STATES)
ax3.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
ax3.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Adjust the ticks
ax3.set_xticks(LonLabel, crs = fig_proj)
ax3.set_yticks(LatLabel, crs = fig_proj)

ax3.set_yticklabels(LatLabel, fontsize = 18)
ax3.set_xticklabels(LonLabel, fontsize = 18)

ax3.xaxis.set_major_formatter(LonFormatter)
ax3.yaxis.set_major_formatter(LatFormatter)

# Determine the areas with statistical significance (correlation)
stipple = (rPVal < alpha/2) | (rPVal > (1-alpha/2))

# Plot the correlation significance
ax3.pcolormesh(FD['lon'], FD['lat'], stipple, vmin = 0, vmax = 1, cmap = cmap_stats, transform = ccrs.PlateCarree(), zorder = 1)

# Set the map extent to the U.S.
ax3.set_extent([-129, -65, 25-1.5, 50-1.5])




# Composite Mean Difference Plot Significance
ax4 = fig.add_subplot(2, 2, 4, projection = fig_proj)    

# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax4.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
ax4.add_feature(cfeature.STATES)
ax4.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
ax4.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Adjust the ticks
ax4.set_xticks(LonLabel, crs = fig_proj)
ax4.set_yticks(LatLabel, crs = fig_proj)

ax4.set_yticklabels(LatLabel, fontsize = 18)
ax4.set_xticklabels(LonLabel, fontsize = 18)

ax4.xaxis.set_major_formatter(LonFormatter)
ax4.yaxis.set_major_formatter(LatFormatter)

# Determine the areas with statistical significance (composite difference)
stipple = (CompDiffPVal < alpha/2) | (CompDiffPVal > (1-alpha/2))

# Plot the composite difference significance
ax4.pcolormesh(FD['lon'], FD['lat'], stipple, vmin = 0, vmax = 1, cmap = cmap_stats, transform = ccrs.PlateCarree(), zorder = 1)

# Set the map extent to the U.S.
ax4.set_extent([-129, -65, 25-1.5, 50-1.5])

# Save the figure
plt.savefig('./Figures/Climatology_Stats_maps.png')
plt.show(block = False)
    
    

#%%
# cell 34
# Create a compound plot for the percentage of areal coverage time series
# This figure is used in the paper

# Create the figure
fig = plt.figure(figsize = [18, 10])

# Adjust some of the figure parameters
plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.95, wspace = 0.25, hspace = 0.25)


### Time Series all U.S. These are a seperate figure
# Drought coverage time series for all U.S.
ax1 = fig.add_subplot(2, 2, 1)

# Set the left title
ax1.set_title('CONUS', size = 18)

# Plot the drought data
ax1.plot(FD['ymd'][tmpind], perDC, 'r-', label = 'Drought (C2)')
ax1.fill_between(FD['ymd'][tmpind], perDC-DCstd, perDC+DCstd, alpha = 0.5, edgecolor = 'r', facecolor = 'r')

# Set the axis labels
ax1.set_xlabel('Time', size = 18)
ax1.set_ylabel('Percent Areal Coverage (%)', size = 14)

# Set the ticks
ax1.xaxis.set_major_formatter(DateFMT)
#ax1.set_yticks(np.round(np.arange(21.900, 22.00, 0.020), 2))
ax1.set_yticks(np.round(np.arange(10.00, 36.00, 5.00), 2))
ax1.set_ylim([10, 35])

for i in ax1.xaxis.get_ticklabels() + ax1.yaxis.get_ticklabels():
    i.set_size(18)



# Rapid intensification and flash drought time series for all U.S.
ax3 = fig.add_subplot(2, 2, 3)

# Plot the rapid intensification and flash drought time series
ax3.plot(FD['ymd'][tmpind], perFC, 'b-.', label = 'Rapid Drying (C4)')
ax3.fill_between(FD['ymd'][tmpind], perFC-FCstd, perFC+FCstd, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
ax3.plot(FD['ymd'][tmpind], perFD, 'k--', label = 'Flash Drought')
ax3.fill_between(FD['ymd'][tmpind], perFD-FDstd, perFD+FDstd, alpha = 0.5, edgecolor = 'k', facecolor = 'k')

# Add a legend
ax3.legend(loc = 'upper right', fontsize = 18)

# Set the label
ax3.set_ylabel('Percent Areal Coverage (%)', size = 18)

# Set the ticks
ax3.xaxis.set_major_formatter(DateFMT)
ax3.set_yticks(np.round(np.arange(0, 2.6, 0.5), 1))
ax3.set_ylim([0, 2.5])

for i in ax3.xaxis.get_ticklabels() + ax3.yaxis.get_ticklabels():
    i.set_size(18)
    
        




### Time series over "eastern" U.S.
# Drought coverage 'east' of the Rockies
ax2 = fig.add_subplot(2, 2, 2)

# Set the right title
ax2.set_title('"East" of Rocky Mountains', size = 18)

# Plot the drought data
ax2.plot(FD['ymd'][tmpind], perDCeast, 'r-', label = 'Drought (C2)')
ax2.fill_between(FD['ymd'][tmpind], perDCeast-DCstdeast, perDCeast+DCstdeast, alpha = 0.5, edgecolor = 'r', facecolor = 'r')

# Set the axis labels
ax2.set_xlabel('Time', size = 18)
ax2.set_ylabel('Percent Areal Coverage (%)', size = 18)

# Set the ticks
ax2.xaxis.set_major_formatter(DateFMT)
#ax2.set_yticks(np.round(np.arange(21.630, 21.675, 0.01), 2))
ax2.set_yticks(np.round(np.arange(10.00, 36.00, 5.00), 2))
ax2.set_ylim([10, 35])

for i in ax2.xaxis.get_ticklabels() + ax2.yaxis.get_ticklabels():
    i.set_size(18)



# Rapid intensification and flash drought time series 'east' of the Rockies
ax4 = fig.add_subplot(2, 2, 4)

# Plot the rapid intensification and flash drought time series
ax4.plot(FD['ymd'][tmpind], perFCeast, 'b-.', label = 'Rapid Drying (C4)')
ax4.fill_between(FD['ymd'][tmpind], perFCeast-FCstdeast, perFCeast+FCstdeast, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
ax4.plot(FD['ymd'][tmpind], perFDeast, 'k--', label = 'Flash Drought')
ax4.fill_between(FD['ymd'][tmpind], perFDeast-FDstdeast, perFDeast+FDstdeast, alpha = 0.5, edgecolor = 'k', facecolor = 'k')

# Add a legend
ax4.legend(loc = 'upper right', fontsize = 18)

# Set the label
ax4.set_ylabel('Percent Areal Coverage (%)', size = 18)

# Set the ticks
ax4.xaxis.set_major_formatter(DateFMT)
ax4.set_yticks(np.round(np.arange(0, 2.6, 0.5), 1))
ax4.set_ylim([0, 2.5])

for i in ax4.xaxis.get_ticklabels() + ax4.yaxis.get_ticklabels():
    i.set_size(18)

# Save the figure
plt.savefig('./Figures/Climatology_Time_series.png')
plt.show(block = False)


#%%
# cell 35
# Create a contingency table for the rapid intensification and flash drought

# Note only scenario (rapid intensification occurred, flash drought did not) yields non-zero or non-100% values
I, J, T = c4['c4'].shape

RInd = np.ones((I, J, T)) * np.nan # RInd: Rapid Intensification, no drought occurred
RId  = np.ones((I, J, T)) * np.nan # RId: Rapid Intensification, drought occurred

# Reshape the variables for less embedded loops
RInd = RInd.reshape(I*J, T, order = 'F')
RId  = RId.reshape(I*J, T, order = 'F')

FD2d = FD['FD'].reshape(I*J, T, order = 'F')
RI2d = c4['c4'].reshape(I*J, T, order = 'F')

# Find the rapid intensification without drought (in which case flash drought is 0)
for ij in range(I*J):
    for t in range(T):
        if (RI2d[ij,t] == 1) & (FD2d[ij,t] == 0):
            RInd[ij,t] = 1
            RId[ij,t]  = 0
        elif (RI2d[ij,t] == 1) & (FD2d[ij,t] == 1):
            RInd[ij,t] = 0
            RId[ij,t]  = 1
        else:
            RInd[ij,t] = np.nan
            RId[ij,t]  = np.nan
            
# Reshape the variables back into a 3D array
RInd = RInd.reshape(I, J, T, order = 'F')
RId  = RId.reshape(I, J, T, order = 'F')

# Apply thte land-sea mask
RInd[maskSub[:,:,0] == 0] = np.nan
RId[maskSub[:,:,0] == 0] = np.nan

#%%
# cell 36
# Plot the contingency table

# Focus on the growing season
ind = np.where( (c4['month'] >= 4) & (c4['month'] <= 10) )[0]


# Set colorbar information
cmin = 0; cmax = 100; cint = 1
clevs = np.arange(cmin, cmax + cint, cint)
nlevs = len(clevs)
cmap  = plt.get_cmap(name = 'Reds', lut = nlevs)


# Shapefile information
# ShapeName = 'Admin_1_states_provinces_lakes_shp'
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



# Create the plots
fig = plt.figure(figsize = [12, 10])


# Rapid intensification without drought plot
ax1 = fig.add_subplot(1, 2, 1, projection = fig_proj)

# Left plot; Correct identification frequency

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax1.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax1.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax1.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set a local title
ax1.set_title('Frequency of Rapid Intensification Events' + '\n' + 'without Drought', fontsize = 16)

# Set the tick information
ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax1.set_yticklabels(LatLabel, fontsize = 16)
ax1.set_xticklabels(LonLabel, fontsize = 16)

ax1.xaxis.set_major_formatter(LonFormatter)
ax1.yaxis.set_major_formatter(LatFormatter)

ax1.tick_params(axis = 'both', which = 'major', length = 3, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = True, labeltop = False,
                labelleft = True, labelright = False)

# Plot the data
cs = ax1.pcolormesh(c4['lon'], c4['lat'], np.nanmean(RInd[:,:,ind], axis = -1)*100, vmin = cmin, vmax = cmax, cmap = cmap, transform = ccrs.PlateCarree())

# Set the map extent over the U.S.
ax1.set_extent([-129, -65, 25-1.5, 50-1.5])



# Rapid intensification with drought plot
ax2 = fig.add_subplot(1, 2, 2, projection = fig_proj)

# Left plot; Correct identification frequency

# Add features
# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax2.add_feature(cfeature.STATES, edgecolor = 'black', zorder = 6)
ax2.add_geometries(NonUSGeom, crs = ccrs.PlateCarree(), facecolor = 'white', edgecolor = 'white', zorder = 2)
ax2.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Set a local title
ax2.set_title('Frequency of Rapid Intensification Events' + '\n' + 'with Drought', fontsize = 16)

# Set the tick information
ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax2.set_yticklabels(LatLabel, fontsize = 16)
ax2.set_xticklabels(LonLabel, fontsize = 16)

ax2.xaxis.set_major_formatter(LonFormatter)
ax2.yaxis.set_major_formatter(LatFormatter)

ax2.tick_params(axis = 'both', which = 'major', length = 3, width = 1,
                labelsize = 16, bottom = True, top = True, left = True,
                right = True, labelbottom = True, labeltop = False,
                labelleft = False, labelright = True)

# Plot the data
cs = ax2.pcolormesh(c4['lon'], c4['lat'], np.nanmean(RId[:,:,ind], axis = -1)*100, vmin = cmin, vmax = cmax, cmap = cmap, transform = ccrs.PlateCarree())

# Set the map extent over the U.S.
ax2.set_extent([-129, -65, 25-1.5, 50-1.5])


# Set the colorbar location and size
cbax = fig.add_axes([0.125, 0.345, 0.775, 0.022])

# Create the colorbar
cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

# Set the colorbar ticks and labels
cbar.set_ticks(np.round(np.arange(0, 100+20, 20)))
cbar.ax.set_xticklabels(np.round(np.arange(0, 100+20, 20)), fontsize = 18)

cbar.ax.set_xlabel('Frequency of Events (%)', fontsize = 18)

# Save the figure
plt.savefig('./Figures/Rapid_Intensification_Contingency_Table.png')
plt.show(block = False)

#%%
# cell 37
# Some early plots to plot/represent statistical significance

# Statistical significance level
alpha = 0.05

# Some ways to represent signficance (i.e., hatches)
mpl.rcParams['hatch.linewidth'] = 5.0
mpl.rcParams['hatch.color'] = 'c'


# Set colorbar information
cmin = 0; cmax = 1; cint = 1
clevs = np.arange(cmin, cmax + cint, cint)
nlevs = len(clevs)
cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "red"], 2)


# Shapefile information
# ShapeName = 'Admin_1_states_provinces_lakes_shp'
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
fig = plt.figure(figsize = [12, 10])

# Set some of the figure parameters
plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.95, wspace = 0.35, hspace = 0.25)


# Correlation Plot
ax1 = fig.add_subplot(2, 1, 1, projection = fig_proj)

# Set the correlation title
ax1.set_title('Statistical Significance of Correlation Coefficient of FD and FC', fontsize = 18)

# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax1.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'none', zorder = 2)
ax1.add_feature(cfeature.STATES)
ax1.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 2)
ax1.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'none', zorder = 2)

# Adjust the ticks
ax1.set_xticks(LonLabel, crs = fig_proj)
ax1.set_yticks(LatLabel, crs = fig_proj)

ax1.set_yticklabels(LatLabel, fontsize = 18)
ax1.set_xticklabels(LonLabel, fontsize = 18)

ax1.xaxis.set_major_formatter(LonFormatter)
ax1.yaxis.set_major_formatter(LatFormatter)

# Determine the significance locations
stipple = (rPVal < alpha/2) | (rPVal > (1-alpha/2))
stipple[stipple == False] = 0
stipple[stipple == True] = 1

# Plot the significance
ax1.contourf(FD['lon'], FD['lat'], stipple, cmap = cmap, transform = ccrs.PlateCarree(), zorder = 1)
#ax1.plot(FD['lon'][stipple][::15], FD['lat'][stipple][::15], 'o', color = 'Gold', markersize = 1.5, zorder = 1)

# Set the map extent to the U.S.
ax1.set_extent([-129, -65, 25-1.5, 50-1.5])




# Composite Mean Difference Plot
ax2 = fig.add_subplot(2, 1, 2, projection = fig_proj)

# Set the composite difference title    
ax2.set_title('Statistical Significance of Composite Mean Difference between FC and FD', fontsize = 18)

# Ocean and non-U.S. countries covers and "masks" data outside the U.S.
ax2.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'none', zorder = 2)
ax2.add_feature(cfeature.STATES)
ax2.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 2)
ax2.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'none', zorder = 2)

# Adjust the ticks
ax2.set_xticks(LonLabel, crs = fig_proj)
ax2.set_yticks(LatLabel, crs = fig_proj)

ax2.set_yticklabels(LatLabel, fontsize = 18)
ax2.set_xticklabels(LonLabel, fontsize = 18)

ax2.xaxis.set_major_formatter(LonFormatter)
ax2.yaxis.set_major_formatter(LatFormatter)

# Determine the significance locations
stipple = (CompDiffPVal < alpha/2) | (CompDiffPVal > (1-alpha/2))

# Plot the significance
ax2.contourf(FD['lon'], FD['lat'], stipple, cmap = cmap, transform = ccrs.PlateCarree(), zorder = 1)
#ax2.plot(FD['lon'][stipple][::15], FD['lat'][stipple][::15], 'o', color = 'Gold', markersize = 1.5, zorder = 1)

# Set the map extent to the U.S.
ax2.set_extent([-129, -65, 25-1.5, 50-1.5])

# Save the figure
plt.savefig('./Figures/Climatology_Significance_maps.png')
plt.show(block = False)
    

#%%
# cell 38
# Creating some figures for a seminar

# Define some variables
year = 2012
month = 5

month_names = ['Janruary', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

tind = np.where( (FD['year'] == year) & (FD['month'] == month) )[0]

analysis = np.nanmean(DI['DI'][:,:,tind], axis = -1)
#analysis = np.nanmean(c4['c4'][:,:,tind], axis = -1)


# Set colorbar information
cmin = -0; cmax = 20; cint = 1
clevs = np.arange(-20, cmax + cint, cint)
nlevs = len(clevs)
cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)

# Get the normalized color values
norm = mcolors.Normalize(vmin = 0, vmax = cmax)

# Generate the colors from the orginal color map in range from [0, cmax]
colors = cmap(np.linspace(1 - (cmax - 0)/(cmax - cmin), 1, cmap.N))  ### Note, in the event cmin and cmax share the same sign, 1 - (cmax - cmin)/cmax should be used
colors[:4,:] = np.array([1., 1., 1., 1.]) # Change the value of 0 to white

# Create a new colorbar cut from the colors in range [o, cmax.]
ColorMap = mcolors.LinearSegmentedColormap.from_list('cut_hot_r', colors)
#cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "yellow", "orange", "red", "black"], nlevs)

colorsNew = cmap(np.linspace(0, 1, cmap.N))
colorsNew[abs(cmin)-1:abs(cmin)+1, :] = np.array([1., 1., 1., 1.]) # Change the value of 0 in the plotted colormap to white
cmap = mcolors.LinearSegmentedColormap.from_list('hot_r', colorsNew)

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

# Shapefile information
# ShapeName = 'Admin_1_states_provinces_lakes_shp'
ShapeName = 'admin_0_countries'
CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

CountriesReader = shpreader.Reader(CountriesSHP)

USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']

# Create the figure
fig = plt.figure(figsize = [12, 10])
ax  = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Set the title
ax.set_title('Drought Component for ' + month_names[month-1] + ', ' + str(year), size = 18)

# Add features
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'none', zorder = 2)
ax.add_feature(cfeature.STATES)
ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 2)
ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'none', zorder = 2)

# Adjust the ticks
ax.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax.set_yticklabels(LatLabel, fontsize = 10)
ax.set_xticklabels(LonLabel, fontsize = 10)

ax.xaxis.set_major_formatter(LonFormatter)
ax.yaxis.set_major_formatter(LatFormatter)

# Plot the data
cs = ax.pcolormesh(FD['lon'], FD['lat'], analysis*100, vmin = cmin, vmax = cmax,
                  cmap = cmap, transform = data_proj, zorder = 1)

# Set the extent to the U.S.
ax.set_extent([-130, -65, 23.5, 48.5])

# Set the colorbar size and location
cbax = fig.add_axes([0.95, 0.28, 0.015, 0.43])

# Create the colorbar
cbar = mcolorbar.ColorbarBase(cbax, cmap = ColorMap, norm = norm, orientation = 'vertical')

# Set the colorbar label
cbar.ax.set_ylabel('% of days with rapid intensification', fontsize = 14)

#%%
# # cell 39
# # Multiple month plots for the seminar slides

# # Initialize some variables
# year = 2017

# I, J, T = FD['FD'].shape
# months = np.unique(FD['month'])

# MonC2 = np.ones((I, J, months.size)) * np.nan
# MonC4 = np.ones((I, J, months.size)) * np.nan

# MonFD = np.ones((I, J, months.size)) * np.nan

# # Calculate the monthly mean drought and flash components and flash drought for a given year
# for m in range(months.size):
#     mInd = np.where( (months[m] == FD['month']) & (FD['year'] == year) )[0]
    
#     MonC2[:,:,m] = np.nanmean(c2['c2'][:,:,mInd], axis = -1)
#     MonC4[:,:,m] = np.nanmean(c4['c4'][:,:,mInd], axis = -1)
    
#     MonFD[:,:,m] = np.nanmean(FD['FD'][:,:,mInd], axis = -1)

# # Set colorbar information
# cmin = 0; cmax = 20; cint = 1
# clevs = np.arange(-20, cmax + cint, cint)
# nlevs = len(clevs)
# cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)

# # Get the normalized color values
# norm = mcolors.Normalize(vmin = 0, vmax = cmax)

# # Generate the colors from the orginal color map in range from [0, cmax]
# colors = cmap(np.linspace(1 - (cmax - 0)/(cmax - cmin), 1, cmap.N))  ### Note, in the event cmin and cmax share the same sign, 1 - (cmax - cmin)/cmax should be used
# colors[:4,:] = np.array([1., 1., 1., 1.]) # Change the value of 0 to white

# # Create a new colorbar cut from the colors in range [o, cmax.]
# ColorMap = mcolors.LinearSegmentedColormap.from_list('cut_hot_r', colors)
# #cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "yellow", "orange", "red", "black"], nlevs)

# colorsNew = cmap(np.linspace(0, 1, cmap.N))
# colorsNew[abs(cmin)-1:abs(cmin)+1, :] = np.array([1., 1., 1., 1.]) # Change the value of 0 in the plotted colormap to white
# cmap = mcolors.LinearSegmentedColormap.from_list('hot_r', colorsNew)


# # Lonitude and latitude tick information
# lat_int = 10
# lon_int = 20

# LatLabel = np.arange(-90, 90, lat_int)
# LonLabel = np.arange(-180, 180, lon_int)

# LonFormatter = cticker.LongitudeFormatter()
# LatFormatter = cticker.LatitudeFormatter()

# # Projection information
# data_proj = ccrs.PlateCarree()
# fig_proj  = ccrs.PlateCarree()

# # Create the plot
# fig, axes = plt.subplots(figsize = [12, 18], nrows = 2, ncols = 2, 
#                          subplot_kw = {'projection': fig_proj})

# # Adjust some of the figure parameters
# plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.15, hspace = -0.75)

# # Set the main figure title
# fig.suptitle('Flash Drought for ' + str(year), y = 0.66, size = 18)

# ax1 = axes[0,0]; ax2 = axes[0,1]; ax3 = axes[1,0]; ax4 = axes[1,1]
        

# # Plot for May
# ax1.set_title('June', size = 16)

# # Add features
# ax1.coastlines()
# ax1.add_feature(cfeature.BORDERS)
# ax1.add_feature(cfeature.STATES)

# # Adjust the ticks
# ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
# ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())

# ax1.set_yticklabels(LatLabel, fontsize = 10)
# ax1.set_xticklabels(LonLabel, fontsize = 10)

# ax1.xaxis.set_major_formatter(LonFormatter)
# ax1.yaxis.set_major_formatter(LatFormatter)

# # Plot the data
# cs = ax1.pcolormesh(FD['lon'], FD['lat'], MonFD[:,:,5]*100, vmin = cmin, vmax = cmax,
#                   cmap = cmap, transform = data_proj)

# # Set the map extent for the U.S.
# ax1.set_extent([-130, -65, 25, 50])


# # Plot for June
# ax2.set_title('July', size = 16)

# # Add features
# ax2.coastlines()
# ax2.add_feature(cfeature.BORDERS)
# ax2.add_feature(cfeature.STATES)

# # Adjust the ticks
# ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
# ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())

# ax2.set_yticklabels(LatLabel, fontsize = 10)
# ax2.set_xticklabels(LonLabel, fontsize = 10)

# ax2.xaxis.set_major_formatter(LonFormatter)
# ax2.yaxis.set_major_formatter(LatFormatter)

# # Plot the data
# cs = ax2.pcolormesh(FD['lon'], FD['lat'], MonFD[:,:,6]*100, vmin = cmin, vmax = cmax,
#                   cmap = cmap, transform = data_proj)

# # Set the map extent to the U.S.
# ax2.set_extent([-130, -65, 25, 50])


# # Plot for July
# ax3.set_title('August', size = 16)

# # Add features
# ax3.coastlines()
# ax3.add_feature(cfeature.BORDERS)
# ax3.add_feature(cfeature.STATES)

# # Adjust the ticks
# ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
# ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())

# ax3.set_yticklabels(LatLabel, fontsize = 10)
# ax3.set_xticklabels(LonLabel, fontsize = 10)

# ax3.xaxis.set_major_formatter(LonFormatter)
# ax3.yaxis.set_major_formatter(LatFormatter)

# # Plot the data
# cs = ax3.pcolormesh(FD['lon'], FD['lat'], MonFD[:,:,7]*100, vmin = cmin, vmax = cmax,
#                   cmap = cmap, transform = data_proj)

# # Set the map extent for the U.S.
# ax3.set_extent([-130, -65, 25, 50])


# # Plot for August
# ax4.set_title('September', size = 16)

# # Add the features
# ax4.coastlines()
# ax4.add_feature(cfeature.BORDERS)
# ax4.add_feature(cfeature.STATES)

# # Adjust the ticks
# ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
# ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())

# ax4.set_yticklabels(LatLabel, fontsize = 10)
# ax4.set_xticklabels(LonLabel, fontsize = 10)

# ax4.xaxis.set_major_formatter(LonFormatter)
# ax4.yaxis.set_major_formatter(LatFormatter)

# # Plot the data
# cfd = ax4.pcolormesh(FD['lon'], FD['lat'], MonFD[:,:,8]*100, vmin = cmin, vmax = cmax,
#                   cmap = cmap, transform = data_proj)

# # Set the map extent for the U.S.
# ax4.set_extent([-130, -65, 25, 50])

# # Set the colorbar location and size
# cbax = fig.add_axes([0.95, 0.365, 0.015, 0.275])

# # Create the colorbar
# #cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
# cbar = mcolorbar.ColorbarBase(cbax, cmap = ColorMap, norm = norm, orientation = 'vertical')

# # Set the colorbar label
# cbar.ax.set_ylabel('% of days with flash drought', fontsize = 14)


#%%
# cell 40
# One more seminar figure
# This the climatological figure for FD and FC

# Initialize some variables
I, J, T = FD['FD'].shape
years  = np.unique(FD['year'])

AnnC4 = np.ones((I, J, years.size)) * np.nan
AnnFD = np.ones((I, J, years.size)) * np.nan

# Calculate the total number of years that experience flash drought/rapid intensification
for y in range(years.size):
    yInd = np.where( (years[y] == FD['year']) & ((FD['month'] >= 4) & (FD['month'] <=10)) )[0] # Second set of conditions ensures only growing season values are cons
    
    # Calculate the mean number of rapid intensification and flash drought for each year
    AnnC4[:,:,y] = np.nanmean(c4['c4'][:,:,yInd], axis = -1)
    
    AnnFD[:,:,y] = np.nanmean(FD['FD'][:,:,yInd], axis = -1)
    
    # Turn nonzero values to nan (each year gets 1 count to the total)
    AnnC4[:,:,y] = np.where(( (AnnC4[:,:,y] == 0) | (np.isnan(AnnC4[:,:,y])) ), 
                            AnnC4[:,:,y], 1) # This changes nonzero  and nan (sea) values to 1.
    
    AnnFD[:,:,y] = np.where(( (AnnFD[:,:,y] == 0) | (np.isnan(AnnFD[:,:,y])) ), 
                            AnnFD[:,:,y], 1) # This changes nonzero  and nan (sea) values to 1.
    
    #### This part needs to be commented out if this code is run without additional help, as the land-sea mask was read in seperately from this
    #AnnC4[:,:,y] = np.where(np.isnan(maskSub[:,:,0]), AnnC4[:,:,y], np.nan)

# Calculate the percentage number of years with rapid intensifications and flash droughts   
PerAnnC4 = np.nansum(AnnC4[:,:,:], axis = -1)/years.size
PerAnnFD = np.nansum(AnnFD[:,:,:], axis = -1)/years.size
#PerAnnC4 = np.nanmean(c4['c4'][:,:,:], axis = -1)

# Turn 0 values into nan
PerAnnC4 = np.where(PerAnnC4 != 0, PerAnnC4, np.nan)
PerAnnFD = np.where(PerAnnFD != 0, PerAnnFD, np.nan)

# Set colorbar information
cmin = -20; cmax = 80; cint = 1
clevs = np.arange(-20, cmax + cint, cint)
nlevs = len(clevs)
cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)

# Get the normalized color values
norm = mcolors.Normalize(vmin = 0, vmax = cmax)

# Generate the colors from the orginal color map in range from [0, cmax]
colors = cmap(np.linspace(1 - (cmax - 0)/(cmax - cmin), 1, cmap.N))  ### Note, in the event cmin and cmax share the same sign, 1 - (cmax - cmin)/cmax should be used
colors[:4,:] = np.array([1., 1., 1., 1.]) # Change the value of 0 to white

# Create a new colorbar cut from the colors in range [o, cmax.]
ColorMap = mcolors.LinearSegmentedColormap.from_list('cut_hot_r', colors)
#cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "yellow", "orange", "red", "black"], nlevs)

colorsNew = cmap(np.linspace(0, 1, cmap.N))
colorsNew[abs(cmin)-1:abs(cmin)+1, :] = np.array([1., 1., 1., 1.]) # Change the value of 0 in the plotted colormap to white
cmap = mcolors.LinearSegmentedColormap.from_list('hot_r', colorsNew)

# Shapefile information
# ShapeName = 'Admin_1_states_provinces_lakes_shp'
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
fig = plt.figure(figsize = [12, 10])
ax  = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Set the title
ax.set_title('Percent of Years from 1979 - 2019 with Rapid Intensification', size = 18)

# Add features
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
ax.add_feature(cfeature.STATES)
ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Adjust the ticks
ax.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax.set_yticklabels(LatLabel, fontsize = 10)
ax.set_xticklabels(LonLabel, fontsize = 10)

ax.xaxis.set_major_formatter(LonFormatter)
ax.yaxis.set_major_formatter(LatFormatter)

# Plot the data
cs = ax.pcolormesh(FD['lon'], FD['lat'], PerAnnC4*100, vmin = cmin, vmax = cmax,
                  cmap = cmap, transform = data_proj, zorder = 1)

# Set the map extent to the U.S.
ax.set_extent([-130, -65, 23.5, 48.5])

# Set the colorbar location and size
cbax = fig.add_axes([0.95, 0.28, 0.015, 0.43])

# Create the colorbar
#cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
cbar = mcolorbar.ColorbarBase(cbax, cmap = ColorMap, norm = norm, orientation = 'vertical')

# Set the colorbar label and ticks
cbar.ax.set_ylabel('% of years with Rapid Intensification', fontsize = 14)

cbar.set_ticks(np.arange(0, 90, 10))


#%%
# cell 41
# Some checks to ensure the accuracy of the data and calculations 
# (namely that the FD algorithm in Flash_Drought_Criteria_Analysis is correct)

# Perform some subsectioning for a time series of SESR percentiles to check it and the algorithm.

# Some constants to adjust when and where the series is made
YearSelect = 2012
#MonthSelect = 6
#DaySelect   = 10
StateInitial = 'GA'

# Coordinates; KS: 39.18 -98.34, OK: 36.37 -96.78, MO: 39.18 -92.73, NE: 41.71 -98.65, IA: 41.85 -95.22, MN: 44.81 -93.66
# These coordinates are for the SESR time series plots (Figure 4) in the Basara et al. 2019 paper
if StateInitial == 'KS':
    LatSelect  = 39.18
    LonSelect  = -98.34
    
elif StateInitial == 'OK':
    LatSelect  = 35.50
    LonSelect  = -98.50

elif StateInitial == 'MO':
    LatSelect  = 39.18
    LonSelect  = -92.73
    
elif StateInitial == 'NE':
    LatSelect  = 41.71
    LonSelect  = -98.65
    
elif StateInitial == 'IA':
    LatSelect  = 41.85
    LonSelect  = -95.22
    
elif StateInitial == 'MN':
    LatSelect  = 44.81
    LonSelect  = -93.66
    
elif StateInitial == 'MS':
    LatSelect  = 31.56
    LonSelect  = -91.40
    
elif StateInitial == 'GA':
    LatSelect  = 34.43
    LonSelect  = -84.24
    
else:
    LatSelect  = 41.85
    LonSelect  = -95.22

# Load SESR
sesr = load_full_climatology('sesr', 'GFS_grid_sesr_pentad.nc', path = './Data/SESR_Climatology_NARR_grid/') # Load the function from Flash_Drought_Criteria_Analysis.py

# Subset the data to the same values as the criteria data
LatMin = 25
LatMax = 50
LonMin = -130
LonMax = -65
sesrSub, LatSub, LonSub = SubsetData(sesr['sesr'], sesr['lat'], sesr['lon'], LatMin = LatMin, LatMax = LatMax,
                                     LonMin = LonMin, LonMax = LonMax)

# Define a path and filename to save the calculated data to
OutPath  = './Figures/SESR_TimeSeries/'
SaveName = 'dp_2012_' + str(StateInitial) + '.png'


# Collect the data for 2012
tind = np.where( DP['year'] == YearSelect )[0]
#tind = np.where( (DP['month'] == MonthSelect) & (DP['day'] == DaySelect) )[0]

DPYear = DP['DP'][:,:,tind]
CPYear = dsesrp['dsesrp'][:,:,tind]
SESRYear = sesrSub[:,:,tind]

c3Year = c3['c3'][:,:,tind]
c4Year = c4['c4'][:,:,tind]
mdYear = mdsesrp['mdsesrp'][:,:,tind]

# Turn the lat and lon into 1d arrays for selection
I, J, T = DPYear.shape
DP2D    = DPYear.reshape(I*J, T, order = 'F')
CP2D    = CPYear.reshape(I*J, T, order = 'F')
SESR2D  = SESRYear.reshape(I*J, T, order = 'F')
Lat1D   = LatSub.reshape(I*J, order = 'F')
Lon1D   = LonSub.reshape(I*J, order = 'F')

c32d = c3Year.reshape(I*J, T, order = 'F')
c42d = c4Year.reshape(I*J, T, order = 'F')
md2d = mdYear.reshape(I*J, T, order = 'F')

# Select the sesr for the desired time series
ind = np.where( (np.abs(Lat1D - LatSelect) < 0.2) & (np.abs(Lon1D - LonSelect) < 0.2) )[0]
DPSelect = np.nanmean(DP2D[ind,:], axis = 0)
CPSelect = np.nanmean(CP2D[ind,:], axis = 0)
SESRSelect = np.nanmean(SESR2D[ind,:], axis = 0)

c3Select = np.nanmean(c32d[ind,:], axis = 0)
c4Select = np.nanmean(c42d[ind,:], axis = 0)
mdSelect = np.nanmean(md2d[ind,:], axis = 0)

TimeYear = DP['ymd'][tind]

#%%
# cell 42
# Plot the SESR time series


DateFMT = DateFormatter('%b')

# Create the figure
fig = plt.figure(figsize = [12, 10])
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

# SESR plot
# Set the titlee
ax1.set_title(str(YearSelect) + ' SESR (unitless) and various SESR Percentiles (all unitless) time series for ' + str(StateInitial), size = 18)#'%4.2f N, %4.2f E (' + StateInitial +  ')' %(LatSelect, LonSelect), size = 18)

# Plot the SESR data
ax1.plot(TimeYear, SESRSelect, 'b-')

# Set the labels
ax1.set_xlabel(' ', size = 16)
ax1.set_ylabel('SESR', size = 16)

# Set the ticks
ax1.set_ylim([-3.5, 1.5])
ax1.set_xlim(datetime(YearSelect, 4, 1), datetime(YearSelect, 10, 31))

ax1.xaxis.set_major_formatter(DateFMT)
    
for i in ax1.xaxis.get_ticklabels() + ax1.yaxis.get_ticklabels():
    i.set_size(16)


# SESR percentiles plot
# Plot the SESR percentiles
ax2.plot(TimeYear, DPSelect, 'b-')

# Set the labels
ax2.set_xlabel(' ', size = 16)
ax2.set_ylabel('SESR Percentiles', size = 16)

# Set the ticks
ax2.set_ylim([0, 65])
ax2.set_xlim(datetime(YearSelect, 4, 1), datetime(YearSelect, 10, 31))

ax2.xaxis.set_major_formatter(DateFMT)
    
for i in ax2.xaxis.get_ticklabels() + ax2.yaxis.get_ticklabels():
    i.set_size(16)
    
    
# Change in SESR percentile plot
# Plot the change in SESR percentiles
ax3.plot(TimeYear, CPSelect, 'b-')

# Set the labels
ax3.set_xlabel('Time', size = 16)
ax3.set_ylabel(r'$\Delta$SESR Percentiles', size = 16)

# Set the ticks
ax3.set_ylim([0, 100])
ax3.set_xlim(datetime(YearSelect, 4, 1), datetime(YearSelect, 10, 31))

ax3.xaxis.set_major_formatter(DateFMT)
    
for i in ax3.xaxis.get_ticklabels() + ax3.yaxis.get_ticklabels():
    i.set_size(16)

# Save the figure
plt.savefig(OutPath + SaveName, bbox_inches = 'tight')
plt.show(block = False)

#%%
# cell 43
# Write the time series to a .csv for additional comparisons with the Christian et al. 2019 data

# Define the filename and path to save the data to
Path = './Data/TimeSeries/'
filename = 'PercentileTimeSeries-' + str(StateInitial) + '-' + str(YearSelect) + '.csv'

# Create and open the file to save the data to
file = open(Path + filename, 'w')

# Write the variable/header information
file.write('Date' + ',' + 'SESR' + ',' + 'Drought Percentiles' + ',' + 'Change in SESR Percentiles' + '\n') # Headers
file.write('Day' + ','+ 'Unitless' + ',' + 'Unitless' + ',' + 'Unitless' + '\n') # Units

# Write the data
for n in range(len(DPSelect)):
    file.write(TimeYear[n].strftime('%Y-%m-%d') + ',' + str(SESRSelect[n]) + ',' + str(DPSelect[n]) + ',' + str(CPSelect[n]) + '\n') # Add the data

file.close() # Close the file after it is written

#%%
# cell 44
# Plot the cumulative flash drought with a point to see where the coordinate point is

# Some user defined constants
YearSelect = 2012
LatSelect  = 35.50
LonSelect  = -98.50

# Calculate the cumulative flash drought.
tind = np.where(FD['year'] == YearSelect)[0]

FDyear = FD['FD'][:,:,tind]

FDcumul = CalculateCumulative(FDyear) # Note index 60 is where the end of October is

# Lonitude and latitude tick information
lat_int = 10
lon_int = 15

LatLabel = np.arange(-90, 90, lat_int)
LonLabel = np.arange(-180, 180, lon_int)

LonFormatter = cticker.LongitudeFormatter()
LatFormatter = cticker.LatitudeFormatter()

 # Color information
cmin = 0; cmax = 2; cint = 0.5
clevs = np.arange(cmin, cmax + cint, cint)
nlevs = len(clevs)
cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "black", "red"], 3)

# Projection information
data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Shapefile information
# ShapeName = 'Admin_1_states_provinces_lakes_shp'
ShapeName = 'admin_0_countries'
CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

CountriesReader = shpreader.Reader(CountriesSHP)

USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']

# Create the figure
fig = plt.figure(figsize = [16, 18], frameon = True)

# Create the title
fig.suptitle('Cumulative Flash Drought Identified', y = 0.68, size = 20)

# Set the first part of the figure
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Add features
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
ax.add_feature(cfeature.STATES)
ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)

# Adjust the ticks
ax.set_xticks(LonLabel, crs = fig_proj)
ax.set_yticks(LatLabel, crs = fig_proj)

ax.set_yticklabels(LatLabel, fontsize = 18)
ax.set_xticklabels(LonLabel, fontsize = 18)

ax.xaxis.set_major_formatter(LonFormatter)
ax.yaxis.set_major_formatter(LatFormatter)

# Plot the cumulative data
cs = ax.pcolormesh(FD['lon'], FD['lat'], FDcumul[:,:,60], vmin = cmin, vmax = cmax,
                          cmap = cmap, transform = data_proj, zorder = 1)

# Plot the coordinate point
ax.plot(LonSelect, LatSelect, color = 'blue', marker = 'o', markersize = 10, transform = ccrs.Geodetic(), zorder = 2)

# Add a label to the coordinate point giving its coordinates
ax.text(np.round(LonSelect), np.round(LatSelect), str(LatSelect) + '\n' + str(LonSelect), color = 'blue',
        fontsize = 16, transform = ccrs.Geodetic(), zorder = 2)

# Set the map extent to the U.S.
ax.set_extent([-129, -65, 25-1.5, 50-1.5])

plt.show(block = False)

#%%
# cell 45
# Calculate the monthly drought intensity. This was used to examine drought intensity and develop and test the current CaseStudy function

# Define the year to be examined
year = 2012

# Initialize some variables and focus on the year to be examined
I, J, T = DI['DI'].shape

ind = np.where(FD['year'] == year)[0]
YearDates = FD['ymd'][ind]

months = np.unique(FD['month'])
T = len(months)

MonDI = np.ones((I, J, T)) * np.nan

# Calculate the monthly means
for m in range(len(months)):
    MonDI[:,:,m] = MonthAverage(DI['DI'], year, m+1, FD['year'], FD['month'])

# Plots for the monthly persistence maps
    
DroughtIntensityMaps(MonC2, MonDI, FD['lon'], FD['lat'], -130, -65, 25, 50, False, year, 
                cmin = 0, cmax = 5, cint = 1, title = 'Drought Severity for ' + str(year), 
                DClabel = 'Drought Component', DIlabel = 'Drought Intensity', savename = 'Drought-Severity-' + str(year) + '.png')


#%%
# cell 46
# Plot the monthly drought intensity to check the function and ensure it works. 
# This was used to examine drought intensity and develop and test the current CaseStudy function

# Determine the date of focus
day = datetime(2012, 11, 2)
dind = np.where(YearDates == day)[0]

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

# Create the figure
fig = plt.figure(figsize = [16, 18], frameon = True)

# Set the title
fig.suptitle('Title', y = 0.68, size = 28)

# Set the first part of the figure
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Add features
ax.coastlines()
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)

# Adjust the ticks
ax.set_xticks(LonLabel, crs = ccrs.PlateCarree())
ax.set_yticks(LatLabel, crs = ccrs.PlateCarree())

ax.set_yticklabels(LatLabel, fontsize = 22)
ax.set_xticklabels(LonLabel, fontsize = 22)

ax.xaxis.set_major_formatter(LonFormatter)
ax.yaxis.set_major_formatter(LatFormatter)

# Plot the data
cs = ax.pcolormesh(FD['lon'], FD['lat'], MonDI[:,:,6], vmin = cmin, vmax = cmax,
                 cmap = cmap, transform = data_proj)

# Set the map extent
ax.set_extent([-129, -65, 25, 50])

# Set the colorbar location and size
cbax = fig.add_axes([0.12, 0.29, 0.78, 0.015])

# Create the colorbar
cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

# Set the colorbar ticks
cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
cbar.ax.set_xticklabels(['No Drought', 'Moderate', 'Severe', 'Extreme', 'Exceptional'], size = 18)
    

plt.show(block = False)
    
#%%
# cell 47
# Plot the cumulative FC
# This is used to examine the cumulative flash drought and compare it to cumulative map (Figure 2) in Basara et al. 2019
# Note this is used as a supplementary figure in the thesis

# Plot C2 and C4 maps
year = 2012

# Define some constants
title = 'Average Flash and Drought Components for the Growing Season in ' + str(year)
savename = 'FD_Component_panel_map_' + str(year) + '_pentad2.png'

I, J, T = FD['FD'].shape

# Determine the year focus on
ind = np.where(FD['year'] == year)[0]
YearDates = FD['year'][ind]
    
months = np.unique(FD['month'])
T = len(months)

# Initialize some variables
MonC2 = np.ones((I, J, T)) * np.nan
MonC4 = np.ones((I, J, T)) * np.nan
MonFD = np.ones((I, J, T)) * np.nan

# Calculate the monthly means
for m in range(len(months)):
    MonC2[:,:,m] = MonthAverage(c2['c2'], year, m+1, FD['year'], FD['month'])
    MonC4[:,:,m] = MonthAverage(c4['c4'], year, m+1, FD['year'], FD['month'])
    MonFD[:,:,m] = MonthAverage(FD['FD'], year, m+1, FD['year'], FD['month'])

# Set non-zero values to 1
FCMon = np.where(MonC4 == 0, MonC4, 1)
FDMon = np.where(MonFD == 0, MonFD, 1)

# Calculate the cumulative flash component and flash drought
FCCumul = CalculateCumulative(FCMon)
FDCumul = CalculateCumulative(FDMon)

# Re-apply the land-sea mask
FCCumul = np.where(maskSub == 1, FCCumul, np.nan)
FDCumul = np.where(maskSub == 1, FDCumul, np.nan)
    

# CaseMonthlyMaps(MonC2[:,:,3:], MonC4[:,:,3:], MonFD[:,:,3:], FD['lon'], FD['lat'], -105, -85, 30, 50, True, year, 0, 1, 0.1, 0.0, 0.2, 
#                 title = title, DClabel = 'Drought Component', FClabel = 'Flash Component', FDlabel = 'Flash Drought',
#                 savename = savename)

# CumulativeMaps(FCCumul[:,:,3:], FDCumul[:,:,3:], FD['lon'], FD['lat'], -105, -85, 30, 50, True, year,
#                 title = 'Cumulative Flash Component and Flash Drought',
#                 savename = 'cumulative_FD_for_' + str(year) + '2.png')


LonMin = -105
LonMax = -82
LatMin = 30
LatMax = 50
            
FCMon[maskSub[:,:,0] == 0] = np.nan

# Month names
# month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_names = ['May', 'Jun', 'Jul', 'Aug']

# Color information
cmin = 0; cmax = 1; cint = 1
clevs = np.arange(cmin, cmax + cint, cint)
nlevs = len(clevs)
cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "red"], 2)

# Lonitude and latitude tick information
lat_int = 10
lon_int = 15

LatLabel = np.arange(-90, 90, lat_int)
LonLabel = np.arange(-180, 180, lon_int)

LonFormatter = cticker.LongitudeFormatter()
LatFormatter = cticker.LatitudeFormatter()

# Projection information
data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Shapefile information
# ShapeName = 'Admin_1_states_provinces_lakes_shp'
ShapeName = 'admin_0_countries'
CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

CountriesReader = shpreader.Reader(CountriesSHP)

USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']

# Create the figure
# fig, axes = plt.subplots(figsize = [12, 18], nrows = len(month_names), ncols = 1, 
#                          subplot_kw = {'projection': fig_proj})
fig, axes = plt.subplots(figsize = [12, 18], nrows = 1, ncols = len(month_names),
                          subplot_kw = {'projection': fig_proj})

# Adjust some of the figure parameters   
plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.2, hspace = 0.1)

# Set the main title
fig.suptitle('FC (C4) for May - August 2012', y = 0.605, size = 18)


for m in range(len(month_names)):
    ax1 = axes[m]#; ax2 = axes[m,1]
    
    # Flash Component plot
    
    # Add features
    # ax1.coastlines()
    # ax1.add_feature(cfeature.BORDERS)
    ax1.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'none', zorder = 2)
    ax1.add_feature(cfeature.STATES)
    ax1.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 2)
    ax1.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'none', zorder = 2)
    
    # Adjust the ticks
    ax1.set_xticks(LonLabel, crs = fig_proj)
    ax1.set_yticks(LatLabel, crs = fig_proj)
    
    ax1.set_yticklabels(LatLabel, fontsize = 16)
    ax1.set_xticklabels(LonLabel, fontsize = 16)
    
    ax1.xaxis.set_major_formatter(LonFormatter)
    ax1.yaxis.set_major_formatter(LatFormatter)
    
    # Plot the data
    cfc = ax1.pcolormesh(FD['lon'], FD['lat'], FCMon[:,:,m+4], vmin = cmin, vmax = cmax,
                      cmap = cmap, transform = data_proj, zorder = 1)
    
    # Set the map extent
    ax1.set_extent([LonMin, LonMax, LatMin, LatMax])
    
    # Set the title (month for the current map)
    ax1.set_title(month_names[m], size = 16)
    # ax1.set_ylabel(month_names[m], size = 16, labelpad = 20.0, rotation = 0)
    
    # Add coordinate points to respective maps where the time series plots in Figure 4 of Basara et al. 2019 are for comparison
    if m == 0: # Month of May
        ax1.plot(-98.34, 39.18, color = 'k', marker = 'o', markersize = 7, transform = ccrs.Geodetic(), zorder = 2) # Kansas Point
        ax1.text(-98.84, 40.58, 'a)', fontsize = 20, transform = ccrs.Geodetic(), zorder = 2)
        
        ax1.plot(-92.73, 39.18, color = 'k', marker = 'o', markersize = 7, transform = ccrs.Geodetic(), zorder = 2) # Missouri Point
        ax1.text(-92.53, 37.28, 'c)', fontsize = 20, transform = ccrs.Geodetic(), zorder = 2)
        
    elif m == 1: # Month of June
        ax1.plot(-98.65, 41.71, color = 'k', marker = 'o', markersize = 7, transform = ccrs.Geodetic(), zorder = 2) # Nebraska Point
        ax1.text(-101.05, 42.41, 'd)', fontsize = 20, transform = ccrs.Geodetic(), zorder = 2)
        
        ax1.plot(-96.78, 36.37, color = 'k', marker = 'o', markersize = 7, transform = ccrs.Geodetic(), zorder = 2) # Oklahoma Point
        ax1.text(-98.88, 34.37, 'b)', fontsize = 20, transform = ccrs.Geodetic(), zorder = 2)
        
    elif m == 2: # Month of July
        ax1.plot(-95.22, 41.85, color = 'k', marker = 'o', markersize = 7, transform = ccrs.Geodetic(), zorder = 2) # Iowa Point
        ax1.text(-94.62, 41.45, 'e)', fontsize = 20, transform = ccrs.Geodetic(), zorder = 2)
        
    elif m == 3: # Month of August
        ax1.plot(-93.66, 44.81, color = 'k', marker = 'o', markersize = 7, transform = ccrs.Geodetic(), zorder = 2) # Minnesota Point
        ax1.text(-94.16, 41.31, 'f)', fontsize = 20, transform = ccrs.Geodetic(), zorder = 2)

    
    if m == 0:
        # Set some extra tick parameters
        ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                        labelsize = 16, bottom = True, top = True, left = True,
                        right = True, labelbottom = True, labeltop = True,
                        labelleft = True, labelright = False)
        
    elif m == len(month_names)-1:
        # Set some extra tick parameters for the last iteration
        ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                        labelsize = 16, bottom = True, top = True, left = True,
                        right = True, labelbottom = True, labeltop = True,
                        labelleft = False, labelright = True)
        
    else:
        # Set some extra tick parameters for the remaining iterations
        ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                        labelsize = 16, bottom = True, top = True, left = True,
                        right = True, labelbottom = True, labeltop = True,
                        labelleft = False, labelright = False)

plt.show(block = False)