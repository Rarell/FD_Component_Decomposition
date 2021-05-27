#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 6 11:11:52 2019

@author: Rarrell/Rarell (Stuart Edris)

This script is designed to take the daily ESR NARR file and calculate the ESR
  (evaporative stress ration; evapotranspiration divided by potential 
  evaportanspiration) and SESR (standardized esr) climatologies, including 
  the mean, standard deviation, and quantiles for ESR and SESR. The calculated
  data will be written for easier and future use.
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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from scipy import stats
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from matplotlib.ticker import MultipleLocator

#%% 
# cell 2
# Examine the files

path = './Data/'

print(Dataset(path + 'esr_narr.nc', 'r'))
print(Dataset(path + 'lat_narr.nc', 'r'))
print(Dataset(path + 'lon_narr.nc', 'r'))
print(Dataset(path + 'sesr_narr.nc', 'r'))

#%% 
# cell 3
# Function to load the files
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

def load3Dnc(filename, SName, path = './Data/'):
    '''
    This function loads 3 dimensional .nc files (with space and time/e.g., the
    esr data). Function is simple as these files only contain the raw data.
    
    
    Inputs:
    - filename: The filename of the .nc file to be loaded.
    - SName: The short name of the variable in the .nc file (i.e., the name to
             call when loading the data)
    - Path: The path from the present direction to the directory the file is in.
    
    Outputs:
    - var: The main variable in the .nc file.
    '''
    
    with Dataset(path + filename, 'r') as nc:
        var = nc.variables[SName][:,:,:]
        
    return var

#%% 
# cell 4
# Calculate the climatological means and standard deviations
  
def CalculateClimatology(var, pentad = False):
    '''
    The function takes in a 3 dimensional variable (2 dimensional space and time)
    and calculates climatological values (mean and standard deviation) for each
    grid point and day in the year.
    
    Inputs:
    - var: 3 dimensional variable whose mean and standard deviation will be
           calculated.
    - pentad: Boolean (True/False) value giving if the time scale of var is 
              pentad (5 day average) or daily.
              
    Outputs:
    - ClimMean: Calculated mean of var for each day/pentad and grid point. 
                ClimMean as the same spatial dimensions as var and 365 (73)
                temporal dimension for daily (pentad) data.
    - ClimStd: Calculated standard deviation for each day/pentad and grid point.
               ClimStd as the same spatial dimensions as var and 365 (73)
               temporal dimension for daily (pentad) data.
    '''
    
    # Obtain the dimensions of the variable
    I, J, T = var.shape
    
    # Count the number of years
    if pentad is True:
        yearLen = int(365/5)
    else:
        yearLen = int(365)
        
    NumYear = int(np.ceil(T/yearLen))
    
    # Create a variable for each day, assumed starting at Jan 1 and no
    #   leap years (i.e., each year is only 365 days each)
    day = np.ones((T)) * np.nan
    
    n = 0
    for i in range(1, NumYear+1):
        if i >= NumYear:
            day[n:T+1] = np.arange(1, len(day[n:T+1])+1)
        else:
            day[n:n+yearLen] = np.arange(1, yearLen+1)
        
        n = n + yearLen
    
    # Initialize the climatological mean and standard deviation variables
    ClimMean = np.ones((I, J, yearLen)) * np.nan
    ClimStd  = np.ones((I, J, yearLen)) * np.nan
    
    # Calculate the mean and standard deviation for each day and at each grid
    #   point
    for i in range(1, yearLen+1):
        ind = np.where(i == day)[0]
        ClimMean[:,:,i-1] = np.nanmean(var[:,:,ind], axis = -1)
        ClimStd[:,:,i-1]  = np.nanstd(var[:,:,ind], axis = -1)
    
    return ClimMean, ClimStd

#%% 
# cell 5
# Create functions to create datetime datasets

def DateRange(StartDate, EndDate):
    '''
    This function takes in two dates and outputs all the dates inbetween
    those two dates.
    
    Inputs:
    - StartDate: A datetime. The starting date of the interval.
    - EndDate: A datetime. The ending date of the interval.
        
    Outputs:
    - All dates between StartDate and EndDate (inclusive)
    '''
    for n in range(int((EndDate - StartDate).days) + 1):
        yield StartDate + timedelta(n) 

#%%
# cell 6
# Create a function to write a variable to a .nc file
  
def WriteNC(var, lat, lon, dates, filename = 'tmp.nc', VarName = 'tmp'):
    '''
    This function writes the 3 dimensionsal data in var into a .nc file. The
    .nc file also contains the latitude and longitude information and dates for
    each time step (in a string with a %Y-%m-%d format).
    
    Input:
    - var: The 3 dimensional (2 spatial and time) data to be written.
    - lat: The latitude (2 dimensional) data for each grid point in var.
    - lon: The longitude (2 dimensional) data for each grid point in var.
    - dates: String of dates (%Y-%m-%d format) for each time step in var.
    - filename: The filename of the .nc file to be written.
    - VarName: The variable name of the main variable for .nc file to be written.
               I.e., what to call to get the main variable for the .nc file being
               written.
    
    Outputs:
    - None. Data is writen the path indicated below with the name specified in
      filename.
    '''
    
    # Define the path
    path = './Data/SESR_Climatology/'
    
    # Determine the spatial and temporal lengths
    J ,I, T = var.shape
    T = len(dates)
    
    with Dataset(path + filename, 'w', format = 'NETCDF4') as nc:
        # Write a description for the .nc file
        nc.description = 'This is one of six .nc files containing synthetic ' +\
                         'data needed to calculate flash drought using SESR. ' +\
                         'These files contain the daily mean and standard ' +\
                         'deviation of ESR, daily SESR, daily change in SESR ' +\
                         'and daily mean and standard deviation of the change ' +\
                         'in SESR. Note all these variables are unitless. ' +\
                         'The variable this file contains ' +\
                         'is ' + str(VarName) + '.\n' +\
                         'Variable: ' + str(VarName) + ' (unitless). This is the ' +\
                         'main variable for this file. It is in the format ' +\
                         'lon x lat x time.\n' +\
                         'lat: The latitude (vector form).\n' +\
                         'lon: The longitude (vector form).\n' +\
                         'date: List of dates starting from 01-01 to 12-31 ' +\
                         'for climatology variables (%m-%d format), or from ' +\
                         '01-01-1979 to 12-31-2016 for SESR/change in SESR ' +\
                         'datasets (%Y-%m-%d format). Leap year additions are excluded.' +\
                         'For change in SESR, date ' +\
                         'is the start date of the change.'
        
        # Create the spatial and temporal dimensions
        nc.createDimension('lat', size = J)
        nc.createDimension('lon', size = I)
        nc.createDimension('time', size = T)
        
        # Create and write the lat and lon variables
        nc.createVariable('lat', lat.dtype, ('lat', 'lon'))
        nc.createVariable('lon', lon.dtype, ('lat', 'lon'))
        
        nc.variables['lat'][:,:] = lat[:,:]
        nc.variables['lon'][:,:] = lon[:,:]
        
        # Create and write the date variable
        nc.createVariable('date', str, ('time', ))
        for n in range(len(dates)):
            nc.variables['date'][n] = np.str(dates[n])
            
        # Create and write the main variable
        nc.createVariable(VarName, var.dtype, ('lat', 'lon', 'time'))
        nc.variables[str(VarName)][:,:,:] = var[:,:,:]
        

#%%
# cell 7
# Load the files
  
TMPesr = load3Dnc('esr_narr.nc', 'esr') # Dataset is time x lat x lon
lat = load2Dnc('lat_narr.nc', 'lat') # Dataset is lat x lon
lon = load2Dnc('lon_narr.nc', 'lon') # Dataset is lat x lon

print(TMPesr)
print(lat)
print(lon)

#%%
# cell 8
# Turn positive lon values to negative (positive values are a sign error and 
#   distort maps).
  
for i in range(len(lon[:,0])):
    ind = np.where( lon[i,:] > 0 )[0]
    lon[i,ind] = -1*lon[i,ind]

# Turn esr into a lat x lon x time variable.
T, I, J = TMPesr.shape
esr = np.ones((I, J, T)) * np.nan
for t in range(T):
    esr[:,:,t] = TMPesr[t,:,:]
    
#%%
# cell 9
# Create a sample plot of ESR to see what the map/grid looks like

# Lonitude and latitude tick information
lat_int = 15
lon_int = 10

lat_label = np.arange(-90, 90, lat_int)
lon_label = np.arange(-180, 180, lon_int)

#lon_formatter = cticker.LongitudeFormatter()
#lat_formatter = cticker.LatitudeFormatter()

# Colorbar information
cmin = 0.0; cmax = 1.0; cint = 0.10
clevs = np.arange(cmin, cmax+cint, cint)
nlevs = len(clevs) - 1
cmap = plt.get_cmap(name = 'Reds', lut = nlevs)
#cmap  = plt.get_cmap(name = 'RdBu_r', lut = nlevs)

data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Figure
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

ax.coastlines()

ax.set_xticks(lon_label, crs = ccrs.PlateCarree())
ax.set_yticks(lat_label, crs = ccrs.PlateCarree())
ax.set_xticklabels(lon_label, fontsize = 16)
ax.set_yticklabels(lat_label, fontsize = 16)

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

#ax.xaxis.set_major_formatter(lon_formatter)
#ax.yaxis.set_major_formatter(lat_formatter)

cs = ax.contourf(lon, lat, esr[:,:,-1], levels = clevs, cmap = cmap, 
                  transform = data_proj, extend = 'both', zorder = 1)

cbax = fig.add_axes([0.92, 0.325, 0.02, 0.35])
cbar = fig.colorbar(cs, cax = cbax)

ax.set_extent([np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], 
                crs = fig_proj)

plt.show(block = False)
  

#%%
# cell 10
# Calculate the climatologies and test/examine by plotting a map for June 30
  
esr_mean, esr_std = CalculateClimatology(esr)

# Reuse the same color map, labels, and projections as the first map on lines
#   284 - 292

data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Figure
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

ax.coastlines()

ax.set_xticks(lon_label, crs = ccrs.PlateCarree())
ax.set_yticks(lat_label, crs = ccrs.PlateCarree())
ax.set_xticklabels(lon_label, fontsize = 16)
ax.set_yticklabels(lat_label, fontsize = 16)

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

#ax.xaxis.set_major_formatter(lon_formatter)
#ax.yaxis.set_major_formatter(lat_formatter)

cs = ax.contourf(lon, lat, esr_mean[:,:,181], levels = clevs, cmap = cmap, 
                  transform = data_proj, extend = 'both', zorder = 1)

cbax = fig.add_axes([0.92, 0.325, 0.02, 0.35])
cbar = fig.colorbar(cs, cax = cbax)

ax.set_extent([np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], 
                crs = fig_proj)

plt.show(block = False)  
#%%
# cell 11
# Calculate SESR

# Initialize variables
sesr = np.ones((esr.shape)) * np.nan
climate_days = np.arange(1, 365+1)
NumYear = int(np.ceil(sesr.shape[-1]/365))
    
# Create a variable for each day, assumed starting at Jan 1 and no
#   leap years (i.e., each year is only 365 days each)
days = np.ones((sesr.shape[-1])) * np.nan
T = len(days)

n = 0
for i in range(1, NumYear+1):
    if i >= NumYear:
        days[n:T+1] = np.arange(1, len(days[n:T+1])+1)
    else:
        days[n:n+365] = np.arange(1, 365+1)
                                 
    n = n + 365

# Calculate SESR
for day in climate_days:
    print('Working on day %i' %day)
    ind = np.where( days == day )[0]
    for t in ind:
        sesr[:,:,t] = (esr[:,:,t] - esr_mean[:,:,day-1])/esr_std[:,:,day-1]


#%%
# cell 12
# Make an example plot of SESR

# Colorbar information
cmin = -3.0; cmax = 3.0; cint = 0.20
clevs = np.arange(cmin, cmax+cint, cint)
nlevs = len(clevs) - 1
# cmap = plt.get_cmap(name = 'Reds', lut = nlevs)
cmap  = plt.get_cmap(name = 'RdBu_r', lut = nlevs)

data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Figure
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

ax.coastlines()

ax.set_xticks(lon_label, crs = ccrs.PlateCarree())
ax.set_yticks(lat_label, crs = ccrs.PlateCarree())
ax.set_xticklabels(lon_label, fontsize = 16)
ax.set_yticklabels(lat_label, fontsize = 16)

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

#ax.xaxis.set_major_formatter(lon_formatter)
#ax.yaxis.set_major_formatter(lat_formatter)

cs = ax.contourf(lon, lat, sesr[:,:,-1], levels = clevs, cmap = cmap, 
                  transform = data_proj, extend = 'both', zorder = 1)

cbax = fig.add_axes([0.92, 0.325, 0.02, 0.35])
cbar = fig.colorbar(cs, cax = cbax)

ax.set_extent([np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], 
                crs = fig_proj)

plt.show(block = False)
#%%
# cell 13
# Calculate the change in SESR
  
delta_sesr = sesr[:,:,1:] - sesr[:,:,:-1]
  
#%%
# cell 14
# Calculate the climatology in the change in SESR
  
delta_sesr_mean, delta_sesr_std = CalculateClimatology(delta_sesr)

#%%
# cell 15
# Write two sets of datetime variables. One for sesr and delta_sesr (every day)
#   date from the start point to the end point.
#   The other is for the climatologies and goes from Jan 01 to Dec 31
  
narr_start = datetime(1979, 1, 1)
narr_end   = datetime(2016, 12, 31)

narr_dates_gen = DateRange(narr_start, narr_end)
narr_dates = ['tmp'] * sesr.shape[-1]
n = 0
for date in narr_dates_gen:
    if date.strftime('%m-%d') == '02-29': # Exclude leap years
        pass
    else:
        narr_dates[n] = date.strftime('%Y-%m-%d')
        n = n + 1


year_start = datetime(1981, 1, 1)
year_end   = datetime(1981, 12, 31)

year_dates_gen = DateRange(year_start, year_end)
year_dates = ['tmp'] * 365
n = 0
for day in year_dates_gen:
    year_dates[n] = day.strftime('%m-%d')
    n = n + 1

#%%
# cell 16
# Write a .nc file for a ESR climatology, change in SESR, and change in SESR
#   climatology
  
# Write the ESR climatology files
WriteNC(esr_mean, lat, lon, year_dates, 'esr_climatology_mean.nc',
         'esrm')

WriteNC(esr_std, lat, lon, year_dates, 'esr_climatology_std.nc',
         'esrstd')

# Write the SESR file
WriteNC(sesr, lat, lon, narr_dates, 'GFS_grid_sesr.nc', 'sesr')

# Write the change in SESR file
WriteNC(delta_sesr, lat, lon, narr_dates[:-1], 'GFS_grid_delta_sesr.nc',
         'dsesr')

# Write the change in SESR climatology files
WriteNC(delta_sesr_mean, lat, lon, year_dates, 'delta_sesr_climatology_mean.nc',
         'dsesrm')

WriteNC(delta_sesr_std, lat, lon, year_dates, 'delta_sesr_climatology_std.nc',
         'dsesrsstd')

#%%
# cell 17
# Print the datasets and see if everything seems to have been made properly

path = './Data/SESR_Climatology/'
print(Dataset(path + 'esr_climatology_mean.nc', 'r'))
print(Dataset(path + 'esr_climatology_std.nc', 'r'))
print(Dataset(path + 'GFS_grid_sesr.nc', 'r'))
print(Dataset(path + 'GFS_grid_delta_sesr.nc', 'r'))
print(Dataset(path + 'delta_sesr_climatology_mean.nc', 'r'))
print(Dataset(path + 'delta_sesr_climatology_std.nc', 'r'))


#%%
# cell 18
# Test the calculated SESR to the already given SESR pentad

# Find and load the given sesr pentad data
sesrPentad = load3Dnc('sesr_narr.nc', 'sesr')

# Plot the data to ensure it was interpolated correctly
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

ax.coastlines()

ax.set_xticks(lon_label, crs = ccrs.PlateCarree())
ax.set_yticks(lat_label, crs = ccrs.PlateCarree())
ax.set_xticklabels(lon_label, fontsize = 16)
ax.set_yticklabels(lat_label, fontsize = 16)

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

#ax.xaxis.set_major_formatter(lon_formatter)
#ax.yaxis.set_major_formatter(lat_formatter)

cs = ax.contourf(lon, lat, sesrPentad[-1,:,:], levels = clevs, cmap = cmap, 
                  transform = data_proj, extend = 'both', zorder = 1)

cbax = fig.add_axes([0.92, 0.325, 0.02, 0.35])
cbar = fig.colorbar(cs, cax = cbax)

ax.set_extent([np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], 
                crs = fig_proj)

#%%
# cell 19
# Create the pentad date data

pentad_dates = ['tmp'] * sesrPentad.shape[0]
n = 0
m = 5
narr_dates_gen = DateRange(narr_start, narr_end) # Note that this date generator has to be remade for every loop
for date in narr_dates_gen:
    if date.strftime('%m-%d') == '02-29': # Exclude leap years
        pass
    elif m < 4:
        m = m + 1
    else:
        pentad_dates[n] = date.strftime('%Y-%m-%d')
        n = n + 1
        m = 0

# Examine the pentad dates to ensure they were calculated correctly.
print(pentad_dates)
    
#%%
# cell 20
# Create a plot for Iowa from May to August (the output of pentad
#   and daily can be compared to Fig. 6a in Christian et al. 2019 to
#   check for correctness)

#### NOTE: This cell needs to be reworked and tested now that lat and lon are 2D matrices and no longer vecors ####

dateFMT = DateFormatter('%m-%d')

# Convert the strings to datetimes
pentad_datetime = np.asarray([datetime.strptime(d, '%Y-%m-%d') for d in pentad_dates])
narr_datetime   = np.asarray([datetime.strptime(d, '%Y-%m-%d') for d in narr_dates])

# Find the dates corresponding to the summer of 2012
pentadind = np.where( (pentad_datetime >= datetime(2012, 5, 1)) &
                     (pentad_datetime <= datetime(2012, 8, 31)) )[0]
narrdatesind = np.where( (narr_datetime >= datetime(2012, 5, 1)) &
                        (narr_datetime <= datetime(2012, 8, 31)) )[0]

# Find the coordinates for Iowa
lonind = np.where( (lon >= -96) & (lon <= -91) )[0]
latind = np.where( (lat >= 41) & (lat <= 43) )[0]


# Create some temporary variables that has latitudinally averaged values
narrtmp   = np.nanmean(sesr[lonind,:,:], axis = 0)
pentadtmp = np.nanmean(sesrPentad[lonind,:,:], axis = 0)

# Create the variable that will be plotted
narrsesr   = np.nanmean(narrtmp[latind,:], axis = 0)
pentadsesr = np.nanmean(pentadtmp[latind,:], axis = 0)

# Create the plot
fig = plt.figure(figsize = [18, 12])
ax  = fig.add_subplot(1, 1, 1)

ax.set_title('Average SESR for Iowa during the summer of 2012', size = 25)

ax.plot(narr_datetime[narrdatesind], narrsesr[narrdatesind], 'b-', label = 'Daily SESR')
ax.plot(pentad_datetime[pentadind], pentadsesr[pentadind], 'r-.', label = 'Pentad SESR')

ax.legend(fontsize = 22, shadow = True)

ax.set_ylim([-3.5, 1])
ax.set_yticks(np.arange(-3, 1+1, 1))
ax.xaxis.set_major_formatter(dateFMT)

ax.set_ylabel('SESR (unitless)', size = 22)
ax.set_xlabel('Time', size = 22)

for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
    i.set_size(22)

for i in ax.xaxis.get_ticklabels():
    i.set_rotation(0)

path = './Figures/Tests/'
savename = 'Daily_SESR_vs_Pentad_SESR.png'
plt.savefig(path + savename, bbox_inches = 'tight')
plt.show(block = False)


#%%
# cell 21
# Calculate pentads to compare to the given values

pentadTest = np.ones((sesrPentad.shape)) * np.nan

n = 0
for i in range(0, sesr.shape[-1], 5):
    pentadTest[n,:,:] = np.nanmean(sesr[:,:,i:i+5], axis = -1) 
    n = n + 1
    
# Plot the test pentad to see how it appears
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

ax.coastlines()

ax.set_xticks(lon_label, crs = ccrs.PlateCarree())
ax.set_yticks(lat_label, crs = ccrs.PlateCarree())
ax.set_xticklabels(lon_label, fontsize = 16)
ax.set_yticklabels(lat_label, fontsize = 16)

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

#ax.xaxis.set_major_formatter(lon_formatter)
#ax.yaxis.set_major_formatter(lat_formatter)

cs = ax.contourf(lon, lat, pentadTest[-1,:,:], levels = clevs, cmap = cmap, 
                  transform = data_proj, extend = 'both', zorder = 1)

cbax = fig.add_axes([0.92, 0.325, 0.02, 0.35])
cbar = fig.colorbar(cs, cax = cbax)

ax.set_extent([np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], 
                crs = fig_proj)


#%%
# cell 22
# Plot the test pentad to see how it compares to the given

#### NOTE: This cell needs to be reworked and tested now that lat and lon are 2D matrices and no longer vecors ####

# Create some temporary variables that has latitudinally averaged values
testtmp   = np.nanmean(pentadTest[lonind,:,:], axis = 0)
pentadtmp = np.nanmean(sesrPentad[lonind,:,:], axis = 0)

# Create the variable that will be plotted
testsesr   = np.nanmean(testtmp[latind,:], axis = 0)
pentadsesr = np.nanmean(pentadtmp[latind,:], axis = 0)

# Create the plot
fig = plt.figure(figsize = [18, 12])
ax  = fig.add_subplot(1, 1, 1)

ax.set_title('Average SESR for Iowa during the summer of 2012, calculated pentads from daily SESR', size = 18)

ax.plot(pentad_datetime[pentadind], testsesr[pentadind], 'b-', label = 'Calculated Pentad SESR')
ax.plot(pentad_datetime[pentadind], pentadsesr[pentadind], 'r-.', label = 'Pentad SESR')

ax.legend(fontsize = 18, shadow = True)

ax.set_ylim([-3.5, 1])

ax.set_ylabel('SESR (unitless)', size = 18)
ax.set_xlabel('Time', size = 18)

for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
    i.set_size(18)
    
for i in ax.xaxis.get_ticklabels():
    i.set_rotation(45)

path = './Figures/Tests/'
savename = 'Calculated_Pentad_SESR_vs_Pentad_SESR.png'
plt.savefig(path + savename, bbox_inches = 'tight')
    
plt.show(block = False)

#%%
# cell 23
# The pentads will be used in criteria 3 and 4, so calculated dsesr for pentads, and climatologies

# Turn esr into a lat x lon x time variable.
T, I, J = pentadTest.shape
sesrTest = np.ones((I, J, T)) * np.nan
for t in range(T):
    sesrTest[:,:,t] = pentadTest[t,:,:]

dsesr_pentad = sesrTest[:,:,1:] - sesrTest[:,:,:-1]

dsesr_pentad_mean, dsesr_pentad_std = CalculateClimatology(dsesr_pentad, pentad = True)

# Create the pentad climatology dates
pentad_year = ['tmp'] * int(365/5)
n = 0
m = 5
pentad_year_gen = DateRange(datetime(1900, 1, 1), datetime(1900, 12, 31)) # Note that this has to be redone for every loop
for date in pentad_year_gen:
    if date.strftime('%m-%d') == '02-29': # Exclude leap years
        pass
    elif m < 4:
        m = m + 1
    else:
        pentad_year[n] = date.strftime('%m-%d')
        n = n + 1
        m = 0


#%%
# cell 24
# Write the pentad sesr, dsesr, and dsesr climatologies to .nc files

# Write the SESR pentad file
WriteNC(sesrTest, lat, lon, pentad_dates, 'GFS_grid_sesr_pentad.nc',
         'sesr')

# Write the Delta SESR pentad file
WriteNC(dsesr_pentad, lat, lon, pentad_dates[:-1], 'GFS_grid_delta_sesr_pentad.nc',
         'dsesr')

# Write the change in SESR pentad climatology files
WriteNC(dsesr_pentad_mean, lat, lon, pentad_year, 'delta_sesr_climatology_mean_pentad.nc',
         'dsesrm')

WriteNC(dsesr_pentad_std, lat, lon, pentad_year, 'delta_sesr_climatology_std_pentad.nc',
         'dsesrsstd')

#%%
# cell 25
# Calculate pentads in esr, then sesr to see if this changes things

ESRpentadTest = np.ones((I, J, T)) * np.nan

n = 0
for i in range(0, esr.shape[-1], 5):
    ESRpentadTest[:,:,n] = np.nanmean(esr[:,:,i:i+5], axis = -1) 
    n = n + 1


ESRpentadm, ESRpentadstd = CalculateClimatology(ESRpentadTest, pentad = True)


# Calculate SESR
sesrTest = np.ones((ESRpentadTest.shape)) * np.nan
climate_days = np.arange(1, 365/5+1)
yearLen = int(365/5)
NumYear = int(np.ceil(sesrTest.shape[-1]/yearLen))
    
# Create a variable for each day, assumed starting at Jan 1 and no
#   leap years (i.e., each year is only 365 days each)
daysTest = np.ones((sesrTest.shape[-1])) * np.nan
T = len(daysTest)

n = 0
for i in range(1, NumYear+1):
    if i >= NumYear:
        daysTest[n:T+1] = np.arange(1, len(daysTest[n:T+1])+1)
    else:
        daysTest[n:n+yearLen] = np.arange(1, yearLen+1)
                                 
    n = n + yearLen

for day in climate_days:
    print('Working on day %i' %day)
    ind = np.where( daysTest == day )[0]
    for t in ind:
        sesrTest[:,:,t] = (ESRpentadTest[:,:,t] - ESRpentadm[:,:,int(day-1)])/ESRpentadstd[:,:,int(day-1)]
        
#%%
# cell 26
# plot the test SESR with given sesr pentad
        
#### NOTE: This cell needs to be reworked and tested now that lat and lon are 2D matrices and no longer vecors ####

# Create some temporary variables that has latitudinally averaged values
testtmp   = np.nanmean(sesrTest[lonind,:,:], axis = 0)
pentadtmp = np.nanmean(sesrPentad[lonind,:,:], axis = 0)

# Create the variable that will be plotted
testsesr   = np.nanmean(testtmp[latind,:], axis = 0)
pentadsesr = np.nanmean(pentadtmp[latind,:], axis = 0)

# Create the plot
fig = plt.figure(figsize = [18, 12])
ax  = fig.add_subplot(1, 1, 1)

ax.set_title('Average SESR for Iowa during the summer of 2012', size = 18)

ax.plot(pentad_datetime[pentadind], testsesr[pentadind], 'b-', label = 'Daily SESR')
ax.plot(pentad_datetime[pentadind], pentadsesr[pentadind], 'r-.', label = 'Pentad SESR')

ax.legend(fontsize = 18, shadow = True)

ax.set_ylim([-3.5, 1])

ax.set_ylabel('SESR (unitless)', size = 18)
ax.set_xlabel('Time', size = 18)

for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
    i.set_size(18)
    
plt.show(block = False)













