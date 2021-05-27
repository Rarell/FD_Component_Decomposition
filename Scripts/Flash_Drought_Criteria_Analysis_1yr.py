#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:58:03 2019

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
  # Name some user defined variables to change for the GFS forecast date and model run

esrmFN       = 'esr_climatology_mean.nc'
esrstdFN     = 'esr_climatology_std.nc'
sesrClimFN   = 'GFS_grid_sesr.nc'
sesrPentadFN = 'GFS_grid_sesr_pentad.nc'
dsesrClimFN  = 'GFS_grid_delta_sesr_pentad.nc'
dsesrmFN     = 'delta_sesr_climatology_mean_pentad.nc'
dsesrstdFN   = 'delta_sesr_climatology_std_pentad.nc'

esrmSName      = 'esrm'
esrstdSName    = 'esrstd'
sesrClimSName  = 'sesr'
dsesrClimSName = 'dsesr'
dsesrmSName    = 'dsesrm'
dsesrstdSName  = 'dsesrsstd'

OutPath = './Figures/'

#%%
# cell 3
  # Examine the .nc files

# Examine climatology values
print(Dataset('./Data/SESR_Climatology/esr_climatology_mean.nc', 'r'))
print(Dataset('./Data/SESR_Climatology/esr_climatology_std.nc', 'r'))
print(Dataset('./Data/SESR_Climatology/GFS_grid_sesr.nc', 'r'))
print(Dataset('./Data/SESR_Climatology/GFS_grid_delta_sesr.nc', 'r'))
print(Dataset('./Data/SESR_Climatology/delta_sesr_climatology_mean.nc', 'r'))
print(Dataset('./Data/SESR_Climatology/delta_sesr_climatology_std.nc', 'r'))


#%%
# cell 4
  # Create a function to import the climatology and annual data
def load_climatology(SName, file, path = './Data/SESR_Climatology_NARR_grid/'):
    '''

    '''
    
    X = {}
    DateFormat = '%m-%d'
    
    with Dataset(path + file, 'r') as nc:
        # Load the grid
        lat = nc.variables['lat'][:,:]
        lon = nc.variables['lon'][:,:]
        # lat = nc.variables['lat'][:]
        # lon = nc.variables['lon'][:]
        
        # lon, lat = np.meshgrid(lon, lat)
        
        X['lat'] = lat
        X['lon'] = lon
        
        # Collect the time information
        time = nc.variables['date'][:]
        dates = np.asarray([datetime.strptime(time[d], DateFormat) for d in range(len(time))])
        
        X['date'] = dates
        X['month'] = np.asarray([d.month for d in dates])
        X['day']   = np.asarray([d.day for d in dates])
        X['ymd']   = np.asarray([datetime(d.year, d.month, d.day) for d in dates])
        
        # Collect the data itself
        X[str(SName)] = nc.variables[str(SName)][:,:,:]
        
    return X

#%%
# cell 5
  # Create a function to import the climatology and annual data
def load_full_climatology(SName, file, path = './Data/SESR_Climatology_NARR_grid/'):
    '''

    '''
    
    X = {}
    DateFormat = '%Y-%m-%d'
    
    with Dataset(path + file, 'r') as nc:
        # Load the grid
        lat = nc.variables['lat'][:,:]
        lon = nc.variables['lon'][:,:]
        # lat = nc.variables['lat'][:]
        # lon = nc.variables['lon'][:]
        
        # lon, lat = np.meshgrid(lon, lat)
        
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
# cell 6
  # Create a function to write a variable to a .nc file
  
def WriteNC(var, lat, lon, dates, filename = 'tmp.nc', VarName = 'tmp', VarSName = 'tmp'):
    '''
    '''
    
    # Define the path
    path = './Data/2012/'
    
    # Determine the spatial and temporal lengths
    I, J, T = var.shape
    T = len(dates)
    
    with Dataset(path + filename, 'w', format = 'NETCDF4') as nc:
        # Write a description for the .nc file
        ##### May need to change to lat x lon format for the lat and lon descriptions
        nc.description = 'This file contains one of the criteria identified by ' +\
                         'the NARR for identifying flash drought (see Christian et al. 2019 ' +\
                         'for the meaning of each criteria). The file FD contains ' +\
                         'the flash drought identified by all four criteria. These files ' +\
                         'contain the identified flash drought/criteria for all days within ' +\
                         'the 1979 to 2016 time range and for all grid points in the NARR. ' +\
                         'This file contains ' + str(VarName) + '.\n' +\
                         'Variable: ' + str(VarSName) + ' (unitless). This is the ' +\
                         'main variable for this file. It is in the format ' +\
                         'lat x lon x time.\n' +\
                         'lat: The latitude (vector form).\n' +\
                         'lon: The longitude (vector form).\n' +\
                         'date: List of dates starting ' +\
                         '01-01-1979 to 12-31-2016 (%Y-%m-%d format). ' +\
                         'Leap year additions (2-29) are excluded.'

        
        # Create the spatial and temporal dimensions
        nc.createDimension('lat', size = I)
        nc.createDimension('lon', size = J)
        nc.createDimension('time', size = T)
        
        # Create the lat and lon variables
        # nc.createVariable('lat', lat.dtype, ('lat', ))
        # nc.createVariable('lon', lon.dtype, ('lon', ))
        
        nc.createVariable('lat', lat.dtype, ('lat', 'lon'))
        nc.createVariable('lon', lon.dtype, ('lat', 'lon'))
        
        # nc.variables['lat'][:] = lat[:]
        # nc.variables['lon'][:] = lon[:]
        
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
# cell 7
  # Load the data

esrm      = load_climatology(esrmSName, esrmFN)
sesr      = load_full_climatology(sesrClimSName, sesrClimFN)
dsesr     = load_full_climatology(dsesrClimSName, dsesrClimFN)
Pensesr   = load_full_climatology(sesrClimSName, sesrPentadFN)
dsesrm    = load_climatology(dsesrmSName, dsesrmFN)
dsesrstd  = load_climatology(dsesrstdSName, dsesrstdFN)

#%%
# cell 8
  # Create a sample plot of SESR to see what the map/grid looks like

# Lonitude and latitude tick information
lat_int = 15
lon_int = 10

lat_label = np.arange(-90, 90, lat_int)
lon_label = np.arange(-180, 180, lon_int)

#lon_formatter = cticker.LongitudeFormatter()
#lat_formatter = cticker.LatitudeFormatter()

# Projection information
data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Colorbar information
cmin = -3.0; cmax = 3.0; cint = 0.20
clevs = np.arange(cmin, cmax+cint, cint)
nlevs = len(clevs) - 1
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


cs = ax.contourf(sesr['lon'], sesr['lat'], sesr['sesr'][:,:,-1], levels = clevs, cmap = cmap, 
                  transform = data_proj, extend = 'both', zorder = 1)

cbax = fig.add_axes([0.92, 0.325, 0.02, 0.35])
cbar = fig.colorbar(cs, cax = cbax)

ax.set_extent([np.nanmin(sesr['lon']), np.nanmax(sesr['lon']), np.nanmin(sesr['lat']), np.nanmax(sesr['lat'])], 
                crs = fig_proj)

plt.show(block = False)

#%%
# cell 9
# Standardize the narr dsesr.
I, J, T = dsesr['dsesr'].shape

Sdsesr = np.ones((I, J, dsesr['dsesr'].shape[-1])) * np.nan
for n, t in enumerate(dsesr['date']):
    date = datetime(1900, dsesr['date'][n].month, dsesr['date'][n].day)
    ind  = np.where( dsesrm['date'] == date )[0]
    Sdsesr[:,:,n] = (dsesr['dsesr'][:,:,n] - dsesrm['dsesrm'][:,:,ind[0]])/dsesrstd['dsesrsstd'][:,:,ind[0]]

#%%
# cell 10
  # Use the same algorithm to identify flash drought for the NARR.

# Initialize the criteria variables for flash drought identification
I, J, T   = sesr['sesr'].shape

crit1 = np.ones((I, J, 365)) * np.nan
crit2 = np.ones((I, J, 365)) * np.nan
crit3 = np.ones((I, J, 365)) * np.nan
crit4 = np.ones((I, J, 365)) * np.nan

FD = np.ones((I, J, 365)) * np.nan

time = np.asarray([datetime(2012, d.month, d.day) for d in esrm['ymd']]) # Create a time variable for a full year


#%%
# cell 11
# Turn the necessary variables into sapce x time arrays to see if that optamizes code
I, J, T = sesr['sesr'].shape

print('Starting')
sesr2d = sesr['sesr'].reshape(I*J, T, order = 'F')
print('1')
crit1 = crit1.reshape(I*J, 365, order = 'F')
print('2')
crit2 = crit2.reshape(I*J, 365, order = 'F')
print('3')
crit3 = crit3.reshape(I*J, 365, order = 'F')
print('4')
crit4 = crit4.reshape(I*J, 365, order = 'F')
print('5')
FD = FD.reshape(I*J, 365, order = 'F')

I, J, T = Sdsesr.shape
Sdsesr2d = Sdsesr.reshape(I*J, T, order = 'F')

#%%
# cell 12
  # Find criteria for the NARR

IJ, T = sesr2d.shape
StartDate  = dsesr['ymd'][-1] # Initialize start_date so it always has some value
MinChange  = timedelta(days = 30)
mdDates    = np.asarray([datetime(1900, d.month, d.day) for d in dsesr['date']]) # Create a month/day array with all the months and days in the pentad data
CritPercentile = 40
NumExceptions  = 1
count = -99
FiveDays = timedelta(days = 5)
OneDay   = timedelta(days = 1)
ZeroDays = timedelta(days = 0)
criteria3 = 0 # Start by assuming criterias 3 and 4 are false
criteria4 = 0

for ij in range(IJ):
    print('Currently %4.2f %% done.' %(ij/IJ * 100))

    StartDate = dsesr['ymd'][-1] # Reset the start date so it does not carry over to the next grid point.
    count = -99 # Reset the counter
    criteria3 = 0 # Reset criteria 3 and 4 at the start of each grid point.
    criteria4 = 0
    for t, date in enumerate(time): # Exclude last pentad since there is not a dsesr for it.
        # Find all days in the full dataset equal to the current day
        ind = np.where( (sesr['month'] == time[t].month) & 
                       (sesr['day'] == time[t].day) )[0]
        
        # Find all days in the pentad dataset equal to the current day
        penind = np.where( (dsesr['month'] == time[t].month) & 
                          (dsesr['day'] == time[t].day) )[0]
        
        # Find the current date indice
        tind = np.where(time[t] == sesr['ymd'])[0]
        
        # Find the current pentad indice
        DateDelta = date - Pensesr['ymd']
        if (np.mod(date.year, 4) == 0) & (date.month == 3) & (date.day == 1): # Exclude leap years. Feburary 29 can add an extra day, making time delta = 5 days at March 1.
            DateDelta = DateDelta - OneDay
        else:
            DateDelta = DateDelta
            
        pentind   = np.where( (DateDelta < FiveDays) & (DateDelta >= ZeroDays) )[0]
#            print(Climdsesr['date'][ind])
        
        ## Calculate the 20th quantile for criteria 2 ##
        percent20 = np.nanpercentile(sesr2d[ij,ind], 20)
        
        
        ## Calculate the quantile for criteria 3 that location and time #
        Crit3Quant = np.nanpercentile(Sdsesr2d[ij,penind], CritPercentile)
        
        # Determine if this loop is the start of a flash drought
        Crit3Part2 = (criteria3 == 0)
        
        
        ## Determine the mean change in SESR between start and end dates ##
        StartInd   = np.where(dsesr['ymd'] == StartDate)[0]
        MeanChange = np.nanmean(Sdsesr2d[ij,StartInd[0]:pentind[0]]) # Note this exclude the current pentad, which is still being tested
                                                                     # Also note if StardInd > pentind (as in the initialized case), the slice returns an empty array.
        
        # Determine the mean change in SESR between start and end date for all years
        tmpDateStart = datetime(1900, dsesr['month'][StartInd], dsesr['day'][StartInd])
        tmpDateEnd   = datetime(1900, dsesr['month'][pentind], dsesr['day'][pentind])
        Crit4Ind = np.where( (mdDates >= tmpDateStart) & (mdDates <= tmpDateEnd) )[0] # This could cause problems for winter droughts (month 12 turns back 1), but should be fine since they are unimportant.
        
        # Calculate the 25th percentile for the mean changes
        percent25 = np.nanpercentile(Sdsesr2d[ij,Crit4Ind], 25)
        
        ### Determine criteria 3 for the given pentad ###
        if (Sdsesr2d[ij,pentind] <= Crit3Quant) & (Crit3Part2 == 1):
            criteria3 = 1
            count = 0
            StartDate = date
        elif (Sdsesr2d[ij,pentind] <= Crit3Quant):
            criteria3 = 1
        elif (Sdsesr2d[ij,pentind] > Crit3Quant) & (abs(count) < NumExceptions):
            criteria3 = 1
            count = count + 1
        elif (Sdsesr2d[ij,pentind] > Crit3Quant) & (abs(count) >= NumExceptions): # This should catch all points when criteria 3 fails
            criteria3 = 0
            count = -99
            StartDate = dsesr['ymd'][-1] # Reset the start date when criteria 3 ends.
        else:
            criteria3 = criteria3 # Do nothing. Between pentads, Crit3Quant is nan (penind is empty), so if block should come to here.
                                  #   Pentad dates mark the beginning of the pentad, so between pentad dates (i.e., within the pentad
                                  #   in which criteria 3 is true or false) leave criteria 3 unchanged as it remains the same within the pentad.
        
        # Assign criteria 3 for each day
        if criteria3 == 1:
            crit3[ij,t] = 1
        else:
            crit3[ij,t] = 0
           
        
        ### Determine criteria 4 for the given pentad ###
        if (MeanChange <= percent25):# | (narrcrit3[j,i,t] == 0):
            criteria4 = 1
        elif (MeanChange > percent25):
            criteria4 = 0
        else:
            criteria4 = criteria4
        
        # Assign criteria 4 for each day
        if criteria4 == 1:
            crit4[ij,t] = 1
        else:
            crit4[ij,t] = 0
            
            
        #### Determine Criteria 1 ###
        # Five days are added here because, while date marks the beginning of Delta SESR, the pentad at the end of Delta SESR
        #   must be included as well, which is five days. 
        if (( (date - StartDate) + FiveDays ) < MinChange):# | (narrcrit3[j,i,t] == 0) ):
            crit1[ij,t] = 0
        else:
            crit1[ij,t] = 1
            
            
        ### Determine criteria 2 ###
        if sesr2d[ij,tind] < percent20:
            crit2[ij,t] = 1
        else:
            crit2[ij,t] = 0
            
        # if (j == latind[0]) & (i == lonind[0]):
        #     print(date - StartDate, ClimSdsesr[i,j,pentind], Crit3Quant, date)
        #     print(count, criteria3)
            # print(ClimSdsesr[i,j,pentind], Crit3Quant, date)
            # print(Climdsesr['dsesr'][i,j,pentind])
        
                        
print(crit1)
print('\n')
print(crit2)
print('\n')                                         
print(crit3)
print('\n')
print(crit4)


#%%
# cell 13
  # Initialize and identify the flash drought with the NARR data



for t in range(365):
    print('Currently %4.2f %% done.' %(t/T * 100))
    
    for ij in range(IJ):
        
        FlashDrought = ((crit1[ij,t] == 1) & (crit2[ij,t] == 1) & 
                        (crit3[ij,t] == 1) & (crit4[ij,t] == 1))
        # Determine if all criteria have tested true for a given grid point
        if FlashDrought == 1:
            FD[ij,t] = 1
        # elif  (t != 0) & (FD[j,i,t-1] != 0):
        #     # narrFD[j,i,t] = 2
        #     FD[j,i,t] = 1
        else:
            FD[ij,t] = 0
            
        # if ((narr_time[t] >= datetime(2012, 7, 15)) & (narr_time[t] <= datetime(2012, 7, 25)) & 
        #     (FlashDrought == 1)):
        #     FD[j,i,t] = 2
                
print(FD)

#%%
# cell 14
# Restore the criteria and flash drought information to 3D arrays
I, J, T = sesr['sesr'].shape
crit1 = crit1.reshape(I, J, 365, order = 'F')
crit2 = crit2.reshape(I, J, 365, order = 'F')
crit3 = crit3.reshape(I, J, 365, order = 'F')
crit4 = crit4.reshape(I, J, 365, order = 'F')
FD = FD.reshape(I, J, 365, order = 'F')

#%%
# cell 14
  # Plot the flash drought for the NARR

# Use this to select the desired date to examine
PlotDate = datetime(2012, 7, 20)
dateind  = np.where(time == PlotDate)[0]

# Color information
cmin = 0; cmax = 1; cint = 0.5
clevs = np.arange(cmin, cmax + cint, cint)
nlevs = len(clevs)
cmap = plt.get_cmap(name = 'binary', lut = nlevs)

# Projection information
data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

fig = plt.figure(figsize = [12, 18])
ax  = fig.add_subplot(1, 1, 1, projection = fig_proj)

ax.coastlines()
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)

cs = ax.contourf(sesr['lon'], sesr['lat'], FD[:,:,dateind[0]], levels = clevs,
                 cmap = cmap, transform = data_proj, extend = 'max')

ax.set_extent([-130, -65, 25, 50])

# cbax = fig.add_axes([0.12, 0.35, 0.78, 0.02])
# cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

plt.show()

#%%
# cell 15
# Save the different criteria and FD files for ease of use without long calculations

# Save the criteria
WriteNC(crit1, sesr['lat'][:,:], sesr['lon'][:,:], time, 'criteria1.nc',
         'criteria 1', 'c1')

WriteNC(crit2, sesr['lat'][:,:], sesr['lon'][:,:], time, 'criteria2.nc',
         'criteria 2', 'c2')

WriteNC(crit3, sesr['lat'][:,:], sesr['lon'][:,:], time, 'criteria3.nc',
         'criteria 3', 'c3')

WriteNC(crit4, sesr['lat'][:,:], sesr['lon'][:,:], time, 'criteria4.nc',
         'criteria 4', 'c4')


# Save the identified flash drought
WriteNC(FD, sesr['lat'][:,:], sesr['lon'][:,:], time, 'FlashDrought.nc',
         'Flash Drought', 'FD')

#%%
  # The next few cells are specifically for creating two figures to be used on the poster for this project

# # Lat/Lon tick information
# LatInt = 5
# LonInt = 10

# LatLabel = np.arange(5, 80, LatInt)
# LonLabel = np.arange(-170, -70, LonInt)

# LatFormatter = cticker.LatitudeFormatter()
# LonFormatter = cticker.LongitudeFormatter()

# # Use this to select the desired date to examine
# PlotDate = datetime(2012, 7, 20)
# dateind  = np.where(time == PlotDate)[0]

# # Color information
# cmin = 0; cmax = 2; cint = 0.5
# clevs = np.arange(cmin, cmax + cint, cint)
# nlevs = len(clevs)
# cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "black", "red"], 3)

# fig = plt.figure(figsize = [12, 18], frameon = True)
# fig.suptitle('Flash Drought Criteria for ' + time[dateind[0]].strftime('%Y/%m/%d') + ' (0 hour forecast for GFS)', y = 0.6, size = 20)

# ### Begin the plots for the GFS data ###
# # Set the first part of the figure
# ax1 = fig.add_subplot(1, 4, 1, projection = fig_proj)

# ax1.set_title('Criteria 1', size = 18)

# ax1.coastlines()
# ax1.add_feature(cfeature.BORDERS)
# ax1.add_feature(cfeature.STATES)

# ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
# ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())

# ax1.set_yticklabels(LatLabel, fontsize = 14)

# ax1.xaxis.tick_bottom()
# ax1.yaxis.tick_left()

# ax1.xaxis.set_major_formatter(LonFormatter)
# ax1.yaxis.set_major_formatter(LatFormatter)

# ax1.set_xticklabels(' ')

# cs = ax1.contourf(Climsesr['lon'], Climsesr['lat'], crit1[:,:,dateind[0]].T, levels = clevs,
#                  cmap = cmap, transform = data_proj, extend = 'max')

# # cbax = fig.add_axes([0.12, 0.35, 0.78, 0.02])
# # cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

# ax1.set_extent([-105, -85, 30, 50])

# ax1.set_ylabel('GFS', size = 18)


# # Set the second part of the figure
# ax2 = fig.add_subplot(1, 4, 2, projection = fig_proj)

# ax2.set_title('Criteria 2', size = 18)

# ax2.coastlines()
# ax2.add_feature(cfeature.BORDERS)
# ax2.add_feature(cfeature.STATES)

# ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
# ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())

# ax2.xaxis.tick_bottom()
# ax2.yaxis.tick_left()

# ax2.xaxis.set_major_formatter(LonFormatter)
# ax2.yaxis.set_major_formatter(LatFormatter)

# ax2.set_xticklabels(' ')
# ax2.set_yticklabels(' ')

# cs = ax2.contourf(Climsesr['lon'], Climsesr['lat'], crit2[:,:,dateind[0]].T, levels = clevs,
#                  cmap = cmap, transform = data_proj, extend = 'max')

# ax2.set_extent([-105, -85, 30, 50])


# # Set the third part of the figure
# ax3 = fig.add_subplot(1, 4, 3, projection = fig_proj)

# ax3.set_title('Criteria 3', size = 18)

# ax3.coastlines()
# ax3.add_feature(cfeature.BORDERS)
# ax3.add_feature(cfeature.STATES)

# ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
# ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())

# ax3.set_yticklabels(LatLabel, fontsize = 14)

# ax3.xaxis.tick_bottom()
# ax3.yaxis.tick_left()

# ax3.xaxis.set_major_formatter(LonFormatter)
# ax3.yaxis.set_major_formatter(LatFormatter)

# ax3.set_xticklabels(' ')
# ax3.set_yticklabels(' ')

# cs = ax3.contourf(Climsesr['lon'], Climsesr['lat'], crit3[:,:,dateind[0]].T, levels = clevs,
#                  cmap = cmap, transform = data_proj, extend = 'max')

# ax3.set_extent([-105, -85, 30, 50])


# # Set the fourth part of the figure
# ax4 = fig.add_subplot(1, 4, 4, projection = fig_proj)

# ax4.set_title('Criteria 4', size = 18)

# ax4.coastlines()
# ax4.add_feature(cfeature.BORDERS)
# ax4.add_feature(cfeature.STATES)

# ax4.set_xticks(LonLabel, crs = ccrs.PlateCarree())
# ax4.set_yticks(LatLabel, crs = ccrs.PlateCarree())

# ax4.xaxis.tick_bottom()
# ax4.yaxis.tick_right()

# ax4.set_yticklabels(LatLabel, fontsize = 14)

# ax4.xaxis.set_major_formatter(LonFormatter)
# ax4.yaxis.set_major_formatter(LatFormatter)

# ax4.set_xticklabels(' ')

# cs = ax4.contourf(Climsesr['lon'], Climsesr'lat'], crit4[:,:,dateind[0]].T, levels = clevs,
#                  cmap = cmap, transform = data_proj, extend = 'max')

# ax4.set_extent([-105, -85, 30, 50])

# savename = 'GFS_criteria_20120720.png'
# plt.savefig('./Figures/' + savename, bbox_inches = 'tight')

# plt.show(block = False)

