#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:58:03 2019

@author: stuartedris

This script is designed to take SESR data from the North American Regional 
Renalysis dataset (NARR) and identify flash drought at every grid point in the
U.S. and for each pentad using the method detailed in Christian et al. (2019).
The identification of each criteria, flash drought identified, and some general
drought information using SESR are all written as .nc files for analysis
when their calculations are completed.
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
# Name some user defined variables

esrmFN       = 'esr_climatology_mean.nc'
esrstdFN     = 'esr_climatology_std.nc'
sesrClimFN   = 'GFS_grid_sesr_pentad.nc'
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

def load_climatology(SName, filename, path = './Data/SESR_Climatology_NARR_grid/'):
    '''
    This function loads climatological .nc files. This function takes in the
    name of the data, and short name of the variable to load the .nc file. Note
    this function is differentiated the laod_full_climatology in that the data
    it loads has an entry for each pentad in a year (i.e., 1 year of data),
    characteristic of a climatological mean and standard deviation dataset.
    
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
    DateFormat = '%m-%d'
    
    with Dataset(path + filename, 'r') as nc:
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
def load_full_climatology(SName, filename, path = './Data/SESR_Climatology_NARR_grid/'):
    '''
    This function loads full .nc files. This function takes in the
    name of the data, and short name of the variable to load the .nc file. Note
    this function is differentiated the laod_climatology in that the data this
    function loads has an entry for each pentad in the full 40 year dataset.
    
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
    DateFormat = '%Y-%m-%d'
    
    with Dataset(path + filename, 'r') as nc:
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
    path = './Data/pentad_NARR_grid/'
    
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
                         'the 1979 to 2019 time range and for all grid points in the NARR. ' +\
                         'This file contains ' + str(VarName) + '.\n' +\
                         'Variable: ' + str(VarSName) + ' (unitless). This is the ' +\
                         'main variable for this file. It is in the format ' +\
                         'lat x lon x time.\n' +\
                         'lat: The latitude (lat x lon format).\n' +\
                         'lon: The longitude (lat x lon format).\n' +\
                         'date: List of dates starting ' +\
                         '01-01-1979 to 12-31-2019 (%Y-%m-%d format). ' +\
                         'Leap year days (2-29) are excluded.'

        
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
# Function to subset any dataset.
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
#%%
# cell 8
# Load the data

esrm      = load_climatology(esrmSName, esrmFN)
sesr      = load_full_climatology(sesrClimSName, sesrClimFN)
dsesr     = load_full_climatology(dsesrClimSName, dsesrClimFN)
Pensesr   = load_full_climatology(sesrClimSName, sesrPentadFN)
dsesrm    = load_climatology(dsesrmSName, dsesrmFN)
dsesrstd  = load_climatology(dsesrstdSName, dsesrstdFN)

#%%
# cell 9
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
# cell 10
# Load and subset the mask dataset

#### Remove this cell when the .nc files have the mask set included in them (this cell will then become unnecessary)

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

# Load the land-sea mask and associated lat/lon
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

#%%
# cell 11
# Subset data to U.S. (25N, -130E to 50Nm -65E) for easier and quicker calculations
LatMin = 25
LatMax = 50
LonMin = -130
LonMax = -65

sesrSub, LatSub, LonSub     = SubsetData(sesr['sesr'], sesr['lat'], sesr['lon'], LatMin, LatMax, LonMin, LonMax)
dsesrSub, LatSub, LonSub    = SubsetData(dsesr['dsesr'], dsesr['lat'], dsesr['lon'], LatMin, LatMax, LonMin, LonMax)
PenSESRSub, LatSub, LonSub  = SubsetData(Pensesr['sesr'], Pensesr['lat'], Pensesr['lon'], LatMin, LatMax, LonMin, LonMax)
dsesrMSub, LatSub, LonSub   = SubsetData(dsesrm['dsesrm'], dsesrm['lat'], dsesrm['lon'], LatMin, LatMax, LonMin, LonMax)
dsesrSTDSub, LatSub, LonSub = SubsetData(dsesrstd['dsesrsstd'], dsesrstd['lat'], dsesrstd['lon'], LatMin, LatMax, LonMin, LonMax)
maskSub, LatSub, LonSub     = SubsetData(maskNew, lat, lon, LatMin = LatMin, LatMax = LatMax, LonMin = LonMin, LonMax = LonMax)
#%%
# cell 12
# Standardize the NARR dsesr.
I, J, T = dsesrSub.shape

Sdsesr = np.ones((I, J, dsesrSub.shape[-1])) * np.nan
for n, t in enumerate(dsesr['date']):
    date = datetime(1900, dsesr['date'][n].month, dsesr['date'][n].day)
    ind  = np.where( dsesrm['date'] == date )[0]
    Sdsesr[:,:,n] = (dsesrSub[:,:,n] - dsesrMSub[:,:,ind[0]])/dsesrSTDSub[:,:,ind[0]]

#%%
# cell 13
# Initialize the criteria and percentile variables for flash drought identification
I, J, T   = sesrSub.shape

crit1 = np.ones((I, J, T)) * np.nan
crit2 = np.ones((I, J, T)) * np.nan
crit3 = np.ones((I, J, T)) * np.nan
crit4 = np.ones((I, J, T)) * np.nan

FD = np.ones((I, J, T)) * np.nan

dsesrP = np.ones((I, J ,T)) * np.nan
mdsesrP = np.ones((I, J, T)) * np.nan

# Create a time variable for a full year (2012; used to test the algorithm)
time = np.asarray([datetime(2012, d.month, d.day) for d in sesr['ymd']]) 



#%%
# cell 14
# Turn the necessary variables into sapce x time arrays to see if that optamizes code
I, J, T = sesrSub.shape

print('Starting')
sesr2d = sesrSub.reshape(I*J, T, order = 'F')
print('1')
crit1 = crit1.reshape(I*J, T, order = 'F')
print('2')
crit2 = crit2.reshape(I*J, T, order = 'F')
print('3')
crit3 = crit3.reshape(I*J, T, order = 'F')
print('4')
crit4 = crit4.reshape(I*J, T, order = 'F')
print('5')
FD = FD.reshape(I*J, T, order = 'F')

mask2d = maskSub.reshape(I*J, 1, order = 'F')

dsesrP = dsesrP.reshape(I*J, T, order = 'F')
mdsesrP = mdsesrP.reshape(I*J, T, order = 'F')

I, J, T = Sdsesr.shape
Sdsesr2d = Sdsesr.reshape(I*J, T, order = 'F')

#%%
# cell 15
# Use the algorithm to identify flash drought for the NARR.
# Find criteria for the NARR

# Next three lines are used to run the algorithm for only 2012 and test the algorithm
YearSelect = 2012
YearInd = np.where(dsesr['year'] == 2012)[0]
YearSelection = sesr['ymd'][YearInd]

# Initialize some variables.
IJ, T = sesr2d.shape
StartDate  = dsesr['ymd'][-1] # Initialize start date so it always has some value
MinChange  = timedelta(days = 30) # Minimum number of days for criteria 1 to be true
mdDates    = np.asarray([datetime(1900, d.month, d.day) for d in dsesr['date']]) # Create a month/day array with all the months and days in the pentad data
CritPercentile = 40 # Percentile dsesr pentads need to be below for criteria 3 to be true
NumExceptions  = 1 # Number of exceptions allowed for criteria 3
count = -99 # Initial count. Count is used to count exceptions for criteria 3
SixDays  = timedelta(days = 6)
FiveDays = timedelta(days = 5)
OneDay   = timedelta(days = 1)
ZeroDays = timedelta(days = 0)
criteria3 = 0 # Start by assuming criterias 3 and 4 are false
criteria4 = 0
Offset = 100/(2*len(np.unique(sesr['year']))) # Offset between percentile calculations in Python and MatLab (Used to compare results to Christian et al. 2019)
#Offset = 0

for ij in range(IJ):
    print('Currently %4.2f %% done.' %(ij/IJ * 100))

    # Skip over ocean values
    if mask2d[ij,0] == 0:
      continue
    else:
      pass
    
    StartDate = dsesr['ymd'][-1] # Reset the start date so it does not carry over to the next grid point.
    count = -99 # Reset the counter
    criteria3 = 0 # Reset criteria 3 and 4 at the start of each grid point.
    criteria4 = 0
    # for t, date in enumerate(YearSelection): # For looping through only 1 year. Below is for looping through all years.
    for t, date in enumerate(dsesr['ymd'][:-3]): # Note, using dsesr will exclude the last 5 days (pentad), but there is no dsesr for that last pentad so it cannot be used. 
                                                 # An additional pentad is omitted at the end so step t+1 can be examined.
        
        # ### Use these when not looping over all years
        # # Find all days in the full dataset equal to the current day
        # ind = np.where( (sesr['month'] == YearSelection[t].month) & 
        #                 (sesr['day'] == YearSelection[t].day) )[0]
        
        # # Find all days in the pentad dataset equal to the current day
        # penind = np.where( (dsesr['month'] == YearSelection[t].month) & 
        #                   (dsesr['day'] == YearSelection[t].day) )[0]
        
        # # Find the current date indice. This is to be used instead of t when not looping over all years.
        # tind = np.where(YearSelection[t] == sesr['ymd'])[0]
        
        # Find all days in the full dataset equal to the current day
        ind = np.where( (sesr['month'] == sesr['ymd'][t].month) & 
                        (sesr['day'] == sesr['ymd'][t].day) )[0]
        
        # Find all days in the pentad dataset equal to the current day
        penind = np.where( (dsesr['month'] == sesr['ymd'][t].month) & 
                          (dsesr['day'] == sesr['ymd'][t].day) )[0]
        
        # Find the current date indice. This is to be used instead of t when not looping over all years.
        tind = np.where(sesr['ymd'][t] == sesr['ymd'])[0]
        
        # Find the current pentad indice
        DateDelta = date - Pensesr['ymd']
        if (np.mod(date.year, 4) == 0) & (date.month == 3) & (date.day == 1): # Exclude leap years. Feburary 29 can add an extra day, making time delta = 5 days at March 1.
            DateDelta = DateDelta - OneDay
        else:
            DateDelta = DateDelta
            
        pentind   = np.where( (DateDelta < FiveDays) & (DateDelta >= ZeroDays) )[0]
        
        # Note the loop is over Delta SESR, which is marked at the start of the chagne (the  p-1 pentad). Then the current pentad where the FD ends is the i th pentad (current + five days) 
        if (np.mod(date.year, 4) == 0) & (date.month == 3) & (date.day == 2): # Exclude leap days.
            PentadDate = date + SixDays
        else:
            PentadDate = date + FiveDays
        
        ## Calculate the 20th quantile for criteria 2 ##
        percent20 = np.nanpercentile(sesr2d[ij,ind], 20 + Offset)
        
        
        ## Calculate the percentile for criteria 3 that location and time ##
        Crit3Quant = np.nanpercentile(Sdsesr2d[ij,penind], CritPercentile + Offset)
        
        # Determine if this loop is the start of a flash drought
        Crit3Part2 = (criteria3 == 0)

        
        ## Determine the mean change in SESR between start and end dates ##
        #  Note PentadDate is not used as an end date because Delta SESR being addressed, not SESR
        StartInd   = np.where(dsesr['ymd'] == StartDate)[0]
        MeanChange = np.nanmean(Sdsesr2d[ij,StartInd[0]:tind[0]]) # Note this exclude the current pentad, which is still being tested
                                                                     # Also note if StardInd > pentind (as in the initialized case), the slice returns an empty array.
                                                                     # Use tind for pentads; pentind for daily
        
        # Calculate the mean change in SESR for all years
        MeanSESRChanges = np.ones((len(np.unique(sesr['year'])))) * np.nan
        for n, year in enumerate(np.unique(sesr['year'])):
            tmpDateStart = datetime(year, dsesr['month'][StartInd], dsesr['day'][StartInd])
            tmpDateEnd   = datetime(year, dsesr['month'][tind[0]], dsesr['day'][tind[0]]) # Use tind for pentads; pentind for daily
            Crit4Ind = np.where( (dsesr['ymd'] >= tmpDateStart) & (dsesr['ymd'] <= tmpDateEnd) )[0] # This could cause problems for winter droughts (month 12 turns back 1), but should be fine since they are unimportant.
            MeanSESRChanges[n] = np.nanmean(Sdsesr2d[ij,Crit4Ind])
            
        
        # Calculate the 25th percentile for the mean changes
        percent25 = np.nanpercentile(MeanSESRChanges, 25 + Offset)
        
        
        ### Determine criteria 3 for the given pentad ###
        if (Sdsesr2d[ij,tind[0]] <= Crit3Quant) & (Crit3Part2 == 1): # Use tind for pentads; pentind for daily
            criteria3 = 1
            count = 0
            
            StartDate = date
            
            
        elif (Sdsesr2d[ij,tind[0]] <= Crit3Quant): # Use tind for pentads; pentind for daily
            criteria3 = 1
        elif (Sdsesr2d[ij,tind[0]] > Crit3Quant) & (abs(count) < NumExceptions) & (Sdsesr2d[ij,tind[0]+1] <= Crit3Quant) & (sesr2d[ij,tind[0]+1] > sesr2d[ij,tind[0]+3]): # Use tind for pentads; pentind for daily
            # Note this marks the start of the moderation period.
            # The third check ensures this moderation point is not actually recovery
            # The last check is an additional minor rule: SESR at the end of the moderation period must be less than SESR at start of the moderation period
            # (moderation period is peaked at tind+1, so it should end at tind+2 for SESR, since dsesr is at the start of the change)
            # (moderation period must have begun in this pentad since only 1 is allowed, and dsesr marks the start of the change, so it is the tind pentad)
            # Note this minor rule uses SESR, not percentiles
            criteria3 = 1
            count = count + 1
        elif (Sdsesr2d[ij,tind[0]] > Crit3Quant) & (abs(count) >= NumExceptions): # Use tind for pentads; pentind for daily
            # This should catch all points when criteria 3 fails
            criteria3 = 0
            count = -99
            StartDate = dsesr['ymd'][-1] # Reset the start date when criteria 3 ends.
        else:
            # The above should cover all cases in which criteria 3 is true. Anything that comes to here should then be false.
            criteria3 = 0
            count = -99
            StartDate = dsesr['ymd'][-1] # Reset the start date when criteria 3 ends.
            # criteria3 = criteria3 # Do nothing. Between pentads, Crit3Quant is nan (penind is empty), so if block should come to here.
            #                       #   Pentad dates mark the beginning of the pentad, so between pentad dates (i.e., within the pentad
            #                       #   in which criteria 3 is true or false) leave criteria 3 unchanged as it remains the same within the pentad.
        
        # Assign criteria 3 for each day
        if criteria3 == 1:
            crit3[ij,tind[0]] = 1
        else:
            crit3[ij,tind[0]] = 0
            
            
        #### Determine Criteria 1 ###
        # Five days are added here (specifically to PentadDate) because, while date marks the beginning of Delta SESR, the pentad at the end of Delta SESR
        #   must be included as well, which is five days. 
        if ( (PentadDate - StartDate) < MinChange ):# | (narrcrit3[j,i,t] == 0) ):
            crit1[ij,tind[0]+1] = 0
        else:
            crit1[ij,tind[0]+1] = 1
            
            
        ### Determine criteria 4 for the given pentad ###
        if (MeanChange <= percent25) & (crit1[ij,tind[0]+1] == 1):# Seccond part ensures only means of 30 days or more are considered (avoid any skewing of < 30 day means)
            criteria4 = 1
        else:
            criteria4 = 0
        
        # Assign criteria 4 for each day
        if criteria4 == 1:
            crit4[ij,tind[0]] = 1
        else:
            crit4[ij,tind[0]] = 0
            
            
        ### Determine criteria 2 ###
        if sesr2d[ij,tind[0]] <= percent20:
            crit2[ij,tind[0]] = 1
        else:
            crit2[ij,tind[0]] = 0
            

        # Calculate the percentile values for dsesr and mean dsesr 
        dsesrP[ij,tind[0]] = stats.percentileofscore(Sdsesr2d[ij,penind], Sdsesr2d[ij,tind[0]])
        mdsesrP[ij,tind[0]] = stats.percentileofscore(MeanSESRChanges, MeanChange)
        
        
# Print output to examine the variables                     
print(crit1)
print('\n')
print(crit2)
print('\n')                                         
print(crit3)
print('\n')
print(crit4)


#%%
# cell 16
# Initialize and identify the flash drought with the NARR data

I, J, T = dsesrSub.shape

# for t, date in enumerate(YearSelection): # For looping through only 1 year. Below is for looping through all years.
for t in range(T):
    print('Currently %4.2f %% done.' %(t/T * 100))
    
    for ij in range(IJ):
        
        # Skip over ocean values
        if mask2d[ij,0] == 0:
          continue
        else:
          pass
        
        # # Find the current date indice
        # tind = np.where(YearSelection[t] == sesr['ymd'])[0]
        
        # Flash drought is considered at the end of the pentad change (hence, the p+1 pentad for criteria 1 and 2, which does not consider Delta SESR)
        FlashDrought = ((crit1[ij,t+1] == 1) & (crit2[ij,t+1] == 1) & 
                        (crit3[ij,t] == 1) & (crit4[ij,t] == 1))
        # Determine if all criteria have tested true for a given grid point and pentad
        if FlashDrought != 0:
            FD[ij,t] = 1

        else:
            FD[ij,t] = 0
            
                
print(FD)

# FDb = FD
# Use FDb to come back to this point quickly and try to figure out issue with FD analysis


#%%
# cell 17
# Now create a variable to measure the intensity of drought
I, J, T = sesrSub.shape
crit2int = np.ones((I, J, T)) * np.nan
crit2p   = np.ones((I, J, T)) * np.nan

crit2int = crit2int.reshape(I*J, T, order = 'F')
crit2p   = crit2p.reshape(I*J, T, order = 'F')

for ij in range(IJ):
    print('Currently %4.2f %% done.' %(ij/IJ * 100))
    
    # Skip over ocean values
    if mask2d[ij,0] == 0:
      continue
    else:
      pass
    
    for t, date in enumerate(sesr['ymd']):
        # Find all days in the full dataset equal to the current day
        ind = np.where( (sesr['month'] == sesr['ymd'][t].month) & 
                       (sesr['day'] == sesr['ymd'][t].day) )[0]
        
        # Find the current date indice
        tind = np.where(sesr['ymd'][t] == sesr['ymd'])[0]
        
        ## Calculate the 20th quantile for category D1 drought ##
        percent20 = np.nanpercentile(sesr2d[ij,ind], 20 + Offset)
        
        ## Calculate the 11th quantile for category D1 drought ##
        percent11 = np.nanpercentile(sesr2d[ij,ind], 11 + Offset)
        
        ## Calculate the 10th quantile for category D2 drought ##
        percent10 = np.nanpercentile(sesr2d[ij,ind], 10 + Offset)
        
        ## Calculate the 6th quantile for category D2 drought ##
        percent6 = np.nanpercentile(sesr2d[ij,ind], 6 + Offset)
        
        ## Calculate the 5th quantile for category D3 drought ##
        percent5 = np.nanpercentile(sesr2d[ij,ind], 5 + Offset)
        
        ## Calculate the 3rd quantile for category D3 drought ##
        percent3 = np.nanpercentile(sesr2d[ij,ind], 3 + Offset)
        
        ## Calculate the 2nd quantile for category D4 drought ##
        percent2 = np.nanpercentile(sesr2d[ij,ind], 2 + Offset)
        
        
        ### Find the percentile of the current SESR value ###
        crit2p[ij,t] = stats.percentileofscore(sesr2d[ij,ind], sesr2d[ij,t])
        
        
        ### Determine drought severity ###
        if (sesr2d[ij,tind] <= percent20) & (sesr2d[ij,tind] >= percent11):
            crit2int[ij,t] = 1
        elif (sesr2d[ij,tind] <= percent10) & (sesr2d[ij,tind] >= percent6):
            crit2int[ij,t] = 2
        elif (sesr2d[ij,tind] <= percent5) & (sesr2d[ij,tind] >= percent3):
            crit2int[ij,t] = 3
        elif (sesr2d[ij,tind] <= percent2):
            crit2int[ij,t] = 4
        else:
            crit2int[ij,t] = 0


#%%
# cell 18
# Restore the criteria and flash drought information to 3D arrays
I, J, T = sesrSub.shape
crit1 = crit1.reshape(I, J, T, order = 'F')
crit2 = crit2.reshape(I, J, T, order = 'F')
crit3 = crit3.reshape(I, J, T, order = 'F')
crit4 = crit4.reshape(I, J, T, order = 'F')
FD = FD.reshape(I, J, T, order = 'F')

dsesrP = dsesrP.reshape(I, J, T, order = 'F')
mdsesrP = mdsesrP.reshape(I, J, T, order = 'F')

crit2int = crit2int.reshape(I, J, T, order = 'F')
crit2p   = crit2p.reshape(I, J, T, order = 'F')

#%%
# cell 19
# Plot the flash drought for the NARR

# Use this to select the desired date to examine
PlotDate = datetime(2012, 7, 25)
dateind  = np.where(sesr['ymd'] == PlotDate)[0]

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

cs = ax.contourf(LonSub, LatSub, FD[:,:,dateind[0]], levels = clevs,
                 cmap = cmap, transform = data_proj, extend = 'max')

ax.set_extent([-130, -65, 25, 50])

# cbax = fig.add_axes([0.12, 0.35, 0.78, 0.02])
# cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

plt.show()

#%%
# cell 20
# Plot the drought intensity for the NARR

# Use this to select the desired date to examine
PlotDate = datetime(2011, 7, 25)
dateind  = np.where(sesr['ymd'] == PlotDate)[0]

# Color information
cmin = 0; cmax = 4; cint = 1
clevs = np.arange(cmin, cmax + cint, cint)
nlevs = len(clevs)
cmap = plt.get_cmap(name = 'hot_r', lut = nlevs)

# Projection information
data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

fig = plt.figure(figsize = [12, 18])
ax  = fig.add_subplot(1, 1, 1, projection = fig_proj)

ax.coastlines()
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)

cs = ax.contourf(LonSub, LatSub, crit2int[:,:,dateind[0]], levels = clevs,
                 cmap = cmap, transform = data_proj, extend = 'max')

ax.set_extent([-130, -65, 25, 50])

# cbax = fig.add_axes([0.12, 0.35, 0.78, 0.02])
# cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

plt.show()

#%%
# cell 21
# Save the different criteria and FD files for ease of use without long calculations

# Save the criteria
WriteNC(crit1, LatSub[:,:], LonSub[:,:], sesr['ymd'], 'criteria1.nc',
         'criteria 1', 'c1')

WriteNC(crit2, LatSub[:,:], LonSub[:,:], sesr['ymd'], 'criteria2.nc',
         'criteria 2', 'c2')

WriteNC(crit3, LatSub[:,:], LonSub[:,:], sesr['ymd'], 'criteria3.nc',
         'criteria 3', 'c3')

WriteNC(crit4, LatSub[:,:], LonSub[:,:], sesr['ymd'], 'criteria4.nc',
         'criteria 4', 'c4')


# Save the identified flash drought
WriteNC(FD, LatSub[:,:], LonSub[:,:], sesr['ymd'], 'FlashDrought.nc',
         'Flash Drought', 'FD')

# Save the drought intensity
WriteNC(crit2int, LatSub[:,:], LonSub[:,:], sesr['ymd'], 'DroughtIntensity.nc',
         'Drought Intensity', 'DI')

WriteNC(crit2p, LatSub[:,:], LonSub[:,:], sesr['ymd'], 'DroughtPercentile.nc',
         'Drought Percentile', 'DP')

# Save the save the delta SESR percentiles
WriteNC(dsesrP, LatSub[:,:], LonSub[:,:], sesr['ymd'], 'DeltaSesrPercentiles.nc',
         'Delta SESR Percentiles', 'dsesrp')

WriteNC(mdsesrP, LatSub[:,:], LonSub[:,:], sesr['ymd'], 'MeanDeltaSesrPercentiles.nc',
         'Mean Delta SESR Percentiles', 'mdsesrp')

#%%
### The following cells are used to test functions and data

# Dummy cell to test the FD identification
sys.path.append('./FlashDroughtFunctions/')
from FlashDroughtFunctions.FlashDroughtCaseStudies import *

I, J, T = sesrSub.shape
crit1 = crit1.reshape(I, J, T, order = 'F')
crit2 = crit2.reshape(I, J, T, order = 'F')
crit3 = crit3.reshape(I, J, T, order = 'F')
crit4 = crit4.reshape(I, J, T, order = 'F')
FD = FD.reshape(I, J, T, order = 'F')

crit2int = np.ones((I, J, T)) * np.nan
crit2p   = np.ones((I, J, T)) * np.nan

year = 2012
subset = False

CaseStudy(crit2, crit4, FD, crit2int, crit2p, year, LatSub, LonSub, sesr['ymd'], sesr['year'], sesr['month'], maskSub, subset)


#%%
# Perform some subsectioning for a time series of SESR to help see how the criteria analysis uses it.

# Some constants to adjust when and where the series is made

# Coordinates; KS: 39.18 -98.34, OK: 36.37 -96.78, MO: 39.18 -92.73, NE: 41.71 -98.65, IA: 41.85 -95.22, MN: 44.81 -93.66
YearSelect = 2012
LatSelect  = 41.85
LonSelect  = -95.22

OutPath  = './Figures/SESR_TimeSeries/'
SaveName = 'sesr_ts_2003_ia.png'


# Collect the data for 2012
tind = np.where ( sesr['year'] == YearSelect )[0]

sesrYear = sesr2d[:,tind]

# Turn the lat and lon into 1d arrays for selection
I, J, T = sesrSub.shape
Lat1D   = LatSub.reshape(I*J, order = 'F')
Lon1D   = LonSub.reshape(I*J, order = 'F')

# Select the sesr for the desired time series
ind = np.where( (np.abs(Lat1D - LatSelect) < 0.2) & (np.abs(Lon1D - LonSelect) < 0.2) )[0]
sesrSelect = np.nanmean(sesrYear[ind,:], axis = 0)

# Next reshape the pentad sesr to examine that.
PentInd = np.where ( Pensesr['year'] == YearSelect )[0]

PenSESR = Pensesr['sesr'][:,:,PentInd]

I, J, T = PenSESR.shape

PenLat1D = Pensesr['lat'].reshape(I*J, order = 'F')
PenLon1D = Pensesr['lon'].reshape(I*J, order = 'F')
PenSESR2D = PenSESR.reshape(I*J, T, order = 'F')

# Select the sesr for the desired time series
Penind = np.where( (np.abs(PenLat1D - LatSelect) < 0.2) & (np.abs(PenLon1D - LonSelect) < 0.2) )[0]
PensesrSelect = np.nanmean(PenSESR2D[Penind,:], axis = 0)

#%%
# Plot the SESR time series
DateFMT = DateFormatter('%b')

fig = plt.figure(figsize = [12, 10])
ax  = fig.add_subplot(1, 1, 1)

ax.set_title('2003 SESR time series for %4.2f N, %4.2f E' %(LatSelect, LonSelect), size = 18)

ax.plot(sesr['ymd'][tind], sesrSelect, 'b-', label = 'Daily SESR')
# ax.plot(Pensesr['ymd'][PentInd], PensesrSelect, 'r--', label = 'Pentad SESR')
# ax.legend(fontsize = 16)

ax.set_xlabel('Time', size = 16)
ax.set_ylabel('SESR (unitless)', size = 16)

ax.set_ylim([-4, 1.5])
ax.set_xlim(datetime(YearSelect, 4, 1), datetime(YearSelect, 10, 31))

ax.xaxis.set_major_formatter(DateFMT)
    
for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
    i.set_size(16)

plt.savefig(OutPath + SaveName, bbox_inches = 'tight')
plt.show(block = False)

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

