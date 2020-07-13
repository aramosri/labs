#########################################################################################################################
# Jake K Carr
# The Ohio State University; Moody's Analytics
# Completely open source
# Estimates Local K-function for a vector of distances and simulates a set of point patterns relative/proportional
# to a given reference measure. Outputs are a set of P-values, K-counts and Buffer Cluster saved to ESRI Shapefiles.

# Close out of Spyder and open Anaconda Prompt. Run:
# conda install -c conda-forge plotnine
# conda install -c conda-forge geopandas

# Modified: 07/01/2018
#########################################################################################################################
# IMPORT MODULES
import time
t0 = time.time()
import os, sys, logging
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import warnings
sys.path.append(r"G:\MAX-Filer\Collab\Labs-kbuzard-S18\Admin\kfunctions\Three Stage K-functions") #directory where kfunction_3Stage.py is
import kfunction_3Stage
t1 = time.time()
print("Import Modules: " + str(t1-t0) + " Seconds.")
# Ignore Warnings: PLOTNINE throws many unecessary warnings 
warnings.filterwarnings("ignore")


#########################################################################################################################
# Read In Data


# Set Local Variables
blocks = r"G:\MAX-Filer\Collab\Labs-kbuzard-S18\Admin\Block Level Analysis\CA_Block_New.shp"
zips = r"G:\MAX-Filer\Collab\Labs-kbuzard-S18\Admin\Block Level Analysis\CA_ZCTA_Man.shp"
points = r"G:\MAX-Filer\Collab\Labs-kbuzard-S18\Admin\Block Level Analysis\GoodLabs_New.shp"
bFeatures = gpd.read_file(blocks).filter(['C000', 'Block_ID', 'geometry'], axis = 1)
zFeatures = gpd.read_file(zips).filter(['ZCTA5CE00', 'Manufa_Emp', 'geometry'], axis = 1)
pFeatures = gpd.read_file(points).filter(['REF_ID', 'SIC', 'geometry'], axis = 1)

# fieldZ is the 'Reference Measure' Variable from zFeatures [aka the null distribution]
fieldZ = "Manufa_Emp"

# ZIPFIELD is the ID Field from zFeatures
ZIPFIELD = "ZCTA5CE00"

# fieldB is the 'Reference Measure' Variable from bFeatures [aka the null distribution]
fieldB = "C000"


#########################################################################################################################
# Distances of Interest

# K-function Distance Vector in Miles
distVec = [0.5, 1, 5, 10]

# How Many Simulations to Run?
simulations = 2 # Perform N simulations

# Collect Variable Names for K-counts
K = []
for i in distVec:
    K.append("K_"+str(i))

del i

logging.basicConfig(filename = os.path.join(os.path.dirname(points), fieldZ +"_"+ fieldB + "_local.log"), level = logging.DEBUG, format = '%(asctime)s %(message)s', datefmt = '%m/%d/%Y %H:%M:%S')
logging.info("Import Modules: " + str(t1-t0) + " Seconds.")


#########################################################################################################################
# Observed K-function Values

# Pull X and Y Coordinates from pointFeatures [Observed Point Shapefile]
pFeatures["X"] = pFeatures.geometry.x
pFeatures["Y"] = pFeatures.geometry.y

# To NUMPY ARRAY
array = pFeatures.loc[:,["X", "Y"]].values

# Number of Points
npts = len(array)

# Observed Distance Matrix Calculation
t2 = time.time()
full = kfunction_3Stage.distanceMatrix(array, array)
t3 = time.time()
print("Distance Matrix Computed: " + str(t3-t2) + " Seconds.")

# K-function Estimates
t4 = time.time()
obsk = kfunction_3Stage.kFunction(full, distVec, Obs = True, Global = False)
t5 = time.time()
print("K-functions Estimated: " + str(t5-t4) + " Seconds.")
    
t6 = time.time()
print("Total Time To Construct Observed Estimates: " + str(t6-t2) + " Seconds.")
logging.info("Total Time To Construct Observed Estimates: " + str(t6-t2) + " Seconds.")


#######################################################################################################################
# Perform Simulations via kfunction_3Stage module
t7 = time.time()
print("Begin Simulations")
print("     Percent Complete:  0") 
index = 10

# Need Containers for Simulation Results
cnts = np.zeros(shape = (len(array), len(distVec))) # Empty matrix to contain simulated K-function counts

# Simulation Runs
sim = 1
simulation = simulations
while sim <= simulation:
    if 100*(float(sim)/simulation) >= index:
        print("     Percent Complete: " + str(index))
        index += 10
    try:
        simarray = None
        simarray = kfunction_3Stage.pointSimulation(zFeatures, bFeatures, npts, fieldZ, fieldB, ZIPFIELD, Global = False)
        cnts += kfunction_3Stage.kSimulation(array, simarray, distVec, obsk, Obs = False, Global = False)
        sim += 1
    except Exception:
        print("SIMULATION " + str(sim) + " Threw an error.")
        simulation += 1
        sim += 1
        continue


t8 = time.time()
print("Simulations Complete. Time: " + str((t8-t7)/60) + " Minutes.")
logging.info("Simulations Complete. Time: " + str((t8-t7)/60) + " Minutes.")


#############################################################################################################################
### Save Results to ESRI Shapefile

# P-values indicate the probability of observing simulated K-function values at least as high as the observed distribution
pvals = (cnts + 1)/(simulations + 1.) # Matrix form of size n*distances [point_count*distVec]

# Collect Variable Names for P-values and K-counts:
PVALS, CNTS = [], []
for i in distVec:
    PVALS.append("P_"+str(i).replace(".","_"))
    CNTS.append("C_"+str(i).replace(".","_"))

del i

dfC = pd.DataFrame(cnts, columns = CNTS)
dfP = pd.DataFrame(pvals, columns = PVALS)

X, Y = [], []
for i in range(len(array)):
    X.append(array[i][0])
    Y.append(array[i][1])

del i
    
# Pair Lat/Lon geometry for geopandas
geometry = [shapely.geometry.Point(xy) for xy in zip(X, Y)]

# Set Coordinate Reference System - shapely style
#crs = pFeatures.crs

# Create GEODATAFRAME from pandas DataFrame
geo_df = gpd.GeoDataFrame(dfC.merge(dfP, left_index = True, right_index = True), geometry = geometry)
geo_df.crs = {'init':'epsg:5070'}
# Save to SHAPEFILE - thanks to geopandas
resultPoints = os.path.join(os.path.dirname(points), fieldZ +"_"+ fieldB + "_Points.shp")
geo_df.to_file(resultPoints)
t9 = time.time()

# Shapefile Saved
print("Shapefile Saved:" + str(t9-t8) + " Seconds.")
logging.info("Shapefile Saved:" + str(t9-t8) + " Seconds.")


#############################################################################################################################
### Buffer Clusters

# Convert Miles Distances to Meters - Buffer constructed in meeters
distVecM = [x * 1609.34 for x in distVec]

for i in range(len(distVec)):
    coreBuffers = os.path.join(os.path.dirname(points), fieldZ +"_"+ fieldB +"_"+ PVALS[i][2:].replace("_",".") + "_Buffers.shp")
    geo_df_selection = geo_df.loc[(geo_df[PVALS[i]] <= 0.002) & (geo_df[CNTS[i]] >= 4), ]
    if geo_df_selection.shape[0] == 0:
        logging.info("No clusters identified at " + str(distVec[i]) + " Miles.")
        pass
    else:
        geo_df_selection = geo_df_selection.to_crs("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs")
        geo_df_buffer = geo_df_selection.buffer(distVecM[i])
        geo_df_buffer = geo_df_buffer.to_crs({'init':'epsg:5070'})
        geo_df_buffer =  gpd.GeoDataFrame(geo_df_buffer)
        geo_df_buffer['UNIT'] = 'BUFFER'
        geo_df_buffer = geo_df_buffer.rename(columns={0: "geometry"})
        geo_df_buffer = geo_df_buffer.dissolve(by = 'UNIT')
        geo_df_buffer.to_file(coreBuffers)


#############################################################################################################################
### Fin
elapsed = time.time() - t0
print("Total Run Time: " + str(elapsed/60) + " Minutes.")
logging.info("Total Run Time: " + str(elapsed/60) + " Minutes.")

