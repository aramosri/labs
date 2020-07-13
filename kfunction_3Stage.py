#########################################################################################################################
# Jake K Carr
# The Ohio State University; Moody's Analytics
# Completely open source

# Functions necessary for 3 Stage Global/Local K-function estimation:
# clockwise = check_clockwise(poly)
# bFeatures = countSimulation(fsFeatures, ssFeatures, npts, fieldFS, fieldSS, idFieldFS, Global = True)
# dist = distanceMatrix(locsA, locsB)
# flatlist = flatten(S)
# np.array(k) = kFunction(distMat, distVec, Obs = True, Global = True)
# comps = kSimulation(array, simarray, distVec, obsk, Obs = True, Global = True)
# inside = pip(x, y, poly)
# sims = pointSimulation(fsFeatures, ssFeatures, npts, fieldFS, fieldSS, idFieldFS, Global = True)
# shplst = polygonDefinitions(polyShapes)
# holes = polygonHoles(polyShapes, polyprts)
# polyprts = polygonParts(polyShapes)
# min(lons), min(lats), max(lons), max(lats) = shape_list_decompose(polyList)

# Modified: 07/01/2018
#########################################################################################################################
#########################################################################################################################
# Import Modules
import numpy as np
import geopandas as gpd
import random


#########################################################################################################################
# Check order of vertices; clockwise = outer ring, counter-clockwise = inner ring
def check_clockwise(poly):
    """Checks if a sequence of (x,y) polygon vertice pairs is ordered clockwise or not.
       NOTE: Counter-clockwise (=FALSE) vertice order reserved for inner ring polygons"""
    clockwise = False
    if (sum(x0*y1 - x1*y0 for ((x0, y0), (x1, y1)) in zip(poly, poly[1:] + [poly[0]]))) < 0:
        clockwise = not clockwise
    return clockwise


#######################################################################################################################
# Point Count Simulation Computation
def countSimulation(fsFeatures, ssFeatures, npts, fieldFS, fieldSS, idFieldFS, Global = True):
    """ Simulates point counts, proportionally, within higher geography (ZIP CODES), matches lower geography (BLOCKS)
    
    to the higher geography by centroid locations, and simulates point counts, proportionally, within lower geography.
    
    Returns a geopandas dataframe (bFeatures) with a column (SIM_POINTS) representing the number of points to be simulated within lower geography. """
    # Set Proportions of reference measure for simulation for First Stage Polygons (ZIP_CODES)
    fsFeatures['FS_PROBS'] = fsFeatures[[fieldFS]] / float(sum(fsFeatures[fieldFS]))
    
    # Multinomial Count Simulation of points for each polygon in study area
    if Global:
        fsFeatures['POINTS'] = np.random.multinomial(npts, fsFeatures['FS_PROBS'].tolist(), 1).tolist()[0]
    else:
        fsFeatures['POINTS'] = np.random.multinomial(npts-1, fsFeatures['FS_PROBS'].tolist(), 1).tolist()[0]
    
    # Convert Second Stage Polygons (BLOCKS) to point dataframe by taking centroids
    ssPoints = gpd.GeoDataFrame(ssFeatures.drop(['geometry'], axis = 1), geometry = ssFeatures.centroid)
    
    # Spatial Overlay of Second Stage points on First Stage polygons
    ssinfs = gpd.sjoin(ssPoints, fsFeatures.filter([idFieldFS, 'POINTS','geometry'], axis = 1), how = 'left', op = 'within')
        
    # Merge First Stage Information back to Second Stage polygons
    ssFeatures = ssFeatures.merge(ssinfs.filter([idFieldFS, 'POINTS'], axis = 1), how = 'left', left_index = True, right_index = True)
    ssFeatures = ssFeatures.assign(SIM_POINTS = 0)
            
    # Make Lists to loop through, first stage and second stage
    fslist = fsFeatures[idFieldFS].values.tolist(); fslist = list(set(fslist))
    sslist = ssFeatures[idFieldFS].values.tolist(); sslist = list(set(sslist))
    for i in fslist:
        # Each set of blocks that 'fall' in a given ZIP CODE
        fspts = fsFeatures.loc[fsFeatures[idFieldFS]==i,'POINTS'].item()
        if fspts > 0:
            if i in sslist:
                bsample = ssFeatures.loc[ssFeatures[idFieldFS] == i,[fieldSS]]
                bsample['SS_PROBS'] = bsample[[fieldSS]] / float(sum(bsample[fieldSS]))
                bsample['SIM_POINTS'] = np.random.multinomial(fspts, bsample['SS_PROBS'].tolist(), 1).tolist()[0]
                ssFeatures.update(bsample.loc[:,['SIM_POINTS']])
            else:
                ssFeatures = ssFeatures.append(fsFeatures.loc[fsFeatures[idFieldFS] == i, ["POINTS", idFieldFS, "geometry"]].rename(columns={'POINTS': 'SIM_POINTS'}), ignore_index = True)
        else:
            pass
        
    # Make Counts Retrievable
    return ssFeatures.reset_index(drop = True)


#######################################################################################################################
# Distance Matrix Construction - Miles
def distanceMatrix(locsA, locsB):
    """ Calculates distance matrix (in miles) between the locations in locsA to locations in locsB. Assumes that both locsA/locsB are numpy arrays.

    First column of locsA/locsB must be X or LON; second column must be Y or LAT. Measures all pairwise distances. 
    
    Returns the full distance matrix in numpy matrix form (dist). """    
    # Empty Container Matrix for Distances
    dist = np.zeros((len(locsA), len(locsB)))
    dist = np.sqrt((69.1 * (locsB[:,0][:,np.newaxis] - locsA[:,0]) * np.cos(locsA[:,1]/57.3))**2 + \
                   (69.1 * (locsB[:,1][:,np.newaxis] - locsA[:,1]))**2)
  
    return dist


#######################################################################################################################
# Flatten List of Lists to List
def flatten(S):
    """ Flattens a list of lists, but leaves tuples intact"""
    flatlist = [val for sublist in S for val in sublist]
    
    return flatlist 


#######################################################################################################################
# K-function Calculation
def kFunction(distMat, distVec, Obs = True, Global = True):
    """ Calculates K-function values of a given point pattern based on the pair-wise distance matrix (distMat) for a vector of distances (distVec).

    Global = True provides average K-count of all n points per distance. Local (Global = False) provides K-count for individual points per distance.

    Returns K-function values (k). """
    if Global:
        # Global K-function Estimates
        k = np.zeros((len(distVec)))
        for i in range(len(distVec)):
            k[i] = (sum(sum(distMat <= distVec[i]) - 1)/float(distMat.shape[0]))
        #del i
    else:
        if Obs:
            # Local Obs K-function Estimates
            k = np.zeros([len(distMat), len(distVec)])
            for i in range(len(distMat)):
                for j in range(len(distVec)):
                    k[i,j] = (sum(distMat[i,] <= distVec[j])-1)
        else:
            # Local Simulated K-function Estimates
            k = np.zeros([len(distMat), len(distVec)])
            for i in range(len(distMat)):
                for j in range(len(distVec)):
                    k[i,j] = (sum(distMat[i,] <= distVec[j]))
            
    # Make K-function Values Retrievable
    return np.array(k)


#######################################################################################################################
# K-function Simulation Calculation
def kSimulation(array, simarray, distVec, obsk, Obs = True, Global = True):
    """ Calculates K-function values and P-value counts of simulated point pattern (simarray) relative to observed K-function values (obsk) for a vector of distances (distVec).

    Global = True compares observed K-functions to the K-functions from a full simulated distribution. Local (Global = False) compares all observed K-counts to the K-counts of observed points to simulated distribution.

    Returns K-function values (simk) and P-value counts (cnts/comps). """
    if Global:
        # Distance Matrix of Simulated Distribution
        simfull = distanceMatrix(simarray, simarray)

        # K-function Estimates of Simulation
        simk = kFunction(simfull, distVec)
        
        # Compare Observed Counts to Simulated Counts
        comps = [x <= y for (x,y) in zip(np.array(obsk), np.array(simk))]

        # Counts of Simulations that Result in Higher K-function Values than the Observed Distribution for All Distances
        cnts = np.matrix([int(elem) for elem in comps])
        #del elem, comps, simfull

        # Make Counts and K-function values retrievable
        return cnts, np.matrix(simk)

    else:
        # Distance Matrix of Observed Points to Simulated Points NOTE THAT SIMARRAY IS FIRST ENTRY!
        simfull = distanceMatrix(simarray, array)

        # K-function Estimates from Simulation
        simk = kFunction(simfull, distVec, Obs, Global = False)
        
        # Counts of Simulations that Result in Higher K-function Values than the Observed Distribution for All Distances
        comps = [x <= y for (x,y) in zip(np.array(obsk), np.array(simk))]
        #del simfull, simk

        # Make Counts Retrievable
        return comps


#########################################################################################################################
# Point-in-Polygon
def pip(x, y, poly):
    """Checks if a given point (x,y) falls inside or outside the given polygon (poly).

    Returns True (inside) or False (outside). """
    #if poly[0] != poly[-1]:
    #    return print("Polygon not closed")
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


#########################################################################################################################
# Point Simulation: Both Global and Local Versions
def pointSimulation(fsFeatures, ssFeatures, npts, fieldFS, fieldSS, idFieldFS, Global = True):
    """ Simulates points, proportionally, within higher geography (ZIP CODES), matches lower geography (BLOCKS)
    
    to the higher geography by centroid locations, and simulates points, proportionally, within lower geography.
    
    Assumes a multinomial count distribution (npts = len(point_pattern), p = proportions) to assign points to given polygons, where
    p = proportions are the multinomial probabilities. 
    
    Global = True simulates n points for a full distribution. Local (Global = False) simulates n-1 points for local K-function
    comparison.
    
    Returns array of simulated point locations (sims). """
    
    # Multinomial Count Simulation of points for each polygon in study area
    if Global:
        polySimCounts = countSimulation(fsFeatures, ssFeatures, npts, fieldFS, fieldSS, idFieldFS, Global = True)
    else:
        polySimCounts = countSimulation(fsFeatures, ssFeatures, npts, fieldFS, fieldSS, idFieldFS, Global = False)
    
    # Extract Point Counts
    rpts = polySimCounts['SIM_POINTS'].tolist()
    
    # Identify Single and Multiple Polygons
    polyprts = polygonParts(polySimCounts)
    
    # Identify Holes in Polygons
    holes = polygonHoles(polySimCounts, polyprts)
    
    # Identify Polygon Shapes
    shapes = polygonDefinitions(polySimCounts)
    
    # Point Simulation $\Rightarrow$ locations constrained by given polygons 
    pts = []
    for i in range(len(shapes)):
        if holes[i][0] == True:
            '''Single Part Polygons'''
            total = rpts[i]
            count = 0
            ptssim = []
            minx, miny, maxx, maxy = shape_list_decompose(shapes[i])
            while count < total:
                x = random.uniform(minx, maxx)
                y = random.uniform(miny, maxy)
                if pip(x, y, shapes[i]):
                    count += 1
                    ptssim.append([x, y])
            pts.append(ptssim)
        elif holes[i][0] == False and holes[i][1] == True:
            '''Multipart Part Polygons w/ No Holes'''
            a = [k for k, x in enumerate(holes[i][2]) if x]
            apolys = [] #apolys is a list of all polygons
            for j in a:
                apolys.append([shapes[i][polyprts[i][j]:polyprts[i][j+1]-1]][0])
            total = rpts[i]
            count = 0
            ptssim = []
            minx, miny, maxx, maxy = shape_list_decompose(shapes[i])
            while count < total:
                x = random.uniform(minx, maxx)
                y = random.uniform(miny, maxy)
                gpip = []
                for j in range(len(a)):
                    gpip.append(pip(x, y, apolys[j]))
                if any(gpip):
                    count += 1
                    ptssim.append([x, y])
                else:
                    pass
            pts.append(ptssim)
        elif holes[i][0] == False and holes[i][1] == False:
            '''Multipart Part Polygons w/ Holes'''
            a = [k for k, x in enumerate(holes[i][2]) if x]
            apolys = [] #apolys is a list of all polygons
            for j in a:
                apolys.append([shapes[i][polyprts[i][j]:polyprts[i][j+1]-1]][0])
            h = [k for k, x in enumerate(holes[i][2]) if not x]
            hpolys = [] #hpolys is a list of all 'hole' polygons
            for j in h:
                hpolys.append([shapes[i][polyprts[i][j]:polyprts[i][j+1]-1]][0])
            total = rpts[i]
            count = 0
            ptssim = []
            minx, miny, maxx, maxy = shape_list_decompose(shapes[i])
            while count < total:
                x = random.uniform(minx, maxx)
                y = random.uniform(miny, maxy)
                gpip = []
                for j in range(len(a)):
                    gpip.append(pip(x, y, apolys[j]))
                if any(gpip):
                    bpip = []
                    for j in range(len(h)):
                        bpip.append(pip(x, y, hpolys[j]))
                    if not any(bpip):
                        count += 1
                        ptssim.append([x, y])
                    else:
                        pass
                else:
                    pass
            pts.append(ptssim)
        
    sims = np.array([item for sublist in pts for item in sublist])
    return sims


#######################################################################################################################
# Identify Polygon Parts
def polygonDefinitions(polyShapes):
    """ Reads the geometry from a polygon geodataframe and extracts vertices defining all polygons
    
    Muliport polygons are individually processed, need output from polygonParts() to identify individual parts
    
    Returns a list (shplst) of vertices of every polygon. """
    # Pull polygon geometry
    shapes = polyShapes.geometry.type
    
    shplst = []
    for i in range(len(shapes)):
        if shapes[i] == "Polygon":
            shplst.append(list(zip(*polyShapes.geometry[i].exterior.coords.xy)))
        else:
            mpp = []
            for j in range(len(polyShapes.geometry[i])):
                mpp.append(list(zip(*polyShapes.geometry[i][j].exterior.coords.xy)))
            shplst.append(flatten(mpp))
            
    return shplst


#######################################################################################################################
# Identify Polygon Parts
def polygonHoles(polyShapes, polyprts):
    """ Reads the geometry from a polygon geodataframe and identifies mulipart and hole polygon structures
    
    Returns a list (holes) identifying singlepart vs multipart polygons, and identifies which if any are 'holes.' """
    
    # Pull polygon geometry
    shapes = polygonDefinitions(polyShapes)
    
    holes = []
    for i in range(len(shapes)):
        single, test  = True, []
        for j in range(len(polyprts[i])-1):
            if len(polyprts[i]) > 2:
                single = False
            test.append(check_clockwise(shapes[i][polyprts[i][j]:polyprts[i][j+1]-1]))
        holes.append([single, all(test), test])
        
    return holes


#######################################################################################################################
# Identify Polygon Parts
def polygonParts(polyShapes):
    """ Reads the geometry from a polygon geodataframe and identifies the beginning and end nodes of individual polygons.
    
    Necessary for muliport polygons
    
    Returns a list (polyprts) of lists containing the first vertex of every polygon, as well as the last vertex. """
    # Pull polygon geometry
    shapes = polyShapes.geometry.type
    
    polyprts = []
    for i in range(len(shapes)):
        if shapes[i] == "Polygon":
            polyprts.append([0, len(polyShapes.geometry[i].exterior.coords)])
        else:
            initial = 0
            mpp = [initial]
            for j in range(len(polyShapes.geometry[i])):
                initial += len(polyShapes.geometry[i][j].exterior.coords)
                mpp.append(initial)
            polyprts.append(mpp)

    return polyprts


#########################################################################################################################
# Decompose a list of tuples defining a polygon
def shape_list_decompose(polyList):
    """Decompose a list of tuples containing the LON/LAT pairs defining a polygon
    
    The result is the bounding box components of the polygon"""
    # Pull Longitudes and Latitudes individually
    lons = [i[0] for i in polyList]
    lats = [i[1] for i in polyList]
    return min(lons), min(lats), max(lons), max(lats)


#########################################################################################################################

