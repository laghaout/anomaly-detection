# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:49:52 2016

@author: Amine Laghaout
"""

import pandas as pd
import module as md

#%% SETTINGS

trainDir = 'data/'              # Directory
trainFile = 'data1.csv'         # Data file
showSynopsis = True             # Show the data synopsis?
num_train_data = None           # Number of training data (None for all)
contamination = 1/17            # Assumed proportion of outliers (1/17, 1/10)
distance = 'Mahalanobis'        # Distance measure
plotTitle = '['+trainFile+']'   # Identify the data source in the plots.

#%% IMPORT DATA

trainData = pd.read_csv(trainDir+trainFile)[:num_train_data]
featureNames = list(trainData.columns)

#%% FEATURE ENGINEERING

# Extract the feature names
x = featureNames[2]             # variable1
y = featureNames[3]             # variable2
z = featureNames[0]             # equipment
#d = featureNames[1]            # date

# Remove incomplete records.
trainData = trainData.dropna()  

# Remove data points at the origin as they are assumed to be irrelevant.
trainData = trainData.loc[abs(trainData[x]) + abs(trainData[y]) != 0]

#%% SYNOPSIS AND EXPLORATION

# Short synopsis of the raw data
md.dataSynopsis(trainData, showSynopsis)

# Reconstruct the Gaussian envelope.
dataFit = md.fit2Gaussian(trainData, x, y)

# Plot the data points and their Gaussian envelope.
md.plotScatter(dataFit, trainData, x, y, z, plotLegend = False, 
               plotTitle = 'All data '+plotTitle, markerSize = 10,
               saveAs = 'All_'+plotTitle[1:-5]+'_-_scatter.pdf')

#%% OUTLIER DETECTION

# Measure the distance of each point from the inferred Gaussian envelope and
# update the data frame to include a distance feature.
trainData = md.measureDistance(dataFit, trainData, x, y)

# Sort the data points by their distance form the inferred Gaussian envelope.
md.plotDistance(trainData[distance], contamination, markerSize = 1,
                plotTitle = 'All data '+plotTitle,
                saveAs = 'All_'+plotTitle[1:-5]+'_-_distance.pdf')

# Retain only the data points that fall within the presumed boundary of the 
# Gaussian envelope.
trainDataTrimmed = md.removeOutliers(trainData, x, y, contamination)

md.plotScatter(dataFit, trainDataTrimmed, x, y, z, plotLegend = False, 
               plotTitle = 'Trimmed data '+plotTitle, markerSize = 10,
               saveAs = 'Trimmed_'+plotTitle[1:-5]+'.pdf')
               
# Produce a scatter plot of the linear models for each equipment type.
linearFitClusters = md.equipmentClusters(trainData, x, y, z)
md.plotData(linearFitClusters, 'intercept', 'slope', 'equipment', 
            plotTitle = 'Linear models by equipment '+plotTitle,
            saveAs = 'Linear_models_by_equipment_'+plotTitle[1:-5]+'.pdf')             