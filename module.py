# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:51:02 2016

@author: Amine Laghaout
"""

def dataSynopsis(df, showSynopsis = True):
    
    '''
    This function prints to the screen a brief synopsis of the data in the 
    data frame `df' provided that `showSynopsis' is True.
    '''
    
    if showSynopsis:    
        print(df.head(5))
        print(df.describe())

def plotData(df, x, y, z, xLabel = '', yLabel = '', plotTitle = '', 
             fontSize = 16, saveAs = None, plotLegend = False, 
             markerSize = 40):

    '''
    This functions produces scatter plots of `(df[x], df[y])' for each 
    corresponding category `df[z]'. `df' is a data frame.
    '''

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm    
    import numpy as np
    
    if xLabel == '':
        xLabel = x
    if yLabel == '':
        yLabel = y
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Extract the set of categories to be plotted and assign to each category
    # a unique colour.
    categories = set(list(df[:][z]))
    colors = iter(cm.rainbow(np.linspace(0, 1, len(categories))))
    
    # Print a scatter plot for each category.
    for category in categories:
        
        # Retrieve the indices for the current category.
        iCategory = df[df[z] == category].index.tolist()
        
        # To change the markers, use:
        # marker = r"$ {} $".format(category)
        ax.scatter(list(df.loc[iCategory, x]), list(df.loc[iCategory, y]), 
                         s = markerSize, c = next(colors), lw = 0,
                         label=str(category))        
    
    plt.grid(True)
    plt.xlabel(r'%s' % xLabel, fontsize = fontSize)
    plt.ylabel(r'%s' % yLabel, fontsize = fontSize) 
    
    if plotTitle != '':
        plt.title(plotTitle, fontsize = fontSize)
        
    if plotLegend is True or len(categories) < 10:
        plt.legend(loc='upper left', bbox_to_anchor=(1,1))
 
    plt.xlim([min(df[x]), max(df[x])])
    plt.ylim([min(df[y]), max(df[y])])
 
    if saveAs != None:
        plt.savefig(saveAs, bbox_inches='tight')
        
    return fig
    
    
def fit2Gaussian(df, x, y):
    
    '''
    This function uses Maximum Likelihood to estimate a Gaussian envelope for
    the data points `(df[x], df[y])' where `df' is a data frame.
    '''    
    
    from numpy import array
    from sklearn.covariance import EmpiricalCovariance
    
    clf = EmpiricalCovariance()
     
    return clf.fit(array(df[[x, y]]))    
    
def distanceContour(df, x, y, dataFit, meshSize = 500):

    '''
    This function produces a contour plot for the Gaussian envelope around the 
    data points `(df[x], df[y])'. The envelope is contained in `dataFit'. The
    contour isolines are based on the Mahalanobis distance metric.
    '''

    from numpy import meshgrid, linspace, c_, sqrt
    import matplotlib.pyplot as plt
          
    (minX, maxX) = (min(df[x]), max(df[x]))
    (minY, maxY) = (min(df[y]), max(df[y]))
    xx, yy = meshgrid(linspace(minX, maxX, meshSize), 
                      linspace(minY, maxY, meshSize))
    zz = c_[xx.ravel(), yy.ravel()]
            
    distance_dataFit = dataFit.mahalanobis(zz)
    distance_dataFit = distance_dataFit.reshape(xx.shape)
    distance_contour = plt.contour(xx, yy, sqrt(distance_dataFit),
                                   cmap=plt.cm.YlOrBr_r, linestyles = 'dotted')
    
    return distance_contour
    
def equipmentClusters(df, x, y, z):

    '''
    This function performs linear regression on the data points 
    `(df[x], df[y])' for each category `z'. The results of the fits such as the
    slope, intercept, and quality-of-fit statistics are then returned in a data
    frame indexed by the categories `z'.
    '''

    from scipy import stats
    from pandas import DataFrame

    categories = set(list(df[:][z]))
    LinRegColumns = [z, 'slope', 'intercept', 'r_value', 'p_value', 'std_err']
    clusters = DataFrame(columns = LinRegColumns)
    
    # Print a scatter plot for each category.
    for category in categories:
        
        X = df[df[z] == category][x]
        Y = df[df[z] == category][y]
        
        (slope, intercept, r_value, p_value, std_err) = stats.linregress(X, Y)
        clusters.loc[category] = [category, slope, intercept, r_value, p_value, std_err]

    return clusters
    
def plotFit(x, y):
    
    '''
    This function plots the linear fit of the data `(x, y)'.
    '''

    from scipy import stats
    import matplotlib.pyplot as plt    
    
    (slope, intercept, r_value, p_value, std_err) = stats.linregress(x, y)    
    
    plt.plot(x, slope*x + intercept, '-', color = 'black')
    
    return plt

def measureDistance(dataFit, df, x, y):

    '''
    This function populates a new feature in the data frame `df' with the 
    Mahalanobis distance of each data point `(df[x], df[y])'.
    '''

    from numpy import array

    data = array([df[x], df[y]]).T    
    df['Mahalanobis'] = dataFit.mahalanobis(data) 

    return df    
    
def removeOutliers(df, x, y, contamination = 0):

    '''
    This function removes the proportion `contamination' of outliers from the
    data points in `df' based on the Mahalanobis distance of each. I.e., the 
    most remote points are removed.
    '''

    from numpy import floor

    numOutliers = int(floor(contamination*len(df)))

    distances = list(df['Mahalanobis'])
    distances.sort()

    return df[df['Mahalanobis'] <= distances[-numOutliers]]
    
def plotDistance(distances, contamination, fontSize = 16, saveAs = None,
                 markerSize = 3, plotTitle = ''):
    
    '''
    This function sorts the `distances' and plots their spectrum. A vertical
    line cuts off the spectrum at the proportion `contamination'.
    '''    
    
    import matplotlib.pyplot as plt
    from numpy import floor
    
    distances = list(distances)
    distances.sort()
    
    numOutliers = int(floor(contamination*len(distances)))
    
    plt.axvline(len(distances) - numOutliers, min(distances), max(distances), 
                color = 'red', linewidth = 3)
    plt.semilogy(range(len(distances)), distances, '-x', color = 'black',
                 ms = markerSize)
    plt.xlabel('Data points (sorted by distance)', fontsize = fontSize)
    plt.ylabel('Mahalanobis distance', fontsize = fontSize)
    
    if plotTitle != '':
        plt.title(plotTitle, fontsize = fontSize)
    
    plt.grid(True)
    plt.xlim([-15, len(distances) + 15])
    
    if saveAs != None:
        plt.savefig(saveAs, bbox_inches='tight')

def plotScatter(dataFit, df, x, y, z, plotLegend, plotTitle, markerSize = 10,
                saveAs = None):   
    
    '''
    This function combines the scatter plots, the envelope contour plot, and 
    the linear fit plot into one figure.
    '''    
    
    import matplotlib.pyplot as plt    
    plt.figure()
    plotData(df, x, y, z, plotLegend = plotLegend, markerSize = markerSize,
             plotTitle = plotTitle, saveAs = saveAs)
    distanceContour(df, x, y, dataFit)
    plotFit(df[x], df[y])
    
    
    if saveAs != None:
        
        plt.savefig(saveAs, bbox_inches='tight')
    plt.show()
           