import numpy as np
import glob
import fnmatch
import os
from time import time
import cv2
from scipy.stats import uniform
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV,GridSearchCV
from sklearn import svm
import pickle
from utils import *

def report(results, n_top=3):
    '''
    Utility function to report best scores
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
    '''
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

if __name__ == "__main__":
    # define paths to the dataset
#    car_path = './dataset/vehicles/**/*.png'
#    noncar_path = './dataset/non-vehicles/**/*.png'
    car_path = './dataset/vehicles/'
    noncar_path = './dataset/non-vehicles/'

    # define parameters for feature extraction
    cspace = 'YUV'
    spatial_size = (32,32)
    hist_bins = 32
    hist_range = (0,256)
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'

    # define parameters for model selection
#    parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'),
#                  'C':uniform(loc=1e-4,scale=1e4)}
#    n_iter_search = 1000
#    parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'),
#                  'C':[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]}
#    parameters = {'kernel': ['rbf'],
#                  'C':[5, 1e1, 2e1, 4e1, 8e1, 1e2, 5e2, 1e3, 5e3, 1e4]}
    parameters = {'C':[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]}

    # define model name
    output_file = './svc-clf.sav'

    # load the dataset
    print('Loading the dataset ...')
    cars = load_data(car_path)
    noncars = load_data(noncar_path)

    # print data summary
    print('dataset loaded')
    data_summary = data_look(cars, noncars)
    print(data_summary)

    # prepare X and y
    print()
    print('Preparing features ...')
    print('orientations:',orient,'pixels per cell:',pix_per_cell, 'cells per block:', cell_per_block,
          '\ncolor space:',cspace,
          '\nSpatial size:',spatial_size, 'histogram bins:',hist_bins)
    car_features = extract_features(cars, cspace, spatial_size,
                            hist_bins, hist_range,
                            orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True, hog_channel=hog_channel)
    notcar_features = extract_features(noncars, cspace, spatial_size,
                            hist_bins, hist_range,
                            orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True, hog_channel=hog_channel)
    X = createX([car_features, notcar_features], normalize = False)
    scaled_X, X_scaler = createX([car_features, notcar_features], normalize = True)
    # generate labels
    y = createY(car_features, notcar_features)
    print('feature and labels generated.')
    print('feature vector shape:', scaled_X.shape)
    print('label vector shape:', scaled_X.shape)

    # training the classifier
    # Split up data into randomized training and test sets
    print()
    print('Splitting the dataset ...')
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    print('X_train shape:',X_train.shape, 'y_train shape:', y_train.shape)
    print('X_test shape:',X_test.shape, 'y_test shape:', y_test.shape)


    # model selection
    # use a linear SVC
    svc = svm.LinearSVC(random_state = rand_state)
    # use randomized search for parameter tuning
#    clf = RandomizedSearchCV(svc, param_distributions=parameters,
#                                   n_iter=n_iter_search, verbose=10, n_jobs=-1)
    clf = GridSearchCV(svc, param_grid=parameters, verbose=10, n_jobs=-1)
    # start tuning
    print()
    print('Selecting model parameters ...')
#    print('iterations:',n_iter_search)
    print('parameters:',parameters)
    start = time()
    clf.fit(X_train, y_train)
    print()
#    print("RandomizedSearchCV took %.2f seconds for %d candidates"
#          " parameter settings." % ((time() - start), n_iter_search))
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(clf.cv_results_['params'])))
    report(clf.cv_results_)

    # print the test error
    print()
    print('Accuracy on test data:',clf.score(X_test, y_test))

    # save the classifier
    print()
    print('Saving model ...')
    pickle.dump([clf,X_scaler,cspace,spatial_size,hist_bins,hist_range,orient,pix_per_cell,cell_per_block,hog_channel], open(output_file, 'wb'))
    print('model saved at:', output_file)
    