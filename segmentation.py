import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid', {'axes.grid': False})

from sklearn import cluster

from scipy.ndimage.filters import median_filter, gaussian_filter

import warnings; warnings.simplefilter('ignore')

originals_dir = 'originals'
data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'test')

classes = ['DUM555', 'DUM560', 'DUM562', 'DUM587', 'DUM588']

# Image preprocessing: filters
from scipy.ndimage.filters import median_filter, gaussian_filter
def filters(img, median_filter_size=(5, 5, 1), gaussian_sigma=4):

    median_filted = median_filter(img, size=median_filter_size)
    gaussian_filted = gaussian_filter(median_filted, sigma=gaussian_sigma)
    return cv2.cvtColor(gaussian_filted, cv2.COLOR_BGR2GRAY)

# Extract mahotas features
import mahotas.features
def haralick_features(names, distance=1):
    f = []
    for i in range(len(names)):
        img = cv2.imread(names[i])
        if img.shape == (1768, 2048, 3):
            # img = img[:960, :1280, :]
            img = img[404:1364, 384:1664, :]
            # print('haralick: {:s} is cropped'.format(names[i]))
        if img is None or img.size == 0 or np.sum(img[:]) == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            h = np.zeros((1, 13))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h = mahotas.features.haralick(img, distance=distance, return_mean=True, ignore_zeros=False)
            h = np.expand_dims(h, 0)
        if i == 0:
            f = h
        else:
            f = np.vstack((f, h))
    return f

# Extract LBP features
from skimage.feature import local_binary_pattern
def lbp_features(names, P=10, R=5):
    f = []
    for i in range(len(names)):
        img = cv2.imread(names[i])
        if img.shape == (1768, 2048, 3):
            # img = img[:960, :1280, :]
            img = img[404:1364, 384:1664, :]
            # print('lbp: {:s} is cropped'.format(names[i]))
        if img is None or img.size == 0 or np.sum(img[:]) == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            h = np.zeros((1, 13))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp = local_binary_pattern(img, P=P, R=R)
            h, _ = np.histogram(lbp, normed=True, bins=P + 2, range=(0, P + 2))
        if i == 0:
            f = h
        else:
            f = np.vstack((f, h))
    return f

# Test feature extraction process
haralick = []
lbp = []
Y = []
names = glob.glob('haralick-lbp-data' + '/test/*.tif')
haralick = haralick_features(names)
lbp = lbp_features(names)
Y += [0] * len(names)
assert(haralick.shape == (13, 13))
assert(lbp.shape == (13,12))
assert(len(Y) == 13)

# Extract features from all images
haralick = []
lbp = []
Y = []
for i in range(len(classes)):
    names = glob.glob('haralick-lbp-data' + '/' + classes[i] + '/*.tif')
    if i == 0:
        haralick = haralick_features(names)
        lbp = lbp_features(names)
    else:
        haralick = np.vstack((haralick, haralick_features(names)))
        lbp = np.vstack((lbp, lbp_features(names)))
    Y += [i] * len(names)
Y = np.asarray(Y)
allfeatures = np.column_stack((haralick, lbp))

# Training kMeans model
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import warnings; warnings.simplefilter('ignore')

skf = StratifiedKFold(n_splits=5)
count = 1
acc = []
out = ''
model = None
for train_index, test_index in skf.split(np.zeros((len(Y), 1)), Y):
    print('k-fold: #{}'.format(count))
    out += 'k-fold: #{}\n'.format(count)
    count += 1
    train_labels = Y[train_index]
    test_labels = Y[test_index]
    train = allfeatures[train_index]
    test = allfeatures[test_index]
    rfmodel = RandomForestClassifier(n_estimators=100,
                                   max_features='sqrt',
                                   n_jobs=-1, verbose=0)
    rfmodel.fit(train, train_labels)
    n_nodes = []
    max_depths = []

    for ind_tree in rfmodel.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    print(f'    Average number of nodes {int(np.mean(n_nodes))}')
    print(f'    Average maximum depth {int(np.mean(max_depths))}')

    train_rf_predictions = rfmodel.predict(train)
    train_rf_probs = rfmodel.predict_proba(train)[:, 1]
    rf_predictions = rfmodel.predict(test)
    rf_probs = rfmodel.predict_proba(test)[:, 1]

    # a = sum(np.multiply(rf_predictions - 2.5, test_labels - 2.5) > 0)/len(test)
    a = sum((rf_predictions==test_labels)/len(test))
    '''
    a = 0
    for i, j in zip(rf_predictions, test_labels):
        if i == j:
            a += 1
        if i == 0 and j == 1:
            a += 1
        if i == 1 and j == 0:
            a += 1
        if i == 3 and j == 4:
            a += 1
        if i == 4 and j == 3:
            a += 1
    a = a / len(test)
    '''
    acc.append(a)
    print('    Model accuracy: {}'.format(a))
    print('    Feature importances: {}'.format(rfmodel.feature_importances_))
    out += '    Model accuracy: {}\n'.format(a)
    out += '    Feature importances: {}\n'.format(rfmodel.feature_importances_)
    test_count = [0, 0, 0, 0, 0]
    output_count = []
    for i in range(5):
        output_count.append([0, 0, 0, 0, 0])
    for i in range(len(rf_predictions)):
        test_count[test_labels[i]] += 1
        output_count[test_labels[i]][rf_predictions[i]] += 1
    print('    Prediction details:')
    for i in range(5):
        print('    {} tests for class {}:'.format(test_count[i], classes[i]), end='')
        for j in range(5):
            print('{:3d}'.format(output_count[i][j]), end='')
        print()
    break

# Segmentation using kMeans
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
def kmeans_segmentation(c, f, k=15, pivot=40, verbose=True):
    name = [os.path.join(data_dir, c, f)]
    haralick = haralick_features(name)
    lbp = lbp_features(name)
    lbp = np.expand_dims(lbp, 0)
    features = np.column_stack((haralick, lbp))
    pred_class = rfmodel.predict(features)[0]
    if pred_class == 0:
        pattern = 0
    elif pred_class == 4:
        pattern = 2
    else:
        pattern = 1
    if verbose:
        print('Class prediction: {:s} => k = {:d}'.format(classes[pred_class], 3 if pattern==1 else 2))
    img = cv2.imread(os.path.join(data_dir, c, f))
    gray_img = filters(img)
    # gray_img = clahe.apply(gray_img)
    # seg_img = np.copy(gray_img)
    # seg_img = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)
    seg_img = np.zeros(img.shape)
    gray_img = gray_img - np.mean(gray_img)
    gray_img = gray_img / np.std(gray_img)
    if pattern==1:
        n_clusters=3
    else:
        n_clusters=2
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(gray_img.flatten().reshape(-1, 1))
    l = kmeans.labels_.reshape(gray_img.shape)
    colors = []
    colors.append((219/255, 94/255, 86/255))
    colors.append((86/255, 219/255, 127/255))
    colors.append((86/255, 111/255, 219/255))
    if pattern==1:
        f = [(l==0).sum(), (l==1).sum(), (l==2).sum()]
        f = f / sum(f)
        avg = []
        for c in range(3):
            segc = (l == c)
            count = segc.sum()
            avg.append((segc*gray_img[:,:]).sum() / count)
        ret = [x for x, _ in sorted(zip(f, avg), key=lambda pair:pair[1], reverse=True)]
        for i, col in enumerate([x for x, _ in sorted(zip(list(range(3)), avg), key=lambda pair:pair[1], reverse=True)]):
            segc = (l == col)
            seg_img[:,:,0] += (segc*(colors[i][0]))
            seg_img[:,:,1] += (segc*(colors[i][1]))
            seg_img[:,:,2] += (segc*(colors[i][2]))
    else:
        f = [(l==0).sum(), (l==1).sum()]
        f = f / sum(f)
        avg = []
        for c in range(2):
            segc = (l == c)
            count = segc.sum()
            avg.append((segc*gray_img[:,:]).sum() / count)
        ret = [x for x, _ in sorted(zip(f, avg), key=lambda pair:pair[1], reverse=True)]
        if pattern==0:
            ret.insert(2, 0.)
            if verbose:
                for i, col in enumerate([x for x, _ in sorted(zip([0,1], avg), key=lambda pair:pair[1], reverse=True)]):
                    segc = (l == col)
                    seg_img[:,:,0] += (segc*(colors[i][0]))
                    seg_img[:,:,1] += (segc*(colors[i][1]))
                    seg_img[:,:,2] += (segc*(colors[i][2]))
        elif pattern == 2:
            ret.insert(0, .0)
            if verbose:
                l += 1
                for i, col in enumerate([x for x, _ in sorted(zip([1,2], avg), key=lambda pair:pair[1], reverse=True)]):
                    segc = (l == col)
                    seg_img[:,:,0] += (segc*(colors[i+1][0]))
                    seg_img[:,:,1] += (segc*(colors[i+1][1]))
                    seg_img[:,:,2] += (segc*(colors[i+1][2]))
    if verbose:
        fig = plt.figure(figsize=(15, 5))
        # fig.suptitle(os.path.join(c, f))
        ax = fig.add_subplot(1, 3, 1)
        plt.imshow(img)
        ax = fig.add_subplot(1, 3, 2)
        plt.imshow(gray_img, 'gray')
        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(seg_img)
    # return ret + [x for x, _ in sorted(zip(f, avg), key=lambda pair:pair[1], reverse=True)]
    return ret

# Display segmentation results
kmeans_segmentation('DUM562', 'DUM562_017.tif')

# Extract segmentation features and test segmentation accuracy using 5-class classification
def get_xny():
    f = []
    Y = []
    for idx, c in enumerate(classes):
        count = 0
        for img_name in sorted(os.listdir(os.path.join(data_dir, c))):
            if not img_name.endswith('.tif'):
                continue
            tmp = kmeans_segmentation(c, img_name, k=15, pivot=40, verbose=False)
            if len(f) == 0:
                f = np.expand_dims(tmp, 0)
            else:
                f = np.vstack((f, np.expand_dims(tmp, 0)))
            count += 1
        Y += [idx] * count
        print('collected segmentation features from {} images of class {}'.format(count, c))
    return f, np.asarray(Y)

X, Y = get_xny()

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import warnings; warnings.simplefilter('ignore')

skf = StratifiedKFold(n_splits=5)
count = 1
acc = []
out = ''
for train_index, test_index in skf.split(np.zeros((len(Y), 1)), Y):
    print('k-fold: #{}'.format(count))
    out += 'k-fold: #{}\n'.format(count)
    count += 1
    train_labels = Y[train_index]
    test_labels = Y[test_index]
    train = X[train_index]
    test = X[test_index]
    model = RandomForestClassifier(n_estimators=100,
                                   max_features='sqrt',
                                   n_jobs=-1, verbose=0)
    model.fit(train, train_labels)
    n_nodes = []
    max_depths = []

    for ind_tree in model.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    print(f'    Average number of nodes {int(np.mean(n_nodes))}')
    print(f'    Average maximum depth {int(np.mean(max_depths))}')

    train_rf_predictions = model.predict(train)
    train_rf_probs = model.predict_proba(train)[:, 1]
    rf_predictions = model.predict(test)
    rf_probs = model.predict_proba(test)[:, 1]

    # a = sum(np.multiply(rf_predictions - 2.5, test_labels - 2.5) > 0)/len(test)
    a = sum((rf_predictions==test_labels)/len(test))
    '''
    a = 0
    for i, j in zip(rf_predictions, test_labels):
        if i == j:
            a += 1
        if i == 0 and j == 1:
            a += 1
        if i == 1 and j == 0:
            a += 1
        if i == 3 and j == 4:
            a += 1
        if i == 4 and j == 3:
            a += 1
    a = a / len(test)
    '''
    acc.append(a)
    print('    Model accuracy: {}'.format(a))
    print('    Feature importances: {}'.format(model.feature_importances_))
    out += '    Model accuracy: {}\n'.format(a)
    out += '    Feature importances: {}\n'.format(model.feature_importances_)
    test_count = [0, 0, 0, 0, 0]
    output_count = []
    for i in range(5):
        output_count.append([0, 0, 0, 0, 0])
    for i in range(len(rf_predictions)):
        test_count[test_labels[i]] += 1
        output_count[test_labels[i]][rf_predictions[i]] += 1
    print('    Prediction details:')
    for i in range(5):
        print('    {} tests for class {}:'.format(test_count[i], classes[i]), end='')
        for j in range(5):
            print('{:3d}'.format(output_count[i][j]), end='')
        print()

print('\nAverage accuracy: {}'.format(sum(acc) / len(acc)))
out += '\nAverage accuracy: {}'.format(sum(acc) / len(acc))

