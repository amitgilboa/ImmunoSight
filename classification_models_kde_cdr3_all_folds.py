import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import pandas as pd
import argparse
import os
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.signal import convolve2d, convolve, gaussian
from numba.core.decorators import jit
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier


#######################################################################################################################
#######################################################################################################################
## Define arguments

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Description of your script.')
# Add arguments
parser.add_argument('--server', help='Name of the server.')
parser.add_argument('--cdr12_weight', help='cdr12 weight', type=np.float32, default=0.05)
parser.add_argument('--cdr3_weight', help='cdr3 weight', type=np.float32, default=0.95)
parser.add_argument('--anchors_num', help='Number of anchors', type=int, default=1000)
parser.add_argument('--bins', help='Number of bins', type=int, default=256)
parser.add_argument('--th', help='Mutation threshold', type=int, default=2)
parser.add_argument('--path_rep', help='Path of the repertoires directory', type=str, default='')
parser.add_argument('--dataset', help='Dataset name', type=str, default='simulation')
# parser.add_argument('--model_type', help='Classification model type: DT , RF , LR_elasticnet , LR_l1 , LR_l2', type=str)
parser.add_argument('--seed', help='KFold random state', type=int, default=42)
parser.add_argument('--model_method', help='Classification model method: peak_seq_features , peak_plus_seq_features , kde_hist, kde_pixels, pixels_features, pixels_features_pca', type=str)
parser.add_argument('--feature_selection', help='Feature selection method: RFECV , RFE', type=str, default='RFECV')
parser.add_argument('--min_features', help='Min features to select for feature selection', type=int, default=0) # 20 / 0
parser.add_argument('--kde_adjust', help='Min features to select for feature selection', type=np.float32, default=1.) # 20 / 0

# Parse the command-line arguments
args = parser.parse_args()

# Access the argument values
server_name = args.server
cdr12_weight = args.cdr12_weight
cdr3_weight = args.cdr3_weight
s = args.anchors_num
th = args.th
path_rep = args.path_rep
# model_type = args.model_type  # DT for descision tree, RF for random forest, LR_elasticnet for logistic regression with elasticnet, LR_l1 for logistic regression with lasso, LR_l2 for logistic regression with l2RFrf
seed = args.seed
model_method = args.model_method
feature_selection = args.feature_selection
min_features = args.min_features
bins = args.bins
kde_adjust = args.kde_adjust
dataset = args.dataset
#
# server_name = 'ig03'
# cdr12_weight = 0.05
# cdr3_weight = 0.95
# s = 1000
# th = 2
# path_rep = ''
# seed = 42
# bins = 128
# kde_adjust = 1

print('Starting...')

# if server_name == 'ig01' or server_name == 'ig02' or server_name == 'ig04':
#     os.chdir("/misc/work/amitk/airrmap/simulations/" + path_rep + "/")
# else:
#     os.chdir("/work/amitk/airrmap/simulations/" + path_rep + "/")

if server_name=='ig01'or server_name=='ig02' or server_name=='ig04':
    if dataset=='hiv':
        os.chdir("/misc/work/amitk/airrmap/hiv/")
    else:
        os.chdir("/misc/work/amitk/airrmap/simulations/" + path_rep + "/")
else:
    if dataset=='hiv':
        os.chdir("/work/amitk/airrmap/hiv/")
    else:
        os.chdir("/work/amitk/airrmap/simulations/" + path_rep + "/")


#################
### Functions ###
#################

# Function to extract 3-mers from a string with a gap of 1. for example: "CASSAAA" -> CAS, ASS, SSA, SAA, AAA
def extract_3mers(sequence):
    return [sequence[i:i+3] for i in range(len(sequence) - 2)]

def max_indices(arr, k):
    '''
    Returns the indices of the k first largest elements of arr
    (in descending order in values)
    '''
    assert k <= arr.size, 'k should be smaller or equal to the array size'
    arr_ = arr.astype(float)  # make a copy of arr
    max_idxs = []
    for _ in range(k):
        max_element = np.max(arr_)
        if np.isinf(max_element):
            break
        else:
            idx = np.where(arr_ == max_element)
        max_idxs.append(idx)
        arr_[idx] = -np.inf
    return max_idxs

def format_length(value):
    return f'cdr3aa_len{value}'

######################
### Read df_unique ###
######################

df_unique = pd.read_pickle("data/df_unique.pkl") # the unique sequences after we added a duplicate_count variable
files = df_unique['file'].unique()
metadata = df_unique[['file','status']]
metadata = metadata.drop_duplicates(ignore_index=True)
files = metadata['file']

if dataset=='hiv':
    metadata['status'] = metadata['status'].replace({'Female': 'signal', 'Male': 'control'})

#######################################################################################
### create the histograms by min_x,max_x,min_y,max_y of the train data in each fold ###
#######################################################################################

# seed = 42
# model_type = "DT" for the 2nd solution
n_splits = 5
# kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

max_top_ind = 1000 # 1000 / 10000 / 500
# Set the regularization strength (C)
C = 1 # 0.01 / 0.05 / 0.1 / 1
# bins = 128



model_lr1 = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=seed)
model_adaboost = AdaBoostClassifier()
model_adaboost_lz = AdaBoostClassifier()

# rfe_cv_dt = RFECV(estimator=model_dt, cv=kf, scoring='roc_auc')
# rfe_cv_rf = RFECV(estimator=model_rf, cv=kf, scoring='roc_auc')
# rfe_cv_lr1 = RFECV(estimator=model_lr1, cv=kf, scoring='roc_auc')
# rfe_cv_lr_elas = RFECV(estimator=model_lr_elas, cv=kf, scoring='roc_auc')

accuracy_vec = []
auc_vec = []
f1_vec = []
all_selected_features = np.array([])

metadata = df_unique[['file','status']]
metadata = metadata.drop_duplicates().reset_index(drop=True)

if dataset=='hiv':
    metadata['status'] = metadata['status'].replace({'Female': 'signal', 'Male': 'control'})

# results_dt = {'Fold': [], 'AUC': [], 'F1 Score': [], 'Features': []} # results list. convert to csv file
results_lr1 = {'Fold': [], 'AUC': [], 'F1 Score': []} # results list. convert to csv file
results_adaboost = {'Fold': [], 'AUC': [], 'F1 Score': []} # results list. convert to csv file
results_lz = {'Fold': [], 'AUC': [], 'F1 Score': []} # results list. convert to csv file

# all_selected_features_dt = np.array([])
all_selected_features_lr1 = np.array([])
all_selected_features_adaboost = np.array([])
all_selected_features_lz = np.array([])

if not os.path.exists('figures'):
    os.mkdir('figures')

########################################################################################################################
########################################################################################################################


############################################
### apply kde on the original histograms ###
############################################

@jit
def fastkde(x, y, gridsize=(256, 256), extents=None, nocorrelation=False, weights=None, adjust=kde_adjust):
    # Variable check
    x, y = np.asarray(x), np.asarray(y)
    x, y = np.squeeze(x), np.squeeze(y)
    if x.size != y.size:
        raise ValueError('Input x & y arrays must be the same size!')
    n = x.size
    if weights is None:
        # Default: Weight all points equally
        weights = np.ones(n)
    else:
        weights = np.squeeze(np.asarray(weights))
        if weights.size != x.size:
            raise ValueError('Input weights must be an array of the same size as input x & y arrays!')
    # Optimize gridsize ------------------------------------------------------
    # Make grid and discretize the data and round it to the next power of 2
    # to optimize with the fft usage
    if gridsize is None:
        gridsize = np.asarray([np.max((len(x), 512.)), np.max((len(y), 512.))])
    gridsize = 2 ** np.ceil(np.log2(gridsize))  # round to next power of 2
    nx, ny = gridsize
    # Make the sparse 2d-histogram -------------------------------------------
    # Default extents are the extent of the data
    if extents is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = map(float, extents)
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    # Basically, this is just doing what np.digitize does with one less copy
    # xyi contains the bins of each point as a 2d array [(xi,yi)]
    xyi = np.vstack((x, y)).T
    xyi -= [xmin, ymin]
    xyi /= [dx, dy]
    xyi = np.floor(xyi, xyi).T
    xyi[xyi>(nx-1)] = nx-1  # if xyi values are above the grid i give them the maximum value: nx=256
    xyi[xyi < 0] = 0  # if xyi values are...
    # Next, make a 2D histogram of x & y.
    # Exploit a sparse coo_matrix avoiding np.histogram2d due to excessive
    # memory usage with many points
    grid = coo_matrix((weights, xyi), shape=(int(nx), int(ny))).toarray()
    # Kernel Preliminary Calculations ---------------------------------------
    # Calculate the covariance matrix (in pixel coords)
    xyi = xyi.astype(float)
    cov = np.cov(xyi)
    if nocorrelation:
        cov[1, 0] = 0
        cov[0, 1] = 0
    # Scaling factor for bandwidth
    scotts_factor = n ** (-1.0 / 6.) * adjust  # For 2D
    # Make the gaussian kernel ---------------------------------------------
    # First, determine the bandwidth using Scott's rule
    # (note that Silvermann's rule gives the # same value for 2d datasets)
    std_devs = np.sqrt(np.diag(cov))
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)
    # Determine the bandwidth to use for the gaussian kernel
    inv_cov = np.linalg.inv(cov * scotts_factor ** 2)
# x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
    #xx = np.arange(kern_nx, dtype=np.float) - kern_nx / 2.0
    #yy = np.arange(kern_ny, dtype=np.float) - kern_ny / 2.0
    xx = np.arange(kern_nx, dtype=float) - kern_nx / 2.0
    yy = np.arange(kern_ny, dtype=float) - kern_ny / 2.0
    xx, yy = np.meshgrid(xx, yy)
    # Then evaluate the gaussian function on the kernel grid
    kernel = np.vstack((xx.flatten(), yy.flatten()))
    kernel = np.dot(inv_cov, kernel) * kernel
    kernel = np.sum(kernel, axis=0) / 2.0
    kernel = np.exp(-kernel)
    kernel = kernel.reshape((int(kern_ny), int(kern_nx)))
    # ---- Produce the kernel density estimate --------------------------------
    # Convolve the histogram with the gaussian kernel
    # use boundary=symm to correct for data boundaries in the kde
    grid = convolve2d(grid, kernel, mode='same', boundary='symm')
    # Normalization factor to divide result by so that units are in the same
    # units as scipy.stats.kde.gaussian_kde's output.
    norm_factor = 2 * np.pi * cov * scotts_factor ** 2
    norm_factor = np.linalg.det(norm_factor)
    norm_factor = n * dx * dy * np.sqrt(norm_factor)
    # Normalize the result
    grid /= norm_factor
    return grid, xyi, (xmin, xmax, ymin, ymax)


if model_method == "kde_hist":
    for fold, (train_index, test_index) in enumerate(kf.split(metadata['file'], metadata['status'])):
        print(f"Fold {fold}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        train_files = files[train_index]
        test_files = files[test_index]
        # calculate min_x, min_y, max_x, max_y for the train files:
        min_x_vec = []
        max_x_vec = []
        min_y_vec = []
        max_y_vec = []
        for i, f in enumerate(train_files):
            seq_coords_df = pd.read_pickle("data/all_folds/dist_matrix_" + str(s) + "_cdr3aa_anchors_th" + str(th) + "_mlat/mlat_" + f + ".pkl")
            min_x_vec.append(np.min(seq_coords_df['x']))
            max_x_vec.append(np.max(seq_coords_df['x']))
            min_y_vec.append(np.min(seq_coords_df['y']))
            max_y_vec.append(np.max(seq_coords_df['y']))
        # calculate the paramaeters:
        min_x = np.min(min_x_vec)
        max_x = np.max(max_x_vec)
        min_y = np.min(min_y_vec)
        max_y = np.max(max_y_vec)
        # calculate training histograms
        H_train = np.zeros((len(train_files), bins, bins))
        for i, f in enumerate(train_files):
            # print(i)
            seq_coords_df = pd.read_pickle("data/all_folds/dist_matrix_" + str(s) + "_cdr3aa_anchors_th" + str(th) + "_mlat/mlat_" + f + ".pkl")
            #### add k-means for clustering
            temp = df_unique[df_unique['file'] == f]
            weights = np.asarray(temp['duplicate_count'])
            x = seq_coords_df['x']
            y = seq_coords_df['y']
            kde_grid, xyi, kde_extents = fastkde(x, y, gridsize=(bins, bins), extents=(min_x, max_x, min_y, max_y), nocorrelation=False,weights=weights, adjust=kde_adjust)  ## kde plot # extents=(xmin, xmax, ymin, ymax)
            H_train[i, :, :] = kde_grid
        # calculate test histograms
        H_test = np.zeros((len(test_files), bins, bins))
        for i, f in enumerate(test_files):
            # print(i)
            seq_coords_df = pd.read_pickle("data/all_folds/dist_matrix_" + str(s) + "_cdr3aa_anchors_th" + str(th) + "_mlat/mlat_" + f + ".pkl")
            temp = df_unique[df_unique['file'] == f]
            weights = np.asarray(temp['duplicate_count'])
            x = seq_coords_df['x']
            y = seq_coords_df['y']
            # x_bins = np.linspace(min_x, max_x, bins + 1)
            # y_bins = np.linspace(min_y, max_y, bins + 1)
            kde_grid, xyi, kde_extents = fastkde(x, y, gridsize=(bins, bins), extents=(min_x, max_x, min_y, max_y), nocorrelation=False,weights=weights, adjust=kde_adjust)  ## kde plot # extents=(xmin, xmax, ymin, ymax)
            H_test[i, :, :] = kde_grid
        #### Find the sum of all H_train kdes with and without signal:
        metadata_train = metadata.iloc[train_index]
        # kde_train_signal = np.sum(H_train[metadata_train['status'] == 'signal'], axis=0)
        # kde_train_control = np.sum(H_train[metadata_train['status'] == 'control'], axis=0)
        kde_train_signal = np.mean(H_train[metadata_train['status'] == 'signal'], axis=0)
        kde_train_control = np.mean(H_train[metadata_train['status'] == 'control'], axis=0)
        kde_diff = np.abs(kde_train_signal-kde_train_control)
        ### find the top 1000 pixels with the maximum values?
        max_index = max_indices(kde_diff, max_top_ind) # the top max_top_ind hot areas in the difference between kde averaged signal and kde averaged control. maybe look for absolute difference? 1000 and not 50 top?
        df_top = pd.DataFrame(columns=['file', 'v_gene', 'public_cdr3', 'signal1', 'duplicate_count', 'status', 'train'])
        print('Starting hot areas loop...')
        # x_bins = np.linspace(min_x, max_x, bins + 1)
        # y_bins = np.linspace(min_y, max_y, bins + 1)
        for i, file in enumerate(files):
            print('-i is: ' + str(i))
            if file in train_files.values:
                train = 'train'
            elif file in test_files.values:
                train = 'test'
            # file = files[i]
            # H = histograms_cmv[i, :, :] # the histograms normalized by z-score
            # max_index = max_indices(H, max_top_ind)
            f_df = df_unique[df_unique['file'] == file]
            f_df = f_df.reset_index()
            status = f_df['status'].iloc[0]
            seq_coords_df = pd.read_pickle("data/all_folds/dist_matrix_" + str(s) + "_cdr3aa_anchors_th" + str(th) + "_mlat/mlat_" + file + ".pkl")
            weights = np.asarray(f_df['duplicate_count'])
            x = seq_coords_df['x']
            y = seq_coords_df['y']
            kde_grid, xyi, kde_extents = fastkde(x, y, gridsize=(bins, bins), extents=(min_x, max_x, min_y, max_y), nocorrelation=False,weights=weights, adjust=kde_adjust)  ## kde plot # extents=(xmin, xmax, ymin, ymax)
            for t in range(len(max_index)):
                ### find all the public groups with the similar x axis and y axis:
                rows_idx = np.intersect1d(np.where(xyi[0]==max_index[t][0][0]),np.where(xyi[1]==max_index[t][1][0]))
                if len(rows_idx) > 0:
                    df = pd.DataFrame(columns=['file', 'v_gene', 'public_cdr3', 'signal1', 'duplicate_count', 'status', 'train'],index=range(len(rows_idx)))
                    df['v_gene'].iloc[:] = f_df['v_gene'].loc[rows_idx]  ## find the public groups with the maximum H values
                    df['duplicate_count'].iloc[:] = f_df['duplicate_count'].loc[rows_idx]  ## find the public groups with the maximum H values
                    df['public_cdr3'].iloc[:] = f_df['public_cdr3'].loc[rows_idx]
                    df['signal1'].iloc[:] = f_df['signal1'].loc[rows_idx]
                    df['file'].iloc[:] = file
                    df['status'].iloc[:] = f_df['status'].iloc[0]
                    df['train'].iloc[:] = train
                    df_top = pd.concat([df_top, df], ignore_index=True)
        df_top.to_pickle("data/df_top_cdr3_fold" + str(fold) + ".pkl")
        print('End of hot areas loop')
        ############################ add BOW from LZGRAPH as features ############################
        ###### df_top = pd.read_pickle("data/df_top_fold" + str(fold) + ".pkl")
        # from LZGraphs.BOWEncoder import LZBOW
        # from LZGraphs.AminoAcidPositional import AAPLZGraph
        # from sklearn.svm import SVC
        # from sklearn import metrics
        # from sklearn.ensemble import AdaBoostClassifier
        # from sklearn.model_selection import cross_validate, StratifiedKFold
        # from sklearn.metrics import make_scorer, roc_auc_score, f1_score
        # from sklearn.datasets import make_classification
        # from sklearn.preprocessing import LabelEncoder
        # from sklearn.preprocessing import StandardScaler, MinMaxScaler
        # df_top_train = df_top[df_top['train']=='train']
        # sequence_list = df_top_train.public_cdr3.to_list()
        # # create vectorizer and choose the Nucleotide Double Positional (ndp) encdoing function (default is Naive)
        # vectorizer = LZBOW(encoding_function=AAPLZGraph.encode_sequence)
        # # fit on sequence list
        # vectorizer.fit(sequence_list)
        # # BOW dictionary
        # df = pd.DataFrame(index=range(len(files)), columns=list(vectorizer.dictionary))
        # df['file'] = ""
        # df['train'] = ""
        # for i, f in enumerate(files):
        #     print(i)
        #     temp = df_top[df_top['file'] == f]
        #     temp_sequence_list = temp.public_cdr3.to_list()
        #     bow_vector = vectorizer.transform(temp_sequence_list)
        #     df['file'].iloc[i] = f
        #     df.iloc[i, :len(bow_vector)] = bow_vector
        #     df['train'].iloc[i] = np.unique(temp['train'])
        # ################################################
        # df = df.merge(metadata, how='left', on='file')
        # df_train = df[df['train'] == 'train']
        # df_test = df[df['train'] == 'test']
        # ##
        # X_train = df_train.iloc[:, :-3]
        # y_train = df_train['status']
        # X_test = df_test.iloc[:, :-3]
        # y_test = df_test['status']
        # ##
        # model_adaboost_lz.fit(X_train, y_train)
        # y_pred = model_adaboost_lz.predict(X_test)
        # feature_importances = model_adaboost_lz.feature_importances_
        # selected_features = X.columns[feature_importances != 0]
        # all_selected_features_lz = np.concatenate((all_selected_features_lz, selected_features))
        ###
        # label_encoder = LabelEncoder()
        # y_pred_numeric = label_encoder.fit_transform(y_pred)
        # y_test_numeric = label_encoder.fit_transform(y_test)
        # auc = roc_auc_score(y_test_numeric, y_pred_numeric)
        # f1 = f1_score(y_test_numeric, y_pred_numeric)
        # results_lz['Fold'].append(fold)
        # results_lz['AUC'].append(auc)
        # results_lz['F1 Score'].append(f1)
        # # results_lz['Features'].append(selected_features)
        # print(f'AUC  (Fold {fold + 1}): {auc}')
        # print(f'F1  (Fold {fold + 1}): {f1}')
        #############################################################
        ##### end of calculation of hot areas
        ##### calculate the 3-mer frequency
        ###### df_top = pd.read_pickle("data/df_top_fold" + str(fold) + ".pkl")
        df_top['3mer'] = df_top['public_cdr3'].apply(extract_3mers)
        # Create a Counter for each file
        counters = {}
        for file, group in df_top.groupby('file'):
            counters[file] = Counter([item for sublist in group['3mer'].tolist() for item in sublist])
        all_3mers_df = pd.DataFrame(counters).fillna(0).T
        all_3mers_df['sum'] = np.sum(all_3mers_df.iloc[:, 1:], axis='columns')
        all_3mers_df = all_3mers_df.reset_index()
        all_3mers_df.iloc[:, 1:-1] = all_3mers_df.iloc[:, 1:-1].div(all_3mers_df['sum'], axis=0)
        all_3mers_df = all_3mers_df.rename(columns={'index': 'file'})
        all_3mers_df = all_3mers_df.merge(metadata, how='left', on='file')
        all_3mers_df['status'] = all_3mers_df['status'].replace({'control': 0, 'signal': 1})
        del all_3mers_df['sum']
        ##### add v genes freq
        vgene_df = df_top.groupby('file')['v_gene'].value_counts().reset_index(name='count')
        vgene_df['freq'] = vgene_df['count'] / vgene_df.groupby('file')['count'].transform('sum')
        vgene_df = vgene_df[['file', 'v_gene', 'freq']]
        vgene_df_t = vgene_df.pivot(index='file', columns='v_gene', values='freq')
        vgene_df_t.reset_index(inplace=True)
        ##### add cdr3 aa length frequency
        df_top['cdr3aa_length'] = df_top['public_cdr3'].apply(len)
        cdr3aa_len_df = df_top.groupby('file')['cdr3aa_length'].value_counts().reset_index(name='count')
        cdr3aa_len_df['freq'] = cdr3aa_len_df['count'] / cdr3aa_len_df.groupby('file')['count'].transform('sum')
        cdr3aa_len_df = cdr3aa_len_df[['file', 'cdr3aa_length', 'freq']]
        cdr3aa_len_df['cdr3aa_length'] = cdr3aa_len_df['cdr3aa_length'].apply(format_length)
        cdr3aa_len_df_t = cdr3aa_len_df.pivot(index='file', columns='cdr3aa_length', values='freq')
        cdr3aa_len_df_t.reset_index(inplace=True)
        # cdr3aa_len_df = df_top.groupby('file')['cdr3aa_length'].max().reset_index()  # the median very similar (15,16)
        ##### merge all features
        all_df = pd.merge(all_3mers_df, vgene_df_t, how='left', on='file')
        all_df = pd.merge(all_df, cdr3aa_len_df_t, how='left', on='file')
        # all_df = all_3mers_df
        all_df.fillna(0, inplace=True)  # replace all NaN values in 0
        ##### apply a classification method
        print('Starting classification model')
        X = all_df.drop(['file', 'status'], axis=1)  # all_df / all_3mers_df
        ##################################################################################
        ### plot the most frequent features (ignore CAS, ASS)
        # X['status'] = all_df['status']
        # from scipy.stats import ttest_ind
        # from statsmodels.stats.multitest import multipletests
        # import matplotlib.pyplot as plt
        # # Separate data based on 'status'
        # group_0 = X[X['status'] == 0]
        # group_1 = X[X['status'] == 1]
        # # Initialize an empty list to store feature names with significantly different means
        # different_means_features = []
        # p_values = []
        # # Loop through each feature column and perform t-test
        # for feature in X.columns[:-1]:  # Exclude the 'status' column
        #     _, p_value = ttest_ind(group_0[feature], group_1[feature])
        #     # You can adjust the significance level (e.g., 0.05) based on your requirements
        #     # if p_value < 0.05:
        #     different_means_features.append(feature)
        #     p_values.append(p_value)
        # # Perform multiple hypothesis testing correction (Benjamini-Hochberg)
        # reject, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')
        # ttest_results = pd.DataFrame(
        #     {'Feature': different_means_features, 'P-Value': p_values, 'Corrected P-Value': corrected_p_values,
        #      'Reject Null Hypothesis': reject})
        # ttest_results = ttest_results.sort_values(by='Corrected P-Value')
        # # Select the top 20 features based on corrected p-values
        # top_features = ttest_results.nsmallest(20, 'Corrected P-Value')
        # top_features['Average Frequency status=0'] = X.groupby('status').mean().loc[0, top_features['Feature']].tolist()
        # top_features['Average Frequency status=1'] = X.groupby('status').mean().loc[1, top_features['Feature']].tolist()
        # top_features['Std Frequency status=0'] = X.groupby('status').std().loc[0, top_features['Feature']].tolist()
        # top_features['Std Frequency status=1'] = X.groupby('status').std().loc[1, top_features['Feature']].tolist()
        # # Create a figure and axis
        # fig, ax = plt.subplots(figsize=(12, 6))
        # # Plot dodged error bars for each feature
        # x = np.arange(len(top_features['Feature']))
        # # Set the width of the bars and the distance between them
        # bar_width = 0.35
        # bar_distance = 0.4
        # # Plot the error bars for status=0
        # ax.errorbar(x - bar_distance / 2, top_features['Average Frequency status=0'],
        #             yerr=top_features['Std Frequency status=0'], fmt='o', color='red', label='status=0', capsize=5,
        #             capthick=2)
        # # Plot the error bars for status=1
        # ax.errorbar(x + bar_distance / 2, top_features['Average Frequency status=1'],
        #             yerr=top_features['Std Frequency status=1'], fmt='o', color='blue', label='status=1', capsize=5,
        #             capthick=2)
        # # Add asterisk above the error bars of rejected features
        # for i, reject_status in enumerate(top_features['Reject Null Hypothesis']):
        #     if reject_status:
        #         ax.text(x[i], max(top_features['Average Frequency status=0'].iloc[i],
        #                           top_features['Average Frequency status=1'].iloc[i]) + 0.00001,
        #                 '*', color='black', ha='center', va='center')
        # # Set x-axis labels and title
        # ax.set_xticks(x)
        # ax.set_xticklabels(top_features['Feature'], rotation=45, ha='right')
        # ax.set_title('Top 20 features with lowest corrected P-Value')
        # ax.legend()
        # plt.savefig("figures/top_features_fold" + str(fold + 1) + ".png")
        ##################################################################################
        X = all_df.drop(['file', 'status'], axis=1)  # all_df / all_3mers_df
        scaler = MinMaxScaler() ####### MinMaxScaler() / StandartScaler
        # X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        y = all_df['status']
        # Extract train and test sets for this fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        # Use the same scaler to transform the test data
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        ###### Apply RFECV on the training set for this fold #####
        # X_train_rfe_cv = rfe_cv_lr1.fit_transform(X_train, y_train)
        # selected_features = X.columns[rfe_cv_lr1.support_].to_numpy()
        # all_selected_features_lr1 = np.concatenate((all_selected_features_lr1, selected_features))
        # print(f'Selected Features (Fold {fold + 1}): {selected_features}')
        # model_lr1.fit(X_train_rfe_cv, y_train)
        model_lr1.fit(X_train, y_train)
        # X_test_selected = X_test[selected_features].values
        # X.columns[model_lr1.coef_[0] != 0]
        # y_pred = model_lr1.predict(X_test_selected)
        y_pred = model_lr1.predict(X_test)
        model_lr1.coef_[0][model_lr1.coef_[0] != 0]
        selected_features = X.columns[model_lr1.coef_[0] != 0]
        all_selected_features_lr1 = np.concatenate((all_selected_features_lr1, selected_features))
        ##
        label_encoder = LabelEncoder()
        y_pred_numeric = label_encoder.fit_transform(y_pred)
        y_test_numeric = label_encoder.fit_transform(y_test)
        auc = roc_auc_score(y_test_numeric, y_pred_numeric)
        f1 = f1_score(y_test_numeric, y_pred_numeric)
        results_lr1['Fold'].append(fold)
        results_lr1['AUC'].append(auc)
        results_lr1['F1 Score'].append(f1)
        # results_lr1['Features'].append(selected_features)
        print(f'AUC  (Fold {fold + 1}): {auc}')
        print(f'F1  (Fold {fold + 1}): {f1}')
        # Create a DataFrame from the results
    ########### Apply Adaboost
        model_adaboost.fit(X_train, y_train)
        y_pred = model_adaboost.predict(X_test)
        feature_importances = model_adaboost.feature_importances_
        selected_features = X.columns[feature_importances != 0]
        all_selected_features_adaboost = np.concatenate((all_selected_features_adaboost, selected_features))
        ##
        label_encoder = LabelEncoder()
        y_pred_numeric = label_encoder.fit_transform(y_pred)
        y_test_numeric = label_encoder.fit_transform(y_test)
        auc = roc_auc_score(y_test_numeric, y_pred_numeric)
        f1 = f1_score(y_test_numeric, y_pred_numeric)
        results_adaboost['Fold'].append(fold)
        results_adaboost['AUC'].append(auc)
        results_adaboost['F1 Score'].append(f1)
        # results_lr1['Features'].append(selected_features)
        print(f'AUC  (Fold {fold + 1}): {auc}')
        print(f'F1  (Fold {fold + 1}): {f1}')
        # Create a DataFrame from the results
    results_df = pd.DataFrame(results_lr1)
    results_df.to_csv('output_kde_LR1_without_RFECV_' + str(bins) + '_cdr3.csv', index=False)
    results_df = pd.DataFrame(results_adaboost)
    results_df.to_csv('output_kde_AdaBoost_' + str(bins) + '_cdr3.csv', index=False)
    # Open a file for writing the results for all 5 folds per model_type
    with open('output_kde_LR1_' +  str(bins) + '_cdr3.txt', 'w') as file:
        import sys
        original_stdout = sys.stdout
        sys.stdout = file
        print(f'Number of unique features: {(len(np.unique(all_selected_features_lr1)))}')
        print("\nOverall Selected Features:", np.unique(all_selected_features_lr1))
        sys.stdout = original_stdout
    with open('output_kde_AdaBoost_' + str(bins) + '_cdr3.txt', 'w') as file:
        import sys
        original_stdout = sys.stdout
        sys.stdout = file
        print(f'Number of unique features: {(len(np.unique(all_selected_features_adaboost)))}')
        print("\nOverall Selected Features:", np.unique(all_selected_features_adaboost))
        # Restore the original standard output
        sys.stdout = original_stdout




