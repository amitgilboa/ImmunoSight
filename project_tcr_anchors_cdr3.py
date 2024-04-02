import pandas as pd
import os
import pickle
import math
import sklearn.cluster
import numpy as np
import shutil
import changeo.Gene
import xlrd
import random
import matplotlib.pyplot as plt
# from plotnine import ggplot, aes, geom_line, ggsave
from polyleven import levenshtein
from tqdm import tqdm
from sklearn import manifold
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from Levenshtein import distance
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster
# from scipy.stats import itemfreq
import random
import seaborn as sns
import argparse
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold


#######################################################################################################################
## Define arguments

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Calculate distance matrix between repertoires and anchors')

# Add arguments
parser.add_argument('--server', help='Name of the server', type=str)
parser.add_argument('--cdr12_weight', help='cdr12 weight', type=np.float32, default=0)
parser.add_argument('--cdr3_weight', help='cdr3 weight', type=np.float32, default=1)
parser.add_argument('--anchors_num', help='Number of anchors', type=int, default=1000)
parser.add_argument('--th', help='Mutation threshold', type=int, default=2)
parser.add_argument('--path_rep', help='Path of the repertoires directory', type=str, default='')
parser.add_argument('--n_splits', help='Number of folds', type=int, default=5)
parser.add_argument('--seed', help='KFold random state', type=int, default=42)
parser.add_argument('--dataset', help='Dataset name', type=str, default='simulation')

# Parse the command-line arguments
args = parser.parse_args()

# Access the argument values
server_name = args.server
cdr12_weight = args.cdr12_weight
cdr3_weight = args.cdr3_weight
s = args.anchors_num
th = args.th
path_rep = args.path_rep
n_splits = args.n_splits
seed = args.seed
dataset = args.dataset

# th = 2 # th=1 / th=2
# s = 10000 # 1000 / 10000
# cdr12_weight = 0.05 # 0.25 / 0.05
# cdr3_weight = 0.95 # 0.75 / 0.95

kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

######################################################################################################################
### project all the repertoires by these anchors with the same weights of the weighted distance matrix using MLAT  ###
######################################################################################################################


print('Starting...')
print('Path repertoires: ' + path_rep)
print('Dataset: ' + dataset)


if server_name=='ig01'or server_name=='ig02' or server_name=='ig04':
    all_anchors_df_iter = pd.read_pickle('/misc/work/amitk/airrmap/tcr_anchors_cdr3/all_anchors_df_iter.pkl')
    if dataset=='hiv':
        os.chdir("/misc/work/amitk/airrmap/hiv/")
    else:
        os.chdir("/misc/work/amitk/airrmap/simulations/" + path_rep + "/")
else:
    all_anchors_df_iter = pd.read_pickle('/work/amitk/airrmap/tcr_anchors_cdr3/all_anchors_df_iter.pkl')
    if dataset=='hiv':
        os.chdir("/work/amitk/airrmap/hiv/")
    else:
        os.chdir("/work/amitk/airrmap/simulations/" + path_rep + "/")



## convert v_genes_mat_min matrix to a dictionary:
import itertools
import glob

################################
### read all the repertoires ###
################################
print('Read the repertoires data frame ...')

df_unique = pd.read_pickle("data/df_unique.pkl") # the unique sequences after we added a duplicate_count variable
files = np.unique(df_unique['file'])


## calculate the dist_matrix for each repertoire separatly (the distance between each sequence in the repertoire to the anchors):
anchors_list_cdr3 = all_anchors_df_iter['public_cdr3'].tolist()
# anchors_list_cdr12 = all_anchors_df_iter['v_gene'].tolist()
print('Start the for loop...')

if not os.path.exists("data/all_folds"):
    os.mkdir("data/all_folds")

directory = "data/all_folds/dist_matrix_" + str(s) + "_cdr3aa_anchors_th" + str(th) + "/"
if not os.path.exists(directory):
    os.mkdir(directory)
for f in tqdm(range(len(files))):
    print(f)
    temp = df_unique[df_unique['file'] == files[f]]
    rep_list_cdr3 = temp['public_cdr3'].tolist()
    # rep_list_cdr12 = temp['v_gene'].tolist()
    dist_list_all = []
    for i in tqdm(range(len(temp))):
        dist_list = []
        for j in (range(len(all_anchors_df_iter))):   #all_anchors_df_iter
            # v_gene_pair = rep_list_cdr12[i] + "_" + anchors_list_cdr12[j]
            dist_list.append(cdr3_weight * levenshtein(rep_list_cdr3[i], anchors_list_cdr3[j]))
        dist_list_all.append(dist_list)
    np.save("data/all_folds/dist_matrix_" + str(s) + "_cdr3aa_anchors_th" + str(th) + "/dist_matrix" + "_" + files[f] + ".npy" , np.array(dist_list_all))

