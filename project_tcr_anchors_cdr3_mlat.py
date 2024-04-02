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

#######################################################################################################################
## Define arguments

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Description of your script.')
# Add arguments
parser.add_argument('--server', help='Name of the server', type=str)
parser.add_argument('--cdr12_weight', help='cdr12 weight', type=np.float32, default=0)
parser.add_argument('--cdr3_weight', help='cdr3 weight', type=np.float32, default=1)
parser.add_argument('--anchors_num', help='Number of anchors', type=int, default=1000)
parser.add_argument('--th', help='Mutation threshold', type=int, default=2)
parser.add_argument('--path_rep', help='Path of the repertoires directory', type=str, default='')
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
dataset = args.dataset

print('Starting...')
print('Path repertoires: ' + path_rep)
print('Dataset: ' + dataset)

####################
#### apply MLAT ####
####################
import glob

# th = 2 # th=1 / th=2
# s = 10000 # 1000 / 10000
# cdr12_weight = 0.05 # 0.25 / 0.05
# cdr3_weight = 0.95 # 0.75 / 0.95

#################################
### read all the repertoires ###
################################

## calculate the dist_matirx for each repertoire separatly (the distance between each sequence in the repertoire to the anchors):


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


###### now run the mlat projetion on the repertoires x anchors distance matrix
dist_mat_path = "data/all_folds/dist_matrix_" + str(s) + "_cdr3aa_anchors_th" + str(th) + "/" # distance matrix from ~1,000 anchors

mat_files = os.listdir(dist_mat_path)
mat_files = sorted(mat_files)
ind = int(len(mat_files)/5)
len_ind = mat_files

directory = "data/all_folds/dist_matrix_" + str(s) + "_cdr3aa_anchors_th" + str(th) + "_mlat/"

if not os.path.exists(directory):
    os.mkdir(directory)

# ## read v_dicts dictionary:
# with open('data/v_dicts.pkl', 'rb') as pickle_file:
#     v_dicts = pickle.load(pickle_file)

# all_anchors_df_iter = pd.read_pickle('data/Fold' + str(fold+1) + '/all_anchors_df_iter.pkl')

strings = all_anchors_df_iter['public_cdr3']. to_numpy()
transformed_strings = np.array(strings).reshape(-1,1)
distance_matrix = pdist(transformed_strings,lambda x,y: distance(x[0],y[0]))


print('Calculate the cdr3 distance')

weighted_distance_matrix = cdr3_weight*squareform(distance_matrix) ## weighted distance of v gene and cdr3_aa
dist_matrix = weighted_distance_matrix
model = manifold.MDS(n_components=2, dissimilarity="precomputed") # random_state=random_state
coords = model.fit(dist_matrix).embedding_  # [[x,y]] coordinates
df_coords = pd.DataFrame(coords, columns=['x', 'y'])

#############################
print("Run MLAT function...")

import math
from numba.core.decorators import jit, njit
from scipy.optimize import minimize

# Define calc_distance
@njit
def calculate_distance(x1, y1, x2, y2):
    # This is related to
    # the plot distance (not AIR distance)
    dx = (x2 - x1)**2
    dy = (y2 - y1)**2
    return (dx + dy)**0.5

@njit
def mse(x, locations, distances):
    mse = 0.0
    for location, distance in zip(locations, distances):
        distance_calculated = calculate_distance(
            x[0], x[1], location[0], location[1])
        mse += math.pow(distance_calculated - distance, 2.0)
    return mse / len(distances)

@njit
def mae(x, locations, distances):
    mae = 0.0
    for location, distance in zip(locations, distances):
        distance_calculated = calculate_distance(
            x[0], x[1], location[0], location[1])
        mae += math.fabs(distance_calculated - distance)
    return mae / len(distances)

def calculate_coords(initial_coords, anchor_coords, anchor_distances, use_mae=False):
    # mse or mae
    obj_func = mae if use_mae else mse
    # %% Minimize the distance
    # # REF: https://www.alanzucconi.com/2017/03/13/positioning-and-trilateration/
    result = minimize(
        obj_func,                    # Mean Square Error function
        initial_coords,         # Initial guess
        args=(anchor_coords, anchor_distances),  # Additional parameters for mse
        method='L-BFGS-B',      # The optimisation algorithm
        # callback=callback_minimize, # callback function for history
        options={
            'ftol': 1e-5,        # Tolerance
            'maxiter': 1e+7     # Maximum iterations
        }
    )
    return result

#############################

for f in tqdm(len_ind):
    print(f)
    dist_matrix_data = np.load(dist_mat_path + f)
    seq_coords_df = pd.DataFrame(columns = ['x', 'y'])
    anchor_coords = df_coords.values
    # anchor_coords = np.array(list(df_coords_max.itertuples(index=False, name=None))) # array of df_coords
    for j in tqdm(range(len(dist_matrix_data))):
        anchor_distances = np.asarray(dist_matrix_data[j])
        index_min = min(range(len(anchor_distances)), key=anchor_distances.__getitem__) # the anchor with the minimum distance to the sequence
        initial_coords = tuple((df_coords['x'].iloc[index_min],df_coords['y'].iloc[index_min])) # the initial coords are the initial guess which is the centre (anchor) which detected the closet distance
        # initial_coords = tuple((df_coords[index_min][0], df_coords[index_min][1]))
        temp = calculate_coords(initial_coords, anchor_coords, anchor_distances, use_mae=False)
        seq_coords_df = pd.concat([seq_coords_df, pd.DataFrame([[temp.x[0], temp.x[1]]], columns = ['x', 'y'])], ignore_index=True)
    seq_coords_df.to_pickle("data/all_folds/dist_matrix_" + str(s) + "_cdr3aa_anchors_th" + str(th) + "_mlat/mlat_" + f[12:(len(f) - 4)] + ".pkl")

