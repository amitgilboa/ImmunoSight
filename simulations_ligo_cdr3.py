import pandas as pd
from Bio.Seq import Seq
from tqdm import tqdm
import os
import pickle
import math
import sklearn.cluster
import numpy as np
import shutil
import changeo.Gene
import xlrd
import random
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
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
import glob
import yaml
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

# I created a conda environment on the server to run ligo in python 3.11 :
# conda create -m ligo_simulation python=3.11
# conda activate ligo_simulation
# to exit the conda environment:
# conda deactivate

########################################
### create a yaml files and run ligo ###
########################################

### run ligo in the command line:
# ligo repertoire_sim1.yaml repertoire_sim1

#######################################################################################################################
## Define arguments

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Define Anchors')

# Add arguments
parser.add_argument('--server', help='Name of the server', type=str)
parser.add_argument('--path_rep', help='Path of the repertoires directory', type=str)
# parser.add_argument('--cdr12_weight', help='cdr12 weight', type=np.float32, default=0.05)
# parser.add_argument('--cdr3_weight', help='cdr3 weight', type=np.float32, default=0.95)
# parser.add_argument('--anchors_num', help='Number of anchors', type=int, default=1000)
# parser.add_argument('--th', help='Mutation threshold', type=int, default=2)
# parser.add_argument('--n_splits', help='Number of folds', type=int, default=5)
parser.add_argument('--seed', help='KFold random state', type=int, default=42)

# Parse the command-line arguments
args = parser.parse_args()

# Access the argument values
server_name = args.server
path_rep = args.path_rep
# cdr12_weight = args.cdr12_weight
# cdr3_weight = args.cdr3_weight
# s = args.anchors_num
# th = args.th
# n_splits = args.n_splits
seed = args.seed

###################################
### read LigO output repertoires ###
###################################

print('Starting...')
print('path repertoires: ' + path_rep)

if server_name=='ig01'or server_name=='ig02' or server_name=='ig04':
    os.chdir("/misc/work/amitk/airrmap/simulations/" + path_rep + "/")
    vgene_ref_alignment = pd.read_csv("/misc/work/amitk/airrmap/trbv_ref/vgen_ref_final.csv")
else:
    os.chdir("/work/amitk/airrmap/simulations/" + path_rep + "/")
    vgene_ref_alignment = pd.read_csv("/work/amitk/airrmap/trbv_ref/vgen_ref_final.csv")


### read the repertoires with signal

folder_path_siganl = "my_sim_inst/sim_item1/repertoires/*.npy"
npy_rep_signal = glob.glob(folder_path_siganl)

### read the control repertoires

folder_path_control = "my_sim_inst/sim_item2/repertoires/*.npy"
npy_rep_control = glob.glob(folder_path_control)

### read the yaml files of repertoires with signal

folder_path_yaml = "my_sim_inst/sim_item1/repertoires/*.yaml"
yaml_files_signal = glob.glob(folder_path_yaml)

yaml_path = yaml_files_signal[0]
with open(yaml_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

yaml_df = pd.DataFrame(yaml_data)

### merge all the signal repertoires

dfs = []
for file_name in npy_rep_signal:
    np_array = np.load(file_name)
    df = pd.DataFrame(np_array, columns=yaml_df['field_list'])
    df['file'] = os.path.splitext(os.path.basename(file_name))[0]
    dfs.append(df)

repertoires_signal = pd.concat(dfs, ignore_index=True)
repertoires_signal['status'] = 'signal'

### merge all the control repertoires

dfs = []
for file_name in npy_rep_control:
    np_array = np.load(file_name)
    df = pd.DataFrame(np_array, columns=yaml_df['field_list'])
    df['file'] = os.path.splitext(os.path.basename(file_name))[0]
    dfs.append(df)

repertoires_control = pd.concat(dfs, ignore_index=True)
repertoires_control['status'] = 'control'

all_repertoires = pd.concat([repertoires_signal, repertoires_control], ignore_index=True)

########################################
### find public groups of V_J_CDR3aa ###
########################################

all_repertoires['v_gene'] = all_repertoires['v_call'].apply(lambda x: x.split("*")[0])
all_repertoires['j_gene'] = all_repertoires['j_call'].apply(lambda x: x.split("*")[0])
all_repertoires['v_j_cdr3aa'] = all_repertoires.v_gene + '_' + all_repertoires.j_gene + '_' + all_repertoires.sequence_aa

# create a duplicate_count variable:

all_repertoires['duplicate_count'].value_counts() # all values are -1
all_repertoires['duplicate_count'] = 1 # change all values from -1 to 1

if not os.path.exists('data'):
    os.mkdir('data')

all_repertoires.to_pickle("data/all_repertoires.pkl")
# all_repertoires = pd.read_pickle("data/all_repertoires.pkl")

df_unique = all_repertoires.groupby(['file','status','signal1','v_j_cdr3aa'], as_index=False)['duplicate_count'].sum() # add duplicate_count variable per  v_j_cdr3aa

# df_public = df_unique.groupby(['v_j_cdr3aa', 'signal1'])['file'].nunique().reset_index(name='count')
# df_public = df_unique.groupby(['v_j_cdr3aa'])['file'].count().sort_values().rename_axis('v_j_cdr3aa').reset_index(name='count') # find in how many repertoires each public group appear
# df_public.to_pickle("data/df_public.pkl")

df_unique['v_gene'] = df_unique['v_j_cdr3aa'].apply(lambda x: x.split('_')[0])
df_unique['public_cdr3'] = df_unique['v_j_cdr3aa'].apply(lambda x: x.split('_')[2])

df_unique['v_gene'] = ['TRBV12-3/TRBV12-4' if v == 'TRBV12-3' or v == 'TRBV12-4' else v for v in df_unique['v_gene'] ]
df_unique['v_gene'] = ['TRBV3-1/TRBV3-2' if v == 'TRBV3-1' or v == 'TRBV3-2' else v for v in df_unique['v_gene'] ]
df_unique['v_gene'] = ['TRBV6-2/TRBV6-3' if v == 'TRBV6-2' or v == 'TRBV6-3' else v for v in df_unique['v_gene'] ]
df_unique['v_gene'] = ['TRBV6-5/TRBV6-6' if v == 'TRBV6-5' or v == 'TRBV6-6' else v for v in df_unique['v_gene'] ]

print('Save df_unique...')

df_unique.to_pickle("data/df_unique.pkl")

#




