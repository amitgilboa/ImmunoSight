## Running the ImmunoSight pipeline on server ig03

#!/bin/bash

nice -19 python /work/amitk/airrmap/tcr_anchors_cdr3/scripts/simulations_ligo_cdr3.py --server ig03 --path_rep 'repertoire_sim100_50_01'
nice -19 python /work/amitk/airrmap/tcr_anchors_cdr3/scripts/project_tcr_anchors_cdr3.py --server ig03 --path_rep 'repertoire_sim100_50_01'
nice -19 python /work/amitk/airrmap/tcr_anchors_cdr3/scripts/project_tcr_anchors_cdr3_mlat.py --server ig03 --path_rep 'repertoire_sim100_50_01'
nice -19 python /work/amitk/airrmap/tcr_anchors_cdr3/scripts/classification_models_kde_cdr3_all_folds.py  --server ig03 --path_rep 'repertoire_sim100_50_01' --model_method 'kde_hist' --bins 128
