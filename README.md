## ImmunoSight - A Visualization tool to classify T cell receptor repertoires

ImmunoSight involves a comprehensive exploration of the TCR repertoire landscape to unveil crucial insights into immune responses. 
This is accomplished through a two-stage methodology, wherein we initially project a small set of public TCR anchors to a two-dimensional space using MDS, followed by the computation of coordinates for the remaining study sequences using MLAT.

This two-stage approach, coupled with the utilization of anchors, yields several advantages over traditional forms of dimensionality reduction. Notably, it mitigates the computational burden associated with projecting tens of thousands of receptors to a 2D space by calculating their pairwise distances.
Additionally, by projecting every repertoire using the same set of anchors shared across all TCR repertoires, the use of anchors ensures consistency in the 2D sequence embeddings across plots. This facilitates comparisons across multiple studies and subjects, wherein activity occurring in equivalent regions of space suggests a shared pattern of response.
While there exist numerous hyperparameters to fine-tune, we meticulously selected those that offer the most insightful perspective. By judiciously choosing these parameters, we aim to capture the essence of the immune system's complexity and provide a coherent representation of TCR repertoire dynamics. 

The ImmunoSight workflow begins by identifying anchor sequences. These anchors act as reference points for projecting all the repertoires onto the same informative 2D space. Once established, a dissimilarity matrix is computed for these anchors, utilizing the Levenshtein metric. Next, the anchor sequences undergo projection using MDS. Simultaneously, a dissimilarity computation is performed for the study sequences relative to the anchors, employing the same Levenshtein metric. These study sequences are then projected into the anchor space using the MLAT method, yielding a 2D projection. Following this, dense areas within each repertoire projection are identified. Subsequently, features such as V genes, CDR3 amino acid frequencies, etc., are extracted from sequences in these dense areas. 
Finally, an AdaBoost classifier with 5-fold cross-validation is applied to classify repertoires of patients with a disease from healthy patients.

![ImmunoSight_workflow](https://github.com/amitgilboa/ImmunoSight/assets/58215769/bab86bbe-cc79-4757-bd92-40d5333fb564)
