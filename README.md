# Background

***

This code is the basis of our work aiming to train and apply the group-contribution based GNN in the prediction of vapor-liquid equilibrium data under ambient pressure for binary mixtures with limited descriptors. This is an **alpha version** of the codes used to generate the results in:

[Interpretable vapor-liquid equilibrium prediction model based on graph neural networks and group-contribution concept]

DOI:

The code is preliminarily built for the CPU training. In the future, the code will be update to the GPU version in https://github.com/sungl123456/37lambdaB_GPU

## Prerequisites and dependencies

The code is written in python and can be run from command prompt. This requires the following packages and modules:

* Python 3.10.16 or higher
* torch==2.7.0
* pandas==2.2.3
* numpy==2.0.1
* scikit-learn==1.6.1
* dgl==2.0.0
* matplotlib==3.10.1
* prettytable==3.16.0
* openpyxl==3.1.2
* rdkit==2024.9.6
* tqdm==4.67.1
* umap==0.1.1
* umap_learn==0.5.7

## Data

The data sets that used in the dataset folder are available as .csv documents. Three files are included:

* `VLE_total.csv` All of the VLE data used in this work.
* `Tc_JCIM_normalized.csv` Critical temperature values (Tc) used in the Tc prediction model.
* `Tb_JCIM_normalized.csv` Bubble-point temperature values (Tb) used in the Tc prediction model.
* `azeotrope_classification.csv` All of the mixtures collected with their azetrope/non-azeotrope tags.

## random seed 
To find the suitable random seed at first, do:
```commandline

python Seed_azeotropic_clf.py
python Seed_VLE_total.py

```
## Generate models for prediction
To train the corresponding model, do:
```commandline

python ensemble_azeotropic_clf.py
python ensemble_VLE_total.py

```
The data mentioned above will be disrupted as mixture-based unit for training and evalidation.


## Calculate the attention value
To generate attention value and the prediction results for the azeotropic classification and VLE prediction, do:
```commandline

python ensemble_attention_azeotropic_clf.py
python ensemble_attention_VLE_total.py

```

