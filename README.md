# Background

***

This code is the basis of our work aiming to train and apply the group-contribution based GNN in the prediction of vapor-liquid equilibrium data under ambient pressure for binary mixtures with limited descriptors. This is an **alpha version** of the codes used to generate the results published in:

[Interpretable vapor-liquid equilibrium prediction model based on graph neural networks and group-contribution concept]

DOI:

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

* `VLE_zeotrope.csv` All of the zeotropic VLE data used in the training, the size of data collected was doubled by swapping and rejoining the descriptors of component A and component B.
* `VLE_azeotrope.csv` All of the zeotropic VLE data used in the training, the size of data collected was doubled by swapping and rejoining the descriptors of component A and component B.
* `VLE.csv` All of the VLE data used in this work.
* `Tc_JCIM_normalized.csv` Critical temperature values (Tc) used in the Tc prediction model.
* `mixtures.xlsx` All of the mixtures collected.

## Cross-validation
To validate the feasibility of the method as well as to evaluate the training, do:
```commandline

python cross_validation.py

```

The data mentioned above will be disrupted as mixture-based unit for training and evalidation.

## Generate models for prediction
To generate models for the whole dataset with the descriptors contribute the most, do:
```commandline

cd all

python train_all.py

```

## Prediction
To acquire the T-xy data for target mixture, do:

```commandline

python predict.py

```
The name of the components of the target mixture is required to input sequentially, for example,

water
methanol

The name of the substance is also listed in `all//Tc_Tb.csv`.

The prediction results by ANN and RF will be written in `water_methanol_VLE_predict.csv`. 

We suggest the users will further regressed the prediction results with traditional thermodynamic models such as NRTL functions, for a practical use. 

We also suggest the users will expand the data set in the `all` folder for a more accurate prediction. Only the critical temperature and boiling point and the T-xy data are required.
