# GNNGL_PPI


Codes and models for the paper "GNNGL-PPI: Multi-category Prediction of Protein-Protein Interactions using Graph Neural Networks based on Global Graphs and Local Subgraphs".



## Using GNNGL_PPI

This repository contains:
- Requirements
- Data Processing
- Training
- Testing


### Requirements
    (1) python 3.7
    (2) torch-1.10.2+cu113
    (3) torchaudio-0.10.2
    (4) torchvision-0.11.3+cu113
    (5) dgl-1.0.2+cu113
    (6) cudatoolkit-10.1.168
    (7) numpy-1.19.5
    (8) pandas
    (9) scikit-learn-0.22.2
### Data Processing

The data processing codes in gnn_data.py (Class GNN_DATA), including:
- data reading (**def** \_\_init\_\_)
- protein vectorize (**def** get_feature_pretrain)
- generate pyg data (**def** generate_data)
- Data partition (**def** split_dataset)
    - For the first time, you need to set the parameter random_new=True to generate a new data set division json file. (Otherwise, an error will be reported, No such file or directory: "./xxxx/string.bfs.fold1.json")

### Training

Training codes in gnn_train.py, and the run script in run.py.


#### Dataset Download:


SHS27k and SHS148k: 
- http://yellowstone.cs.ucla.edu/~muhao/pipr/SHS_ppi_beta.zip

This repositorie uses the processed dataset download path:
- https://pan.baidu.com/s/1FU-Ij3LxyP9dOHZxO3Aclw (Extraction code: tibn)


