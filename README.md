# LiverCT: machine learning based Liver Cell Type mapping
## Introduction
We developed LiverCT (machine learning based Liver Cell Type mapping), for mapping new datasets onto the normal liver atlas. The method mainly contains three parts:
- **Cell type classification**: Provide predictions for two levels of cell type labels. 
- **Deviated/Intermediate state identification**: Identify cells potentially in deviated states and intermediate states. 
- **Hepatocyte zonation reconstruction**: Specifically for hepatocytes, provide zonation reconstruction labels along the CV-PV axis at sub-lobule scale.

The code below can be obtained from github.

## Requirements
LiverCT's dependencies can be found in requirements.txt in this repository.

## Quick start
Here, we provide an example data of hcc from uniLIVER. Users can download it and run following scripts to understand the workflow of LiverCT. See demo.ipynb in this repository.