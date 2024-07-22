# DataConsistency: Application 2

In this branch, we develop an efficient AI-based approach to predict if the client will subscribe (yes/no) a term deposit (variable y).

## Table of Contents
- [Prerequisites](#prerequisites)
  - [Anaconda and conda environment](#anaconda-and-conda-environment)
  - [Databases](#databases)
- [Goal of the project](#goal-of-the-project)
- [Project architecture](#project-architecture)
  - [1. Load the data and vizualize the content](#1-load-the-data-and-vizualize-the-content)  
  - [2. Homogeneize the data and train a first classification model](#2-homogeneize-the-data-and-train-a-first-classification-model) 
  - [3. Studying the influence of classification models](#3-studying-the-influence-of-classification-models)  
- [Code and Jupyter notebook available](#code-and-jupyter-notebook-available)

## Prerequisites

### Anaconda

To execute the code, we will set up a specific environment using Anaconda. To install it, visit [Anaconda Installation](https://docs.anaconda.com/free/anaconda/install/).

### Setup conda environment

First, create the conda environment:
```
conda create -n myMarketAnalysis python=3.8
```

Then, activate the conda environment:
```
conda activate myMarketAnalysis
```

Once the environment is properly created, install the necessary Python libraries to execute the code:
```
pip install numpy pandas scikit-learn numpy pandas scipy
```

### Database

In this project, we take the database available on Kaggle (https://www.kaggle.com/datasets/abdelazizsami/bank-marketing/data) and put here as a copy (downloaded on 22-07-2024) and stored in the folder (Databases)[Databases]. 

## Goal of the project

***The goal of the project is to use our tools to predict, if yes or no, a client will subscribe a term deposit based on data*** 

## Project architecture

This project consists of three main tasks:

[***1. Load the data and vizualize the content***](#1-load-the-data-and-vizualize-the-content)  
[***2. Homogeneize the data and train a first classification model***](#2-homogeneize-the-data-and-train-a-first-classification-model) 
[***3. Studying the influence of classification models***](#3-studying-the-influence-of-classification-models)  

Let us see in more details these aspects.

### 1. Load the data and vizualize the content

First, we need to load the data and vizualize them. To do this task, we can use the script [1-Load-data-and-vizualize.py](1-Load-data-and-vizualize.py) by typing:

```bash
python 1-Load-data-and-vizualize.py
```

The code will show you how the data are sparsed. Surprinsingly, depicted on the Figure above, we can observe that most of the data are related to the "no case", which can cause a bias in our analysis. In that case, we will need to homogeneize the data in order to have an equal number between the yes and no data.

### 2. Homogeneize the data and train a first classification model

### 3. Studying the influence of classification models

## Code and jupyter notebook available

The jupyter notebook released on Kaggle is available here: https://www.kaggle.com/code/celerse/bank-marketing-features-selections-and-acc-0-99
If you have any comments, remarks, or questions, do not hesitate to leave a comment or to contact me directly. I would be happy to discuss it directly with you !

