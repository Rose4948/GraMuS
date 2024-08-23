# GraMuS
> GraMuS: a novel Graph representation learning and Multi-information based technique for Statement-level FL. GraMuS is comprised of two key components: a fine-grained fault diagnosis Graph integrally recording enriched Multi-information from various levels of granularity, and a multi-level collaborative suspiciousness measure which utilizes the interactions between FL tasks at various levels of granularity to extract existing/latent useful features from multi sources of information for more precise FL.

## Introduction
* This project corresponds to the paper `GraMuS: Boosting Statement-level Fault Localization via Graph Representation and Multimodal Information` (doi links will be added in the future).



## Provided Resources:

Within this open-source project, we offer essential resources for experimentation:


- **Data Folder**: Contains the necessary data files for conducting experiments.
- **Code Folder**: Includes the source code implementing the GraMuS strategy.
- **detailed information of motivation example Folder**:  The effectiveness of our approach is demonstrated through an example program containing a Cli27 program with critical information.
- **ExperimentalData Folder**: We also conducted further analysis on the SIR dataset, and the experimental results are presented in this folder.

These resources are made available to facilitate reference and reproducibility of the experimental process. Developers can clone the repository to access the code and data required for utilizing GBSR in fault localization experiments.


## Environment
PyTorch: V1.13.0  
OS: CentOS Linux release 7.9.2009 (Core) 

## Using GraMuS
> Example commands  

Find Faults for a specific project (commons lang):  

* ```python runtotalAll.py Lang 0 0.01 60 SpGGAT 15 3```  

where runtotal.py is main entry file. Using the above command, GraMuS would execute the `runAll.py`, `DataCofigAll.py`, `ModelAll.py`, `TransfomerAll.py`, `GGAT.py`, `sum.py`, respectively.    


> Note  
* The third, fourth, fifth, seventh and eighth parameters in the `Example commands` are `random seed`, `learning rate`, `batch size`, `training epoch`, `number of model layers`, respectively.  

* These values in the `Example commands` all are default configuration on GraMuS. If you are making a first attempt at using GraMuS in your project, it is recommended to use the default parameters.  

* `GGAT` is the gated graph attention neural network model. Since the adjacency matrix representing the graph structure is a sparse matrix, we additionally provide a sparse matrix-based gated graph attention neural network `SpGGAT` to reduce the space required at runtime. Both models are configured in `GGAT.py`. Should you wish to employ `SpGGAT`, you can designate the sixth parameter as `SpGGAT`, as exemplified in the `Example command`. 

    
>  Configuration of Multi-head attention  
* We did not use the multi-head attention mechanism in our work. However, the advantages of multi-head attention have been generally acknowledged, and we also retain the interface to configure the multi-head attention mechanism. If you do, go to `modelAll.py`.  

> Important Files  

* `runtotalAll.py` receives the parameters entered by the user and performs the experiment for the user specified project according to the corresponding configuration.

* `runAll.py` is responsible for training the ranking model for each buggy version of the project under test and predicting the fault location based on its graph representation. Each version of the ranking results is saved in a separate pkl file.
  
* `DataCofigAll.py` is a fault diagnosis graph construction file that generates input files for model training and testing. Its responsibility lies in constructing the fault diagnosis graph, representing the multimodal information of the buggy program into the graph structure, graph nodes, and their attributes.

* `ModleAll.py` and `TransformerAll.py` are framework files for model configuration,  overseeing model iteration and optimization. They facilitate the flexible configuration of various network models. To set up a new ranking model x, you can seamlessly align its input and output with the two files, enabling you to start using it effortlessly.

* `GGAT.py` provides detailed code for both `GGAT` and `SpGGAT`.  Please choose which one to use according to your actual situation.

* `sum.py` merges the results for all the buggy version of the project under test and  stores them in a pkl file. In addition, metrics about `top-1`,`top-3`, and `top-5` are displayed in the console.  

> Dataset
URL：https://pan.baidu.com/s/1-eekbS0oGh6c147qdeMkOw 
password：z2dp 

## Experimental data

> We have made detailed experimental data open-source. In this section, we will provide a detailed introduction to the role of each folder and its files in a tree-like structure.
- ExperimentalData  `The experimental results of our approach and baselines.`
  - GraMuS  `The experimental results of our approach and baselines.`
    - Chart.xlsx  ` All detailed data of GraMuS in all metrics on Chart `
    - Cli.xlsx  `   All detailed data of GraMuS in all metrics on Cli`
    - JxPath.xlsx  `All detailed data of GraMuS in all metrics on JxPath`
    - Lang.xlsx  `  All detailed data of GraMuS in all metrics on Lang`
    - Math.xlsx  `  All detailed data of GraMuS in all metrics on Math`
    - The details of ablation experiments on graph-based representation.xlsx  `Detailed ablation experiment results regarding four node representations, five edge representations, and the features as node attributes.`
    - Train and Test Time.xlsx  `Detailed time data regarding model training and inference.`
  -  abconfiguration.xlsx `Experimental results regarding the  different configurations of parameters a and b in our loss function.`
  -  SBFL-MBFL.xlsx  `All detailed data of 34 SBFL formulae and 34 MBFL formulae in all metrics on five subjects`
  -  Grace.xlsx `All detailed data of Grace in all metrics on five subjects`
  -  SmarFL Detailed Worst  Rankt.txt `Detailed ranking information of SmartFL in all metrics  on three subjects`
  -  SmartFL.xlsx `Results of SmartFL in all metrics on three subjects`
- detailed information of motivation example
  - 1 runing results produced by GraMuS `The result files generated by GraMuS running on Cli. Each file contains a ranking list of statements by GraMuS for the corresponding buggy version.`
  - 1 runing results produced by Grace `The result files generated by Grace running on Cli. Each file contains a ranking list of statements by Grace for the corresponding buggy version.`
  - 1 The performance of all approaches in motivation example， including detail suspiciousness and rank .doc
  - 1 code snippets Cli7.pdf `buggy code snippets from the 27th faulty version of Cli`
  - 1 patch.png `patch (correct code snippets) from the 27th faulty version of Cli`


