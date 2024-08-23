# GraMuS
> GraMuS: a novel Graph representation learning and Multi-information based technique for Statement-level FL. GraMuS is comprised of two key components: a fine-grained fault diagnosis Graph integrally recording enriched Multi-information from various levels of granularity, and a multi-level collaborative suspiciousness measure which utilizes the interactions between FL tasks at various levels of granularity to extract existing/latent useful features from multi sources of information for more precise FL.

## Introduction
* This project corresponds to the paper `GraMuS: Boosting Statement-level Fault Localization via Graph Representation and Multimodal Information` (doi links will be added in the future).



## Provided Resources:

Within this open-source project, we offer essential resources for experimentation:


- **Data Folder**: Contains the necessary data files for conducting experiments.
- **Code Folder**: Includes the source code implementing the GraMuS strategy.
- **detailed information of motivation example Folder**:  The effectiveness of our approach is demonstrated using a real faulty program, Cli27 from Defects4J. We provide complete suspiciousness scores and rankings from various fault localization techniques for this example.
- **ExperimentalData Folder**: Provide all experimental data on two benchmarks Defects4J and ConDefects for this study.

These resources are made available to facilitate reference and reproducibility of the experimental process. Developers can clone the repository to access the code and data required for utilizing GraMuS in fault localization experiments.

## Data Files
The project includes a dedicated "Data" folder, housing essential files required for experimentation. These data files are instrumental in supporting various aspects of fault localization strategies. Developers can find and utilize these files within the repository to conduct experiments, analyze results, and enhance their understanding of GraMuS's effectiveness in refining fault localization.

In our experimental process, we focused on seven subjects from Defects4J and 374 Python programs from ConDefects, with a total of 661 faulty programs:

| Subject   | Name                  | #Test | #Loc | #Version | #Faults |
|-----------|-----------------------|-------|------|----------|---------|
| Lang      | commons-lang          | 2,245 | 22K  | 52       | 123     |
| Chart     | jfreechart            | 2,205 | 96K  | 24       | 97      |
| Cli       | commons-cli           | 361   | 4K   | 33       | 83      |
| JxPath    | commons-jxpath        | 401   | 21K  | 22       | 71      |
| Math      | commons-math          | 3,602 | 85K  | 102      | 264     |
| Time      | Joda-Time             | 4,130 | 28K  | 24       | 63      |
| Closure   | Google Closure compile| 7,927 | 90K  | 30       | 77      |
|ConDefects |                       | 34    | 96   | 374      | 374     |

## Code Files

- **runtotalAll.py**: receives the parameters entered by the user and performs the experiment for the user specified project according to the corresponding configuration.

- **runAll.py**: is responsible for training the ranking model for each buggy version of the project under test and predicting the fault location based on its graph representation. Each version of the ranking results is saved in a separate pkl file.
  
- **DataCofigAll.py**: is a fault diagnosis graph construction file that generates input files for model training and testing. Its responsibility lies in constructing the fault diagnosis graph, representing the multimodal information of the buggy program into the graph structure, graph nodes, and their attributes.

- **ModleAll.py** and **TransformerAll.py**: are framework files for model configuration,  overseeing model iteration and optimization. They facilitate the flexible configuration of various network models. To set up a new ranking model x, you can seamlessly align its input and output with the two files, enabling you to start using it effortlessly. Besides, we did not use the multi-head attention mechanism in our work. However, the advantages of multi-head attention have been generally acknowledged, and we also retain the interface to configure the multi-head attention mechanism in the file `ModleAll.py`.  

- **GGAT.py**: provides detailed code for both `GGAT` and `SpGGAT`.  Please choose which one to use according to your actual situation.

- **sum.py**: merges the results for all the buggy version of the project under test and  stores them in a pkl file. In addition, metrics about `top-1`,`top-3`, and `top-5` are displayed in the console.  


## Using GraMuS

To run the code for this project, ensure you have the following runtime environment:

1. **Python Version:** Python 3.X

Install the required dependencies using the following commands:

```bash
pip install numpy
pip install json
pip install pickle
# Add any other necessary dependencies
```
Please note that the provided list includes common dependencies like numpy, json, and pickle. Adjust the dependencies based on specific details within your code. If there are additional dependencies or specific versions required, update the installation commands accordingly.

2. **PyTorch Version:** V1.13.0

```bash
#1. Install Conda
Visit the [Anaconda official website](https://www.anaconda.com/products/distribution) or [Miniconda official website](https://docs.conda.io/en/latest/miniconda.html) to install Conda suitable for your operating system.

#2. Configure PyTorch
After installing Conda, use Conda to configure the PyTorch environment according to the official guidelines and activate it. Note that GPU support is required.
conda activate pytorch
```

3. **Example commands**

  Change to the working directory, run GraMuS to localize faults for a specific project, for example, Lang:
  ```bash
  cd path/GraMuS-master
  python runtotalAll.py Lang 0 0.01 60 SpGGAT 15 3
  ```
where `runtotal.py` is main entry file, `Lang` is the projects under test, `0` is our default random seed, `0.01` is the learning rate, `60`is the batch size, `SpGGAT` is the gated graph attention neural network model, `15` is training epoch, and `3` is the number of model layers. Using the above command, GraMuS would execute the `runAll.py`, `DataCofigAll.py`, `ModelAll.py`, `TransfomerAll.py`, `GGAT.py`, `sum.py`, respectively.    

> Note  
* These values in the `Example commands` all are default configuration on GraMuS. If you are making a first attempt at using GraMuS in your project, it is recommended to use the default parameters.  

* `GGAT` is the gated graph attention neural network model. Since the adjacency matrix representing the graph structure is a sparse matrix, we additionally provide a sparse matrix-based gated graph attention neural network `SpGGAT` to reduce the space required at runtime. Both models are configured in `GGAT.py`. Should you wish to employ `SpGGAT`, you can designate the sixth parameter as `SpGGAT`, as exemplified in the `Example command`. 









