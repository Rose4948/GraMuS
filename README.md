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
   1. **Defects4J**

In our experimental process, we focused on five Defects4J subjects:

| Subject   | Name                  | #Test | #Loc | #Version | #Faults |
|-----------|-----------------------|-------|------|----------|---------|
| Lang      | commons-lang          | 2,245 | 22K  | 52       | 123     |
| Chart     | jfreechart            | 2,205 | 96K  | 24       | 97      |
| Cli       | commons-cli           | 361   | 4K   | 33       | 83      |
| JxPath    | commons-jxpath        | 401   | 21K  | 22       | 71      |
| Math      | commons-math          | 3,602 | 85K  | 102      | 264     |
| Time      | Joda-Time             | 4,130 | 28K  | 24       | 63      |
| Closure   | Google Closure compile| 7,927 | 90K  | 30       | 77      |
|-----------|-----------------------|-------|------|----------|---------|
|ConDefects |                       | 34    | 96   | 374      | 374     |

These subjects collectively encompass a total of 661 faulty programs.

3. **Data Format**

The experiment data is provided in JSON format, offering a structured and versatile representation for ease of use. 

As an illustration, here is an example structure of the JSON data for the subject "Lang" with the version "Lang1":

```json
{   
   "proj": "Lang1", 
    "ans": [1], 
    "methods": {"org/apache/commons/lang3/StringUtils.java@isBlank.finalCharSequencecs": 0, "org/apache/commons/lang3/math/NumberUtils.java@createNumber.finalStringstr": 1,...}, 
    "ftest": {"org.apache.commons.lang3.math.NumberUtilsTest#TestLang747": 0}, 
    "rtest": {"org.apache.commons.lang3.StringUtilsTest#testDefaultIfBlank_StringString": 0, "org.apache.commons.lang3.StringUtilsTest#testDefaultIfBlank_StringBuffers": 1, ...}, 
    "lines": {"org/apache/commons/lang3/StringUtils.java25": 0, "org/apache/commons/lang3/StringUtils.java27": 1, ...}, 
    "ltype": {"0": "IfStatement", "1": "ForStatement",  ...},
"edge": [[0, 0], [1, 0], ...], 
"mutation": {"10": 0, "11": 1, ...}, 
    "mtype": {"0": "ROR.==.FALSE", "1": "LVR.0.POS", ...}, 
    "edge12": [[0, 0], [1, 0],...], 
    "edge13": [[4, 23], [8, 23],...], 
    "edge14": [[4, 0], [8, 0],...], 
}
```
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




