# GraMuS Code Usage Guide

This folder contains the source code for the **GraMuS** framework: a graph-based and multi-information-enhanced fault localization system at the statement level.

## Key Files

- `runtotalAll.py`: Main entry point. Accepts user parameters and initiates the experiment process.
- `runAll.py`: Trains the ranking model for each buggy version and outputs prediction results.
- `DataCofigAll.py`: Constructs the fault diagnosis graph using multimodal information extracted from buggy programs.
- `ModleAll.py` and `TransformerAll.py`: Define model architecture, training loops, and configurable components. Also include an optional multi-head attention interface (not used by default).
- `GGAT.py`: Implements two graph attention network models:
  - `GGAT` (standard gated GAT)
  - `SpGGAT` (sparse version optimized for memory usage)
- `sum.py`: Aggregates prediction results and outputs evaluation metrics (Top-1, Top-3, Top-5).

---

## Runtime Environment

### Python
- Version: **Python 3.x**

Install Python dependencies:
```bash
pip install numpy
pip install json
pip install pickle
# Add any other needed dependencies
```

### PyTorch
- Version: **1.13.0**
- **GPU support required**

Install PyTorch with Conda:
```bash
# Step 1: Install Conda
# Download from: https://www.anaconda.com/products/distribution

# Step 2: Create and activate the environment
conda create -n pytorch python=3.9
conda activate pytorch

# Step 3: Install PyTorch
conda install pytorch==1.13.0 torchvision torchaudio cudatoolkit=11.6 -c pytorch
```

---

## How to Run GraMuS

To run fault localization on a specific project (e.g., Lang), use the following command:

```bash
python runtotalAll.py Lang 0 0.01 60 SpGGAT 15 3
```

This command will automatically call the following modules in sequence:

- `runAll.py`
- `DataCofigAll.py`
- `ModleAll.py`
- `TransformerAll.py`
- `GGAT.py`
- `sum.py`

Each module is responsible for a part of the workflow: preparing data, building the graph, training the model, predicting suspiciousness scores, and summarizing results.

---

###  Parameter Explanation

| Position | Parameter   | Description                                                             |
|----------|-------------|-------------------------------------------------------------------------|
| 1        | `Lang`      | Target project (e.g., Lang, Math, Cli, etc.)                            |
| 2        | `0`         | Random seed                                                             |
| 3        | `0.01`      | Learning rate                                                           |
| 4        | `60`        | Batch size                                                              |
| 5        | `SpGGAT`    | GNN model type: choose from `GGAT` or `SpGGAT`                          |
| 6        | `15`        | Number of training epochs                                               |
| 7        | `3`         | Number of model layers                                                  |

**Example:**  
Run GraMuS using the sparse gated graph attention model (`SpGGAT`) on the Lang project:
```bash
python runtotalAll.py Lang 0 0.01 60 SpGGAT 15 3
```

---

## Notes

- All parameters above are default values and suitable for first-time usage.
- `SpGGAT` is recommended for large-scale graphs to reduce memory consumption. To use the standard version, replace it with `GGAT`.
- The model automatically stores the suspiciousness scores for each buggy version in `.pkl` files for further evaluation or comparison.

---

## Output

- Prediction results for each buggy version are stored in `.pkl` files in the output directory.
