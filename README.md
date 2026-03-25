# AdRo-FL
A Secure and Informed Client Selection Framework for Federated Learning under Adversarial Aggregator
##
```
This repository contains the code, result files, and analysis scripts for the paper:

**AdRo-FL: Secure and Informed Client Selection for Federated Learning under Adversarial Aggregator**

Authors:

Md. Kamrul Hossain{1}, Walid Aljoby{1,2}, Anis Elgabli{2,3}, Ahmed M. Abdelmoniem{4}, Khaled A. Harras{5}

{1}{Information and Computer Science Department, King Fahd University of Petroleum and Minerals, Dhahran 31261, Saudi Arabia}

{2}{IRC for Intelligent Secure Systems, King Fahd University of Petroleum and Minerals, Dhahran 31261, Saudi Arabia}

{3}{Industrial and Systems Engineering Department, King Fahd University of Petroleum and Minerals, Dhahran 31261, Saudi Arabia}

{4}{School of Electronic Engineering and Computer Science, Queen Mary University of London, United Kingdom}

{5}{Department of Computer Science, Carnegie Mellon University, United States}
```

````
The repository includes:
- the core experiment code used in the paper,
- the final result files used to generate the reported figures and tables,
- the notebooks used for post-processing and visualization,
- and the additional scripts used to generate the revision/rebuttal artifacts.

## Repository structure

```text
food101/            # core training scripts for AdRo-FL, Random, Oort, and VRF-based settings
notebooks/          # notebooks used to regenerate paper figures/tables from result files
results/svhn2/      # final result files used in the paper
revision_tools/     # scripts and artifacts added for the revised manuscript / rebuttal
````

### Core paper pipeline

The main paper results were produced from the scripts in `food101/` and post-processed using the notebooks in `notebooks/`.

The three main experiment groups are:

1. **Cluster-oriented local/global selection** using AdRo-FL and Random
2. **Oort baselines**
3. **Distributed non-cluster VRF-based AdRo-FL**

### Revision pipeline

The `revision_tools/` directory contains helper scripts used to:

* regenerate figures,
* summarize result files into cleaner CSV outputs,
* generate adaptive-targeting analysis,
* and run focused ablation studies used in the revised manuscript and reviewer response.

## Environment

The experiments were run in a custom script-based federated learning simulator implemented in **Python** using **PyTorch**.

### Recommended software

* Python 3.9
* PyTorch
* torchvision
* numpy
* pandas
* matplotlib
* scipy
* jupyter

### Optional dependency

* `gurobipy` is required for the Oort baseline scripts.

## Installation

We recommend creating a fresh conda environment.

```bash
conda create -n adrofl python=3.9 -y
conda activate adrofl
pip install torch torchvision numpy pandas matplotlib scipy notebook
```

If you want to run the Oort baselines, also install:

```bash
pip install gurobipy
```

## Datasets

Raw datasets are **not included** in this repository.

Expected datasets:

* MNIST
* FashionMNIST
* CIFAR-10
* SVHN

In the original codebase, datasets were placed under `food101/data/`.

### Notes

* MNIST, FashionMNIST, and CIFAR-10 are typically downloaded automatically by torchvision.
* For SVHN, the scripts may expect:

  * `train_32x32.mat`
  * `test_32x32.mat`
    under `food101/data/`.

If your local setup differs, adjust the dataset paths inside the scripts accordingly.

## Reproducing the published figures and tables (no retraining required)

If you only want to reproduce the paper’s figures and tables from the included result files:

1. Ensure the final result files are present in:

   * `results/svhn2/`

2. Open and run the notebooks in `notebooks/`.

### Main notebooks

* `Last_4 dataset 3 methods acc loss plot local global in cluster setting.ipynb`

  * cluster-oriented accuracy/loss figures
* `find_accuracy_value_for_all_DS.ipynb`

  * time-to-accuracy summaries across datasets
* `find_accuracy_values_for_one_DS.ipynb`

  * time-to-accuracy summaries for one dataset
* `Last_plot_3methods_4dataset_VRF.ipynb`

  * distributed/non-cluster figures
* `Last_plot_privacy_violation.ipynb`

  * privacy-violation plot
* `Last_visualize_bitEnergy_from_file_1_dataset_3_methods.ipynb`

  * communication / energy plots

## Re-running the experiments

### 1) Cluster-oriented AdRo-FL / Random

The cluster scripts are:

* `food101/ClusterFed_mnist_custom_random_Global_local_param.py`
* `food101/ClusterFed_fmnist_custom_random_Global_local_param.py`
* `food101/ClusterFed_cifar10_custom_random_Global_local_param.py`
* `food101/ClusterFed_svhn_custom_random_Global_local_param.py`

These scripts are mainly configured by editing global variables inside each file, especially:

* `client_select_type` (`"custom"` or `"random"`)
* `selection_scope` (`"local"` or `"global"`)
* `VRF_scope` (`False` for cluster-oriented runs)

Example:

* AdRo-FL local cluster run:

  * `client_select_type = "custom"`
  * `selection_scope = "local"`
  * `VRF_scope = False`
* Random global cluster run:

  * `client_select_type = "random"`
  * `selection_scope = "global"`
  * `VRF_scope = False`

Then run:

```bash
python food101/ClusterFed_mnist_custom_random_Global_local_param.py
```

### 2) Oort baselines

The Oort scripts are:

* `food101/OORT_ClusterFed_mnist.py`
* `food101/OORT_ClusterFed_fmnist.py`
* `food101/OORT_ClusterFed_cifar10.py`
* `food101/OORT_ClusterFed_svhn.py`

Typical usage:

* local cluster-aware Oort:

```bash
python food101/OORT_ClusterFed_mnist.py --selection_mode local --K 2
```

* global cluster-aware Oort:

```bash
python food101/OORT_ClusterFed_mnist.py --selection_mode global --K 2
```

* insecure non-cluster Oort:

```bash
python food101/OORT_ClusterFed_mnist.py --selection_mode global --K 0
```

### 3) Distributed / VRF-based AdRo-FL

The distributed scripts are:

* `food101/VRF_informed_selection_mnist.py`
* `food101/VRF_informed_selection_fmnist.py`
* `food101/VRF_informed_selection_cifar10.py`
* `food101/VRF_informed_selection_svhn.py`

Example:

```bash
python food101/VRF_informed_selection_mnist.py
```

## Reproducing the revised-manuscript / rebuttal artifacts

The scripts in `revision_tools/` were used to generate the additional analysis for the revised manuscript, including:

* focused ablation studies,
* adaptive-targeting analysis,
* and styled versions of the main figures.

### Main scripts

* `revision_tools/adrofl_revision_runner.py`
* `revision_tools/adrofl_revision_runner_styled_figures.py`
* `revision_tools/adrofl_revision_runner_styled_figures_v2.py`

### Typical usage

Summaries and styled figures from existing results:

```bash
python revision_tools/adrofl_revision_runner.py \
  --repo-root . \
  --output-root revision_outputs \
  --mode all
```

Focused live ablation (heavy):

```bash
python revision_tools/adrofl_revision_runner.py \
  --repo-root . \
  --output-root revision_outputs_live \
  --mode all \
  --run-live-ablations \
  --datasets mnist,cifar10 \
  --rounds 3000
```

The curated outputs used in the revised manuscript and rebuttal are stored in:

* `revision_tools/artifacts/`

## Notes on paths and portability

This repository originated from a local research codebase, and some scripts may still contain hard-coded paths or assumptions from the original workstation setup. If you encounter path errors:

* update the path variables to your local repository root,
* ensure the dataset files are placed in the expected locations,
* and create output directories before running experiments if needed.

## Secure aggregation note

The experiments in this repository **do not instantiate a concrete secure aggregation protocol**. The paper evaluates a client-selection layer designed to remain compatible with secure aggregation; confidentiality of updates is assumed to be provided by an external compatible secure aggregation mechanism in deployment.

## Citation

If you use this repository, please cite the associated paper.

## Contact

For questions about the code or reproduction, please open a GitHub issue in this repository.

```
