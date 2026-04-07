# Mixed and Irregular Frequency Models: Comparative Analysis with Healthcare Applications

**Thomas B. Arnold**  
Sc.M. in Biostatistics, Brown University, Department of Biostatistics  
May 2025

Advisor: Dr. Alice J. Paul  
Reader: Dr. Stavroula A. Chrysanthopoulou

---

## Overview

This repository contains the experimental code for the master's thesis *Mixed and Irregular Frequency Models: Comparative Analysis with Healthcare Applications* (Brown University, 2025). The thesis evaluates two specialized neural network architectures — **BiTimelyGPT** and **GRU-D** — against traditional baseline methods (**Logistic Regression** and **XGBoost**) on partially observed time series (POTS): clinical time series characterized by irregular sampling intervals, missing values, and variable-length observations.

Contrary to theoretical expectations, the results consistently demonstrate that properly implemented baseline methods often match or exceed the performance of specialized architectures. Model performance was significantly influenced by sample size, process complexity, and feature sampling synchronicity.

---

## Experiments

Three experimental settings are covered, each with three models (BiTimelyGPT, GRU-D, LR/XGB):

| Experiment | Description | Notebooks |
|---|---|---|
| **PhysioNet 2012** | In-hospital mortality prediction on the PhysioNet 2012 Challenge clinical dataset (Sets A/B/C) | `experiments/*_physionet.ipynb` |
| **Simulated Tabular** | Synthetic tabular time series with a factorial design over N, regularity, missingness, synchronicity, process type, and outcome dependence. Run twice (REP0, REP1). | `experiments/*_sim-tab_REP0.ipynb`, `*_sim-tab_REP1.ipynb` |
| **Simulated Signal** | Synthetic biosignal time series under the same factorial design | `experiments/*_sim-signal.ipynb` |

The `sim-tab` experiment was conducted first (REP0 only); REP1 and the `sim-signal` experiment were added for the final thesis. The `debuging/` folder contains intermediate notebooks from model development and is not part of the final analysis.

---

## Repository Structure

```
experiments/        Final experiment notebooks (one per model × dataset)
debuging/           Development/diagnostic notebooks (not part of final analysis)
dependencies/
  BiTimelyGPT-main/ BiTimelyGPT source code
  sim_dgp.R         R data-generating process for simulated experiments
plot/               Notebook for PhysioNet data visualization
```

> **Note on original execution environment:** All experiments were run in Google Colab on T4 GPU instances (May 2025). At the time of running, the notebooks were flat files in a Google Colab folder; `sim_dgp.R` and the `BiTimelyGPT-main/` source were in the root of Google Drive (`MyDrive/`). This repository reorganizes those files for clarity, but the notebooks still reference the original Google Drive paths described below.

---

## Replicating in Google Colab

### 1. Set up Google Drive

The notebooks mount Google Drive and expect the following structure in `MyDrive/`:

```
MyDrive/
  BiTimelyGPT-main/
    BiTimelyGPT/          ← contents of dependencies/BiTimelyGPT-main/ in this repo
  sim_dgp.R               ← dependencies/sim_dgp.R in this repo
  physionet2012/          ← PhysioNet 2012 data (see below)
    set-a/
      Outcomes-a.txt
      (patient .txt files)
    set-b/
      Outcomes-b.txt
    set-c/
      Outcomes-c.txt
```

Copy `dependencies/BiTimelyGPT-main/` and `dependencies/sim_dgp.R` from this repo into the corresponding locations in your Google Drive.

### 2. Obtain the PhysioNet 2012 data

The PhysioNet experiment notebooks require the [PhysioNet Computing in Cardiology Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/) dataset (Sets A, B, and C). Download it from PhysioNet and place it under `MyDrive/physionet2012/` as shown above.

### 3. Runtime settings

- **Hardware accelerator:** T4 GPU (Runtime → Change runtime type → T4 GPU)
- **Python:** 3.11 (default in Colab as of May 2025)
- Required packages are installed at the top of each notebook via `!pip install`

### 4. Run the notebooks

Open any notebook in `experiments/` via Google Colab. Each notebook is self-contained: it mounts Drive, installs dependencies, generates or loads data, trains the model, and reports metrics. For the simulated experiments, `sim_dgp.R` is sourced directly inside the notebook via `rpy2` — no separate R setup step is needed.

Run order within a dataset does not matter; each notebook is independent.

---

## Citation

Arnold, T. B. (2025). *Mixed and Irregular Frequency Models: Comparative Analysis with Healthcare Applications*. Sc.M. Thesis, Brown University, Department of Biostatistics.
