<h1 align="center">scState: Decoding stem cell state transitions through pathway-informed heterogeneous graph representations</h1>

## Description

We developed scState, a pathway-informed graph transformer designed to resolve continuous stem cell state transitions from scRNA-seq data. 

<p align="center">
  <img src="./images/workflow.jpg" alt="scState Flowchart" width="900">
</p>

## Installation

### System Requirements

* Python 3.8.x
* Hardware architecture: x86_64
* Linux system is recommended
* GPU is recommended for faster model training, but CPU installation is also supported

---

### Installation Steps

#### 1. Create a new conda environment

```bash
conda create --name scState python=3.8 -y
conda activate scState
```

---

#### 2. Install PyTorch and PyTorch Geometric

Install PyTorch and PyTorch Geometric before installing scState. Choose either the GPU or CPU version according to your computing environment.

##### GPU version with CUDA 11.8

```bash
pip install https://download.pytorch.org/whl/cu118/torch-2.4.1%2Bcu118-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.4.0%2Bcu118/torch_scatter-2.1.2%2Bpt24cu118-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.4.0%2Bcu118/torch_sparse-0.6.18%2Bpt24cu118-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.4.0%2Bcu118/torch_cluster-1.6.3%2Bpt24cu118-cp38-cp38-linux_x86_64.whl
pip install torch-geometric==2.6.1
pip install torchmetrics==0.9.3
```

##### CPU version

```bash
pip install https://download.pytorch.org/whl/cpu/torch-2.4.1%2Bcpu-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.4.0%2Bcpu/torch_scatter-2.1.2%2Bpt24cpu-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.4.0%2Bcpu/torch_sparse-0.6.18%2Bpt24cpu-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.4.0%2Bcpu/torch_cluster-1.6.3%2Bpt24cpu-cp38-cp38-linux_x86_64.whl
pip install torch-geometric==2.6.1
pip install torchmetrics==0.9.3
```

---

#### 3. Install scState

The package name on PyPI is `scstate`:

```bash
pip install --upgrade scstate
```

To install a specific version:

```bash
pip install scstate==0.1.2
```

Although the package is installed as `scstate`, it should be imported in Python as `scState`:

```python
import scState
```

---

#### 4. Install trajectory-related dependencies

If trajectory-related analysis is needed, install the optional trajectory dependencies:

```bash
python -m pip install "scstate[trajectory]==0.1.2"
```

This installs:

```text
PhenoGraph==1.5.7
fcsparser==0.2.8
```

Then install `MulticoreTSNE` for Palantir:

```bash
conda install -c conda-forge multicore-tsne=0.1 -y
```

Finally, install Palantir:

```bash
pip install palantir==1.0.0
```

If `pip install palantir==1.0.0` attempts to rebuild `MulticoreTSNE` and fails, use:

```bash
pip install palantir==1.0.0 --no-deps
```

---

#### 5. Add the environment to Jupyter Notebook

```bash
pip install ipykernel
python -m ipykernel install --user --name scState --display-name "Python (scState)"
```

---
## Code Contributor
Xue Liu carried out benchmark experiments and wrapped the code.
