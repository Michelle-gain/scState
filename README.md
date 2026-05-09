# scState
<h1 align="center">Deep-learning-enabled multi-omics analyses for prediction of future metastasis in cancer</h1>

## Description

We developed EmitGCL, a graph contrastive learning model that integrates metastatic knowledge to detect subtle differences in cell groups at primary and metastatic sites. 

<p align="center">
  <img src="./images/workflow.png" alt="EmitGCL Flowchart" width="900">
</p>

## Installation

### System Requirements

* Python Version >=3.8.0
* Hardware Architecture: x86_64
* Operating System: GNU/Linux or Windows or MacOS

### Approximate Runtime (example: 10,000 cells)
* On CPU: approximately 8–10 hours
* On GPU: approximately 2–4 hours

### Dependencies, EmitGCL has the following dependencies:

| **Package**         | **Version**           |
|:-------------------:|:---------------------:|
| **seaborn**         | 0.11.2                |
| **numpy**           | 1.22.3                |
| **scipy**           | 1.9.1                 |
| **tqdm**            | 4.64.0                |
| **torch_geometric** | 2.1.0.post1           |
| **pandas**          | 1.4.2                 |
| **bioservices**     | 1.11.2                |
| **torch**           | 1.12.0+cu102          |
| **h5py**            | 3.10.0                |
| **scanpy**          | 1.9.1                 |
| **anndata**         | 0.8.0                 |
| **torchmetrics**    | 0.9.3                 |
| **matplotlib**      | 3.5.1                 |
| **scikit-learn**    | 1.1.2                 |
| **leidenalg**       | 0.8.10                |


### Installation Steps

The installation process involves some optional and necessary steps. Here's the detailed breakdown:

1. **Recommended Step:** Create a new environment, you should use python 3.8.

    ```bash
    conda create --name emitgcl python=3.8
    conda activate emitgcl
    ```

2. **Necessary Step:** You need to install either the CPU or GPU version of PyTorch as per your preference, We recommend using the GPU version, which has a faster running speed compared to the CPU version:

    - **CPU Version**
        - For Linux/Windows/MacOS system (torch-1.12.0+ torch_cluster-1.6.0+ torch_scatter-2.0.9+ torch_sparse-0.6.14):
        
            ```bash
            pip install torch==1.12.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
            pip install torch_scatter==2.0.9 torch_sparse==0.6.14 torch_cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.12.0%2Bcpu/
            ```

    - **GPU Version**
        - Please visit the official PyTorch website at [PyTorch](https://pytorch.org/) to select and download the CUDA-enabled version of PyTorch that best matches your system configuration.
        - For linux system(You need to select the version that is compatible with your system's graphics card. For example: torch-1.12.0+ torch_cluster-1.6.0+ torch_scatter-2.1.0+ torch_sparse-0.6.16):
          
             ```bash
            pip install torch==1.12.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
            pip install torch_scatter==2.1.0 torch_sparse==0.6.16 torch_cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.12.0%2Bcu102/
             ```
        - For Windows system(You need to select the version that is compatible with your system's graphics card. For example: torch-1.12.0+ torch_cluster-1.6.0+ torch_scatter-2.1.0+ torch_sparse-0.6.16):

             ```bash
            pip install torch==1.12.0+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html
            pip install torch_scatter==2.1.0 torch_sparse==0.6.15 torch_cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.12.0%2Bcu116/
            ```
             
        - For MacOS system: According to the official PyTorch documentation, CUDA is not available on MacOS, please use the CPU Version.

3. **Necessary Step:** You can directly install EmitGCL using the pip command:

    ```bash
    pip install --upgrade EmitGCL
    ```
## Code Contributor
Maoteng Duan carried out benchmark experiments and wrapped the code.
  
## Citation
If you use this work in your research, please cite the following publication:
 
```bibtex
@article{wang2025deep,
     author = {Wang, X. and Duan, M. and Su, P. L. and Li, J. and Krull, J. and Jin, J. and Ma, Q.},
     title = {Deep-learning-enabled multi-omics analyses for prediction of future metastasis in cancer},
     journal = {bioRxiv},
     year = {2025},
     doi = {10.1101/2025.05.16.654579}, 
}
```
