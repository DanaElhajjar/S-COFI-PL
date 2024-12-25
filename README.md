# S-COFI-PL

This repository is dedicated to the implementation of Sequential Covariance fitting for InSAR Phase Linking (S-COFI-PL). This approach aims to estimate the phases of a new SAR images based on a block of past images. 

The repository provides reproduction of the results presented in the paper:
> Dana EL HAJJAR, Guillaume GINOLHAC, Yajing YAN, and Mohammed Nabil EL KORSO, " Sequential Covariance fitting for InSAR Phase Linking".

If you use any of the code or data provided here, please cite the above paper.

## Code organisation


├── README.md<br>
├── environment.yml<br>
├── <font color="#3465A4"><b>simulations</b></font><br>
│   ├── script_mse_COFIPL_vs_SCOFIPL_2blocs.py<br>
│   ├── script_mse_COFIPL_vs_SCOFIPL_compar_SCM_PO.py<br>
│   ├── script_mse_COFIPL_vs_SCOFIPL.py<br>
│   ├── script_simulation_COFIPL_vs_SCOFIPL_2blocs.py<br>
│   ├── script_simulation_COFIPL_vs_SCOFIPL_compar_SCM_PO.py<br>
│   └── script_simulation_COFIPL_vs_SCOFIPL.py<br>
└── <font color="#3465A4"><b>src</b></font><br>
    ├── cost_functions.py<br>
    ├── covariance_matrix.py<br>
    ├── generation.py<br>
    ├── gradients.py<br>
    ├── manifold.py<br>
    ├── optimization.py<br>
    ├── __init__.py<br>
    └── utility.py<br>

The main code for the methods is provided in src/ directory. The file optimization.py provides the function for the S-COFI-PL algorithm. The folder simulations/ provides the simulations.

## Environment

A conda environment is provided in the file `environment.yml` To create and use it run:

```console
conda env create -f environment.yml
conda activate s-cofi-pl
```

### Authors

* Dana El Hajjar, mail: dana.el-hajjar@univ-smb.fr,  dana.el-hajjar@centralesupelec.fr
* Guillaume Ginolhac, mail: guillaume.ginolhac@univ-smb.fr
* Yajing Yan, mail: yajing.yan@univ-smb.fr
* Mohammed Nabil El Korso, mail: mohammed.nabil.el-korso@centralesupelec.fr


Copyright @Université Savoie Mont Blanc, 2024
