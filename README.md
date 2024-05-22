# Predictive coding in Pytorch

## Overview
This repository houses the implementation of predictive coding used in the [Bogacz Group](https://www.mrcbndu.ox.ac.uk/groups/bogacz-group). It offers an easy and opensource access to implementing a range of predictive coding networks.

In this repository you can also find tutorials that can be run on google colab for:
- predictive coding networks for supervised and unsupervised learning as discussed in []() and []() respectively,
- monte carlo predictive coding networks as proposed in []()
- recurrent predictive coding networks for memory tasks[]()
- temporal predictive coding networks as propsoed in []()



## Structure of the repository
The repository includes:
- the predictive coding library under `predictive_coding`
- `1_supervised_learning_pc.ipynb`, a tutorial on how to train a predictive coding model to perform classification on MNIST [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bogacz-Group/PredictiveCoding/blob/main/1_supervised_learning_pc.ipynb)
- `2_unsupervised_learning_mcpc.ipynb`, a tutorial on how to train a Monte Carlo predictive coding model on Gaussian data
- `3_memory_rpc.ipynb`, a tutorial on how to train a recurrent predictive coding model on a memory task
- `4_sequential_memory_tpc.ipynb`, a tutorial on how to train a temporal predictive coding model on a sequential memory task.


## Usage
Follow the next steps to clone the code and setup the necessary python libraries:

```bash
git clone https://github.com/gaspardol/MonteCarloPredictiveCoding.git
cd MonteCarloPredictiveCoding
pip install -r requirements.txt
```

Each tutorial can be run on Google Colab by opening 
## Citation
For those who find our work useful, here is how you can cite it:

```bibtex
@article {Oliviers2024,
	author = {Gaspard Oliviers and Rafal Bogacz and Alexander Meulemans},
	title = {Learning probability distributions of sensory inputs with Monte Carlo Predictive Coding},
	elocation-id = {2024.02.29.581455},
	year = {2024},
	doi = {10.1101/2024.02.29.581455},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/03/02/2024.02.29.581455},
	eprint = {https://www.biorxiv.org/content/early/2024/03/02/2024.02.29.581455.full.pdf},
	journal = {bioRxiv}
}

```

## Contact
For any inquiries or questions regarding the project, please feel free to contact Gaspard Oliviers at gaspard.oliviers@pmb.ox.ac.uk.

## Code Aknowledgements
This repository builds upon the following repositories/codes:
- https://github.com/YuhangSong/Prospective-Configuration
- https://github.com/yiyuezhuo/Deep-Latent-Gaussian-Models
- https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518


## Code Acknowledgments
This repository builds upon the following repositories/codes:
- https://github.com/YuhangSong/Prospective-Configuration
