# Predictive coding in Pytorch

## Overview
This repository houses the implementation of predictive coding used in the [Bogacz Group](https://www.mrcbndu.ox.ac.uk/groups/bogacz-group). It offers an easy and opensource access to implementing a range of predictive coding networks.

In this repository you can also find tutorials that can be run on google colab for:
- predictive coding networks for supervised and unsupervised learning as discussed in []() and []() respectively,
- monte carlo predictive coding networks as proposed in []()
- recurrent predictive coding networks for memory tasks[]()
- temporal predictive coding networks as propsoed in []()


## Key Contributions
- **Pytorch implementation**: Our Pytorch implementation of predictive coding offers a fast and efficient implementation that can eb run on CPU and GPU.
- **Extendable framework**: Our model demonstrates how precise generative models can be learned through mechanisms that are implementable within the biological constraints of neural networks.
- **Alignment with Experimental Observations**: MCPC not only offers theoretical contributions but also provides a compelling match to experimental data on neural variability, supporting its relevance and applicability to understanding brain function.


## Structure of the repository
The repository includes:
- `figure_2.py`, `figure_3.py`, `figure_4.py`, `figure_5.py`, `figure_6.py`, and `table_1.py` contain the code to recreate the figures and the table from our paper.
- `requirements.txt` contains the python dependencies of this repository.
- `figures/` contains the output figures of the code.
- `models/` contains trained models to generate figures.
- `utils/` contains utility functions
- `predictive_coding/` contains the code to simulate MCPC- and PC- models.
- `Deep_Latent_Gaussian_Models/` contains code to simulate DLGMs.
- `ResNet.py` contains the code to simulate ResNet-9 models.

## Usage
Follow the following steps to clone the code and setup the necessary python libraries:

```bash
git clone https://github.com/gaspardol/MonteCarloPredictiveCoding.git
cd MonteCarloPredictiveCoding
pip install -r requirements.txt
```

To generate the figures of the paper please run

```bash
python figure_2.py
python figure_3.py
python figure_4.py
python figure_5.py
python figure_6.py
python table_1.py
```

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
