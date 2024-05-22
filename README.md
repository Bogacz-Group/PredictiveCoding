# Predictive coding in Pytorch

## Overview
This repository houses the implementation of [predictive coding](https://www.sciencedirect.com/science/article/pii/S0022249615000759) used in the [Bogacz Group](https://www.mrcbndu.ox.ac.uk/groups/bogacz-group). It offers an easy and opensource access to implementing a range of predictive coding networks.

In this repository you can also find tutorials that can be run on google colab for:
- [predictive coding models for supervised learning](https://pubmed.ncbi.nlm.nih.gov/28333583/)
- [Monte Carlo predictive coding models](https://www.biorxiv.org/content/10.1101/2024.02.29.581455v1.full)
- [recurrent predictive coding models](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010719#:~:text=The%20recurrent%2C%20single%2Dlayer%20network,instead%20of%20the%20signal%20itself.)
- [temporal predictive coding models](https://arxiv.org/abs/2305.11982)



## Structure of the repository
The repository includes:
- the predictive coding library under `predictive_coding`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bogacz-Group/PredictiveCoding/blob/main/1_supervised_learning_pc.ipynb)`1_supervised_learning_pc.ipynb`, a tutorial on how to train a predictive coding model to perform classification on MNIST 
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bogacz-Group/PredictiveCoding/blob/main/2_unsupervised_learning_mcpc.ipynb)`2_unsupervised_learning_mcpc.ipynb`, a tutorial on how to train a Monte Carlo predictive coding model on Gaussian data
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bogacz-Group/PredictiveCoding/blob/main/3_memory_rpc.ipynb)
`3_memory_rpc.ipynb`, a tutorial on how to train a recurrent predictive coding model on a memory task
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bogacz-Group/PredictiveCoding/blob/main/4_sequential_memory_tpc.ipynb)`4_sequential_memory_tpc.ipynb`, a tutorial on how to train a temporal predictive coding model on a sequential memory task.


## Usage
Follow the next steps to clone the code and setup the necessary python libraries:

```bash
git clone https://github.com/Bogacz-Group/PredictiveCoding.git
cd PredictiveCoding
pip install -r requirements.txt
```

## Citation
For those who find our work useful, here is how you can cite it:

```bibtex
@article {SomePaper,
	author = {Authors},
	title = {Title},
	year = {2024},
	doi = {},
	publisher = {Publisher},
	URL = {},
	eprint = {},
	journal = {}
}

```

## Contact
For any inquiries or questions regarding the project, please feel free to contact us at gaspard.oliviers@pmb.ox.ac.uk or mufeng.tang@ndcn.ox.ac.uk.

## Code Aknowledgements
This repository builds upon the following repositories/codes:
- https://github.com/YuhangSong/Prospective-Configuration
