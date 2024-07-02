# SNK_revisited
Implementation of a revisited version of model from paper: "Shape Non-rigid Kinematics (SNK): A Zero-Shot Method for Non-Rigid Shape Matching via Unsupervised Functional Map Regularized Reconstruction" by Attaiki and Ovsjanikov (2024), as final project from Technion CS course "Geometric Computer Vision" (236861).
## Git repositories used for this project:
- SNK modules implemented before 30.06.2024 (loss.py and prism_decoder.py) - modified for experiments and compatibility: https://github.com/pvnieo/SNK.git
- SURFMNet for pytorch implementation of functional mapping: https://github.com/pvnieo/SURFMNet-pytorch.git
- DiffusionNet+ for learnable descriptors computation: https://github.com/pvnieo/diffusion-net-plus.git
- pyFM for non-differentiable functional mapping (based on HKS/WKS features descriptors): https://github.com/RobinMagnet/pyFM.git
## Installation
- Highly recommended to install requirements using a conda environment (as some packages, such as meshplot, can not be installed otherwise).
- I used Python 3.10.14, worked in Windows and implemented notebooks using Jupyter lab.
