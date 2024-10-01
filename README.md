# Single-shot reconstruction of three-dimensional morphology of biological cells in digital holographic microscopy using a physics-driven neural network
Jihwan Kim, Youngdo Kim, Hyo Seung Lee, Eunseok Seo & Sang Joon Lee. Single-shot reconstruction of three-dimensional morphology of biological cells in digital holographic microscopy using a physics-driven neural network. arXiv preprint [arXiv:2409.20013](https://arxiv.org/abs/2409.20013) (2024).

## Development environment
MorpHoloNet was trained using Python 3.6.7, Anaconda3-4.5.11, PyCharm (JetBrains, Czech Republic), TensorFlow-gpu 2.4.1, NVIDIA CUDA toolkit 11.0, and cuDNN 8.2.1. A desktop computer used in this study is composed of Nvidia GeForce RTX 3090 GPU, AMD Ryzen 5950X CPU, and 128 GB RAM. Python packages required to reproduce the results are listed in `requirements.txt`. Typical time required to install this development environment is about 30 minutes.

## Contents
In this repository, we provide source codes for MorpHoloNet, typical holograms and the corresponding trained models.

1. `/MorpHoloNet.py`: Main code for training 3D morphology of biological cells using physics-driven and coordinate-based neural networks.
2. `/positional_encoding.py`: Fourier feature projection for positional encoding, referring to [this work](https://github.com/titu1994/tf_fourier_features/blob/master/tf_fourier_features/fourier_features.py).
3. `/Results.py`: Code for obtaining object arrays and intensity maps reconstructed at different depths.
4. `/requirements.txt`: Python packages required to run the codes.
5. `/holograms`: Typical holograms for training MorpHoloNet.
6. `/holograms/config.txt`: Various parameters of each hologram for training MorpHoloNet.
7. `/trained_model`: Trained model of each hologram. Copy `save_weights` folder and paste it into the same directory as `Results.py` to check trained the object arrays and intensity maps of each hologram.

## Citation
We welcome improvements to the concept of MorpHoloNet and its broad application in various research fields. Please cite our [preprint](https://arxiv.org/abs/2409.20013) when using this code:

Jihwan Kim, Youngdo Kim, Hyo Seung Lee, Eunseok Seo & Sang Joon Lee. Single-shot reconstruction of three-dimensional morphology of biological cells in digital holographic microscopy using a physics-driven neural network. arXiv preprint [arXiv:2409.20013](https://arxiv.org/abs/2409.20013) (2024).

## Contact
Jihwan Kim: jhkim15@postech.ac.kr   
Sang Joon Lee: sjlee@postech.ac.kr
