# XMP: A CROSS-ATTENTION MULTI-SCALE PERFORMER FOR FILE FRAGMENT CLASSIFICATION

Official Pytorch implementation of [XMP: A CROSS-ATTENTION MULTI-SCALE PERFORMER FOR FILE FRAGMENT CLASSIFICATION](https://ieeexplore.ieee.org/abstract/document/10447626?casa_token=fJRE5tpvjKwAAAAA:66uZKJffkVsswFwzmB6UEA0GSd4cWcFXlzHRT5GQ_iUnZYB8hlrPNWUzwUqUEEw17Js-queQoTo) accepted in ICASSP 2024.

Authors: Jeong Gyu Park, Sisung Liu, Je Hyeong Hong

## Abstract
File fragment classification (FFC) is the task of identifying the file type given a small fraction of binary data, and serves a crucial role in digital forensics and cybersecurity. Recent studies have adopted convolutional neural networks (CNNs) for this problem, significantly improving the accuracy over the traditional methods relying on handcrafted features. In this paper, we aim to expand on the recent performance gain by better leveraging the large amount of digital files available for training. We propose to achieve this by employing a Transformer encoder-based network known for its weak inductive bias suited for large-scale training. Our model, XMP, is inspired by the CrossViT architecture for image recognition and utilizes multi-scale self and cross-attentions between CNN features extracted from the byte n-grams of input binary data. Experimental results on the latest public dataset show XMP achieving state-of-the-art accuracies in almost all scenarios without need for additional preprocessing of binary data such as bit shifting, demonstrating the effectiveness of the Transformer-based architecture for FFC. The benefit of each proposed component is assessed through ablation study.

![image](https://github.com/Jaden-kor/xmp/assets/116866770/0d0b8b98-381d-4916-a96d-543765e577e9)

## Contribution
- the **first proper Transformer encoder-based architecture** for file fragment classification with aims to reduce inductive bias and benefit from abundant training data,

- integration of CNN-based feature extraction for byte $n$-gram value embeddings, multi-scale features and Performer approximation of the vanilla Transformer to yield XMP and set new benchmarks on almost all FFT-75 scenarios, and

- ablation study of components proposed as part of XMP demonstrating the usefulness of each module.

## Prerequisites:
````
python=3.8
torch==1.10.0+cu111
torchvision==0.11.0+cu111
torchaudio==0.10.0
pytorch-fast-transformers
performer-pytorch
vit-pytorch
pykernel
wandb
tqdm
````
This code has been tested with Ubuntu 20.04, A100 GPUs with CUDA 12.2, Python 3.8, Pytorch 1.10.

Earlier versions may also work~ :)

## 🏃 How to run our code!
To run our code, use the following command template, adjusting the hyperparameters as needed based on the scenario and dataset size. We have conducted extensive hyperparameter tuning for different scenarios and dataset sizes to ensure optimal performance. The details of these configurations are documented in the comments within the code.

Example command:
````
python train.py -scen 1 -size 512 -name XMP_scen1_512 -gpu 0 -emb_dim 128 -depth 3 -sm_cnn_size 11 -sm_dim 256 -sm_token_size 4 -lg_cnn_size 11 -lg_dim 512 -lg_token_size 16
````



## Citation
````
@inproceedings{park2024xmp,
  title={XMP: A Cross-Attention Multi-Scale Performer for File Fragment Classification},
  author={Park, Jeong Gyu and Liu, Sisung and Hong, Je Hyeong},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4505--4509},
  year={2024},
  organization={IEEE}
}
````
## License
A patent application for XMP has been submitted and is under review for registration. XMP is licensed under the CC-BY-NC-SA-4.0 license limiting any commercial use.

## Acknowledgement
This work was supported by the Korea Research Institute for defense Technology planning and advancement (KRIT) grant funded by the Korea government (DAPA (Defense Acquisition Program Administration)) (No. KRIT-CT-22-021, Space Signal Intelligence Research Laboratory, 2022).
