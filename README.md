# Multimodal Project

Fall 2026 * ECE 532

This package contains bounded cached features and the CPU-only experiment path. No GPU, external download, or paid API is required. 

## Description

Inspired by [1], our goal is to train separate audio and video encoders that map both inputs into a single, shared feature space where cross-modal similarity can be evaluated directly. In this shared space, the objectives are to:
•	Maximize feature similarity between matching audio-video pairs from the same clip while minimizing similarity for non-matching pairs using contrastive loss functions.
•	Perform bidirectional similarity search using nearest-neighbor lookup to find the corresponding video given an audio query or retrieve the matching audio track given a video query.

## Intended Modalities:
The two modalities that will be utilized in this proposal will be video and audio data. The VGG Sound dataset [2] will be utilized since it is a well-documented audio-visual dataset that has been used for both training and evaluating multi-modal models [1][3]. Raw audio data from 10-second-long clips will be transformed into mel-spectrograms which will then be passed through a neural network (e.g. RESNET) to learn representative audio embeddings. Video will be processed frame-by-frame via a spatio-temporal CNN (e.g. VGG or RESNET) to extract visual features. 

## Checkers:
Before computing contrastive losses (e.g., InfoNCE) or nearest-neighbor queries, an automated assertion layer will verify that both audio and visual embeddings match in target vector dimension (d) and pass L_2-normalization (||z||_2=1). This ensures that spatial inner products accurately reflect cosine similarity.

## CPU/Replay Fallback: 
To keep within predefined hardware and resource boundaries, features will be extracted via an encoder for only 10,000 clips. Of these clips, 8000 will be utilized for training, 1000 will be used for validation, and 1000 for testing. After extracting feature embeddings, they will be cached as lightweight Numpy arrays. During training, cached embeddings can be rapidly replayed from system memory or disk on standard CPU infrastructure without requiring repeated forward passes through heavy convolutional backbones.

## Bibliography
[1] Girdhar, Rohit, et al. "ImageBind one embedding space to bind them all." 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023.
[2] Chen, Honglie, et al. "Vggsound: A large-scale audio-visual dataset." ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.
[3] Du, Chenzhuang, et al. "On uni-modal feature learning in supervised multi-modal learning." International Conference on Machine Learning. PMLR, 2023.

# Multimodal-Project
