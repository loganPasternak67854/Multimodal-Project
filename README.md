# Multimodal Project

Fall 2026 * ECE 532

This package contains bounded cached features and the CPU-only experiment path. No GPU, external download, or paid API is required. 

## Problem and decision boundary:
Existing Audio-Text Retrieval frameworks primarily rely on coarse-grained single-vector global representations. These single-vector methods miss local temporal/word details and fail when descriptions in both modalities are similar yet conceptually inconsistent (e.g., distinguishing "car honks three times" from a single honk). Additionally, standard binary contrastive loss ignores fine-grained semantic differences and intra-modal similarity relationships. In this proposal, we build and evaluate the multimodal sensing system discussed in [1] that addresses this issue through a novel local-to-global multiscale matching (LGMM) architecture. 

## Modalities, Data Access, Typed Claims, Risks, and Obligations:
The two modalities that will be utilized in this proposal will be text and audio data. AudioCaps [2] will be utilized since it is a well-documented audio-text dataset that has been used for both training and evaluating multi-modal models [3][4]. Audiocaps is fully accessible via resources such as GitHub [5] or Hugging Face [6]. 
Audio embeddings will be acquired via a Hierarchical Token Semantic Audio Transformer (HT-SAT) encoder. These features will be learned through taking 10-second-long clips, converting them into log mel-spectrograms, and then passing these spectrograms through a HT-SAT to learn representative audio embeddings. Text based features will be acquired by taking input captions and then passing them through a Bidirectional Encoder Representations from Transformers (BERT) model. 
Our claim is that the audio-text embedding space trained using the method described in [1] will achieve at least a Recall@5 (R@5) of 0.45 on cross-modal retrieval (Audio-to-Text and Text-to-Audio). Potential risks include class-imbalance leading to skewed embedding spaces and severe memory bottlenecks during off-chip caching if batch sizes become too large. Our obligations for this work will be to ensure reproducibility by releasing code, fixed train/val/test splits, and pre-extracted feature arrays via our GitHub repository; and report clear quantitative ablation results isolating the impact of textual vs. auditory features.
## CPU baseline with estimated runtime and memory:
Our baseline analysis will be conducted utilizing a four-core CPU and 16 GB of RAM. Evaluation will take no more than two CPU hours. To keep within predefined hardware and resource boundaries, features will be extracted via an encoder for only 10,000 clips. Of these clips, 8000 will be utilized for training, 1000 will be used for validation, and 1000 for testing. After extracting feature embeddings, they will be cached as lightweight Numpy arrays. During training, cached embeddings can be rapidly replayed from system memory or disk on standard CPU infrastructure without requiring repeated forward passes through heavy convolutional backbones.
## Checker and action-gate plan with failure injection:
Normalization and cosine bound checkers will verify that frame-sentence similarity input vectors (f_(a,i) and v_(a,i)^t) strictly comply with L_2-normalization standards (‖z‖_2=1.0±10^(-5)), ensuring that S(f_(a,i),v_(a,i)^t )∈[-1,1]. Synthetic injection of unnormalized vectors or zero vectors will prompt the action-gate to halt training with an explicit normalization error prior to LogSumExp pooling.
A distribution and LogSumExp stability checker will confirm that w_ij rows sum to 1.0 across the text length dimension and verifies that global similarity scores S_g (A,T) and soft label distributions S ˆ_g are free of NaN or Inf values before computing losses. Injecting NaN or Inf values, out-of-bounds similarity scores, or invalid soft-target scales (β∉[0,1]) immediately will cause the action-gate to abort training and log a detailed exception report.
## Metrics, baselines, track ownership, timeline, and fallback:
Our evaluation metric will be the proportion of queries where the true matching item ranks within the top-K nearest neighbors (Recall). We shall evaluate the Recall given K = 1, 5, and 10. We shall compare these results against those of a baseline audio-text retrieval model, a Multi-modal Transformer (MMT) [7] model. This project will be conducted my Logan Pasternak. The timeline for the proposed work will be as follows: 
	Week of November 2: Prepare dataset and begin setup of proposed model.
	Week of November 9: Continue work on proposed model and fully flesh out unit testing.
	Week of November 16: Finish work on model and move on to implementation of baseline model.
	Week of November 23: Implement baseline model.
	Week of November 30: Submit Progress Gate.
If the pace of the project does not flow in a smooth fashion, then as a fallback I will transition from joint training with Cross-Modal Similarity Consistency (L_"Jnt"  and L_"IntraC" ) to training solely with standard inter-modal contrastive loss (L_"InterC" ) over the LGMM architecture or replace LGMM with a simplified Max-Mean pooling mechanism to ensure baseline completion. If the project moves forward without difficulty on the other hand, then optional scale-up plans may include evaluating cross-dataset generalization on the Clotho benchmark dataset or integrating fine-grained audio event representations derived from sound event detection models.

## Bibliography:
[1] Q. Wang, J. -C. Gu and Z. -H. Ling, "Multiscale Matching Driven by Cross-Modal Similarity Consistency for Audio-Text Retrieval," ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Seoul, Korea, Republic of, 2024, pp. 11581-11585, doi: 10.1109/ICASSP48485.2024.10446302.
[2] Kim, Chris Dongjoo, et al. “AudioCaps: Generating Captions for Audios in The Wild.” Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), edited by Jill Burstein et al., Association for Computational Linguistics, 2019, pp. 119–32, https://doi.org/10.18653/v1/N19-1011.
[3] Ruan, Ludan, et al. "Accommodating audio modality in clip for multimodal processing." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 8. 2023.
[4] Changin, Choi, Lim Sungjun, and Rhee Wonjong. "Enhancing retrieval-augmented audio captioning with generation-assisted multimodal querying and progressive learning." arXiv preprint arXiv:2410.10913 (2024).
[5] “Cdjkim/Audiocaps: Repository for Our NAACL-Hlt 2019 Paper: Audiocaps.” GitHub, github.com/cdjkim/audiocaps. Accessed 28 Oct. 2026. 
[6] Balobin, Dmitry. “D0RJ/Audiocaps · Datasets at Hugging Face.” D0rj/Audiocaps · Datasets at Hugging Face, huggingface.co/datasets/d0rj/audiocaps. Accessed 28 Oct. 2026. 
[7] Koepke, A. Sophia, et al. "Audio retrieval with natural language queries: A benchmark study." IEEE Transactions on Multimedia 25 (2022): 2675-2685.

