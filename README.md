# Bidirectional Local-to-Global Multiscale Matching for Audio-Text Retrieval

Fall 2026 * ECE 532

This repository implements a symmetric **Bidirectional Local-to-Global Multiscale Matching (Bi-LGMM)** architecture driven by **Cross-Modal Similarity Consistency (CMSC)** for Audio-Text Retrieval (ATR). It provides pre-cached feature replay, a CPU-only execution path, fail-closed computational integrity gates, schema-validated JSON execution receipts, and automated CSV benchmark analysis. No GPU, external runtime download, or paid API is required during training or evaluation.

---

## Problem and Decision Boundary

Audio-Text Retrieval (ATR) comprises two complementary tasks: text-based audio retrieval (T2A) and audio-based text retrieval (A2T)[cite: 6]. Existing frameworks predominantly compress entire audio clips and text captions into single global vector embeddings, which causes feature collapse and discards localized details such as momentary sound events (e.g., a single car honk vs. three honks) or acoustic modifiers[cite: 1, 6].

While the Local-to-Global Multiscale Matching (LGMM) framework introduced by Wang et al. [1] addresses coarse-grained aggregation via frame-to-token cross-attention and Cross-Modal Similarity Consistency (CMSC), it evaluates similarity asymmetrically in only one direction ($A \rightarrow T$)[cite: 1, 6]. Because audio sequences exhibit high temporal redundancy compared to concise captions, unidirectional querying introduces structural modality bias[cite: 6].

This project investigates whether a symmetric, bidirectional multiscale matching architecture combining parallel $A \rightarrow T$ and $T \rightarrow A$ late-interaction streams—fused into an invariant global similarity score $\bar{S}_g(A, T) = \frac{1}{2}[S_g(A \rightarrow T) + S_g(T \rightarrow A)]$—achieves superior R@1, R@5, and R@10 recall over both asymmetric LGMM [1] and Multi-Modal Transformer (MMT) [7] baselines on AudioCaps[cite: 6, 7].

---

## Modalities, Data Access, Typed Claims, Risks, and Obligations

Audio and text data from the AudioCaps benchmark [2] are accessed via GitHub [5] and Hugging Face (`d0rj/audiocaps`) [6][cite: 7]. Ten-second audio clips are converted into log Mel-spectrograms and encoded into local temporal frame sequences $F_a \in \mathbb{R}^{l_a \times 768}$ via HT-SAT [8], while captions truncated to 30 tokens are encoded into local word representations $F_t \in \mathbb{R}^{l_t \times 768}$ via BERT [9] before projection into a 512-dimensional shared semantic space[cite: 2, 6, 7].

| Contract Element | Specification |
| :--- | :--- |
| **Geometric Claim (`CLM-GEOM-01`)** | All projected frame and word vectors satisfy $L_2$-normalization ($\Vert{}z\Vert{}_2 = 1.0 \pm 10^{-5}$), bounding local and global cosine similarities strictly within $[-1, 1]$. |
| **Fusion & Retrieval Claim (`CLM-FUSION-02`)** | Symmetric bidirectional fusion $\bar{S}_g(A, T)$ trained with CMSC elevates R@1, R@5, and R@10 across both A2T and T2A relative to asymmetric LGMM and MMT baselines (refuted if recall degrades across the board). |
| **Identified Risks** | Implementation complexity across three architectures within a five-week schedule and potential CPU RAM bottlenecks during dual cross-attention matrix computation. |
| **Reproducibility Obligations** | Release reproducible scripts, fixed `8,000 / 1,000 / 1,000` train/val/test splits, pre-extracted `.npy` embedding caches, schema-validated JSON receipts, and quantitative ablation CSVs[cite: 7]. |

---

## CPU Baseline, Runtime, and Memory Budget

All baseline and experimental evaluations are executed on a four-core CPU with 16 GB of RAM within a total runtime budget of two CPU hours[cite: 7]. To remain strictly within these hardware boundaries, features are extracted for 10,000 AudioCaps clips partitioned into 8,000 training, 1,000 validation, and 1,000 test samples[cite: 7]. Pre-extracted HT-SAT and BERT sequences are cached as memory-mapped NumPy arrays (`audio_embeddings.npy` and `text_embeddings.npy`) via `prepare_audiocaps_data.py` and replayed directly during 30-epoch training (`batch_size=128`, Adam `lr=0.001`, $\tau_w=0.25$, $\lambda=10$, $\tau=0.07$, $\beta=0.3$) without repeated transformer forward passes[cite: 3, 7]. During each run, `src/telemetry.py` records wall-clock execution time, peak CPU Resident Set Size (RSS), and Python heap usage into individual JSON receipts under `experiments/`.

---

## Checker and Action-Gate Plan with Failure Injection

Computational integrity enforcement is implemented in `src/integrity.py`, validated by `src/verification.py`, and tested across `tests/` through five deterministic verification stages:

1. **Normalization and Zero-Vector Gates (`L2NormalizationGate`, `ZeroVectorGate`):** Verifies that all frame/token embeddings and context-aware vectors satisfy $\Vert{}z\Vert{}_2 = 1.0 \pm 10^{-5}$ and contain zero null vectors prior to similarity computation and LogSumExp pooling[cite: 7].
2. **Cosine Bounds Gate (`CosineBoundsGate`):** Enforces $S(f, v) \in [-1, 1]$ across both $A \rightarrow T$ and $T \rightarrow A$ interaction streams[cite: 7].
3. **Attention Distribution Gate (`AttentionDistributionGate`):** Confirms softmax attention weights $w_{ij}$ and $w'_{ji}$ sum to $1.0$ across their respective context dimensions[cite: 7].
4. **Finite Value and Soft-Target Gates (`FiniteValueGate`, `SoftTargetScaleGate`):** Ensures global similarities $S_g(A, T)$, soft target distributions $\hat{S}_g$, and CMSC losses contain zero `NaN`/`Inf` values and enforces $\beta \in [0, 1]$[cite: 7].
5. **Fail-Closed Contract Enforcement:** Any injected violation immediately aborts execution with an `IntegrityViolationError` report, and `python -m src.verification --strict <receipt.json>` rejects any receipt admitting `outcome="act"` if a required checker failed, was unavailable, or exceeded the 16 GB RAM budget.

---

## Quickstart Commands

```bash
# 1. Run the full verification & robustness test suite
pytest tests/ -v

# 2. Run the synthetic smoke test and validate its JSON receipt
python train.py --config configs/smoke_test.yaml --model bi_lgmm
python -m src.verification --strict experiments/bi_lgmm_bi_lgmm_smoke_test_*.json

# 3. Extract and cache the 10,000-clip AudioCaps dataset (8k / 1k / 1k)
python prepare_audiocaps_data.py --output_dir ./data/audiocaps_cached --train_size 8000 --val_size 1000 --test_size 1000

# 4. Run full CPU training with automatic OOM batch-size fallback
python run_with_fallback.py --include_mmt

# 5. Export CSV tables to analysis/
python eval_models_csv.py --config configs/cpu_train.yaml
python eval_fine_grained_modes.py --config configs/cpu_train.yaml
python eval_loss_ablation.py --config configs/cpu_train.yaml
```

---

## Metrics, Baselines, Track Ownership, Timeline, and Fallback

Retrieval performance is evaluated using Recall@K ($K \in \{1, 5, 10\}$) for both Text-to-Audio (T2A) and Audio-to-Text (A2T) tasks against two baselines: Asymmetric LGMM [1] and Multi-Modal Transformer (MMT) [7][cite: 7]. This track is owned and executed by Logan Pasternak (Electrical and Computer Engineering Department, Rutgers University)[cite: 6].

| Milestone Window | Deliverable |
| :--- | :--- |
| **Week of November 2** | Finalize proposal submission. |
| **Week of November 9** | Complete AudioCaps dataset preparation/caching and initial Bi-LGMM setup. |
| **Week of November 16** | Conclude Bi-LGMM development and unit/integrity gate testing. |
| **Week of November 23** | Implement and benchmark Asymmetric LGMM and MMT baselines. |
| **Week of November 30** | Submit Progress Gate with validated receipts and `analysis/` CSVs[cite: 7]. |

If implementing all three models threatens the November 30 deadline, the MMT baseline will be removed in favor of the primary Asymmetric LGMM versus Bidirectional LGMM comparison[cite: 7]. If CPU memory limits are reached during joint bidirectional matching, `run_with_fallback.py` automatically halves `batch_size` (`128 -> 64 -> 32 -> 16`) to preserve numerical precision without out-of-memory errors[cite: 7]. Conversely, if execution proceeds ahead of schedule, optional scale-up experiments include evaluating cross-dataset generalization on Clotho [10] or integrating fine-grained sound event detection representations[cite: 7].

---

## Bibliography

[1] Q. Wang, J.-C. Gu, and Z.-H. Ling, "Multiscale Matching Driven by Cross-Modal Similarity Consistency for Audio-Text Retrieval," in *Proc. IEEE ICASSP*, 2024, pp. 11581–11585.  
[2] C. D. Kim, B. Kim, H. Lee, and G. Kim, "AudioCaps: Generating Captions for Audios in The Wild," in *Proc. NAACL-HLT*, 2019, pp. 119–132.  
[3] D. Kim, N. Kim, and S. Kwak, "Improving Cross-Modal Retrieval with Set of Diverse Embeddings," in *Proc. IEEE/CVF CVPR*, 2023, pp. 23422–23431.  
[4] C. Wang et al., "When Audio and Text Disagree: Revealing Text Bias in Large Audio-Language Models," in *Proc. EMNLP*, 2025, pp. 4878–4888.  
[5] C. D. Kim, `cdjkim/audiocaps`, GitHub: https://github.com/cdjkim/audiocaps.  
[6] D. Balobin, `d0rj/audiocaps`, Hugging Face: https://huggingface.co/datasets/d0rj/audiocaps.  
[7] A.-M. Oncescu, A. S. Koepke, J. F. Henriques, Z. Akata, and S. Albanie, "Audio Retrieval with Natural Language Queries: A Benchmark Study," *IEEE Trans. Multimedia*, vol. 25, pp. 2675–2685, 2022.  
[8] K. Chen et al., "HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection," in *Proc. IEEE ICASSP*, 2022, pp. 646–650.  
[9] J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.  
[10] K. Drossos, S. Lipping, and T. Virtanen, "Clotho: An Audio Captioning Dataset," in *Proc. IEEE ICASSP*, 2020, pp. 736–740.