import argparse
import logging
import os
import time
import yaml
import torch
from torch.utils.data import DataLoader

from src.integrity import ComputationalIntegrityChecker
from src.models import BidirectionalLGMM, AsymmetricLGMM, MMTBaseline
from src.losses import CMSCLoss
from src.dataset import CachedAudioCapsDataset, SyntheticATRDataset
from src.metrics import compute_retrieval_metrics
from src.telemetry import ExperimentTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("TrainPipeline")


def build_model(cfg: dict, checker: ComputationalIntegrityChecker):
    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    m_type = m_cfg["type"]

    if m_type == "bi_lgmm":
        return BidirectionalLGMM(
            audio_in_dim=d_cfg["audio_in_dim"],
            text_in_dim=d_cfg["text_in_dim"],
            shared_dim=d_cfg["shared_dim"],
            tau_w=m_cfg["tau_w"],
            lse_lambda=m_cfg["lse_lambda"],
            checker=checker,
        )
    elif m_type == "asym_lgmm":
        return AsymmetricLGMM(
            audio_in_dim=d_cfg["audio_in_dim"],
            text_in_dim=d_cfg["text_in_dim"],
            shared_dim=d_cfg["shared_dim"],
            tau_w=m_cfg["tau_w"],
            lse_lambda=m_cfg["lse_lambda"],
            checker=checker,
        )
    elif m_type == "mmt":
        return MMTBaseline(
            audio_in_dim=d_cfg["audio_in_dim"],
            text_in_dim=d_cfg["text_in_dim"],
            shared_dim=d_cfg["shared_dim"],
        )
    raise ValueError(f"Unknown model type: {m_type}")


def build_dataloaders(cfg: dict):
    d_cfg = cfg["data"]
    bs = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]

    if d_cfg["mode"] == "synthetic":
        train_ds = SyntheticATRDataset(
            d_cfg["train_size"], d_cfg["max_audio_frames"], d_cfg["max_text_tokens"], d_cfg["audio_in_dim"], seed=42
        )
        val_ds = SyntheticATRDataset(
            d_cfg["val_size"], d_cfg["max_audio_frames"], d_cfg["max_text_tokens"], d_cfg["audio_in_dim"], seed=43
        )
        test_ds = SyntheticATRDataset(
            d_cfg["test_size"], d_cfg["max_audio_frames"], d_cfg["max_text_tokens"], d_cfg["audio_in_dim"], seed=44
        )
    else:
        cache_dir = d_cfg["cache_dir"]
        train_ds = CachedAudioCapsDataset(os.path.join(cache_dir, "train"), d_cfg["train_size"])
        val_ds = CachedAudioCapsDataset(os.path.join(cache_dir, "val"), d_cfg["val_size"])
        test_ds = CachedAudioCapsDataset(os.path.join(cache_dir, "test"), d_cfg["test_size"])

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device, tracker: ExperimentTracker = None):
    model.eval()
    all_audio, all_text = [], []
    for audio_b, text_b in loader:
        all_audio.append(audio_b.to(device))
        all_text.append(text_b.to(device))

    audio_full = torch.cat(all_audio, dim=0)
    text_full = torch.cat(all_text, dim=0)

    N = audio_full.size(0)
    chunk_size = 128
    sim_rows = []
    if hasattr(model, "proj") and (hasattr(model, "compute_cross_modal_sim") or hasattr(model, "stream")):
        f_a, f_t = model.proj(audio_full, text_full)
        for i in range(0, N, chunk_size):
            a_chunk = f_a[i : i + chunk_size]
            if hasattr(model, "compute_cross_modal_sim"):
                row = model.compute_cross_modal_sim(a_chunk, f_t)
            else:
                row = model.stream(a_chunk, f_t)
            sim_rows.append(row.cpu())
            if tracker is not None:
                tracker.sample_memory()
        sim_matrix = torch.cat(sim_rows, dim=0)
    else:
        sim_matrix = model(audio_full, text_full)["S_AT"].cpu()
        if tracker is not None:
            tracker.sample_memory()

    return compute_retrieval_metrics(sim_matrix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke_test.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override model type: bi_lgmm, asym_lgmm, mmt")
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.model is not None:
        cfg["model"]["type"] = args.model

    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    tracker = ExperimentTracker(
        experiment_name=cfg["experiment_name"],
        model_type=cfg["model"]["type"],
        config=cfg,
        output_dir=args.experiments_dir,
    )
    tracker.start()

    checker = ComputationalIntegrityChecker(
        norm_tol=cfg["integrity"]["norm_tol"],
        strict=cfg["integrity"]["strict_mode"],
    )
    model = build_model(cfg, checker).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = CMSCLoss(tau=cfg["loss"]["tau"], beta=cfg["loss"]["beta"], checker=checker)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    save_dir = os.path.join(cfg["training"]["save_dir"], cfg["model"]["type"])
    os.makedirs(save_dir, exist_ok=True)
    best_r1 = -1.0

    logger.info(f"Starting training [{cfg['model']['type']}] ({num_params:,} params) on {device}...")
    train_start = time.perf_counter()

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0

        for audio_b, text_b in train_loader:
            audio_b, text_b = audio_b.to(device), text_b.to(device)
            optimizer.zero_grad()

            sim_dict = model(audio_b, text_b)
            losses = criterion(sim_dict)
            losses["total"].backward()
            optimizer.step()

            running_loss += losses["total"].item()
            tracker.sample_memory()

        avg_loss = running_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device, tracker=tracker)
        epoch_sec = time.perf_counter() - epoch_start
        tracker.log_epoch(epoch, avg_loss, val_metrics, epoch_sec)

        mean_r1 = 0.5 * (val_metrics["A2T_R@1"] + val_metrics["T2A_R@1"])
        logger.info(
            f"Epoch {epoch:02d}/{cfg['training']['epochs']:02d} ({epoch_sec:.2f}s) | "
            f"Loss: {avg_loss:.4f} | Val T2A R@1: {val_metrics['T2A_R@1']:.2f}% | "
            f"Val A2T R@1: {val_metrics['A2T_R@1']:.2f}%"
        )

        if mean_r1 > best_r1:
            best_r1 = mean_r1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

    tracker.train_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    test_metrics = evaluate(model, test_loader, device, tracker=tracker)
    tracker.eval_seconds = time.perf_counter() - eval_start

    receipt_path = tracker.stop_and_save(test_metrics=test_metrics, num_parameters=num_params)

    logger.info(f"JSON experiment receipt saved to: {receipt_path}")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.2f}%")

if __name__ == "__main__":
    main()