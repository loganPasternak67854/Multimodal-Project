import json
import os
import platform
import time
import tracemalloc
from datetime import datetime, timezone
import psutil
import torch


class ExperimentTracker:
    """
    Tracks wall-clock execution time (total, training, evaluation) and
    memory consumption (CPU RSS delta/peak, Python heap peak, GPU peak if applicable),
    writing a standalone JSON receipt into the 'experiments/' directory.
    """

    def __init__(self, experiment_name: str, model_type: str, config: dict, output_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.config = config
        self.output_dir = output_dir

        self.process = psutil.Process(os.getpid())
        self.start_rss_bytes = None
        self.peak_rss_bytes = 0
        self.t0 = None
        self.train_seconds = 0.0
        self.eval_seconds = 0.0
        self.epoch_history = []

    def start(self):
        tracemalloc.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.start_rss_bytes = self.process.memory_info().rss
        self.peak_rss_bytes = self.start_rss_bytes
        self.t0 = time.perf_counter()

    def sample_memory(self):
        current_rss = self.process.memory_info().rss
        if current_rss > self.peak_rss_bytes:
            self.peak_rss_bytes = current_rss

    def log_epoch(self, epoch: int, train_loss: float, val_metrics: dict, epoch_wall_sec: float):
        self.sample_memory()
        self.epoch_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_metrics": {k: round(v, 4) for k, v in val_metrics.items()},
            "epoch_wall_time_sec": round(epoch_wall_sec, 4),
            "rss_memory_mb": round(self.process.memory_info().rss / (1024 ** 2), 2),
        })

    def stop_and_save(self, test_metrics: dict, num_parameters: int) -> str:
        total_wall_sec = time.perf_counter() - self.t0
        self.sample_memory()

        _, py_heap_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        end_rss_bytes = self.process.memory_info().rss
        mb = 1024 ** 2
        gb = 1024 ** 3

        gpu_peak_mb = None
        if torch.cuda.is_available():
            gpu_peak_mb = round(torch.cuda.max_memory_allocated() / mb, 2)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"{self.model_type}_{self.experiment_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        receipt = {
            "receipt_metadata": {
                "timestamp_utc": timestamp,
                "experiment_name": self.experiment_name,
                "model_type": self.model_type,
                "num_trainable_parameters": num_parameters,
                "device": self.config.get("device", "cpu"),
                "hardware": {
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "logical_cpus": psutil.cpu_count(logical=True),
                    "total_system_ram_gb": round(psutil.virtual_memory().total / gb, 2),
                },
            },
            "wall_time": {
                "total_wall_time_sec": round(total_wall_sec, 4),
                "total_wall_time_min": round(total_wall_sec / 60.0, 4),
                "total_wall_time_hours": round(total_wall_sec / 3600.0, 6),
                "training_loop_sec": round(self.train_seconds, 4),
                "test_evaluation_sec": round(self.eval_seconds, 4),
                "avg_epoch_sec": round(
                    self.train_seconds / max(len(self.epoch_history), 1), 4
                ),
            },
            "memory_consumption": {
                "initial_rss_mb": round(self.start_rss_bytes / mb, 2),
                "final_rss_mb": round(end_rss_bytes / mb, 2),
                "peak_rss_mb": round(self.peak_rss_bytes / mb, 2),
                "peak_rss_gb": round(self.peak_rss_bytes / gb, 4),
                "net_rss_delta_mb": round((self.peak_rss_bytes - self.start_rss_bytes) / mb, 2),
                "python_heap_peak_mb": round(py_heap_peak_bytes / mb, 2),
                "gpu_peak_allocated_mb": gpu_peak_mb,
            },
            "test_retrieval_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
            "hyperparameters": {
                "data": self.config.get("data", {}),
                "model": self.config.get("model", {}),
                "loss": self.config.get("loss", {}),
                "training": self.config.get("training", {}),
            },
            "epoch_trace": self.epoch_history,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)

        return filepath