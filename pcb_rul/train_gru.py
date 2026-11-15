"""Train five deterministic two-head GRUs on synthetic trajectories."""
import json

import numpy as np
import torch
from torch import nn

if __package__:
    from .config import MEMBER_SEEDS, OBSERVED, OUTPUTS, seed_everything
    from .features import load_split
else:
    from config import MEMBER_SEEDS, OBSERVED, OUTPUTS, seed_everything
    from features import load_split

SEQUENCE = 100
BATCH = 256
EPOCHS = 40


def training_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Histories:
    def __init__(self, data, stats, device="cpu"):
        observed = (data["observed"] - stats["sequence_mean"]) / stats["sequence_scale"]
        self.values = torch.from_numpy(np.column_stack((observed, data["step"] / 1200)).astype(np.float32)).to(device)
        self.steps = torch.from_numpy(data["step"].astype(np.int64)).to(device)
        self.y = torch.from_numpy(data["y"]).to(device)
        self.auxiliary = torch.from_numpy(data["auxiliary"]).to(device)
        self.offsets = torch.arange(SEQUENCE, device=device)

    def batch(self, indices):
        lengths = (self.steps[indices] + 1).clamp(max=SEQUENCE)
        offsets = self.offsets if self.values.is_cuda else self.offsets[:int(lengths.max())]
        mask = offsets[None, :] < lengths[:, None]
        rows = indices[:, None] - lengths[:, None] + 1 + offsets[None, :]
        sequence = self.values[torch.where(mask, rows, 0)]
        return sequence.masked_fill(~mask[:, :, None], 0), mask


class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(len(OBSERVED) + 1, 96, num_layers=2, dropout=0.1, batch_first=True)
        self.rul = nn.Linear(96, 1)
        self.damage = nn.Linear(96, 1)

    def forward(self, sequence, mask):
        states, _ = self.gru(sequence)
        # Right padding cannot influence the last valid recurrent state.
        last = states[torch.arange(len(sequence), device=sequence.device), mask.sum(dim=1) - 1]
        return self.rul(last).squeeze(1), self.damage(last).squeeze(1)


@torch.inference_mode()
def predict(model, history):
    model.eval()
    values = []
    for indices in torch.arange(len(history.steps), device=history.steps.device).split(BATCH):
        rul, _ = model(*history.batch(indices))
        values.append(rul)
    return torch.cat(values).cpu().numpy()


def load_model(index, device="cpu"):
    model = GRU()
    model.load_state_dict(torch.load(OUTPUTS / f"gru_{index}.pt", map_location="cpu", weights_only=True))
    return model.to(device)


def fit(index, seed, train, val):
    seed_everything(seed)
    model = GRU().to(train.steps.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    huber = nn.HuberLoss()
    selected = torch.where(train.steps.remainder(5) == 0)[0]
    validation_targets = val.y.cpu().numpy()
    best, best_epoch, best_predictions = float("inf"), 0, None
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = selected[torch.randperm(len(selected)).to(selected.device)]
        for indices in order.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            rul, damage = model(*train.batch(indices))
            loss = huber(rul, train.y[indices]) + 0.5 * huber(damage, train.auxiliary[indices])
            loss.backward()
            optimizer.step()
        schedule.step()
        predictions = predict(model, val)
        mae = float(np.abs(predictions - validation_targets).mean())
        history.append(mae)
        print(f"GRU {index + 1}/5, seed {seed}, epoch {epoch}: validation MAE {mae:.6f}", flush=True)
        if mae < best:
            best, best_epoch, best_predictions = mae, epoch, predictions
            torch.save(model.state_dict(), OUTPUTS / f"gru_{index}.pt")
        if epoch - best_epoch >= 8:
            break
    return best_predictions, {"seed": seed, "best_epoch": best_epoch, "validation_mae": history}


def main():
    torch.set_num_threads(5)
    seed_everything()
    device = training_device()
    print(f"GRU device: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}", flush=True)
    with np.load(OUTPUTS / "scaling.npz") as stats:
        train, val = (Histories(load_split(split), stats, device) for split in ("train", "val"))
    predictions, logs = [], []
    for i, seed in enumerate(MEMBER_SEEDS):
        values, history = fit(i, seed, train, val)
        predictions.append(values)
        logs.append(history)
    np.save(OUTPUTS / "gru_val.npy", np.stack(predictions))
    (OUTPUTS / "gru_training.json").write_text(json.dumps(logs, indent=2) + "\n")


if __name__ == "__main__":
    main()
