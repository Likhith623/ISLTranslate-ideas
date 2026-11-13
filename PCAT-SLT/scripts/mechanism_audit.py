"""Numerical mechanism evidence, independent of translation or random training success."""
import argparse
import json
from pathlib import Path
import torch
from pcat_slt.operators import ProgressCoupledAreaTransport, cayley


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="mechanism_audit.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    layer = ProgressCoupledAreaTransport(4, 5, 12).double().eval()
    z = torch.randn(2, 7, 4, dtype=torch.float64)
    fine = [z[:, 0]]
    for i in range(6):
        fine.extend([z[:, i] + a * (z[:, i + 1] - z[:, i]) for a in (.17, .49, .85, 1.)])
    fine = torch.stack(fine, 1)
    with torch.no_grad():
        coarse, details = layer(z, torch.tensor([7, 7]), return_details=True)
        refined = layer(fine, torch.tensor([25, 25]))
        q = details["transport"]
        eye = torch.eye(4, dtype=torch.float64)
        # Held-out amplitude/speed order pairs; label means only event order.
        count = 200
        amplitude = .2 + 2 * torch.rand(count, 2, dtype=torch.float64)
        paths = torch.zeros(count, 3, 2, dtype=torch.float64)
        paths[:, 2] = amplitude
        label = torch.arange(count) % 2
        paths[label == 0, 1, 0] = amplitude[label == 0, 0]
        paths[label == 1, 1, 1] = amplitude[label == 1, 1]
        ts = torch.stack([torch.zeros(count), .01 + torch.rand(count), 2 + 10 * torch.rand(count)], 1).double()
        order = ProgressCoupledAreaTransport(2, 1, 8).double()
        stats = order.statistics(paths, torch.full((count,), 3), ts)
        prediction = (stats.area[:, 0, 0, 1] < 0).long()
        # Separate path increments cannot discriminate paired order-swapped examples.
        paired = torch.tensor([[[0., 0.], [1., 0.], [1., 1.]],
                               [[0., 0.], [0., 1.], [1., 1.]]], dtype=torch.float64)
        control = ProgressCoupledAreaTransport(2, 1, 8, mode="first_order").double()
        _, first = control(paired, torch.tensor([3, 3]), return_details=True)
        # The numerical condition bound is a statement about I-A, not the whole net.
        h = details["statistics"].area
        a = layer.raw_gain.tanh() * h / (1 + torch.linalg.matrix_norm(h, keepdim=True))
        result = {"dtype": "float64", "seed": 17,
                  "subdivision_max_abs_token_error": float((coarse - refined).abs().max()),
                  "orthogonality_max_abs_error": float((q.transpose(-1, -2) @ q - eye).abs().max()),
                  "max_solve_condition_number": float(torch.linalg.cond(eye - a).max()),
                  "reversal_max_abs_error": float((cayley(-h, layer.raw_gain.tanh()) - q.transpose(-1, -2)).abs().max()),
                  "nonlinguistic_order_accuracy": float(prediction.eq(label).double().mean()),
                  "nonlinguistic_order_samples": count,
                  "first_order_paired_feature_distance": float((first["features"][0] - first["features"][1]).norm()),
                  "interpretation": "Constructed event-order task and algebraic checks; no natural-language benchmark evidence."}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
