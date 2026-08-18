# Frozen ImageNet ResNet-18 representation smoke

Date: 2026-07-17 (Europe/Warsaw)

Status: **passed** as a two-sample software/weight integration check. This is not a PanNuke experiment and not a medical result.

## Executed checks

- The official torchvision `ResNet18_Weights.IMAGENET1K_V1` checkpoint was initially obtained only after an explicit download-enabled smoke invocation.
- A second durable invocation used the already cached checkpoint with implicit downloads disabled.
- Device: `cuda`; GPU: `NVIDIA GeForce RTX 4070`; AMP: enabled; inference mode: enabled.
- PyTorch: `2.12.1+cu126`; torchvision: `0.27.1+cu126`.
- Both `rgb` and `target_highlighted_rgb` produced finite `2 x 512` embeddings.
- No OOM batch-size backoff occurred.

## Weight provenance

- Official URL: `https://download.pytorch.org/models/resnet18-f37072fd.pth`
- Local checkpoint SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`

## Durable evidence

- `artifacts/embeddings/resnet18_smoke/resnet18_rgb_smoke.npz`
- `artifacts/embeddings/resnet18_smoke/resnet18_rgb_smoke.npz.metadata.json`
- `artifacts/embeddings/resnet18_smoke/resnet18_target_highlighted_rgb_smoke.npz`
- `artifacts/embeddings/resnet18_smoke/resnet18_target_highlighted_rgb_smoke.npz.metadata.json`

The metadata sidecars bind the input arrays, embedding arrays, cache files, preprocessing policy, package versions, checkpoint path, official weight URL, and checkpoint SHA-256.

