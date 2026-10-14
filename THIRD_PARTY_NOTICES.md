# Third-party source notices

This repository preserves upstream source and license notices. It does not claim
ownership of the original Restormer, SwinIR, EDSR, RCAN, or RRDB architectures.

- **Restormer / its BasicSR fork**: Syed Waqas Zamir and contributors,
  upstream `swz30/Restormer`. The bundled MIT license is retained under each
  source tree and in `LICENSES/Restormer-LICENSE.md`.
- **SwinIR**: Jingyun Liang and contributors, upstream
  `JingyunLiang/SwinIR`. The adapted `swinir_holo_arch.py` retains source header
  attribution. The upstream license is in `LICENSES/SwinIR-LICENSE.txt`.
- **BasicSR baseline architecture adapters**: upstream `XPixelGroup/BasicSR`.
  EDSR, RCAN, RRDBNet and MSRResNet files were adapted to this fork's
  `basicsr.models.archs` import layout. A minimal registry shim supports the
  existing dynamic class factory. The upstream license is in
  `LICENSES/BasicSR-LICENSE.txt`.

The baseline configuration determines the exact width, depth, reconstruction
head, normalization and training objective. Names alone do not imply the largest
upstream configuration or the complete training method of an original paper.

The small visual reference sample is included at the project owner's request;
the complete experimental dataset and trained weights are not distributed here.
