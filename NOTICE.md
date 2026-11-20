# Third-party notices

IS3D ships no weights inside the git repository. The installer downloads a
weight file that is a re-packed, inference-only snapshot of the following
projects. Redistributing or using it commercially means honouring their terms.

| Component | Role in the pipeline | Upstream |
| --- | --- | --- |
| Idol / Splat 3D generation stack (`denoiser`, `decoder`, `decoder_gs`) | image -> latent -> Gaussian anchors | speridlabs `idol` SDK |
| FLUX.2 klein VAE (`encoder`, `quant_conv`, `bn`) | image tokens for the denoiser | Black Forest Labs FLUX.2 klein 4B |
| DINOv3 ViT-H/16+ | conditioning features | Meta DINOv3 (via LibreYOLO conversion) |
| BiRefNet | background removal | ZhengPeng7 / speridlabs snapshot |

`validation/fixtures/fig-snowman.webp` is the demo image from the `idol` SDK and
is used by the checks in this repository.

The code in this repository is MIT licensed (see LICENSE).
