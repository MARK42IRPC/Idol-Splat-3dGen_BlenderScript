# Third-party notices

IS3D ships no weights inside the git repository. The installed weight file is a
re-packed, inference-only snapshot of the projects below. Using or redistributing
it means honouring their terms; the code in this repository is MIT (see LICENSE).

| Component | Role in the pipeline | Upstream |
| --- | --- | --- |
| Idol / Splat 3D generation stack (`denoiser`, `encoder`, `decoder`, `decoder_gs`) | image -> latent -> Gaussian anchors | speridlabs `idol` SDK -- <https://github.com/speridlabs/idol-sdk> |
| FLUX.2 klein VAE (`encoder`, `quant_conv`, `bn`) | image tokens for the denoiser | Black Forest Labs FLUX.2 klein 4B -- <https://github.com/black-forest-labs/flux2> |
| DINOv3 ViT-H/16+ | conditioning features | Meta DINOv3 -- <https://github.com/facebookresearch/dinov3> |
| BiRefNet | background removal | Peng Zheng et al. -- <https://github.com/ZhengPeng7/BiRefNet> |

`validation/fixtures/fig-snowman.webp` is the demo image from the `idol` SDK and
is used by the checks in this repository.
