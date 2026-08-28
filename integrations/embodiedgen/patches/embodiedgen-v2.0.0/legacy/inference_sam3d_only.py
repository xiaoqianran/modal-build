from __future__ import annotations

import random
import torch
from PIL import Image

try:
    from embodied_gen.utils.monkey_patch.trellis import monkey_path_trellis
    monkey_path_trellis()
except Exception:
    pass

from embodied_gen.data.utils import trellis_preprocess
from embodied_gen.models.sam3d import Sam3dInference
from embodied_gen.utils.trender import pack_state, unpack_state

try:
    from thirdparty.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
except Exception:
    class TrellisImageTo3DPipeline:  # SAM3D-only runtime sentinel
        pass

__all__ = ["image3d_model_infer"]


def image3d_model_infer(pipe, seg_image: Image.Image, seed: int | None = None, **kwargs):
    if isinstance(pipe, TrellisImageTo3DPipeline) and not isinstance(pipe, Sam3dInference):
        pipe.cuda()
        seg_image = trellis_preprocess(seg_image)
        outputs = pipe.run(
            seg_image,
            preprocess_image=False,
            seed=(random.randint(0, 100000) if seed is None else seed),
            **kwargs,
        )
        pipe.cpu()
    elif isinstance(pipe, Sam3dInference):
        outputs = pipe.run(
            seg_image,
            seed=(random.randint(0, 100000) if seed is None else seed),
            **kwargs,
        )
        state = pack_state(outputs["gaussian"][0], outputs["mesh"][0])
        outputs["gaussian"][0], _ = unpack_state(state, device="cuda")
    else:
        raise ValueError(f"Unsupported pipeline type: {type(pipe)}")
    torch.cuda.empty_cache()
    return outputs
