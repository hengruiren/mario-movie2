"""Optional class-consensus patch refinement of an existing Mario export.

Prediction and acceptance have no label argument. CLIP coordinates are mapped
into Mario coordinates using full-image pairs from the training split only.
"""

import math

import torch
import torch.nn.functional as F


def _vectors(values, name):
    if not torch.is_floating_point(values) or not torch.isfinite(values).all():
        raise ValueError("{} must contain finite floating-point vectors".format(name))
    if (values.norm(dim=-1) <= 1e-12).any():
        raise ValueError("{} contains zero vectors".format(name))


@torch.no_grad()
def fit_clip_to_mario(clip_global, mario_images, train_ids, ridge=1.0):
    """Solve sum_i ||(x_i-x_mean) W - (y_i-y_mean)||² + ridge ||W||².

    Both full-image inputs and targets are unit normalized. The intercept is
    unregularized. No validation/test pairs or category labels enter the solve.
    """
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Bridge ridge must be finite and positive")
    if clip_global.ndim != 2 or mario_images.ndim != 2 or len(clip_global) != len(mario_images):
        raise ValueError("Bridge inputs must be aligned [N, D] full-image matrices")
    ids = torch.as_tensor(train_ids, dtype=torch.long, device="cpu")
    if ids.ndim != 1 or len(ids) < 2 or len(ids.unique()) != len(ids):
        raise ValueError("Bridge needs at least two unique training IDs")
    if int(ids.min()) < 0 or int(ids.max()) >= len(clip_global):
        raise ValueError("Training ID outside bridge inputs")
    x = clip_global.detach().cpu()[ids].double()
    y = mario_images.detach().cpu()[ids].double()
    _vectors(x, "Training CLIP images")
    _vectors(y, "Training Mario images")
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    w = torch.linalg.solve(xc.t() @ xc + ridge * torch.eye(x.shape[1], dtype=x.dtype), xc.t() @ yc)
    return {"weight": w.float(), "input_mean": xm.float(), "target_mean": ym.float()}


@torch.no_grad()
def map_clip_features(features, bridge):
    _vectors(features, "CLIP features")
    x = F.normalize(features.detach().cpu().float(), dim=-1)
    mapped = (x - bridge["input_mean"]) @ bridge["weight"] + bridge["target_mean"]
    if not torch.isfinite(mapped).all():
        raise ValueError("Bridge produced non-finite features")
    return F.normalize(mapped, dim=-1)


@torch.no_grad()
def refine_consensus_batch(text, images, mapped_patches, prototypes,
                           temperature=0.07, min_class_gain=1e-6, patch_query="class"):
    """Keep disagreements bitwise unchanged; gate agreements by two cosines.

    The candidate is the normalized patch sum weighted by the agreed class
    (default) or the node's own Text. Both variants use the SAME class gate.
    Accepted images retain their original norm. There is no confidence filter,
    true-label input, extra class-margin gate, or deletion of nodes/crops.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Patch temperature must be finite and positive")
    if not math.isfinite(min_class_gain) or min_class_gain < 0:
        raise ValueError("Minimum class gain must be finite and nonnegative")
    if patch_query not in ("class", "text"):
        raise ValueError("Patch query must be 'class' or 'text'")
    if (images.ndim != 2 or text.shape != images.shape or prototypes.ndim != 2
            or prototypes.shape[1] != images.shape[1] or len(prototypes) < 2
            or mapped_patches.ndim != 3 or mapped_patches.shape[0] != len(images)
            or mapped_patches.shape[2] != images.shape[1] or mapped_patches.shape[1] < 1):
        raise ValueError("Expected text/images [B,D], patches [B,P,D], prototypes [C,D]")
    for value, name in [(text, "Text"), (images, "Images"), (prototypes, "Prototypes")]:
        _vectors(value, name)
    if not torch.isfinite(mapped_patches).all():
        raise ValueError("Mapped patches must be finite")
    t, v, c = (F.normalize(x.float(), dim=-1) for x in (text, images, prototypes))
    text_class = (t @ c.t()).argmax(-1)
    image_scores = v @ c.t()
    image_class = image_scores.argmax(-1)
    agreement = text_class == image_class
    selected = c[image_class]
    class_before = (v * selected).sum(-1)
    text_before = (v * t).sum(-1)
    class_after, text_after = class_before.clone(), text_before.clone()
    after_class = image_class.clone()
    weights = torch.zeros(mapped_patches.shape[:2], device=images.device)
    candidate_valid = torch.zeros_like(agreement)
    output = images.clone()
    accepted = torch.zeros_like(agreement)
    ids = agreement.nonzero(as_tuple=True)[0]
    if len(ids):
        patches = F.normalize(mapped_patches[ids].float(), dim=-1)
        query = t[ids] if patch_query == "text" else selected[ids]
        w = F.softmax((patches * query[:, None]).sum(-1) / temperature, dim=-1)
        candidate = (w[..., None] * patches).sum(1)
        valid = ((mapped_patches[ids].norm(dim=-1) > 1e-12).all(-1)
                 & (candidate.norm(dim=-1) > 1e-12))
        candidate = F.normalize(candidate, dim=-1)
        weights[ids] = w
        candidate_valid[ids] = valid
        class_after[ids] = (candidate * selected[ids]).sum(-1)
        text_after[ids] = (candidate * t[ids]).sum(-1)
        after_class[ids] = (candidate @ c.t()).argmax(-1)
        use = (valid & (class_after[ids] > class_before[ids] + min_class_gain)
               & (text_after[ids] >= text_before[ids]))
        accepted[ids] = use
        chosen = ids[use]
        output[chosen] = (candidate[use] * images[chosen].float().norm(dim=-1, keepdim=True)).to(images.dtype)
    return output, dict(text_class=text_class, image_class=image_class,
                        candidate_class=after_class, agreement=agreement,
                        candidate_valid=candidate_valid, accepted=accepted,
                        class_before=class_before, class_after=class_after,
                        text_before=text_before, text_after=text_after,
                        patch_weights=weights)
