import torch
import torch.nn.functional as F


def symmetric_infonce(image, text, logit_scale):
    image = F.normalize(image, dim=-1)
    text = F.normalize(text, dim=-1)
    logits = logit_scale.exp() * image @ text.t()
    labels = torch.arange(len(image), device=image.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) * 0.5


def class_alignment(image, prototypes, labels, temperature=0.07):
    if temperature <= 0:
        raise ValueError("Class temperature must be positive")
    logits = F.normalize(image, dim=-1) @ F.normalize(prototypes.detach(), dim=-1).t()
    return F.cross_entropy(logits / temperature, labels)


def patch_similarity(left, right):
    left, right = F.normalize(left, dim=-1), F.normalize(right, dim=-1)
    similarities = (left @ right.transpose(-1, -2)).clamp(0, 1)
    return (similarities.amax(dim=-1).mean(dim=-1)
            + similarities.amax(dim=-2).mean(dim=-1)) * 0.5


def fine_distillation(anchor, neighbors, teacher_scores, valid):
    student = F.cosine_similarity(anchor[:, None, :], neighbors, dim=-1)
    error = (student - teacher_scores.detach()).square() * valid
    counts = valid.sum(dim=-1)
    active = counts > 0
    if not active.any():
        return anchor.sum() * 0.0
    return (error.sum(dim=-1)[active] / counts[active]).mean()


def fine_weight(epoch, maximum, class_only_epochs=5, ramp_epochs=5):
    if epoch < class_only_epochs:
        return 0.0
    return maximum * min((epoch - class_only_epochs + 1) / max(ramp_epochs, 1), 1.0)


@torch.no_grad()
def same_class_topk(embeddings, labels, train_ids, k=5, batch_size=512):
    if k < 1:
        raise ValueError("top_k must be positive")
    train_ids = train_ids.cpu().long().sort().values
    labels = labels.cpu().long().reshape(-1)
    features = F.normalize(embeddings.detach().cpu().float(), dim=-1)
    neighbors = torch.full((len(train_ids), k), -1, dtype=torch.long)
    valid = torch.zeros_like(neighbors, dtype=torch.bool)
    for category in labels[train_ids].unique(sorted=True):
        rows = torch.nonzero(labels[train_ids] == category).flatten()
        ids = train_ids[rows]
        count = min(k, len(ids) - 1)
        if count == 0:
            continue
        candidates = features[ids]
        for start in range(0, len(ids), batch_size):
            stop = min(start + batch_size, len(ids))
            scores = candidates[start:stop] @ candidates.t()
            scores[torch.arange(stop - start), torch.arange(start, stop)] = -torch.inf
            order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :count]
            neighbors[rows[start:stop], :count] = ids[order]
            valid[rows[start:stop], :count] = True
    return {"anchor_ids": train_ids, "neighbor_ids": neighbors, "valid": valid}


@torch.no_grad()
def build_teacher_scores(pairs, patch_ids, patch_features, batch_size=1024):
    patch_ids = patch_ids.cpu().long()
    if not torch.equal(patch_ids, patch_ids.unique(sorted=True)):
        raise ValueError("Patch cache IDs must be sorted and unique")
    anchors = pairs["anchor_ids"]
    if not set(anchors.tolist()) <= set(patch_ids.tolist()):
        raise ValueError("Patch cache does not cover every training anchor")
    neighbor_ids, valid = pairs["neighbor_ids"], pairs["valid"]
    if not set(neighbor_ids[valid].tolist()) <= set(patch_ids.tolist()):
        raise ValueError("Patch cache does not cover every training neighbor")
    scores = torch.zeros_like(neighbor_ids, dtype=torch.float32)
    rows, columns = valid.nonzero(as_tuple=True)
    for start in range(0, len(rows), batch_size):
        r, c = rows[start:start + batch_size], columns[start:start + batch_size]
        left = patch_features[torch.searchsorted(patch_ids, anchors[r])]
        right = patch_features[torch.searchsorted(patch_ids, neighbor_ids[r, c])]
        scores[r, c] = patch_similarity(left, right)
    return scores
