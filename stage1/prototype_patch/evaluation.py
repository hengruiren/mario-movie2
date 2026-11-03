import torch
import torch.nn.functional as F


@torch.no_grad()
def encode_images(model, features, node_ids, device, batch_size=512):
    was_training = model.image_bert.training
    model.image_bert.eval()
    values = []
    for start in range(0, len(node_ids), batch_size):
        ids = node_ids[start:start + batch_size]
        values.append(model.image_infer(None, features[ids].to(device).unsqueeze(1)).float().cpu())
    model.image_bert.train(was_training)
    return torch.cat(values)


@torch.no_grad()
def prototype_metrics(image, prototypes, labels):
    scores = F.normalize(image.float(), dim=-1) @ F.normalize(prototypes.float(), dim=-1).t()
    predictions = scores.argmax(dim=-1)
    correct = predictions == labels
    per_class = {str(int(category)): float(correct[labels == category].float().mean())
                 for category in labels.unique(sorted=True)}
    return {"prototype_accuracy": float(correct.float().mean()),
            "prototype_macro_accuracy": sum(per_class.values()) / len(per_class),
            "prototype_per_class_accuracy": per_class,
            "nodes": len(labels)}


@torch.no_grad()
def retrieval_accuracy(image, text, batch_size=512):
    image, text = F.normalize(image.float(), dim=-1), F.normalize(text.float(), dim=-1)
    correct_image, correct_text = 0, 0
    for start in range(0, len(image), batch_size):
        stop = min(start + batch_size, len(image))
        targets = torch.arange(start, stop)
        correct_image += int(((image[start:stop] @ text.t()).argmax(dim=-1) == targets).sum())
        correct_text += int(((text[start:stop] @ image.t()).argmax(dim=-1) == targets).sum())
    return {"image_to_text_r1": correct_image / len(image),
            "text_to_image_r1": correct_text / len(image)}


@torch.no_grad()
def ridge_probe(train_image, train_labels, val_image, val_labels, num_classes=20, ridge=1.0):
    if ridge <= 0:
        raise ValueError("Probe ridge must be positive")
    train = F.normalize(train_image.double(), dim=-1)
    val = F.normalize(val_image.double(), dim=-1)
    train = torch.cat([train, torch.ones(len(train), 1, dtype=train.dtype)], dim=1)
    val = torch.cat([val, torch.ones(len(val), 1, dtype=val.dtype)], dim=1)
    labels = F.one_hot(train_labels, num_classes=num_classes).double()
    regularizer = torch.eye(train.shape[1], dtype=train.dtype) * ridge
    regularizer[-1, -1] = 0
    weights = torch.linalg.solve(train.t() @ train + regularizer, train.t() @ labels)
    predictions = (val @ weights).argmax(dim=-1)
    return {"ridge_probe_accuracy": float((predictions == val_labels).float().mean()),
            "ridge_probe_regularization": ridge,
            "ridge_probe_training_nodes": len(train_labels)}


@torch.no_grad()
def embedding_metrics(images, frozen_text, data, prototypes, ridge=1.0, batch_size=512):
    result = prototype_metrics(images[data.val_ids], prototypes, data.labels[data.val_ids])
    result.update(retrieval_accuracy(images[data.val_ids], frozen_text[data.val_ids], batch_size))
    result.update(ridge_probe(images[data.train_ids], data.labels[data.train_ids],
                              images[data.val_ids], data.labels[data.val_ids],
                              len(data.names), ridge))
    return result
