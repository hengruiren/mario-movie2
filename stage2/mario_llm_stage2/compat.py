from functools import lru_cache
import inspect


@lru_cache(maxsize=None)
def logits_argument(forward):
    parameters = inspect.signature(forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            return name
    return None


def answer_logits_kwargs(forward, count):
    name = logits_argument(forward)
    return {name: count} if name is not None else {}
