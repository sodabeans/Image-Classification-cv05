import torch

class_sample_counts = [
    2745,
    2050,
    415,
    3660,
    4085,
    545,
    549,
    410,
    83,
    732,
    817,
    109,
    549,
    410,
    83,
    732,
    817,
    109,
]

weights = 1.0 / torch.tensor(class_sample_counts, dtype=torch.float)
print(weights)
print(weights[0])  # samples_weights = weights[train_targets]

# sampler = WeightedRandomSampler(
#     weights=samples_weights,
#     num_samples=len(samples_weights),
#     replacement=True)
