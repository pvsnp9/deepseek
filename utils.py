import numpy as np
import matplotlib.pyplot as plt


def plot_sin_cos(positional_encodings, indices=None):
    num_tokens, d_model = positional_encodings.shape

    # Normalize indices to list
    if indices is None:
        indices = list(range(0, d_model, 2))
    elif isinstance(indices, int):
        indices = [indices]
    else:
        indices = list(indices)

    # Validate indices: must be even and in range
    indices = [i for i in indices if i % 2 == 0 and 0 <= i < d_model - 1]

    plt.figure(figsize=(8, 4))

    for i in indices:
        sine_vals = positional_encodings[:, i]
        cosine_vals = positional_encodings[:, i + 1]
        plt.plot(range(num_tokens), sine_vals, label=f'Sin (dim {i})')
        plt.plot(range(num_tokens), cosine_vals, label=f'Cos (dim {i+1})')

    plt.xlabel('Token Position')
    plt.ylabel('Encoding Value')
    plt.title('Sinusoidal Positional Encoding Values')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True)
    plt.show()
