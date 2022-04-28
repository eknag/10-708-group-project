import os

def create_sample_mosaic(sampler, n_rows, n_cols, fname):
    """
    Create a sample mosaic of the given sampler.

    Parameters
    ----------
    sampler : Sampler
        Sampler to create the mosaic of.
    n_rows : int
        Number of rows in the mosaic.
    n_cols : int
        Number of columns in the mosaic.
    fname : str
        Name of the file to save the mosaic to.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(n_rows, n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            ax = plt.subplot(gs[i, j])
            ax.imshow(sampler.sample(1)[0].cpu().numpy().transpose(1, 2, 0))
            ax.axis('off')
    print(f"saving figure to {fname + '.png'}")
    path = fname.split('/')[:-1].join('/')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(fname + '.png')
