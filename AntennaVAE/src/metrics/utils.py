
def split_batches(adata, batch, hvg=None, return_categories=False):
    split = []
    batch_categories = adata.obs[batch].unique()
    if hvg is not None:
        adata = adata[:, hvg]
    for i in batch_categories:
        split.append(adata[adata.obs[batch] == i].copy())
    if return_categories:
        return split, batch_categories
    return split