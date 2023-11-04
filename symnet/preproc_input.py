import torch
import numpy as np


# TODO: случай когда idx == None
def prepare_batches(u, derivs, shape, idx, additional_tokens=None):
    u = np.reshape(u, (shape[0], shape[1], 1))
    if len(derivs.shape) != 3:
        derivs = np.reshape(derivs, (shape[0], shape[1], derivs.shape[1]))
    mxs = [u, derivs]

    if additional_tokens is not None:
        add_mx = np.array(additional_tokens)
        mxs.append(np.reshape(add_mx, (shape[0], shape[1], len(additional_tokens))))
    input_matrices = np.concatenate(mxs, axis=2)

    return _create_batch(input_matrices, 32, left_idx=idx)


def _create_batch(matrices, batch_size, left_idx):
    n_batch_row = matrices[:, :, 0].shape[0] // batch_size
    in_row_indent = matrices[:, :, 0].shape[0] % batch_size // 2
    n_batch_col = matrices[:, :, 0].shape[1] // batch_size
    in_col_indent = matrices[:, :, 0].shape[1] % batch_size // 2

    def pack_token(k):
        elem_ls = []
        for i in range(n_batch_row):
            for j in range(n_batch_col):
                elem_ls.append(matrices[
                               in_row_indent + i * batch_size:in_row_indent + (i + 1) * batch_size,
                               in_col_indent + j * batch_size:in_col_indent + (j + 1) * batch_size, k])
        return elem_ls

    all_tokens_ls = []
    for l in range(matrices.shape[2]):
        if l == left_idx:
            left_side = torch.from_numpy(np.asarray(pack_token(l)))
        else:
            all_tokens_ls.append(pack_token(l))
    right_side = torch.from_numpy(np.asarray(all_tokens_ls))
    return torch.permute(right_side, (1, 2, 3, 0)), left_side