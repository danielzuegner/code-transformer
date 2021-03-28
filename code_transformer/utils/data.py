import torch


def pad_list(list_to_pad, pad_size, padding):
    assert len(list_to_pad) <= pad_size, f"Cannot pad list with length {len(list_to_pad)} with pad size {pad_size}"

    padded_list = list_to_pad.copy()
    num_padding = pad_size - len(list_to_pad)
    padded_list.extend([padding for _ in range(num_padding)])
    return padded_list


def tensor_to_tuple(tensor):
    tensor = tensor.to_sparse()
    return tensor.indices(), tensor.values(), tensor.size()


def tuple_to_tensor(compressed_tensor):
    return torch.sparse_coo_tensor(compressed_tensor[0], compressed_tensor[1], compressed_tensor[2])


def batch_index_select(tensor: torch.Tensor, dim: int, index: torch.Tensor, batch_dim=0):
    """
    Select a sub tensor for every batch.
    :param tensor: the tensor to select from
    :param dim: the dimension to select from. E.g., if `tensor` has shape 2x3x5 and `dim` = 1 then one can select
        from the 3 rows in the 3x5 matrix in every batch. If `dim` = 2, one can select from the 5 columns in the 3x5
        matrix
    :param index: 1D or 2D tensor. If `index` is 1D then exactly one sub-tensor will be chosen per batch. If `index`
        is 2D, then the sub tensors corresponding to the respective indices on the dimension `dim` will be selected. It
        is possible to select the same sub tensor multiple times. `index` has to have as many rows as the dimension
        `batch_dim`. The columns of `index` can be of arbitrary length
    :param batch_dim: the dimension that corresponds to the batch size
    :return:
        (d1 x ... x db x ... x di x ... x dn) -> (d1 x ... x db x ... x n_select x ... x dn)
        di is the dimension to select from, db is the batch dimension
        n_select is `index.shape[1]`, i.e., how many sub tensors are chosen per batch
    """
    assert index.shape[0] == tensor.shape[batch_dim], "Need to have one index per batch"
    assert not dim == batch_dim, "Cannot use batch_index_select on the same dimension that is the batch dimension"
    index_shape = [1 for _ in range(len(tensor.shape))]
    index_shape[batch_dim] = tensor.shape[batch_dim]
    index_shape[dim] = -1
    index = index.reshape(index_shape)  # batch_size x n_select x 1 x 1 x ...

    expansion = list(tensor.shape)
    expansion[batch_dim] = -1
    expansion[dim] = -1
    index = index.expand(expansion)  # batch_size x tensor[1] x ... x n_select x ... x tensor[n]

    return torch.gather(tensor, dim, index)
