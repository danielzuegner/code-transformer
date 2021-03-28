import torch


def permutation_attention_mask(inputs, is_masked, perm_size, seq_len, sep_id=-1, cls_id=-1):
    """
    Sample a permutation of the factorization order, and create an attention mask accordingly.
    The attention mask ensures that non-target tokens cannot attend target tokens and target tokens can only attend
    other target tokens if they come in the permutation order before them. Non-target tokens can always be attended
    even if they come later in the sequence. The reason is that all the non-target tokens are treated like one block.
    From the XLNet paper:
    "To reduce the optimization difficulty, we choose to only predict the last tokens in a
    factorization order. Formally, we split z into a non-target subsequence z≤c and a target subsequence z>c,
    where c is the cutting point. The objective is to maximize the log-likelihood of the target subsequence
    conditioned on the non-target subsequence."

    :param inputs: int64 Tensor in shape [batch_size, seq_len], input ids. Or [batch_size, seq_len, num_subtokens]
    :param is_masked: bool Tensor in shape [batch_size, seq_len]. True means being selected for partial prediction.
    :param perm_size: the length of longest permutation. Could be set to be reuse_len. Should not be larger than
        reuse_len or there will be data leaks.
    :param seq_len: int, sequence length.
    :return a batch_size x seq_len x seq_len tensor where a 1 at :,i,j indicates that position i cannot attend
    position j
    """
    if inputs.dim() == 3:
        inputs_bkp = inputs
        inputs = inputs[:, :, 0]

    if is_masked.dim() == 1:
        inputs = inputs.unsqueeze(0)
        is_masked = is_masked.unsqueeze(0)
        batch_size = 1
    else:
        batch_size = is_masked.shape[0]

    # Generate permutation indices
    index = torch.arange(seq_len, dtype=torch.int64).to(inputs.device)
    index_batch = index[None, :].expand([batch_size, -1])
    index_batch = torch.reshape(index_batch,
                                [batch_size, -1, perm_size]).transpose(1, 2)

    randperm = torch.stack([torch.randperm(index_batch.shape[1])
                            for _ in range(batch_size)])
    index_batch = index_batch.gather(1, index=randperm.unsqueeze(-1))
    index_batch = torch.reshape(index_batch.transpose(1, 2), [batch_size, -1])

    # `perm_mask` and `target_mask`
    # non-functional tokens
    non_func_tokens_batch = ~(torch.eq(inputs, sep_id) | torch.eq(inputs, cls_id))
    non_mask_tokens_batch = (~is_masked.bool()) & non_func_tokens_batch.bool()
    masked_or_func_tokens_batch = ~non_mask_tokens_batch

    # Set the permutation indices of non-masked (& non-functional) tokens to the
    # smallest index (-1):
    # (1) they can be seen by all other positions
    # (2) they cannot see masked positions, so there won"t be information leak
    smallest_index = -torch.ones([batch_size, seq_len], dtype=torch.int64).to(inputs.device)
    # put -1 if `non_mask_tokens(real token not cls or sep)` not permutation index
    rev_index = torch.where(non_mask_tokens_batch, smallest_index, index_batch)

    # Create `target_mask`: non-functional and masked tokens
    # 1: use mask as input and have loss
    # 0: use token (or [SEP], [CLS]) as input and do not have loss
    target_tokens = masked_or_func_tokens_batch & non_func_tokens_batch.bool()
    target_mask = target_tokens.type(torch.float32)

    # Create `perm_mask`
    # `target_tokens` cannot see themselves
    # put `rev_index` if real mask(not cls or sep) else `rev_index + 1`
    self_rev_index = torch.where(target_tokens, rev_index, rev_index + 1)

    # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
    # 0: can attend if i > j or j is non-masked
    perm_mask = (self_rev_index.unsqueeze(2) <= rev_index.unsqueeze(1)) & masked_or_func_tokens_batch.unsqueeze(1)
    perm_mask = perm_mask.type(torch.float32)

    # construct inputs_k
    if inputs.dim() == 3:
        inputs_k = inputs_bkp
    else:
        inputs_k = inputs

    # construct inputs_q
    inputs_q = target_mask

    return perm_mask, target_mask, inputs_k, inputs_q


def sample_targets(num_predict, seq_len, batch_size, pad_mask=None, ):
    """
    :return: a batch_size x num_predict x seq_len tensor containing the chosen token to be masked per sequence and
    prediction
        a batch_size x seq_len tensor containing the 5 chosen tokens to be masked per sequence
    """
    if pad_mask is not None:
        num_non_padded = pad_mask.sum(-1).long()
    else:
        num_non_padded = torch.ones(batch_size, dtype=torch.long) * seq_len

    import numpy as np
    target_idx = [np.random.choice(np.arange(num_tokens), num_predict, replace=False) for num_tokens in num_non_padded]
    ixs = torch.from_numpy(np.array(target_idx))

    assert (ixs.max(-1)[0] < num_non_padded).all()

    target_mapping = torch.zeros([batch_size, num_predict, seq_len])
    target_mapping = target_mapping.scatter_(-1, ixs.unsqueeze(-1), value=1.0)
    target_mapping_per_token = target_mapping.sum(1).clamp_max(1)
    return target_mapping, target_mapping_per_token


def pad_mask(sequence_lengths, max_len=None):
    if max_len is None:
        max_len = sequence_lengths.max()
    batch_size = sequence_lengths.shape[0]
    pad_mask = torch.arange(0, max_len).expand((batch_size, -1))
    pad_mask = 1 - (pad_mask >= sequence_lengths[:, None]).long()
    assert (pad_mask.sum(-1) == sequence_lengths).all()
    return pad_mask
