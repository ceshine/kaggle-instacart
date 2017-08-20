"""Masked Binary Cross Entropy Objective Function
"""

import torch
from torch.nn import functional
from torch.autograd import Variable

DEBUG = 0


def sequence_mask(start_idx, sequence_lengths, max_len):
    batch_size = len(sequence_lengths)
    seq_range = torch.arange(0, max_len).long().cuda()
    # expand to: (max_len, batch)
    seq_range_expand = Variable(seq_range.unsqueeze(1)
                                .expand(max_len, batch_size))
    sequence_lengths_expand = Variable(sequence_lengths.unsqueeze(0)
                                       .expand_as(seq_range_expand))
    start_idx_expand = Variable(start_idx.unsqueeze(0)
                                .expand_as(seq_range_expand))
    if DEBUG:
        import numpy as np
        target = (
            (seq_range_expand < sequence_lengths_expand) *
            (seq_range_expand >= start_idx_expand)
        ).cpu().data.numpy()
        assert np.array_equal(
            np.sum(target, axis=0),
            sequence_lengths.cpu().numpy() - start_idx.cpu().numpy())
    return (
        (seq_range_expand < sequence_lengths_expand) *
        (seq_range_expand >= start_idx_expand)
    ).float()


def masked_bce(probs, target, sequence_lengths, max_lookback=3):
    """
    Args:
        probs: A Variable containing a FloatTensor of size
            (max_len, batch, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (max_len, batch) which contains the index of the true
            class for each corresponding step.
        sequence_lengths: A array-like object which contains the
            valid sequence length of each row in a batch.
        max_lookback: An integer specifying the maximum lookback
            in the order history

    Returns:
        loss: An average loss value masked by the length.
    """
    sequence_lengths = torch.from_numpy(sequence_lengths).long().cuda()
    target = target[:probs.size(0), :]
    # mask: (batch, max_len)
    mask = sequence_mask(
        start_idx=torch.clamp(
            sequence_lengths - max_lookback, min=0
        ),
        sequence_lengths=sequence_lengths,
        max_len=probs.size(0))
    # losses_flat: (max_len, batch)
    loss = torch.nn.functional.binary_cross_entropy(
        probs * mask, target * mask, size_average=False)
    return loss / torch.sum(mask)
