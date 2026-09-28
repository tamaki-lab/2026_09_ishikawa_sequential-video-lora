"""Select negatives from an old FIFO snapshot, without mutating the queue."""


def select_negatives(entries, sequence_id, negative_policy='different_sequence'):
    if negative_policy == 'different_sequence':
        selected = tuple(entry for entry in entries if entry.sequence_id != sequence_id)
        if not selected:
            raise RuntimeError('No valid different-sequence negatives in the queue')
    elif negative_policy == 'all_past':
        selected = tuple(entries)
        if not selected:
            raise RuntimeError('No valid past negatives in the queue')
    else:
        raise ValueError(f'Unknown negative_policy: {negative_policy}')
    return selected
