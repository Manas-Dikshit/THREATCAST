"""Sliding-window sequence construction: states -> NetworkStateSequences.

Generates S(t-L+1..t) input windows plus the target future state
S(t + horizon) where available. Sequences never span window gaps, and splits
are time-ordered (no shuffling across the temporal boundary).
"""

from ..schemas.network_state import NetworkState, NetworkStateSequence
from .engine import contiguous_runs


def build_sequences(
    states: list[NetworkState],
    *,
    sequence_length: int = 5,
    prediction_horizon: int = 3,
) -> list[NetworkStateSequence]:
    """Slide over contiguous state runs.

    target_state = the state `prediction_horizon` windows after the last input
    state, when it exists inside the same contiguous run; otherwise None.
    """
    if sequence_length < 1 or prediction_horizon < 0:
        raise ValueError("sequence_length must be >=1 and prediction_horizon >=0.")

    sequences: list[NetworkStateSequence] = []
    counter = 1
    for run in contiguous_runs(states):
        n = len(run)
        for i in range(n - sequence_length + 1):
            inputs = run[i : i + sequence_length]
            t_idx = i + sequence_length - 1 + prediction_horizon
            target = run[t_idx] if t_idx < n else None
            sequences.append(
                NetworkStateSequence(
                    sequence_id=f"seq_{counter:06d}",
                    states=inputs,
                    sequence_length=sequence_length,
                    window_seconds=inputs[0].window_seconds,
                    target_state=target,
                )
            )
            counter += 1
    return sequences


def split_time_aware(
    sequences: list[NetworkStateSequence],
    *,
    train: float = 0.7,
    validation: float = 0.15,
) -> dict[str, list[NetworkStateSequence]]:
    """Chronological split of sequences into train/val/test.

    The boundary is the last-input-state timestamp of each sequence; nothing is
    shuffled, so no future information can leak into training.
    """
    if abs(train + validation - 1.0) > 1e-9:
        raise ValueError("train + validation must equal 1.0 (rest goes to test).")
    ordered = sorted(sequences, key=lambda s: s.states[-1].timestamp_start)
    n = len(ordered)
    cut_train = int(n * train)
    cut_val = cut_train + int(n * validation)
    return {
        "train": ordered[:cut_train],
        "validation": ordered[cut_train:cut_val],
        "test": ordered[cut_val:],
    }


__all__ = ["build_sequences", "split_time_aware"]
