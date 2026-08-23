"""Window engine + sequence builder tests: bucketing, gaps, targets, validation."""

from datetime import datetime, timedelta, timezone

import pytest

from data_pipeline.src.schemas.network_state import NetworkState, NetworkStateSequence
from data_pipeline.src.windows.engine import build_states, contiguous_runs
from data_pipeline.src.windows.sequences import build_sequences, split_time_aware

T0 = datetime(2026, 1, 1, 10, 0, 1, tzinfo=timezone.utc)   # window anchor: 10:00:00


def test_window_bucketing(sample_records):
    offsets = [0, 2, 3, 11, 15, 22]     # -> windows [00,10) x3, [10,20) x2, [20,30) x1
    states = build_states(
        [sample_records(o, label="BENIGN" if o < 20 else "ATTACK") for o in offsets],
        window_seconds=10.0,
    )
    assert len(states) == 3
    assert [s.state_id for s in states] == ["state_000001", "state_000002", "state_000003"]
    assert states[0].timestamp_start == datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    assert states[0].features["flow_count"] == 3
    assert states[2].window_seconds == 10.0
    assert states[0].timestamp_end - states[0].timestamp_start == timedelta(seconds=10)


def test_label_gating_requires_source_name(sample_records):
    labeled = [sample_records(0, label="ATTACK")]
    without_source = build_states(labeled, window_seconds=10.0)
    assert without_source[0].label is None            # no verified source -> null

    with_source = build_states(labeled, window_seconds=10.0, source_name="cic_ids2018")
    assert with_source[0].label == "ATTACK"
    assert with_source[0].label_source == "cic_ids2018"


def test_empty_dataset_yields_no_states():
    assert build_states([], window_seconds=10.0) == []


def test_gap_breaks_contiguity():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def state(minute: int) -> NetworkState:
        return NetworkState(
            state_id=f"state_{minute:06d}",
            timestamp_start=t + timedelta(minutes=minute),
            timestamp_end=t + timedelta(minutes=minute, seconds=60),
            window_seconds=60.0,
        )

    runs = contiguous_runs([state(0), state(1), state(3)])
    assert [len(r) for r in runs] == [2, 1]


def _states(n: int, start_offset_s: float = 0.0) -> list[NetworkState]:
    base = T0.replace(second=0)
    return [
        NetworkState(
            state_id=f"state_{i + 1:06d}",
            timestamp_start=base + timedelta(seconds=start_offset_s + i * 10),
            timestamp_end=base + timedelta(seconds=start_offset_s + (i + 1) * 10),
            window_seconds=10.0,
            features={"flow_count": float(i + 1)},
        )
        for i in range(n)
    ]


def test_sequence_creation_and_target():
    states = _states(8)
    seqs = build_sequences(states, sequence_length=5, prediction_horizon=3)
    # n=8, L=5 -> i in 0..3; target index i+4+3 must stay < 8 -> only i=0 qualifies
    assert len(seqs) == 4
    assert seqs[0].target_state is not None
    assert seqs[0].target_state.state_id == "state_000008"
    assert all(s.target_state is None for s in seqs[1:])
    assert seqs[0].sequence_length == 5 and seqs[0].window_seconds == 10.0
    assert [s.state_id for s in seqs[1].states] == ["state_000002", "state_000003", "state_000004", "state_000005", "state_000006"]


def test_sequences_never_span_gaps():
    states = _states(3) + _states(3, start_offset_s=100)   # gap between the two runs
    seqs = build_sequences(states, sequence_length=2, prediction_horizon=0)
    assert seqs
    for seq in seqs:
        stamps = [s.timestamp_start for s in seq.states]
        deltas = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
        assert all(d == 10.0 for d in deltas)              # never mixes the two runs


def test_horizon_zero_targets_last_input_state():
    """K=0 means target == S(t) itself (no future window beyond the inputs)."""
    seqs = build_sequences(_states(3), sequence_length=2, prediction_horizon=0)
    assert seqs[0].target_state.state_id == "state_000002"
    assert seqs[1].target_state.state_id == "state_000003"


def test_horizon_one_targets_next_state():
    seqs = build_sequences(_states(4), sequence_length=2, prediction_horizon=1)
    assert [s.state_id for s in seqs[0].states] == ["state_000001", "state_000002"]
    assert seqs[0].target_state.state_id == "state_000003"   # one window after S(t)
    assert seqs[2].target_state is None                      # beyond run end


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        build_sequences(_states(5), sequence_length=0)
    with pytest.raises(ValueError):
        build_sequences(_states(5), prediction_horizon=-1)
    with pytest.raises(ValueError):
        split_time_aware([], train=0.5, validation=0.6)


def test_split_time_aware_orders_chronologically():
    seqs = build_sequences(_states(12), sequence_length=2, prediction_horizon=1)
    splits = split_time_aware(seqs, train=0.7, validation=0.15)
    total = sum(len(v) for v in splits.values())
    assert total == len(seqs)
    last_train = splits["train"][-1].states[-1].timestamp_start
    first_test = splits["test"][0].states[-1].timestamp_start if splits["test"] else None
    if first_test:
        assert last_train <= first_test


def test_schema_validation_errors():
    from pydantic import ValidationError

    t = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        NetworkState(timestamp_start=t, timestamp_end=t)      # missing state_id
    seq = NetworkStateSequence(sequence_id="seq_000001")       # contract defaults
    assert seq.sequence_length == 5 and seq.target_state is None and seq.states == []


def test_open_feature_dict_extension():
    st = _states(1)[0]
    st.features["some_future_feature"] = 1.23      # additive extension must be allowed
    parsed = NetworkState.model_validate_json(st.model_dump_json())
    assert parsed.features["some_future_feature"] == pytest.approx(1.23)
