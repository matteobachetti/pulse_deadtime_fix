import pytest
from pulse_deadtime_fix.simulate import (
    simulate_pulsed_events,
    mask_fraction_of_data,
    apply_deadtime_and_calculate_prior,
)


def test_peaks_fraction_high():
    with pytest.raises(ValueError, match="Pulse fractions must be non-negative"):
        simulate_pulsed_events(peak_flux_fraction=1.2)


def test_peaks_fraction_and_width_inconsistent():
    with pytest.raises(
        ValueError, match="Pulse fraction and peak width must have same length."
    ):
        simulate_pulsed_events(peak_flux_fraction=1.2, peak_width=[0.2, 0.3])


def test_mask_fraction_invalid():
    with pytest.raises(
        ValueError,
        match="The fraction of data to be eliminated must be between 0 and 1.",
    ):
        mask_fraction_of_data([1, 2, 3], 1.2)


def test_mask_all():
    mask = mask_fraction_of_data([1, 2, 3], 1)
    assert len(mask) == 3
    assert not mask.any()


def test_mask_none():
    mask = mask_fraction_of_data([1, 2, 3], 0)
    assert len(mask) == 3
    assert mask.all()
