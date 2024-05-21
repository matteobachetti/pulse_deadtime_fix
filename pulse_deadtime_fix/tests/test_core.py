import numpy as np
import pytest

from pulse_deadtime_fix.core import fold_and_correct_profile
from pulse_deadtime_fix.simulate import (
    simulate_pulsed_events,
    apply_deadtime_and_calculate_prior,
    mask_fraction_of_data,
)


@pytest.mark.parametrize("period", [0.1, 0.01])
@pytest.mark.parametrize("fraction", [0.7, 0.3])
def test_consistent_results(period, fraction):
    nbin = 512
    times = simulate_pulsed_events(
        period=period,
        peak_phases=[0.2, 0.8],
        peak_width=[0.01, 0.05],
        peak_flux_fraction=[0.015, 0.015],
        nbin=nbin,
    )

    profile, _ = np.histogram((times / period) % 1, bins=np.linspace(0, 1, nbin + 1))
    times_filt, priors = apply_deadtime_and_calculate_prior(
        times, 2.5e-3, paralyzable=False
    )
    mask = mask_fraction_of_data(times_filt, fraction)
    _, _, profile_corr = fold_and_correct_profile(
        times_filt[mask], priors[mask], 0, [1 / period, 0, 0], nbin=nbin
    )

    profile_corr *= np.mean(profile / profile_corr)

    assert (profile - profile_corr).std() < profile_corr.std() * 5
