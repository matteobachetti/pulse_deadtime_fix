import numpy as np
import pytest

from pulse_deadtime_fix.core import fold_and_correct_profile, get_deadtime_mask


@pytest.mark.parametrize("period", [0.1, 0.01])
def test_consistent_results(period):
    nbin = 512
    deadtime = 2.5e-3
    tstart = 0
    ctrate = 1000
    tstop = 1000 * nbin / ctrate

    nphots = ctrate * (tstop - tstart)
    pulse_flux_fraction = 0.03
    phases_main = np.random.normal(0.2, 0.01, int(pulse_flux_fraction / 2 * nphots))
    phases_sec = np.random.normal(0.8, 0.05, int(pulse_flux_fraction / 2 * nphots))
    base = np.random.uniform(0, 1, int((1 - pulse_flux_fraction) * nphots))
    phases = np.concatenate((phases_main, phases_sec, base))

    profile, _ = np.histogram(phases, bins=np.linspace(0, 1, nbin + 1))
    phases = np.sort(
        (np.random.randint(tstart / period, tstop / period, phases.size) + phases)
    )
    times = phases * period

    mask = get_deadtime_mask(
        times,
        deadtime,
        paralyzable=False,
        return_all=False,
    )
    times_filt = times[mask]
    priors = np.zeros_like(times_filt)
    priors[1:] = np.diff(times_filt) - deadtime

    # Now, eliminate a chunk of data
    mask = np.ones_like(times_filt, dtype=bool)

    phase, profile_raw, profile_corr = fold_and_correct_profile(
        times_filt[mask], priors[mask], 0, [1 / period, 0, 0], nbin=nbin
    )

    profile_corr *= np.mean(profile / profile_corr)

    assert (profile - profile_corr).std() < profile_corr.std() * 5
