import numpy as np
import numba as nb
from stingray.pulse.pulsar import pulse_phase
from stingray.filters import get_deadtime_mask


@nb.njit()
def fast_phase(times, tref, f0, f1, f2):
    """
    Compute the pulse phase of the events.

    Parameters
    ----------
    times : np.array
        The photon times
    tref : float
        The reference time to use for the folding
    f0 : float
        The frequency of the pulsar
    f1 : float
        The first frequency derivative
    f2 : float
        The second frequency derivative

    Returns
    -------
    phase : np.array
        The pulse phase of the events
    """
    phase_array = np.zeros_like(times)

    ONE_SIXTH = 1 / 6
    for nt in range(times.size):
        t = times[nt] - tref
        phase = t * f0 + 0.5 * t**2 * f1 + ONE_SIXTH * t**3 * f2
        phase_array[nt] = phase

    return phase_array


@nb.njit()
def phase_to_1(phases):
    """
    Convert phases to the range [0, 1).

    Parameters
    ----------
    phases : np.array
        The phases to convert
    new_phases : np.array
        The array to store the converted phases
    """
    new_phases = np.zeros_like(phases)
    for i in range(phases.size):
        new_phases[i] = phases[i] - np.floor(phases[i])
    return new_phases


@nb.njit()
def _create_weights(phase_start, phase_end, phase_edges):
    phase_mid = 0.5 * (phase_edges[1:] + phase_edges[:-1])
    weights = np.zeros_like(phase_mid)
    for ph_start, ph_end in zip(phase_start, phase_end):
        while ph_end - ph_start > 1:
            ph_end -= 1
            weights += 1

        ph_start = ph_start - np.floor(ph_start)
        ph_end = ph_end - np.floor(ph_end)

        startbin, endbin = np.searchsorted(phase_edges, [ph_start, ph_end])
        # if endbin == phase_mid.size:
        #     endbin = 0
        if endbin < startbin:
            weights[:endbin] += 1
            weights[startbin:] += 1
        else:
            weights[startbin:endbin] += 1
    weights = 1 / weights
    weights /= weights.max()
    return weights


def fold_and_correct_profile(
    times: np.array,
    prior: np.array,
    tref: float,
    frequency_derivatives: list,
    nbin: int = 128,
):
    """
    Fold and correct the events using the prior and the frequency derivatives.

    Parameters
    ----------
    times : np.array
        The photon times that will form the pulsed profile
    prior : np.array
        The livetime recorded before each event
    tref : float
        The reference time to use for the folding and correction.
    frequency_derivatives : list
        The frequency derivatives to use for the folding and correction.

    Other Parameters
    ----------------
    nbin : int
        The number of bins to use for the folded profile.

    Returns
    -------
    phase: np.array
        The phases corresponding to the mean phase of each bin
    profile_raw: np.array
        The raw pulse profile
    profile_corr: np.array
        The folded and corrected events.
    """
    f1 = f2 = 0
    f0 = frequency_derivatives[0]
    if len(frequency_derivatives) > 1:
        f1 = frequency_derivatives[1]
    if len(frequency_derivatives) > 2:
        f2 = frequency_derivatives[2]

    phases = fast_phase(times, tref, f0, f1, f2)
    phases_livetime_start = fast_phase(times - prior, tref, f0, f1, f2)
    phase_edges = np.linspace(0, 1, nbin + 1)
    phase_mid = 0.5 * (phase_edges[1:] + phase_edges[:-1])

    profile_raw, _ = np.histogram(phase_to_1(phases), bins=phase_edges)

    weights = _create_weights(phases_livetime_start, phases, phase_edges)

    return phase_mid, profile_raw, profile_raw * weights


import pytest


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


def make_plot(period=0.02):
    nbin = 256
    deadtime = 2.5e-3
    tstart = 0
    ctrate = 1000
    tstop = 1000 * nbin / ctrate

    rdet_over_rin = 1 / (1 + deadtime * ctrate)

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
    # mask[np.random.randint(0, mask.size, size=mask.size)] = False
    # print(times_filt.size, times_filt[mask].size, priors[mask].size)
    phase, profile_raw, profile_corr = fold_and_correct_profile(
        times_filt[mask], priors[mask], 0, [1 / period, 0, 0], nbin=nbin
    )

    import matplotlib.pyplot as plt

    weights = profile_corr / profile_raw

    allph = np.concatenate((phase, phase + 1))

    def duplicate(prof):
        return np.concatenate((prof, prof))

    profile_corr *= np.mean(profile / profile_corr)
    plt.plot(
        allph, duplicate(profile), label="Original", ds="steps-mid", zorder=10, lw=2
    )
    plt.plot(allph, duplicate(profile_raw), label="Raw", ds="steps-mid")
    plt.plot(
        allph,
        duplicate(profile_corr),
        label="Corrected",
        ds="steps-mid",
        lw=2,
    )
    plt.plot(
        allph,
        duplicate(weights * profile.max()),
        alpha=0.1,
        label="Weights",
        ds="steps-mid",
    )
    print(weights.max())

    plt.legend()
    plt.show()


if __name__ == "__main__":
    make_plot(0.1)
    make_plot(0.01)
