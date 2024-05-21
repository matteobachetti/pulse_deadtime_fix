from collections.abc import Iterable
import numpy as np
from stingray.filters import get_deadtime_mask


def apply_deadtime_and_calculate_prior(times, deadtime, paralyzable=False):
    """
    Apply deadtime to the given times and calculate the prior values.

    Parameters
    ----------
    times : array-like
        The input times.
    deadtime : float
        The deadtime value.

    Returns
    -------
    times_filt : array-like
        The filtered times
    priors : array-like
        The livetime-before-event values for each time.

    """
    mask = get_deadtime_mask(
        times,
        deadtime,
        paralyzable=paralyzable,
        return_all=False,
    )
    times_filt = times[mask]
    priors = np.zeros_like(times_filt)
    priors[1:] = np.diff(times_filt) - deadtime
    return times_filt, priors


def mask_fraction_of_data(times, fraction_of_data_to_eliminate):
    """
    Masks a fraction of the data based on the given fraction_of_data_to_eliminate.

    Parameters
    ----------
    times : array-like
        Array of times.
    fraction_of_data_to_eliminate : float
        Fraction of data to be eliminated. Must be between 0 and 1.

    Returns
    -------
    mask : array of bools
        Masked data array.

    Raises
    ------
    ValueError: If ``fraction_of_data_to_eliminate`` is not between 0 and 1.
    """

    if (fraction_of_data_to_eliminate > 1) | (fraction_of_data_to_eliminate < 0):
        raise ValueError(
            "The fraction of data to be eliminated must be between 0 and 1."
        )
    saved_data = np.ones_like(times, dtype=bool)
    if fraction_of_data_to_eliminate == 0:
        return saved_data
    if fraction_of_data_to_eliminate == 1:
        return ~saved_data

    saved_data[: int(fraction_of_data_to_eliminate * saved_data.size)] = False
    np.random.shuffle(saved_data)
    return saved_data


def simulate_pulsed_events(
    period=0.02,
    ctrate=1000,
    nbin=256,
    tstart=0,
    min_length=0,
    min_ct_per_bin=1000,
    peak_flux_fraction=0.03,
    peak_width=0.01,
    peak_phases=None,
):
    """
    Simulate pulsed events.

    Parameters
    ----------
    period : float
        The period of the pulsar in seconds.
    ctrate : float
        The count rate of the pulsar in counts per second.
    nbin : int
        The number of bins to use for the folded profile.
    tstart : float
        The start time of the observation.
    min_length : float
        The minimum length of the observation.
    min_ct_per_bin : int
        The minimum number of counts per bin.
    peak_flux_fraction : float or array-like
        The fraction of the peak flux of the pulse profile.
    peak_width : float or array-like
        The width of the peak flux of the pulse profile.
    peak_phases : float or array-like
        The phases of the peak flux of the pulse profile.

    Returns
    -------
    times : array-like
        The simulated times of the pulsar.
    """
    if not isinstance(peak_flux_fraction, Iterable):
        peak_flux_fraction = [peak_flux_fraction]
    if not isinstance(peak_width, Iterable):
        peak_width = [peak_width]
    if not len(peak_flux_fraction) == len(peak_width):
        raise ValueError("Pulse fraction and peak width must have same length.")
    if peak_phases is None:
        peak_phases = 0.5 + np.arange(0, 1, 1 / len(peak_flux_fraction))

    peak_flux_fraction = np.asanyarray(peak_flux_fraction)

    if np.any(peak_flux_fraction < 0) or np.sum(peak_flux_fraction) > 1:
        raise ValueError("Pulse fractions must be non-negative and sum to less than 1.")

    peak_width = np.asanyarray(peak_width)
    base_flux = 1 - np.sum(peak_flux_fraction)

    tstop = max(min_length + tstart, tstart + min_ct_per_bin * nbin / ctrate)

    nphots = ctrate * (tstop - tstart)
    phase_arrays = [np.random.uniform(0, 1, int(base_flux * nphots))]
    for phas, pf, wid in zip(peak_phases, peak_flux_fraction, peak_width):
        phase_arrays.append(np.random.normal(phas, wid, int(pf * nphots)))

    phases = np.concatenate(phase_arrays)

    phases = np.sort(
        (np.random.randint(tstart / period, tstop / period, phases.size) + phases)
    )

    times = phases * period
    return times
