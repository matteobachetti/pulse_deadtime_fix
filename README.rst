Fix dead time distortion of pulse profiles
------------------------------------------

Usage
-----

Here is a simple example of how to correct the dead time distortion of a pulse profile.
The data in the example are from the Crab pulsar, observed with NuSTAR (ObsID 10302001004).

The following ephemeris from the Jodrell Bank Observatory is used to fold the data:

.. code-block::

    PSRJ            J0534+2200
    RAJ             05:34:31.973
    DECJ            +22:00:52.06
    PEPOCH           58011.000000379725
    F0               29.6384226073
    F1               -3.6865813e-10
    F2               9.171123484933526e-21
    TZRMJD           58012.000000349
    TZRSITE          0
    TZRFRQ          0
    EPHEM           DE200
    UNITS           TDB
    CLK             TT(TAI)

.. code-block::python

    from pulse_deadtime_fix.core import fold_and_correct_profile
    from stingray import EventList

    ev = EventList.read("nu10302001004A01_bary.evt", additional_columns=["prior"], fmt="hea")
    phas, prof, prof_corr = fold_and_correct_profile(
        ev.time,
        ev.prior,
        (58011.000000379725 - ev.mjdref) * 86400,
        [29.6384226073, -3.6865813e-10, 9.171123484933526e-21]
    )


License
-------

This project is Copyright (c) Matteo B and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Openastronomy packaging guide <https://github.com/OpenAstronomy/packaging-guide>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.

Contributing
------------

We love contributions! pulse_deadtime_fix is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
pulse_deadtime_fix based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.
