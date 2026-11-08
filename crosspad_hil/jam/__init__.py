"""``live_jam``: a whole live set on the board, listened to sample by sample.

The scenario (``crosspad_hil.scenarios.live_jam``) drives the board; this
package says what it should have sounded like and where it did not:

* ``spec``      -- the fixture kits and the spectrum plan (stdlib only);
* ``script``    -- the set itself, step by step, deterministic for a seed;
* ``pilot``     -- the "phone's" stream: a pilot tone and a timecode;
* ``model``     -- the sequencer, the voices, swaps and the mixer, as expected audio;
* ``bands`` / ``detect`` -- what the take holds against what was expected;
* ``clocks`` / ``telemetry`` / ``attribute`` -- when, on every clock, and why;
* ``analysis`` / ``report`` -- one pure function from a run directory to a verdict.
"""
