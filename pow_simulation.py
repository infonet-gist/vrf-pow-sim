"""Backward-compatible imports for the simulation package.

New code should import from :mod:`vrf_pow_sim` directly. The plotting and
analytical-figure helpers from the research notebook are intentionally excluded.
"""

from vrf_pow_sim import *  # noqa: F401,F403
from vrf_pow_sim import calculate_submission_times

# Historical public name used by the experiment notebook.
calculate_submission_time_vectorized = calculate_submission_times
