"""Preflight: assert the loaded pywarpx can actually run this fixture.

WHY THIS EXISTS
---------------
This working tree is shared with other threads of work (PR splitting, cherry-
picks, branch surgery). A `cmake --build` issued while HEAD is briefly on
another branch compiles THAT branch and overwrites the installed module. The
failure is silent and dangerous: the module still imports, the run still
completes, and it produces plausible numbers from a code path you did not
intend to test.

That happened on 2026-08-11. A rebuild landed in a two-minute window while
HEAD was on `test-add-magnetic-pressure-to-cylinder-compression-output`, which
has none of the electrode bindings, and overwrote a working module. See
`electrode-potential-maintenance/reference/absorption_potential_maintenance.md`
section 6.

Call `require(...)` at the top of a fixture, BEFORE building the simulation.
It costs milliseconds and turns a silent wrong-binary run into an immediate,
legible failure.
"""

import subprocess
import sys


def _git(args, cwd="/home/mgarten/src/warpx"):
    try:
        return subprocess.run(["git"] + args, cwd=cwd, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return "<git unavailable>"


def require(bindings=(), fields=(), verbose=True):
    """Abort unless the LOADED pywarpx module provides `bindings`.

    Parameters
    ----------
    bindings : iterable of str
        Method names required on the WarpX pybind class, e.g.
        ("compute_eb_charge", "solve_poisson_efield").
    fields : iterable of str
        Informational only -- registry field names the fixture expects.
        Not checked here (the registry is not populated until after init);
        listed in the banner so a later failure is easier to read.
    verbose : bool
        Print the provenance banner.

    Returns
    -------
    dict with the resolved module path and git branch, for logging.
    """
    import pywarpx

    mod = None
    for name in ("warpx_pybind_3d", "warpx_pybind_rz", "warpx_pybind_2d",
                 "warpx_pybind_1d"):
        mod = getattr(pywarpx, name, None)
        if mod is not None:
            break
    if mod is None:
        try:
            from pywarpx import warpx_pybind_3d as mod   # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "preflight: no pywarpx pybind module could be resolved; "
                f"PYTHONPATH is probably wrong ({exc})") from exc

    cls = getattr(mod, "WarpX", None)
    missing = [b for b in bindings if not hasattr(cls, b)]

    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    head = _git(["rev-parse", "--short", "HEAD"])
    dirty = "dirty" if _git(["status", "--porcelain"]) else "clean"

    if verbose:
        print("=" * 88)
        print("PREFLIGHT")
        print(f"  module     : {getattr(mod, '__file__', '?')}")
        print(f"  git HEAD   : {branch} @ {head} ({dirty})")
        if fields:
            print(f"  expects    : {', '.join(fields)}")
        print(f"  bindings   : {len(bindings) - len(missing)}/{len(bindings)} present")
        print("=" * 88)

    if missing:
        sys.stderr.write(
            "\n" + "!" * 88 + "\n"
            "PREFLIGHT FAILED -- the loaded pywarpx module is missing bindings\n"
            f"this fixture requires: {', '.join(missing)}\n\n"
            f"  module    : {getattr(mod, '__file__', '?')}\n"
            f"  git HEAD  : {branch} @ {head}\n\n"
            "The installed module was almost certainly built from a different\n"
            "branch than the one currently checked out -- this tree is shared,\n"
            "and a build issued while HEAD was elsewhere silently overwrites it.\n"
            "Rebuild with the intended branch checked out:\n\n"
            "  cd /home/mgarten/src/warpx && git rev-parse --abbrev-ref HEAD\n"
            "  cmake --build build -j 20\n\n"
            "Refusing to run: a wrong-binary run completes normally and yields\n"
            "plausible but meaningless numbers.\n"
            + "!" * 88 + "\n")
        raise SystemExit(2)

    return {"module": getattr(mod, "__file__", "?"), "branch": branch,
            "head": head, "dirty": dirty}
