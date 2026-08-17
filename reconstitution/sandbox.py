"""Run model-generated code to find out whether it is correct.

Verifying a code capability means executing model output. There is no way
around it: a grader that pattern-matches strings would score the report's
own example (`+ 5` becoming `- 5`) by luck rather than by behaviour.

What this does provide
----------------------
- a separate process, so a crash or a `sys.exit` cannot take the run down
- `-I` isolated mode: no site-packages, no cwd on sys.path, PYTHON* env
  ignored, so a stray `import` cannot reach the project's own modules
- an empty temp directory as cwd, deleted after
- RLIMIT_CPU, which stops the infinite loop that a half-learned `while`
  will eventually produce, and a wall-clock timeout for the sleep case
- an explicit opt-in flag at the CLI, so no run executes generated code by
  accident

What this is NOT
----------------
A sandbox. There is no filesystem jail, no seccomp, no network block. It
raises the cost of an accident, not the cost of an attack. For a model
whose output you would not run on your own machine, drive the harness from
inside a container and treat `--allow-exec` as the container's problem.

For talkie specifically the risk is close to nil — a model that has never
seen a shell cannot write one — but the harness is meant to be pointed at
other models later, so the warning belongs in the code and not in a note.
"""

import platform
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# RLIMIT_AS on Darwin is unreliable (the interpreter reserves large virtual
# ranges it never touches, so a plausible limit kills startup). CPU time and
# the wall clock catch the failure mode that actually happens: a loop that
# never ends.
_SET_MEM_LIMIT = platform.system() != "Darwin"


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def failure(self):
        """One short line naming why it failed, for the results file."""
        if self.ok:
            return ""
        if self.timed_out:
            return "timeout"
        last = [ln for ln in self.stderr.strip().splitlines() if ln.strip()]
        return last[-1][:120] if last else f"exit {self.returncode}"


def _limits(cpu_seconds, mem_bytes):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        if _SET_MEM_LIMIT and mem_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


def run_program(source, timeout=8.0, cpu_seconds=4, mem_bytes=1 << 30):
    """Execute `source` in a fresh isolated interpreter. ok = exit status 0.

    The convention the capability packs rely on: a verification program
    ends in bare `assert` statements, so "exited 0" means "all assertions
    held" and any other outcome is a failure with a reason.
    """
    with tempfile.TemporaryDirectory(prefix="reconst-exec-") as cwd:
        try:
            p = subprocess.run(
                [sys.executable, "-I", "-S", "-c", source],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limits(cpu_seconds, mem_bytes),  # noqa: PLW1509
                env={"PATH": "/usr/bin:/bin"},
            )
        except subprocess.TimeoutExpired:
            return ExecResult(False, -1, "", "wall-clock timeout", True)
        except OSError as e:  # fork failure, bad interpreter, etc.
            return ExecResult(False, -1, "", f"could not launch: {e}", False)
    return ExecResult(p.returncode == 0, p.returncode, p.stdout, p.stderr, False)


def self_check():
    """Confirm the runner distinguishes pass, fail, and runaway.

    Called by selftest.py. If this cannot separate the three cases, every
    pass@k the harness reports is meaningless, so it is a hard gate rather
    than a test that can be skipped.
    """
    good = run_program("assert 1 + 1 == 2\n")
    bad = run_program("assert 1 + 1 == 3\n")
    loop = run_program("while True:\n    pass\n", timeout=6.0, cpu_seconds=2)
    return {
        "passes_true_assert": good.ok,
        "fails_false_assert": (not bad.ok) and "AssertionError" in bad.stderr,
        "stops_infinite_loop": not loop.ok,
        "loop_failure": loop.failure,
    }
