"""pytest plugin — collects YAML eval cases as pytest items and runs the engine once per session."""

from __future__ import annotations

import asyncio
import warnings
from pathlib import Path

import pytest

from evalgate.schema import CaseResult


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--evalgate",
        action="store",
        default=None,
        metavar="CASES_DIR",
        help="Directory of EvalGate YAML cases to run as pytest items.",
    )


def pytest_load_initial_conftests(early_config, parser, args: list[str]) -> None:
    """Make pytest traverse the cases dir even though it is outside testpaths.

    Without this, `pytest --evalgate examples/sec_agent/cases` would only
    collect from testpaths (tests/) and never reach the YAML files.
    """
    cases_dir = early_config.known_args_namespace.evalgate
    # Note: cases_dir already appears in `args` as the --evalgate option's
    # value, so a plain "not in args" check would wrongly treat it as an
    # existing positional root. Always append; pytest tolerates the same
    # directory being named as both an option value and a collection root.
    if cases_dir:
        args.append(cases_dir)


def pytest_collect_file(file_path: Path, parent: pytest.Collector) -> pytest.Collector | None:
    cases_dir = parent.config.getoption("--evalgate")
    if not cases_dir or file_path.suffix not in (".yaml", ".yml"):
        return None
    if Path(cases_dir).resolve() not in file_path.resolve().parents:
        return None
    return EvalGateFile.from_parent(parent, path=file_path)


# Session-scoped cache: the engine runs ONCE for all items, on first demand.
# Re-running the agent per pytest item would multiply LLM cost by the case count.
_RESULTS_KEY = pytest.StashKey[dict]()


def _engine_results(session: pytest.Session) -> dict[str, CaseResult]:
    """Run the whole eval engine once and cache {case_id: CaseResult}."""
    if _RESULTS_KEY in session.stash:
        return session.stash[_RESULTS_KEY]

    from evalgate.cli import _build_sec_agent_adapter
    from evalgate.config import load_config
    from evalgate.loader import load_cases
    from evalgate.runner import run_cases

    cfg = load_config()
    cfg.validate()
    cases = load_cases(Path(session.config.getoption("--evalgate")))
    adapter = _build_sec_agent_adapter(cfg)
    report = asyncio.run(run_cases(cases, adapter, cfg))

    results = {cr.case_id: cr for cr in report.cases}
    session.stash[_RESULTS_KEY] = results
    return results


class EvalGateFailure(Exception):
    """A case's pass rate fell below its threshold."""


class EvalGateFile(pytest.File):
    def collect(self):
        from evalgate.loader import load_case_file

        case = load_case_file(self.path)
        yield EvalGateItem.from_parent(self, name=case.id, case=case)


class EvalGateItem(pytest.Item):
    def __init__(self, *, case, **kwargs) -> None:
        super().__init__(**kwargs)
        self.case = case

    def runtest(self) -> None:
        from evalgate.config import load_config
        from evalgate.stats import is_flaky, wilson_interval

        result = _engine_results(self.session)[self.case.id]
        n = len(result.trials)
        ci_low, ci_high = wilson_interval(result.passes, n)

        if is_flaky(result.passes, n, load_config().flaky_ci_width):
            warnings.warn(
                f"EvalGate case {self.case.id!r} is FLAKY: "
                f"{result.passes}/{n} passed, CI [{ci_low:.2f}, {ci_high:.2f}] — "
                "investigate rather than trusting pass or fail.",
                pytest.PytestWarning,
                stacklevel=1,
            )

        threshold = self.case.thresholds.pass_rate_min
        if result.pass_rate < threshold:
            failed = [t for t in result.trials if not t.passed]
            details = "\n".join(
                f"  trial {t.trial_idx}: {(t.failure_reason or 'scorer failure')[:200]}"
                for t in failed
            )
            raise EvalGateFailure(
                f"{self.case.id}: pass rate {result.pass_rate:.0%} "
                f"({result.passes}/{n}) below threshold {threshold:.0%}, "
                f"CI [{ci_low:.2f}, {ci_high:.2f}]\n{details}"
            )

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, EvalGateFailure):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, 0, f"evalgate: {self.case.id}"
