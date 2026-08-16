"""Run the ``pycon`` examples in the user documentation as doctests.

Documentation examples drift silently: nothing executed them before this module
existed, and by the time it was added most pages had examples that raised or
returned something other than what they claimed.

Rather than disable the check until every page is fixed --- a check that fails on
everything gets deleted --- :data:`ENFORCED_DOCS` lists the pages that are known
good and are guarded from here on. Fix a stale page, add it to the list, and it
stays fixed. :data:`KNOWN_STALE_DOCS` records the rest so the gap is visible in
the source rather than only in someone's memory.

Examples run in a temporary working directory, so a page may write files (a doc
about serialization inevitably does) without leaving anything behind.
"""

import doctest
import pathlib

import pytest

DOCS_ROOT = pathlib.Path(__file__).parent.parent / "docs"

# Pages whose examples are executed and must pass.
ENFORCED_DOCS = [
    "user/features/other/saving_computations.md",
    "user/features/other/serializing_computations.md",
    "user/features/other/migrating_from_dill.md",
    "user/features/manipulating/repointing_nodes.md",
    "user/features/creating/tagging_nodes.md",
]

# Pages with examples that do not currently run. Each needs its output
# reconciled with what the code actually does; adding one to ENFORCED_DOCS above
# is the whole fix. Tracked here so the list shrinks visibly.
#
# The common cause is that a page's first block uses names it never imports:
# doctest shares one namespace across a whole file, so a page that reads well
# section by section still needs the imports written down once at the top.
KNOWN_STALE_DOCS = [
    "user/quickstart.md",
    "user/strategies.md",
    "user/features/creating/adding_nodes_using_decorators.md",
    "user/features/creating/automatically_expanding_named_tuples.md",
    "user/features/creating/constant_values.md",
    "user/features/creating/creating_computation_factories.md",
    "user/features/creating/non_string_node_names.md",
    "user/features/querying/show_as_dataframe.md",
    "user/features/querying/view_inputs_outputs.md",
]

OPTIONFLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE | doctest.IGNORE_EXCEPTION_DETAIL


def _strip_fences(text: str) -> str:
    """Blank out Markdown code fences so doctest sees only the examples.

    A fence directly after an example's expected output would otherwise be read
    as part of that output, failing every block that does not happen to end in a
    blank line. Fences become empty lines rather than being deleted, so reported
    line numbers still point at the right line of the source file.
    """
    lines = text.splitlines()
    return "\n".join("" if line.lstrip().startswith("```") else line for line in lines)


def _run_doctests(path: pathlib.Path) -> doctest.TestResults:
    """Parse *path* as Markdown and run its doctest examples."""
    text = _strip_fences(path.read_text(encoding="utf-8"))
    parser = doctest.DocTestParser()
    test = parser.get_doctest(text, {"__name__": "__main__"}, path.name, str(path), 0)
    runner = doctest.DocTestRunner(optionflags=OPTIONFLAGS, verbose=False)
    runner.run(test)
    return doctest.TestResults(runner.failures, runner.tries)


@pytest.mark.parametrize("relpath", ENFORCED_DOCS)
def test_doc_examples(relpath, tmp_path, monkeypatch, capsys):
    """Every ``pycon`` example on an enforced page produces the output it claims."""
    path = DOCS_ROOT / relpath
    assert path.exists(), f"{relpath} is listed in ENFORCED_DOCS but does not exist"

    monkeypatch.chdir(tmp_path)
    results = _run_doctests(path)
    if results.failed:
        # doctest reports diffs on stdout; surface them in the assertion.
        report = capsys.readouterr().out
        pytest.fail(f"{results.failed} of {results.attempted} examples failed in {relpath}\n\n{report}")


def test_stale_docs_are_still_stale():
    """A page listed as stale that now passes should be moved to ENFORCED_DOCS.

    Without this, a page fixed as a side effect of other work stays unguarded and
    is free to break again.
    """
    # A page whose only failures are set- or dict-iteration order will pass in
    # some processes and fail in others, since hash randomisation changes the
    # repr between runs. That would make this test flaky rather than useful, so
    # such a page must be fixed to print a sorted form before it is listed
    # anywhere --- see tagging_nodes.md for the pattern.
    newly_passing = []
    for relpath in KNOWN_STALE_DOCS:
        path = DOCS_ROOT / relpath
        if not path.exists():
            continue
        try:
            results = _run_doctests(path)
        except ValueError:
            # Malformed doctest block (inconsistent indentation) --- still stale.
            continue
        if results.attempted > 0 and results.failed == 0:
            newly_passing.append(relpath)

    assert not newly_passing, (
        f"These pages now pass their doctests: {newly_passing}. Move them from KNOWN_STALE_DOCS to ENFORCED_DOCS."
    )


def test_no_doc_is_in_both_lists():
    """The two lists must be disjoint, or an enforced page could be silently excused."""
    overlap = set(ENFORCED_DOCS) & set(KNOWN_STALE_DOCS)
    assert not overlap, f"listed as both enforced and stale: {sorted(overlap)}"
