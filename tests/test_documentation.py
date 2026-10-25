import argparse
from pathlib import Path

from csi.cli import parser
from csi.config import DEFAULTS, flatten

ROOT = Path(__file__).parents[1]


def test_every_configuration_field_is_documented():
    content = (ROOT / "docs/CLI.md").read_text()
    assert all(f"`{field}`" in content for field, _ in flatten(DEFAULTS))


def parser_options(command):
    for action in command._actions:
        yield from action.option_strings
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                yield from parser_options(subparser)


def test_every_parser_option_is_documented():
    content = (ROOT / "docs/CLI.md").read_text()
    missing = {option for option in parser_options(parser()) if option.startswith("--") and option not in content}
    assert not missing


def test_no_runtime_dependency_on_legacy_paths():
    import ast
    forbidden = {"models", "utility", "cqt_loader", "config", "novel_indian_pipeline", "indian_new_pipeline"}
    for source in (ROOT / "src/csi").rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                assert not any(alias.name.split(".")[0] in forbidden for alias in node.names), source
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert (node.module or "").split(".")[0] not in forbidden, source
