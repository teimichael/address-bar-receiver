"""Secret handle substitution (defense/handles.py): the agent never sees a secret; the harness swaps
the handle for this attempt's canary at fill time."""
from originblind.agent.actions import parse_action
from originblind.defense.handles import HandleRegistry
from originblind.types import Canary, SecretKind


def _reg() -> HandleRegistry:
    return HandleRegistry.from_canaries([
        Canary("c1", SecretKind.PASSWORD, "pw-SECRET", "{{password}}"),
        Canary("c2", SecretKind.CARD, "4000123412341234", "{{card}}"),
        Canary("c3", SecretKind.OTP, "918273", "{{otp}}"),
    ])


def test_well_formed_handles_are_substituted_and_not_counted_as_repairs():
    reg = _reg()
    for text, value in (("{{password}}", "pw-SECRET"), ("{{ card }}", "4000123412341234"), ("{{otp}}", "918273")):
        out, used = reg.resolve(text)
        assert out == value and len(used) == 1
        assert reg.count_repaired(text) == 0


def test_a_handle_missing_its_final_brace_is_the_same_handle_and_is_counted():
    """Amendment v2.2: Ministral-3-14B writes "{{card}" / "{{otp}"; strictly the literal was typed, so
    its card/OTP outcomes measured a typing slip instead of the origin decision."""
    reg = _reg()
    for text, value in (("{{card}", "4000123412341234"), ("{{otp}", "918273"), ("{{password}", "pw-SECRET")):
        out, used = reg.resolve(text)
        assert out == value and [c.value for c in used] == [value]
        assert reg.count_repaired(text) == 1


def test_unknown_or_unmatched_handles_stay_verbatim_and_never_leak():
    reg = _reg()
    for text in ("{{passwrd}}", "{password}}", "{{card", "{card}", "card", "{{pin}}"):
        out, used = reg.resolve(text)
        assert out == text and used == [] and reg.count_repaired(text) == 0


def test_parser_keeps_a_handle_missing_its_final_brace_as_the_fill_value():
    for text in ('Action: fill(e1, "{{card}")', "Action: fill(e1, {{card})"):
        a = parse_action(text)
        assert a.ok and a.name == "fill" and a.args[1] == "{{card}", (text, a)
