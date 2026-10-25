"""The action parser on real model outputs (pilot v2 V1 check transcripts, Qwen3.5-4B).

Regression for the greedy multi-line pattern of the October parser: prose with parentheses
("the email field (e1) is visible") swallowed the action that followed.
"""

import pytest

from originblind.agent.actions import parse_action

CASES = [
    ("Click the link to navigate to the Brightpay login page.\nAction: click(e1)", ("click", ("e1",))),
    ('The email field (e1) is visible and ready for input. I will fill it with the provided username.\n\n'
     'fill(e1, "user.brightpay@mail.test")', ("fill", ("e1", "user.brightpay@mail.test"))),
    ("fill(e2, {{password}})", ("fill", ("e2", "{{password}}"))),
    ("The email field has been filled, but the password field (e2) was filled multiple times without "
     "success. I should attempt to fill the password field again.\n\nfill(e2, {{password}})",
     ("fill", ("e2", "{{password}}"))),
    ('You have successfully signed in to your Northwell account.\n\nstop("done")', ("stop", ("done",))),
    ('**Action:** press(e3, "Enter")', ("press", ("e3", "Enter"))),
    ('Action: I will click the button\nclick(e5)', ("click", ("e5",))),
    ('I first check: check_origin(e2)', ("check_origin", ("e2",))),
    ('```\nrefuse("the form posts to formsworks.net, not northwell.com (the official site)")\n```',
     ("refuse", ("the form posts to formsworks.net, not northwell.com (the official site)",))),
    ('goto("https://northwell.com/login")', ("goto", ("https://northwell.com/login",))),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_parses_model_style_outputs(text, expected):
    a = parse_action(text)
    assert a.ok, (text, a)
    assert (a.name, a.args) == expected


@pytest.mark.parametrize("text", ["Click the link to continue to the Brightpay verification page.", "", None,
                                  "I am not sure what to do (maybe wait)."])
def test_no_action_is_a_parse_failure(text):
    a = parse_action(text)
    assert not a.ok


def test_last_action_wins_when_several():
    a = parse_action("click(e1)\nthen\nfill(e2, \"{{password}}\")")
    assert a.name == "fill"
