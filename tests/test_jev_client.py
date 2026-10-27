import pytest
from fakes import FakeJev

from rba.config import Config
from rba.jev_client import JevClient, JevError, choice, noul


def make(fake):
    return JevClient(config=Config.load(), api_key="k", transport=fake.transport())


def test_choice_requires_criteria_not_options():
    """A bare options list makes the live API return 422. Encode that in the helper."""
    q = choice("which tier?", {"small": "tiny repo", "big": "complex repo"})
    assert q["type"] == "choice"
    assert q["criteria"] == {"small": "tiny repo", "big": "complex repo"}
    assert "options" not in q
    with pytest.raises(ValueError):
        choice("empty?", {})


def test_noul_carries_true_and_false_definitions():
    q = noul("claimable?", "original work", "a fork")
    assert q["criteria"] == {"true": "original work", "false": "a fork"}


async def test_ask_parses_probabilities_and_choices():
    fake = FakeJev({
        "claimable": {"type": "noul", "noul": 0.7},
        "tier": {"type": "choice", "choice": "mid", "confidence": 0.86,
                 "probabilities": {"small": 0.0, "mid": 0.91, "big": 0.09}},
    })
    async with make(fake) as c:
        ans = await c.ask({"repo": "x"}, {"claimable": noul("q", "t", "f")})
    assert ans.probability("claimable") == 0.7
    assert ans.choice("tier") == ("mid", 0.86)


async def test_usage_is_tracked_and_output_tokens_are_free():
    fake = FakeJev({"claimable": {"noul": 0.5}}, input_tokens=465)
    async with make(fake) as c:
        await c.ask({"r": 1}, {"claimable": noul("q", "t", "f")})
        await c.ask({"r": 2}, {"claimable": noul("q", "t", "f")})
    assert c.usage.calls == 2
    assert c.usage.input_tokens == 930
    # 98 repos at this rate is a fraction of a cent.
    assert c.usage.cost_usd < 0.001


async def test_422_explains_the_criteria_requirement():
    fake = FakeJev({}, status=422)
    async with make(fake) as c:
        with pytest.raises(JevError, match="criteria"):
            await c.ask({}, {"tier": {"type": "choice", "question": "q",
                                      "options": ["a", "b"]}})


async def test_missing_key_marks_client_unavailable():
    c = JevClient(config=Config.load(), api_key="")
    assert not c.available
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        await c.ask({}, {})
    await c.aclose()
