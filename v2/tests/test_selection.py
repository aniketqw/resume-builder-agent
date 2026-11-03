"""Ops must patch a copy, address entries the way the rest of the code does,
and refuse to silently miss."""
from __future__ import annotations

from pathlib import Path

import pytest

from rba.resume.ingest import parse_resume
from rba.resume.selection import (
    Op,
    UnknownTarget,
    apply_ops,
    copy_doc,
    entries_by_key,
    signature,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def doc():
    return parse_resume((FIXTURES / "aniket_bits.txt").read_bytes(),
                        filename="aniket_bits.txt")


def test_apply_ops_does_not_touch_the_original(doc):
    """The property the whole concurrency story rests on."""
    target = doc.enabled()[0]
    before_enabled = target.enabled
    before_text = target.text

    patched = apply_ops(doc, [{"op": "disable", "bullet_id": target.id}])

    assert doc.by_id(target.id).enabled is before_enabled
    assert doc.by_id(target.id).text == before_text
    assert patched.by_id(target.id).enabled is False


def test_disable_then_enable_round_trips(doc):
    target = doc.enabled()[0]
    off = apply_ops(doc, [Op("disable", bullet_id=target.id)])
    back = apply_ops(off, [Op("enable", bullet_id=target.id)])
    assert back.as_text() == doc.as_text()


def test_set_text_keeps_the_bullet_id_stable(doc):
    """Editing wording must not rename the line the client is holding."""
    target = doc.enabled()[0]
    patched = apply_ops(doc, [Op("set_text", bullet_id=target.id,
                                 text="Completely different wording here.")])
    bullet = patched.by_id(target.id)
    assert bullet is not None
    assert bullet.text == "Completely different wording here."


def test_remove_bullet_drops_it(doc):
    target = doc.enabled()[0]
    patched = apply_ops(doc, [Op("remove_bullet", bullet_id=target.id)])
    assert patched.by_id(target.id) is None
    assert doc.by_id(target.id) is not None


def test_add_bullet_lands_in_the_named_entry(doc):
    key = next(iter(entries_by_key(doc)))
    before = len(entries_by_key(doc)[key].bullets)
    patched = apply_ops(doc, [Op("add_bullet", entry_key=key, text="A new line.")])
    assert len(entries_by_key(patched)[key].bullets) == before + 1
    assert len(entries_by_key(doc)[key].bullets) == before


def test_ops_apply_in_order(doc):
    target = doc.enabled()[0]
    patched = apply_ops(doc, [Op("set_text", bullet_id=target.id, text="first"),
                              Op("set_text", bullet_id=target.id, text="second")])
    assert patched.by_id(target.id).text == "second"


# ------------------------------------------------------------- failure ----
def test_an_unknown_bullet_raises_rather_than_being_ignored(doc):
    """A dropped op would score a document the user is not looking at."""
    with pytest.raises(UnknownTarget):
        apply_ops(doc, [Op("disable", bullet_id="does-not-exist")])


def test_an_unknown_entry_raises(doc):
    with pytest.raises(UnknownTarget):
        apply_ops(doc, [Op("add_bullet", entry_key="NOPE#9", text="x")])


def test_an_unknown_op_kind_is_rejected(doc):
    with pytest.raises(ValueError):
        apply_ops(doc, [{"op": "delete_everything"}])


def test_no_ops_is_a_faithful_copy(doc):
    assert apply_ops(doc, []).to_dict() == doc.to_dict()


# ----------------------------------------------------------- signature ----
def test_signature_changes_when_a_line_is_disabled(doc):
    target = doc.enabled()[0]
    other = apply_ops(doc, [Op("disable", bullet_id=target.id)])
    assert signature(doc) != signature(other)


def test_signature_changes_when_wording_changes(doc):
    target = doc.enabled()[0]
    other = apply_ops(doc, [Op("set_text", bullet_id=target.id, text="new")])
    assert signature(doc) != signature(other)


def test_signature_is_stable_across_an_identical_copy(doc):
    assert signature(doc) == signature(copy_doc(doc))


def test_entry_keys_match_the_scheme_the_rest_of_the_code_uses(doc):
    """`matching`, `additions`, `variants` and `grounding` all build
    f"{section.title}#{index}". A second scheme here would silently fail to
    join against any of them."""
    keys = entries_by_key(doc)
    expected = {f"{s.title}#{i}"
                for s in doc.sections for i, _ in enumerate(s.entries)}
    assert set(keys) == expected
