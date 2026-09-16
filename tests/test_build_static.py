"""Guards on the static demo (M8, D43): it must reproduce every published score and never let
dataset text or model output become markup."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_static", ROOT / "demo" / "build_static.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)


@pytest.fixture(scope="module")
def payload():
    return bs.build_payload()


def test_every_golden_item_is_published_once(payload):
    counts = {t: len(v["items"]) for t, v in payload["tasks"].items()}
    assert counts == {"intent": 770, "urgency": 300, "pii": 300, "drafting": 300}


def test_pii_shows_the_adapter_manifest_v7_serves(payload):
    pii = payload["tasks"]["pii"]
    assert "manifest v7" in pii["caption"] and "a10-v7-pii-realistic" in pii["caption"]
    stats = {s["label"]: s for s in pii["summary"]}
    flagged = stats["Adapter · PII-free texts it flags"]
    assert flagged["value"] == "4 of 928" and "5 of 928" in flagged["note"]
    served = json.loads((ROOT / "runs" / "regression__a10-v7-pii-realistic.json").read_text())
    assert stats["Adapter · strict span F1"]["value"] == (
        f"{served['per_split']['pii']['random']['span_f1_strict']:.4f}")


def test_a_recorded_score_the_rows_do_not_reproduce_is_refused():
    with pytest.raises(ValueError, match="contradict"):
        bs.check("intent adapter accuracy", 0.9, 0.9286)


def test_classification_uses_the_scoring_rule(payload):
    item = payload["tasks"]["intent"]["items"][0]
    assert item["adapter_ok"] == (item["adapter"] == item["gold"])
    assert bs.first_line(" card_arrival\nextra") == "card_arrival"


def test_drafting_items_carry_all_three_graded_sides(payload):
    sides = payload["tasks"]["drafting"]["items"][0]["sides"]
    assert set(sides) == {"adapter", "prompted", "frontier"}
    assert all(1 <= s["grade"] <= 5 for s in sides.values())


def test_no_string_can_close_the_data_script_tag():
    out = bs.embed({"text": "</script><script>alert(1)</script>"})
    assert "</script>" not in out
    assert json.loads(out.replace("<\\/", "</"))["text"].startswith("</script>")


def test_dashboard_markdown_cannot_inject_html():
    results, failure = bs.render_dashboard()
    assert "<script" not in results and "<script" not in failure
    assert failure.startswith("<h2>Failure demo</h2>")


def test_the_space_card_is_static_and_attributes_the_nc_data():
    assert "sdk: static" in bs.CARD
    assert "CC BY-NC 4.0" in bs.CARD


def test_recorded_routing_decisions_reproduce_the_published_operating_point():
    """§5 step 3 on the static page: every decision shown must add up to the published curve."""
    r = bs.routing()
    confidence = r["policies"]["confidence"]
    assert r["pairs"] == 392
    assert confidence["quality"] == pytest.approx(0.7806, abs=1e-4)
    assert sum(i["confidence"]["escalated"] for i in r["items"]) == confidence["escalated"] == 78
    assert sum(i["router"]["misroute"] == "harmful_escalation" for i in r["items"]) == 14
