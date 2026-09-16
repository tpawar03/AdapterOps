"""Guards on relabelling confusable PII spans by the cue word before them."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import pii_relabel as rl


def test_a_gender_cue_relabels_sex_and_keeps_the_text():
    text = "Please confirm your gender: Male, and your date of birth."
    assert rl.relabel(text, "SEX: Male") == "GENDER: Male"


def test_a_driver_licence_cue_overrides_an_id_card_label():
    text = "Scan your driver's licence 48674ZYT6Q at the counter."
    assert rl.relabel(text, "IDCARDNUM: 48674ZYT6Q") == "DRIVERLICENSENUM: 48674ZYT6Q"


def test_the_nearest_cue_wins_and_labels_outside_the_groups_are_untouched():
    text = "Tax records aside, your passport XQ0032178 belongs to Dana."
    out = rl.relabel(text, "TAXNUM: XQ0032178\nGIVENNAME: Dana")
    assert out.splitlines() == ["PASSPORTNUM: XQ0032178", "GIVENNAME: Dana"]


def test_the_grouped_score_forgives_only_confusions_within_a_group():
    from adapterops.eval import regression as reg

    text = "Sex: Male. ID 48674ZYT6Q. Name Dana."
    gold = "SEX: Male\nIDCARDNUM: 48674ZYT6Q\nGIVENNAME: Dana"
    within = "GENDER: Male\nDRIVERLICENSENUM: 48674ZYT6Q\nGIVENNAME: Dana"
    across = "SEX: Male\nIDCARDNUM: 48674ZYT6Q\nEMAIL: Dana"
    assert reg.score("pii", [text], [gold], [within])["span_f1_grouped"] == 1.0
    assert reg.score("pii", [text], [gold], [within])["span_f1_strict"] < 1.0
    assert reg.score("pii", [text], [gold], [across])["span_f1_grouped"] < 1.0


def test_the_grouped_score_treats_a_name_split_either_way_as_one_name():
    from adapterops.eval import regression as reg

    text = "Signed, Arja Bahra Alvear, and Kim."
    gold = "GIVENNAME: Arja Bahra\nSURNAME: Alvear\nGIVENNAME: Kim"
    split_other_way = "GIVENNAME: Arja\nSURNAME: Bahra Alvear\nGIVENNAME: Kim"
    half_the_name = "GIVENNAME: Arja\nGIVENNAME: Kim"
    assert reg.score("pii", [text], [gold], [split_other_way])["span_f1_grouped"] == 1.0
    assert reg.score("pii", [text], [gold], [half_the_name])["span_f1_grouped"] < 1.0


def test_a_span_without_a_cue_keeps_its_label():
    text = "Reference 48674ZYT6Q was logged."
    assert rl.relabel(text, "IDCARDNUM: 48674ZYT6Q") == "IDCARDNUM: 48674ZYT6Q"
