"""Guards on the realistic-format PII training split."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.data import pii_realistic_split as rs


def test_addresses_split_into_building_number_and_street_either_way_round():
    assert rs.address_parts("879 15th St W") == [("BUILDINGNUM", "879"), ("STREET", "15th St W")]
    assert rs.address_parts("Ulitsa Tverskaya, 14") == [("STREET", "Ulitsa Tverskaya"), ("BUILDINGNUM", "14")]
    assert rs.address_parts("County Road 141 118") == [("STREET", "County Road 141"), ("BUILDINGNUM", "118")]
    assert rs.address_parts("Flat 8B, Green Park Extension") == [("STREET", "Flat 8B, Green Park Extension")]


def test_targets_use_the_adapters_labels_in_text_order_and_drop_other_categories():
    text = "Dana Reyes lives at 146 County Rd 86 and works at Acme; email dana@x.io."
    spans = [
        {"start": text.index("dana@"), "end": text.index("dana@") + 9, "label": "email"},
        {"start": 0, "end": 4, "label": "first_name"},
        {"start": 5, "end": 10, "label": "last_name"},
        {"start": text.index("146"), "end": text.index("146") + 16, "label": "street_address"},
        {"start": text.index("Acme"), "end": text.index("Acme") + 4, "label": "company_name"},
    ]
    assert rs.target(text, spans).splitlines() == [
        "GIVENNAME: Dana", "SURNAME: Reyes", "BUILDINGNUM: 146", "STREET: County Rd 86", "EMAIL: dana@x.io"]


def test_a_document_with_only_other_categories_gets_an_empty_answer():
    text = "Visit https://acme.com today."
    assert rs.target(text, [{"start": 6, "end": 23, "label": "url"}]) == ""
