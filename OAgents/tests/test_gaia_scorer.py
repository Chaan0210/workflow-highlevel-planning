import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "example" / "oagents_deep_research" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gaia_scorer import question_scorer  # noqa: E402


def test_plain_number_matches():
    assert question_scorer("0.1777", "0.1777")


def test_number_with_trailing_unit_matches():
    assert question_scorer("0.1777 m^3", "0.1777")


def test_number_with_currency_and_commas_matches():
    assert question_scorer("$1,234.56", "1234.56")


def test_number_with_percent_matches():
    assert question_scorer("17.77%", "17.77")


def test_number_inside_short_phrase_matches():
    assert question_scorer("approximately 42", "42")


def test_wrong_number_with_unit_does_not_match():
    assert not question_scorer("0.2 m^3", "0.1777")


def test_non_numeric_answer_for_numeric_truth_fails():
    assert not question_scorer("no idea", "42")


def test_list_elements_with_units_match():
    assert question_scorer("0.5 kg; 2 m", "0.5;2")


def test_list_length_mismatch_fails():
    assert not question_scorer("0.5;2;3", "0.5;2")


def test_string_answers_unaffected():
    assert question_scorer("Sea Gull", "seagull")
    assert not question_scorer("penguin", "seagull")
