import pytest

from src.evals.ifbench.eval import DATA_PATH, EXPECTED_PROMPTS, load_prompts, score_response

pytestmark = pytest.mark.skipif(not DATA_PATH.exists(), reason="checksum-pinned IFBench data not fetched")


def test_ifbench_data_shape_and_official_scorer_smoke():
    rows = load_prompts()
    assert len(rows) == EXPECTED_PROMPTS
    scores = score_response(rows[0], "This response deliberately ignores the constraints.")
    assert set(scores) == {
        "strict_prompt_pass",
        "strict_instruction_passes",
        "loose_prompt_pass",
        "loose_instruction_passes",
    }
    assert scores["loose_prompt_pass"] is False
