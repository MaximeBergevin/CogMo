# tests/test_get_condition_lookup.py

from get_condition_lookup import get_condition_lookup

def test_get_condition_lookup(mock_condition_data):
    """
    Tests that get_condition_lookup correctly summarizes a condition DataFrame.
    
    Args:
        mock_condition_data: A fixture providing a default mock condition DataFrame.
    """
    # Fixture provides input DataFrame
    input_df = mock_condition_data
    result  = get_condition_lookup(input_df)

    # 1. Check results is a dict
    assert isinstance(result, dict)

    # 2. Check participant ID
    assert result['participant_id'] == "p01_test"

    # 3. Check structure of the condition_counts dict
    counts = result['condition_counts']
    expected_keys = ["cognitive_demand", "motor_demand", "n_blocks"]
    assert list(counts.keys()) == expected_keys

    # 4. Check values within the condition_counts
    assert counts['cognitive_demand'] == ["congruent", "incongruent"]
    assert counts['motor_demand'] == ["highForce", "lowForce"]
    assert counts['n_blocks'] == 2

