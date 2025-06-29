import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_engineering import merge_downtime_features


def test_merge_downtime_sums_and_modes():
    df_prod = pd.DataFrame({
        'line': ['A', 'B'],
        'build_start_date': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-03')],
        'build_complete_date': [pd.Timestamp('2023-01-05'), pd.Timestamp('2023-01-04')],
    })

    df_down = pd.DataFrame({
        'line': ['A', 'A', 'B'],
        'date': [pd.Timestamp('2023-01-02'), pd.Timestamp('2023-01-04'), pd.Timestamp('2023-01-05')],
        'downtime_min': [10, 5, 8],
        'opportunity_cost': [100, 50, 80],
        'failure_mode': ['FM1', 'FM2', 'FM3']
    })

    result = merge_downtime_features(df_prod, df_down)

    # First build should sum downtime on line A within window
    row0 = result.loc[0]
    assert row0['downtime_min'] == 15
    assert row0['opportunity_cost'] == 150
    assert row0['failure_modes'] == ['FM1', 'FM2']
    assert row0['failure_mode'] == 'FM1'

    # Second build has no downtime within its window
    row1 = result.loc[1]
    assert row1['downtime_min'] == 0
    assert row1['opportunity_cost'] == 0
    assert row1['failure_modes'] == []
    assert row1['failure_mode'] == 'NONE'
