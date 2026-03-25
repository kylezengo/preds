"""Tests for download_gt_data.py"""

from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

import pandas as pd
import pytest
import requests
from pytrends.exceptions import ResponseError

import download_gt_data


# ──────────────────────────────────────────────────────────────────────────────
# load_existing_data
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadExistingData:
    """Tests for load_existing_data."""

    def test_returns_empty_dataframes_when_no_files(self):
        """Returns empty DataFrames and empty list when no CSV files exist."""
        with patch('download_gt_data.glob.glob', return_value=[]):
            monthly, weekly, daily, empty_params = download_gt_data.load_existing_data()

        assert monthly.empty
        assert weekly.empty
        assert daily.empty
        assert empty_params == []

    def test_loads_latest_monthly_file(self):
        """Picks the most recent monthly file when multiple exist."""
        files = ['data/gt_monthly_20240101.csv', 'data/gt_monthly_20240301.csv']
        expected_df = pd.DataFrame(columns=['start_date', 'end_date'])

        def mock_glob(pattern):
            if 'monthly' in pattern:
                return files
            return []

        with patch('download_gt_data.glob.glob', side_effect=mock_glob), \
             patch('download_gt_data.pd.read_csv', return_value=expected_df) as mock_read:
            download_gt_data.load_existing_data()
            calls = [str(c) for c in mock_read.call_args_list]
            assert any('20240301' in c for c in calls)

    def test_loads_params_return_empty_df_file(self):
        """Reads lines from the latest params_return_empty_df file."""
        daily_df = pd.DataFrame(
            columns=['date', 'index', 'isPartial', 'search_term', 'pytrends_params',
                     'request_datetime']
        )

        def mock_glob(pattern):
            if 'params_return_empty' in pattern:
                return ['data/params_return_empty_df_20240301.txt']
            return []

        file_contents = "param_line_1\nparam_line_2\n"
        with patch('download_gt_data.glob.glob', side_effect=mock_glob), \
             patch('download_gt_data.pd.read_csv', return_value=daily_df), \
             patch('builtins.open', mock_open(read_data=file_contents)), \
             patch('download_gt_data.os.path.getctime', return_value=0):
            _, _, _, params = download_gt_data.load_existing_data()

        assert params == ['param_line_1', 'param_line_2']


# ──────────────────────────────────────────────────────────────────────────────
# clean_up
# ──────────────────────────────────────────────────────────────────────────────

def _make_monthly(search_term='SPY', idx=100, date_range='2024-01-01 2024-12-31'):
    return pd.DataFrame([{
        'start_date': pd.Timestamp('2024-01-01'),
        'end_date': pd.Timestamp('2024-01-31'),
        'index': idx,
        'isPartial': False,
        'search_term': search_term,
        'pytrends_params': f'{{"timeframe": "{date_range}"}}',
    }])


def _make_weekly(search_term='SPY', idx=80):
    return pd.DataFrame([{
        'start_date': pd.Timestamp('2023-12-31'),  # week containing 2024-01-02 (W-SAT period)
        'end_date': pd.Timestamp('2024-01-06'),
        'index': idx,
        'isPartial': False,
        'search_term': search_term,
        'pytrends_params': '{"timeframe": "2024-01-01 2024-12-31"}',
    }])


def _make_daily(search_term='SPY', idx=50, is_partial=False, request_dt=None):
    return pd.DataFrame([{
        'date': pd.Timestamp('2024-01-02'),
        'index': idx,
        'isPartial': is_partial,
        'search_term': search_term,
        'pytrends_params': '{"timeframe": "2023-12-31 2024-01-06"}',
        'request_datetime': request_dt or pd.Timestamp('2024-01-03 10:00:00'),
    }])


class TestCleanUp:
    """Tests for clean_up."""

    def test_returns_expected_columns(self):
        """Output has exactly date, day_of_week, search_term, index columns."""
        result = download_gt_data.clean_up(_make_monthly(), _make_weekly(), _make_daily())
        assert list(result.columns) == ['date', 'day_of_week', 'search_term', 'index']

    def test_index_math_is_correct(self):
        """Adjusted index = daily * weekly/100 * monthly/100."""
        # daily=50, weekly=80, monthly=100 → 50 * 0.8 * 1.0 = 40
        result = download_gt_data.clean_up(
            _make_monthly(idx=100),
            _make_weekly(idx=80),
            _make_daily(idx=50)
        )
        assert result['index'].iloc[0] == pytest.approx(40.0)

    def test_partial_rows_excluded(self):
        """Rows where isPartial=True are excluded from the output."""
        result = download_gt_data.clean_up(
            _make_monthly(),
            _make_weekly(),
            _make_daily(is_partial=True)
        )
        assert result.empty

    def test_day_of_week_is_correct(self):
        """day_of_week matches the actual day of the date (2024-01-02 is Tuesday)."""
        result = download_gt_data.clean_up(_make_monthly(), _make_weekly(), _make_daily())
        assert result['day_of_week'].iloc[0] == 'Tuesday'

    def test_latest_request_datetime_wins(self):
        """When duplicate date/search_term rows exist, the latest request_datetime is kept."""
        daily_old = _make_daily(idx=10, request_dt=pd.Timestamp('2024-01-03 08:00:00'))
        daily_new = _make_daily(idx=99, request_dt=pd.Timestamp('2024-01-03 12:00:00'))
        daily = pd.concat([daily_old, daily_new], ignore_index=True)

        result = download_gt_data.clean_up(_make_monthly(), _make_weekly(), daily)
        # Only one row, and it should use idx=99
        assert len(result) == 1
        assert result['index'].iloc[0] == pytest.approx(99 * 80 / 100 * 100 / 100)


# ──────────────────────────────────────────────────────────────────────────────
# custom_retry
# ──────────────────────────────────────────────────────────────────────────────

class TestCustomRetry:
    """Tests for custom_retry."""

    def _make_pytrends(self, response_df=None):
        mock_pt = MagicMock()
        mock_pt.token_payload = {'timeframe': '2024-01-01 2024-01-07'}
        mock_pt.interest_over_time.return_value = (
            response_df if response_df is not None
            else pd.DataFrame({'SPY': [50], 'isPartial': [False]},
                              index=pd.to_datetime(['2024-01-01']))
        )
        return mock_pt

    def test_success_appends_to_df_list(self):
        """A successful response is processed and appended to df_list."""
        df_list, no_resp_list = [], []
        pytrends = self._make_pytrends()

        with patch('download_gt_data.time.sleep'):
            download_gt_data.custom_retry('SPY', pytrends, df_list, no_resp_list, 3)

        assert len(df_list) == 1
        assert len(no_resp_list) == 0
        assert 'index' in df_list[0].columns
        assert 'search_term' in df_list[0].columns

    def test_empty_response_appends_to_no_resp_list(self):
        """An empty DataFrame response is recorded in no_resp_list."""
        df_list, no_resp_list = [], []
        pytrends = self._make_pytrends(response_df=pd.DataFrame())

        with patch('download_gt_data.time.sleep'):
            download_gt_data.custom_retry('SPY', pytrends, df_list, no_resp_list, 3)

        assert len(df_list) == 0
        assert len(no_resp_list) == 1

    def test_request_exception_retries(self):
        """Retries after a RequestException and succeeds on second attempt."""
        df_list, no_resp_list = [], []
        good_df = pd.DataFrame(
            {'SPY': [50], 'isPartial': [False]},
            index=pd.to_datetime(['2024-01-01'])
        )
        mock_pt = MagicMock()
        mock_pt.token_payload = {'timeframe': '2024-01-01 2024-01-07'}
        mock_pt.interest_over_time.side_effect = [
            requests.exceptions.RequestException("timeout"),
            good_df,
        ]

        with patch('download_gt_data.time.sleep'):
            download_gt_data.custom_retry('SPY', mock_pt, df_list, no_resp_list, 3)

        assert len(df_list) == 1

    def test_response_error_retries(self):
        """Retries after a ResponseError and succeeds on second attempt."""
        df_list, no_resp_list = [], []
        good_df = pd.DataFrame(
            {'SPY': [50], 'isPartial': [False]},
            index=pd.to_datetime(['2024-01-01'])
        )
        mock_pt = MagicMock()
        mock_pt.token_payload = {'timeframe': '2024-01-01 2024-01-07'}
        mock_pt.interest_over_time.side_effect = [
            ResponseError("rate limited", MagicMock(status_code=429)),
            good_df,
        ]

        with patch('download_gt_data.time.sleep'):
            download_gt_data.custom_retry('SPY', mock_pt, df_list, no_resp_list, 3)

        assert len(df_list) == 1


# ──────────────────────────────────────────────────────────────────────────────
# review_past_requests
# ──────────────────────────────────────────────────────────────────────────────

class TestReviewPastRequests:
    """Tests for review_past_requests."""

    def _empty_weekly(self):
        return pd.DataFrame(columns=['search_term', 'pytrends_params'])

    def _empty_daily(self):
        return pd.DataFrame(columns=['search_term', 'pytrends_params', 'isPartial'])

    def test_current_year_always_in_year_ranges_to_do(self):
        """Current year is always re-queued even if previously completed."""
        current_year = datetime.now().year
        current_year_range = f"{current_year}-01-01 {current_year}-12-31"

        gt_weekly = pd.DataFrame([{
            'search_term': 'SPY',
            'pytrends_params': f'{{"timeframe": "{current_year_range}"}}',
        }])

        kw_yrtd, _, _ = download_gt_data.review_past_requests(
            {'SPY'}, [], gt_weekly, self._empty_daily()
        )

        assert current_year_range in kw_yrtd['SPY']

    def test_completed_past_years_excluded(self):
        """Year ranges already fetched (non-current year) are not re-queued."""
        gt_weekly = pd.DataFrame([{
            'search_term': 'SPY',
            'pytrends_params': '{"timeframe": "2020-01-01 2020-12-31"}',
        }])

        kw_yrtd, _, _ = download_gt_data.review_past_requests(
            {'SPY'}, [], gt_weekly, self._empty_daily()
        )

        assert '2020-01-01 2020-12-31' not in kw_yrtd['SPY']

    def test_partial_weeks_are_requeued(self):
        """Week ranges with any isPartial=True rows are included in weeks to do."""
        week_range = '2024-01-07 2024-01-13'
        gt_daily = pd.DataFrame([
            {
                'search_term': 'SPY',
                'pytrends_params': f'{{"timeframe": "{week_range}"}}',
                'isPartial': True,
            },
            {
                'search_term': 'SPY',
                'pytrends_params': f'{{"timeframe": "{week_range}"}}',
                'isPartial': False,
            },
        ])

        _, kw_wrtd, _ = download_gt_data.review_past_requests(
            {'SPY'}, [], self._empty_weekly(), gt_daily
        )

        assert week_range in kw_wrtd['SPY']

    def test_params_return_empty_df_mapped_by_keyword(self):
        """params_return_empty_df entries are correctly mapped to their keyword."""
        params_str = '{"kw_list": ["SPY"], "timeframe": "2024-01-07 2024-01-13"}'

        _, _, params_dict = download_gt_data.review_past_requests(
            {'SPY'}, [params_str], self._empty_weekly(), self._empty_daily()
        )

        assert '2024-01-07 2024-01-13' in params_dict['SPY']
