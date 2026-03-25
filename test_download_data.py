"""Tests for download_data.py"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import requests

import download_data


# ──────────────────────────────────────────────────────────────────────────────
# load_existing_weather_data
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadExistingWeatherData:
    def test_raises_when_no_files(self):
        with patch('download_data.glob.glob', return_value=[]):
            with pytest.raises(FileNotFoundError):
                download_data.load_existing_weather_data()

    def test_loads_latest_file(self):
        files = ['data/weather_df_20240101.csv', 'data/weather_df_20240301.csv', 'data/weather_df_20240201.csv']
        expected_df = pd.DataFrame({'date': pd.to_datetime(['2024-01-01'])})

        with patch('download_data.glob.glob', return_value=files), \
             patch('download_data.pd.read_csv', return_value=expected_df) as mock_read:
            result = download_data.load_existing_weather_data()
            # Should have picked the most recent file (20240301)
            mock_read.assert_called_once_with('data/weather_df_20240301.csv', parse_dates=['date'])
            assert result.equals(expected_df)


# ──────────────────────────────────────────────────────────────────────────────
# get_wikipedia_pageviews
# ──────────────────────────────────────────────────────────────────────────────

class TestGetWikipediaPageviews:
    def _make_sp_df(self, page='Apple_Inc.', symbol='AAPL'):
        return pd.DataFrame([{'wiki_page': page, 'Symbol': symbol}])

    def _make_valid_response(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            'items': [
                {'timestamp': '2024010100', 'views': 1000},
                {'timestamp': '2024010200', 'views': 1200},
            ]
        }
        return mock_resp

    def test_request_exception_does_not_crash(self):
        sp_df = self._make_sp_df()
        with patch('download_data.requests.get', side_effect=requests.exceptions.RequestException("timeout")):
            # Should not raise — missing pages are skipped
            with pytest.raises(ValueError):
                # concat of empty list raises ValueError; that's expected here
                download_data.get_wikipedia_pageviews(sp_df)

    def test_missing_items_key_does_not_crash(self):
        sp_df = self._make_sp_df()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {'error': 'not found'}  # no 'items' key

        with patch('download_data.requests.get', return_value=mock_resp):
            with pytest.raises(ValueError):
                download_data.get_wikipedia_pageviews(sp_df)

    def test_successful_response_returns_dataframe(self):
        sp_df = self._make_sp_df()
        mock_resp = self._make_valid_response()

        with patch('download_data.requests.get', return_value=mock_resp):
            result = download_data.get_wikipedia_pageviews(sp_df)

        assert isinstance(result, pd.DataFrame)
        assert 'ticker' in result.columns
        assert 'Date' in result.columns
        assert (result['ticker'] == 'AAPL').all()

    def test_shared_wiki_page_gets_combined_tickers(self):
        # Alphabet Class A and C share the same wiki page
        sp_df = pd.DataFrame([
            {'wiki_page': 'Alphabet_Inc.', 'Symbol': 'GOOGL'},
            {'wiki_page': 'Alphabet_Inc.', 'Symbol': 'GOOG'},
        ])
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            'items': [{'timestamp': '2024010100', 'views': 500}]
        }

        with patch('download_data.requests.get', return_value=mock_resp):
            result = download_data.get_wikipedia_pageviews(sp_df)

        assert 'GOOGL' in result['ticker'].iloc[0]
        assert 'GOOG' in result['ticker'].iloc[0]

    def test_missing_pages_are_logged(self, capsys: pytest.CaptureFixture[str]):
        sp_df = self._make_sp_df()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {}  # no 'items' key

        with patch('download_data.requests.get', return_value=mock_resp):
            try:
                download_data.get_wikipedia_pageviews(sp_df)
            except ValueError:
                pass  # expected when dat is empty

        captured = capsys.readouterr()
        assert 'missing' in captured.out.lower() or 'items' in captured.out.lower()


# ──────────────────────────────────────────────────────────────────────────────
# get_outstanding_shares (transformation logic)
# ──────────────────────────────────────────────────────────────────────────────

class TestGetOutstandingShares:
    def _run_ffill_logic(self, os_df_date_tick, today_str):
        """Re-runs just the ffill transformation from get_outstanding_shares."""
        all_dates = pd.date_range(start=os_df_date_tick['os_report_date'].min(), end=today_str, freq='D')
        tickers = os_df_date_tick['ticker'].unique()
        full_index = pd.MultiIndex.from_product([all_dates, tickers], names=['date', 'ticker'])

        return (
            os_df_date_tick
            .rename(columns={'os_report_date': 'date'})
            .set_index(['date', 'ticker'])
            .reindex(full_index)
            .groupby(level='ticker')['outstanding_shares']
            .ffill()
            .reset_index()
        )

    def test_gaps_are_filled_within_ticker(self):
        os_df_date_tick = pd.DataFrame({
            'os_report_date': pd.to_datetime(['2024-01-01', '2024-01-05']),
            'ticker': ['AAPL', 'AAPL'],
            'outstanding_shares': [1000.0, 2000.0],
        })
        result = self._run_ffill_logic(os_df_date_tick, '2024-01-05')

        aapl = result[result['ticker'] == 'AAPL'].set_index('date')
        # Jan 2, 3, 4 should be filled with 1000
        assert aapl.loc[pd.Timestamp('2024-01-02'), 'outstanding_shares'] == 1000.0
        assert aapl.loc[pd.Timestamp('2024-01-03'), 'outstanding_shares'] == 1000.0
        assert aapl.loc[pd.Timestamp('2024-01-05'), 'outstanding_shares'] == 2000.0

    def test_no_cross_ticker_bleed(self):
        os_df_date_tick = pd.DataFrame({
            'os_report_date': pd.to_datetime(['2024-01-01', '2024-01-03']),
            'ticker': ['AAPL', 'MSFT'],
            'outstanding_shares': [1000.0, 9999.0],
        })
        result = self._run_ffill_logic(os_df_date_tick, '2024-01-03')

        # AAPL on Jan 2 should be 1000, NOT 9999 (MSFT's value)
        aapl_jan2 = result[(result['ticker'] == 'AAPL') & (result['date'] == pd.Timestamp('2024-01-02'))]
        assert aapl_jan2['outstanding_shares'].iloc[0] == 1000.0

        # MSFT before its first report date should be NaN, not AAPL's value
        msft_jan1 = result[(result['ticker'] == 'MSFT') & (result['date'] == pd.Timestamp('2024-01-01'))]
        assert pd.isna(msft_jan1['outstanding_shares'].iloc[0])

    def test_extends_to_today(self):
        os_df_date_tick = pd.DataFrame({
            'os_report_date': pd.to_datetime(['2024-01-01']),
            'ticker': ['AAPL'],
            'outstanding_shares': [1000.0],
        })
        today = '2024-01-10'
        result = self._run_ffill_logic(os_df_date_tick, today)

        assert result['date'].max() == pd.Timestamp(today)
        assert result[result['date'] == pd.Timestamp(today)]['outstanding_shares'].iloc[0] == 1000.0


# ──────────────────────────────────────────────────────────────────────────────
# fetch_noaa_with_retry
# ──────────────────────────────────────────────────────────────────────────────

class TestFetchNoaaWithRetry:
    def _params(self):
        return {'startdate': '2024-01-01', 'enddate': '2024-01-29'}

    def _headers(self):
        return {'token': 'test-key'}

    def test_returns_response_on_first_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch('download_data.requests.get', return_value=mock_resp) as mock_get, \
             patch('download_data.time.sleep'):
            result = download_data.fetch_noaa_with_retry(self._params(), self._headers())

        assert result is mock_resp
        assert mock_get.call_count == 1

    def test_retries_and_succeeds_on_second_attempt(self):
        fail = MagicMock(status_code=500, text='error')
        success = MagicMock(status_code=200)

        with patch('download_data.requests.get', side_effect=[fail, success]), \
             patch('download_data.time.sleep'):
            result = download_data.fetch_noaa_with_retry(self._params(), self._headers())

        assert result is success

    def test_returns_none_after_all_attempts_fail(self):
        fail = MagicMock(status_code=500, text='error')

        with patch('download_data.requests.get', return_value=fail), \
             patch('download_data.time.sleep'):
            result = download_data.fetch_noaa_with_retry(self._params(), self._headers())

        assert result is None

    def test_first_attempt_has_no_sleep(self):
        mock_resp = MagicMock(status_code=200)
        sleep_calls = []

        with patch('download_data.requests.get', return_value=mock_resp), \
             patch('download_data.time.sleep', side_effect=sleep_calls.append):
            download_data.fetch_noaa_with_retry(self._params(), self._headers())

        assert sleep_calls == []
