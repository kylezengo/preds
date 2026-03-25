"""Tests for prep_data.py"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import prep_data
from prep_data import (
    IndicatorConfig,
    calculate_rsi_long,
    calculate_rsi_wide,
    calculate_technical_indicators,
    calculate_vwap_long,
    calculate_vwap_wide,
    gen_stocks_w,
)


# ──────────────────────────────────────────────────────────────────────────────
# load_data
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadData:
    """Tests for load_data."""

    def test_raises_when_stocks_missing(self):
        """Raises FileNotFoundError when no stocks_df files exist."""
        with patch('prep_data.glob.glob', return_value=[]):
            with pytest.raises(FileNotFoundError, match="stocks_df"):
                prep_data.load_data()

    def test_raises_when_wiki_missing(self):
        """Raises FileNotFoundError when no wiki_pageviews files exist."""
        def mock_glob(pattern):
            if 'stocks_df' in pattern:
                return ['data/stocks_df_20240301.csv']
            return []

        dummy = pd.DataFrame({'Date': pd.to_datetime(['2024-01-01'])})
        with patch('prep_data.glob.glob', side_effect=mock_glob), \
             patch('prep_data.pd.read_csv', return_value=dummy):
            with pytest.raises(FileNotFoundError, match="wiki_pageviews"):
                prep_data.load_data()

    def test_raises_when_weather_missing(self):
        """Raises FileNotFoundError when no weather files exist."""
        def mock_glob(pattern):
            if 'weather' in pattern:
                return []
            return ['data/some_file_20240301.csv']

        dummy = pd.DataFrame({'Date': pd.to_datetime(['2024-01-01']), 'date': pd.to_datetime(['2024-01-01'])})
        with patch('prep_data.glob.glob', side_effect=mock_glob), \
             patch('prep_data.pd.read_csv', return_value=dummy):
            with pytest.raises(FileNotFoundError, match="weather"):
                prep_data.load_data()


# ──────────────────────────────────────────────────────────────────────────────
# calculate_rsi
# ──────────────────────────────────────────────────────────────────────────────

class TestCalculateRsi:
    """Tests for RSI calculation functions."""

    def _make_long_data(self, prices):
        return pd.DataFrame({'price': prices})

    def _make_wide_data(self, prices, ticker='SPY'):
        return pd.DataFrame({f'Adj Close_{ticker}': prices, f'Volume_{ticker}': [1] * len(prices)})

    def test_rsi_long_range_0_to_100(self):
        """RSI values should be between 0 and 100."""
        prices = [100, 101, 102, 101, 100, 99, 100, 101, 103, 104, 102, 101, 100, 99, 98]
        data = self._make_long_data(prices)
        rsi = calculate_rsi_long(data, 'price', window=5)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_wide_range_0_to_100(self):
        """RSI values should be between 0 and 100 (wide format)."""
        prices = [100, 101, 102, 101, 100, 99, 100, 101, 103, 104, 102, 101, 100, 99, 98]
        data = self._make_wide_data(prices)
        rsi = calculate_rsi_wide(data, 'Adj Close', 'SPY', window=5)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_all_gains_approaches_100(self):
        """RSI should approach 100 when prices only go up."""
        prices = list(range(100, 140))  # 40 consecutive up days
        data = self._make_long_data(prices)
        rsi = calculate_rsi_long(data, 'price', window=5)
        assert rsi.dropna().iloc[-1] > 90

    def test_rsi_all_losses_approaches_0(self):
        """RSI should approach 0 when prices only go down."""
        prices = list(range(140, 100, -1))  # 40 consecutive down days
        data = self._make_long_data(prices)
        rsi = calculate_rsi_long(data, 'price', window=5)
        assert rsi.dropna().iloc[-1] < 10

    def test_rsi_long_and_wide_agree(self):
        """Wide and long RSI should produce the same values given the same prices."""
        prices = [100, 102, 101, 103, 105, 104, 106, 107, 105, 103, 102, 104, 106, 108, 107]
        long_data = self._make_long_data(prices)
        wide_data = self._make_wide_data(prices)

        rsi_long = calculate_rsi_long(long_data, 'price', window=5)
        rsi_wide = calculate_rsi_wide(wide_data, 'Adj Close', 'SPY', window=5)

        np.testing.assert_array_almost_equal(
            rsi_long.dropna().values,
            rsi_wide.dropna().values
        )


# ──────────────────────────────────────────────────────────────────────────────
# calculate_vwap
# ──────────────────────────────────────────────────────────────────────────────

class TestCalculateVwap:
    """Tests for VWAP calculation functions."""

    def test_vwap_long_constant_price(self):
        """VWAP should equal the price when price is constant."""
        data = pd.DataFrame({'price': [100.0] * 5, 'Volume': [1000] * 5})
        vwap = calculate_vwap_long(data, 'price')
        assert (vwap == 100.0).all()

    def test_vwap_wide_constant_price(self):
        """VWAP should equal the price when price is constant (wide format)."""
        data = pd.DataFrame({'Adj Close_SPY': [200.0] * 5, 'Volume_SPY': [500] * 5})
        vwap = calculate_vwap_wide(data, 'Adj Close', 'SPY')
        assert (vwap == 200.0).all()

    def test_vwap_long_weighted_correctly(self):
        """VWAP should weight higher-volume days more heavily."""
        # Day 1: price=100, volume=1; Day 2: price=200, volume=9
        # Cumulative after day 2: (100*1 + 200*9) / (1+9) = 1900/10 = 190
        data = pd.DataFrame({'price': [100.0, 200.0], 'Volume': [1, 9]})
        vwap = calculate_vwap_long(data, 'price')
        assert vwap.iloc[-1] == pytest.approx(190.0)


# ──────────────────────────────────────────────────────────────────────────────
# calculate_technical_indicators
# ──────────────────────────────────────────────────────────────────────────────

class TestCalculateTechnicalIndicators:
    """Tests for calculate_technical_indicators."""

    def _make_data(self, n=60):
        prices = [100 + i * 0.5 + (i % 3) for i in range(n)]
        return pd.DataFrame({
            'Adj Close_SPY': prices,
            'Volume_SPY': [1_000_000] * n,
        })

    def test_does_not_mutate_input(self):
        """Input DataFrame should not be modified."""
        data = self._make_data()
        original_cols = list(data.columns)
        calculate_technical_indicators(data, IndicatorConfig())
        assert list(data.columns) == original_cols

    def test_adds_expected_columns(self):
        """Output contains all expected indicator columns."""
        result = calculate_technical_indicators(self._make_data(), IndicatorConfig())
        for col in ['RSI', 'MA_S', 'MA_L', 'MA_B', 'Bollinger_Upper', 'Bollinger_Lower',
                    'VWAP', 'short_ema', 'long_ema', 'macd_line']:
            assert col in result.columns, f"Missing column: {col}"

    def test_bollinger_upper_above_lower(self):
        """Bollinger upper band should always be >= lower band."""
        result = calculate_technical_indicators(self._make_data(), IndicatorConfig())
        valid = result.dropna(subset=['Bollinger_Upper', 'Bollinger_Lower'])
        assert (valid['Bollinger_Upper'] >= valid['Bollinger_Lower']).all()

    def test_macd_line_is_short_minus_long_ema(self):
        """macd_line should equal short_ema - long_ema."""
        result = calculate_technical_indicators(self._make_data(), IndicatorConfig())
        expected = result['short_ema'] - result['long_ema']
        pd.testing.assert_series_equal(result['macd_line'], expected, check_names=False)


# ──────────────────────────────────────────────────────────────────────────────
# gen_stocks_w
# ──────────────────────────────────────────────────────────────────────────────

class TestGenStocksW:
    """Tests for gen_stocks_w."""

    def _make_stocks(self, tickers=('SPY', 'AAPL', 'GOOG', 'GOOGL')):
        rows = []
        for ticker in tickers:
            for i in range(3):
                rows.append({
                    'Date': pd.Timestamp(f'2024-01-0{i+1}'),
                    'ticker': ticker,
                    'Open': 100.0, 'High': 105.0, 'Low': 95.0,
                    'Close': 102.0, 'Adj Close': 101.0, 'Volume': 1_000_000,
                })
        return pd.DataFrame(rows)

    def _make_wiki(self, tickers=('SPY', 'AAPL', 'GOOG', 'GOOGL')):
        rows = []
        for ticker in tickers:
            for i in range(3):
                rows.append({
                    'Date': pd.Timestamp(f'2024-01-0{i+1}'),
                    'ticker': ticker,
                    'views_prev_day': 1000,
                })
        return pd.DataFrame(rows)

    def test_duplicate_tickers_dropped(self):
        """GOOG, FOX, and NWS should be dropped from the output."""
        stocks = self._make_stocks(tickers=('SPY', 'AAPL', 'GOOG', 'GOOGL', 'FOX', 'FOXA'))
        wiki = self._make_wiki(tickers=('SPY', 'AAPL', 'GOOG', 'GOOGL', 'FOX', 'FOXA'))
        result = gen_stocks_w('AAPL', stocks, wiki)

        cols = result.columns.tolist()
        assert not any('_GOOG' in c and not '_GOOGL' in c for c in cols)
        assert not any(c.endswith('_FOX') for c in cols)
        assert any('_GOOGL' in c for c in cols)
        assert any('_FOXA' in c for c in cols)

    def test_output_is_pivoted_wide(self):
        """Output should have one row per date (wide format)."""
        stocks = self._make_stocks(tickers=('SPY', 'AAPL'))
        wiki = self._make_wiki(tickers=('SPY', 'AAPL'))
        result = gen_stocks_w('AAPL', stocks, wiki)

        assert 'Date' in result.columns
        assert len(result) == 3  # 3 dates

    def test_does_not_mutate_input(self):
        """Input stocks_df should not be modified."""
        stocks = self._make_stocks(tickers=('SPY', 'AAPL'))
        wiki = self._make_wiki(tickers=('SPY', 'AAPL'))
        original_len = len(stocks)
        gen_stocks_w('AAPL', stocks, wiki)
        assert len(stocks) == original_len


# ──────────────────────────────────────────────────────────────────────────────
# prep_data (Target and streak logic)
# ──────────────────────────────────────────────────────────────────────────────

class TestPrepDataTargetAndStreaks:
    """Tests for Target column and streak logic in prep_data."""

    def _make_inputs(self, prices):
        """Build minimal inputs for prep_data given a list of SPY Adj Close prices."""
        dates = pd.date_range('2024-01-01', periods=len(prices), freq='B')
        stocks = pd.DataFrame({
            'Date': dates, 'ticker': 'SPY',
            'Open': prices, 'High': prices, 'Low': prices,
            'Close': prices, 'Adj Close': prices, 'Volume': [1_000_000] * len(prices),
        })
        wiki = pd.DataFrame({'Date': dates, 'ticker': 'SPY', 'views': 1000})
        ffr = pd.DataFrame({'Date': dates, 'federal_funds_rate': 5.0})
        weather = pd.DataFrame({
            'date': dates,
            'high_temp_nyc': 20.0, 'low_temp_nyc': 10.0,
            'precipitation_PRCP_nyc': 0.0, 'precipitation_SNOW_nyc': 0.0,
        })
        gt = pd.DataFrame({'date': dates, 'search_term': 'SPY', 'index': 50.0})
        return stocks, wiki, ffr, weather, gt

    def test_target_is_1_when_next_day_up(self):
        """Target=1 when the next day's price is higher."""
        prices = [100.0, 105.0, 103.0] + [103.0] * 60
        stocks, wiki, ffr, weather, gt = self._make_inputs(prices)
        result = prep_data.prep_data(stocks, wiki, ffr, weather, gt, IndicatorConfig())
        # Day 0: price=100, next=105 → up → Target=1
        assert result.loc[result['Adj Close_SPY'] == 100.0, 'Target'].iloc[0] == 1

    def test_target_is_0_when_next_day_down(self):
        """Target=0 when the next day's price is lower."""
        prices = [105.0, 100.0] + [100.0] * 61
        stocks, wiki, ffr, weather, gt = self._make_inputs(prices)
        result = prep_data.prep_data(stocks, wiki, ffr, weather, gt, IndicatorConfig())
        # Day 0: price=105, next=100 → down → Target=0
        assert result.loc[result['Adj Close_SPY'] == 105.0, 'Target'].iloc[0] == 0

    def test_streak_resets_on_direction_change(self):
        """Streak counter should reset to 1 when direction changes."""
        # 3 up days, then 1 down day
        prices = [100.0, 101.0, 102.0, 103.0, 102.0] + [102.0] * 58
        stocks, wiki, ffr, weather, gt = self._make_inputs(prices)
        result = prep_data.prep_data(stocks, wiki, ffr, weather, gt, IndicatorConfig())
        result = result.sort_values('Date').reset_index(drop=True)
        # After the direction change on day 4 (down), streak should be 1
        assert result.loc[4, 'streak0'] == 1

    def test_streak_increments_on_same_direction(self):
        """Streak counter should increment on consecutive same-direction days."""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0] + [104.0] * 58
        stocks, wiki, ffr, weather, gt = self._make_inputs(prices)
        result = prep_data.prep_data(stocks, wiki, ffr, weather, gt, IndicatorConfig())
        result = result.sort_values('Date').reset_index(drop=True)
        # Days 1-4 are all up, streak1 should be increasing
        assert result.loc[2, 'streak1'] > result.loc[1, 'streak1']
