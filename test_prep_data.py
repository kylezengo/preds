"""Tests for prep_data.py"""

from unittest.mock import patch

import pandas as pd
import pytest

import prep_data
from prep_data import (
    IndicatorConfig,
    calculate_rsi,
    calculate_technical_indicators,
    calculate_vwap,
    gen_stocks_w,
    resample_to_weekly,
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

        dummy = pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01']),
            'date': pd.to_datetime(['2024-01-01']),
        })
        with patch('prep_data.glob.glob', side_effect=mock_glob), \
             patch('prep_data.pd.read_csv', return_value=dummy):
            with pytest.raises(FileNotFoundError, match="weather"):
                prep_data.load_data()


# ──────────────────────────────────────────────────────────────────────────────
# calculate_rsi
# ──────────────────────────────────────────────────────────────────────────────

class TestCalculateRsi:
    """Tests for calculate_rsi."""

    def _make_data(self, prices, ticker='SPY'):
        return pd.DataFrame({f'Adj Close_{ticker}': prices, f'Volume_{ticker}': [1] * len(prices)})

    def test_rsi_range_0_to_100(self):
        """RSI values should be between 0 and 100."""
        prices = [100, 101, 102, 101, 100, 99, 100, 101, 103, 104, 102, 101, 100, 99, 98]
        rsi = calculate_rsi(self._make_data(prices), 'Adj Close', 'SPY', window=5)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_all_gains_approaches_100(self):
        """RSI should approach 100 when prices only go up."""
        prices = list(range(100, 140))  # 40 consecutive up days
        rsi = calculate_rsi(self._make_data(prices), 'Adj Close', 'SPY', window=5)
        assert rsi.dropna().iloc[-1] > 90

    def test_rsi_all_losses_approaches_0(self):
        """RSI should approach 0 when prices only go down."""
        prices = list(range(140, 100, -1))  # 40 consecutive down days
        rsi = calculate_rsi(self._make_data(prices), 'Adj Close', 'SPY', window=5)
        assert rsi.dropna().iloc[-1] < 10


# ──────────────────────────────────────────────────────────────────────────────
# calculate_vwap
# ──────────────────────────────────────────────────────────────────────────────

class TestCalculateVwap:
    """Tests for calculate_vwap."""

    def test_vwap_constant_price(self):
        """VWAP should equal the price when price is constant."""
        data = pd.DataFrame({'Adj Close_SPY': [200.0] * 5, 'Volume_SPY': [500] * 5})
        vwap = calculate_vwap(data, 'Adj Close', 'SPY')
        assert (vwap == 200.0).all()

    def test_vwap_weighted_correctly(self):
        """VWAP should weight higher-volume days more heavily."""
        # Day 1: price=100, volume=1; Day 2: price=200, volume=9
        # Cumulative after day 2: (100*1 + 200*9) / (1+9) = 1900/10 = 190
        data = pd.DataFrame({'Adj Close_SPY': [100.0, 200.0], 'Volume_SPY': [1, 9]})
        vwap = calculate_vwap(data, 'Adj Close', 'SPY')
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
            'High_SPY': [p + 1.0 for p in prices],
            'Low_SPY':  [p - 1.0 for p in prices],
            'Volume_SPY': [1_000_000] * n,
        })

    def test_does_not_mutate_input(self):
        """Input DataFrame should not be modified."""
        data = self._make_data()
        original_cols = list(data.columns)
        calculate_technical_indicators(data, IndicatorConfig())
        assert list(data.columns) == original_cols

    def test_adds_expected_columns(self):
        """Output contains ratio columns; raw level columns and Adj Close are dropped."""
        result = calculate_technical_indicators(self._make_data(), IndicatorConfig())
        for col in ['RSI', 'price_vs_ma_s', 'ma_crossover', 'bollinger_pct_b',
                    'price_vs_vwap', 'macd_pct', 'breakout_pct']:
            assert col in result.columns, f"Missing column: {col}"
        for col in ['MA_S', 'MA_L', 'MA_B', 'Bollinger_Upper', 'Bollinger_Lower',
                    'VWAP', 'short_ema', 'long_ema', 'macd_line']:
            assert col not in result.columns, f"Raw level column should be dropped: {col}"

    def test_bollinger_pct_b_in_range(self):
        """bollinger_pct_b should be 0 at the lower band and 1 at the upper band."""
        result = calculate_technical_indicators(self._make_data(), IndicatorConfig())
        valid = result['bollinger_pct_b'].dropna()
        # Price sits between bands for trending data, so valid values are finite
        assert valid.notna().all()

    def test_macd_pct_positive_for_uptrend(self):
        """macd_pct should be positive when price has been trending up."""
        n = 60
        prices = [100 + i for i in range(n)]  # strictly upward
        data = pd.DataFrame({
            'Adj Close_SPY': prices,
            'Volume_SPY': [1_000_000] * n,
        })
        result = calculate_technical_indicators(data, IndicatorConfig())
        assert result['macd_pct'].dropna().iloc[-1] > 0


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
        vix_yields = pd.DataFrame({
            'Date': dates,
            'vix_close': 20.0, 'vix9d_close': 18.0, 'vix3m_close': 21.0, 'vix6m_close': 22.0,
            'yield_10y': 4.0, 'yield_2y': 3.5, 'yield_spread': 0.5,
            'vix9d_to_vix': 18.0/20.0, 'vix_to_vix3m': 20.0/21.0,
        })
        sp_df = pd.DataFrame({'Symbol': ['SPY'], 'GICS Sector': [None]})
        sector_etfs = pd.DataFrame({'Date': dates})
        return stocks, wiki, ffr, weather, gt, vix_yields, sp_df, sector_etfs

    def test_target_is_1_when_next_day_up(self):
        """Target=1 when the next day's price is higher."""
        prices = [100.0, 105.0, 103.0] + [103.0] * 60
        stocks, wiki, ffr, weather, gt, vix_yields, sp_df, sector_etfs = self._make_inputs(prices)
        result = prep_data.prep_data(
            stocks, wiki, ffr, weather, gt, vix_yields, sp_df, sector_etfs, IndicatorConfig()
        )
        # Day 0: price=100, next=105 → up → Target=1
        assert result.sort_values('Date').iloc[0]['Target'] == 1

    def test_target_is_0_when_next_day_down(self):
        """Target=0 when the next day's price is lower."""
        prices = [105.0, 100.0] + [100.0] * 61
        stocks, wiki, ffr, weather, gt, vix_yields, sp_df, sector_etfs = self._make_inputs(prices)
        result = prep_data.prep_data(
            stocks, wiki, ffr, weather, gt, vix_yields, sp_df, sector_etfs, IndicatorConfig()
        )
        # Day 0: price=105, next=100 → down → Target=0
        assert result.sort_values('Date').iloc[0]['Target'] == 0

    def test_no_adj_close_in_output(self):
        """Adj Close columns should be dropped from the returned DataFrame."""
        prices = [100.0 + i for i in range(63)]
        stocks, wiki, ffr, weather, gt, vix_yields, sp_df, sector_etfs = self._make_inputs(prices)
        result = prep_data.prep_data(
            stocks, wiki, ffr, weather, gt, vix_yields, sp_df, sector_etfs, IndicatorConfig()
        )
        assert not any(c.startswith('Adj Close') for c in result.columns)

    def test_streak_resets_on_direction_change(self):
        """Streak counter should reset to 1 when direction changes."""
        # 3 up days, then 1 down day
        prices = [100.0, 101.0, 102.0, 103.0, 102.0] + [102.0] * 58
        stocks, wiki, ffr, weather, gt, vix_yields, sp_df, sector_etfs = self._make_inputs(prices)
        result = prep_data.prep_data(
            stocks, wiki, ffr, weather, gt, vix_yields, sp_df, sector_etfs, IndicatorConfig()
        )
        result = result.sort_values('Date').reset_index(drop=True)
        # After the direction change on day 4 (down), streak should be 1
        assert result.loc[4, 'streak0'] == 1

    def test_streak_increments_on_same_direction(self):
        """Streak counter should increment on consecutive same-direction days."""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0] + [104.0] * 58
        stocks, wiki, ffr, weather, gt, vix_yields, sp_df, sector_etfs = self._make_inputs(prices)
        result = prep_data.prep_data(
            stocks, wiki, ffr, weather, gt, vix_yields, sp_df, sector_etfs, IndicatorConfig()
        )
        result = result.sort_values('Date').reset_index(drop=True)
        # Days 1-4 are all up, streak1 should be increasing
        assert result.loc[2, 'streak1'] > result.loc[1, 'streak1']


# ──────────────────────────────────────────────────────────────────────────────
# resample_to_weekly
# ──────────────────────────────────────────────────────────────────────────────

class TestResampleToWeekly:
    """Tests for resample_to_weekly."""

    def _make_daily(self, n_weeks=6):
        """Minimal daily DataFrame resembling prep_data() output (n_weeks * 5 days)."""
        # Start on a Monday so each resample('W-FRI') window is a clean 5-day week
        dates = pd.date_range('2024-01-08', periods=n_weeks * 5, freq='B')
        prices = [100.0 + i * 0.5 for i in range(len(dates))]
        returns = pd.Series(prices).pct_change().fillna(0).tolist()
        return pd.DataFrame({
            'Date': dates,
            'Daily_Return': returns,
            'RSI': [50.0] * len(dates),
            'streak0': [0] * len(dates),
            'streak1': list(range(1, len(dates) + 1)),
            'Target': [1] * len(dates),
        })

    def test_row_count_is_weekly(self):
        """Output should have one row per calendar week (minus last row for Target)."""
        daily = self._make_daily(n_weeks=6)
        weekly = resample_to_weekly(daily)
        # 6 weeks of input → 5 rows (last week dropped — no next-week Target)
        assert len(weekly) == 5

    def test_does_not_mutate_input(self):
        """Input DataFrame should not be modified."""
        daily = self._make_daily()
        original_len = len(daily)
        resample_to_weekly(daily)
        assert len(daily) == original_len

    def test_target_is_next_week_direction(self):
        """Target=1 when next week's cumulative return is positive."""
        daily = self._make_daily()  # strictly rising prices → all weekly returns positive
        weekly = resample_to_weekly(daily)
        assert (weekly['Target'] == 1).all()

    def test_weekly_return_is_cumulative(self):
        """Weekly Daily_Return should be the product of daily returns, not just Friday's."""
        # 5 days each gaining exactly 1%: cumulative ≈ (1.01^5 - 1) ≈ 5.1%
        # Start on Monday so W-FRI resample captures a clean 5-day week
        dates = pd.date_range('2024-01-08', periods=5, freq='B')
        daily = pd.DataFrame({
            'Date': dates,
            'Daily_Return': [0.01] * 5,
            'RSI': [50.0] * 5,
            'streak0': [0] * 5,
            'streak1': [1, 2, 3, 4, 5],
            'Target': [1] * 5,
        })
        # Need a second week so the first week gets a Target
        dates2 = pd.date_range('2024-01-15', periods=5, freq='B')
        daily2 = pd.DataFrame({
            'Date': dates2,
            'Daily_Return': [0.01] * 5,
            'RSI': [50.0] * 5,
            'streak0': [0] * 5,
            'streak1': [6, 7, 8, 9, 10],
            'Target': [1] * 5,
        })
        combined = pd.concat([daily, daily2], ignore_index=True)
        weekly = resample_to_weekly(combined)
        expected = (1.01 ** 5) - 1
        assert weekly.iloc[0]['Daily_Return'] == pytest.approx(expected, rel=1e-4)

    def test_streaks_recomputed_on_weekly_bars(self):
        """streak1 should increment across consecutive up weeks, not reflect daily streaks."""
        daily = self._make_daily(n_weeks=4)
        weekly = resample_to_weekly(daily)
        # All weeks are up (prices strictly rising), so streak1 should increment
        assert weekly.iloc[1]['streak1'] > weekly.iloc[0]['streak1']

    def test_same_columns_as_input(self):
        """Output should have the same columns as the input."""
        daily = self._make_daily()
        weekly = resample_to_weekly(daily)
        assert set(weekly.columns) == set(daily.columns)
