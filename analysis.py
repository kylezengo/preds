"""Model edge analysis and statistical significance testing."""

import numpy as np
import pandas as pd
from scipy import stats


def _sharpe(returns, periods_per_year=252):
    """Annualized Sharpe ratio assuming 0 risk-free rate."""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return (r.mean() / r.std()) * np.sqrt(periods_per_year)


def _max_drawdown(returns):
    """Maximum peak-to-trough drawdown of a return series."""
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return drawdown.min()


def _win_loss_ratio(oos):
    """
    Average return on correct trade-day calls vs. average loss on incorrect ones.
    Only meaningful when the model actually changes its signal (i.e., makes a trade).
    """
    trade_days = oos[oos['Signal'].diff().abs() > 0].dropna(subset=['Strategy_Return'])
    if len(trade_days) < 2:
        return np.nan
    wins = trade_days.loc[trade_days['Signal'] == trade_days['Target'], 'Strategy_Return']
    losses = trade_days.loc[trade_days['Signal'] != trade_days['Target'], 'Strategy_Return']
    if len(wins) == 0 or len(losses) == 0:
        return np.nan
    return wins.mean() / abs(losses.mean())


def edge_report(strat_bds, mod_mod_dic=None, initial_train_period=0, ticker=None, significance_level=0.05):
    """
    Analyze model edge vs base rate with statistical significance testing.

    Metrics per strategy:
        Accuracy     — % of out-of-sample days predicted correctly
        vs Base      — accuracy minus the base rate (% up days)
        Trades       — number of position changes in the out-of-sample period
        Trade Acc    — accuracy specifically on days the model changes its signal
        Sharpe       — annualized Sharpe ratio of Strategy_Return (0 risk-free rate)
        Max DD       — maximum peak-to-trough drawdown of cumulative Strategy_Return
        Win/Loss     — avg return on correct trade-day calls / avg loss on incorrect ones
        Acc p-val      — one-sided binomial test: is accuracy significantly above base rate?
        Sig          — * if Acc p-val < significance_level

    Parameters:
        strat_bds (dict): Dict of strategy name -> backtested DataFrame (from backtest_strategy).
        mod_mod_dic (dict): Dict of ensemble name -> DataFrame (from build_ensemble). Optional.
        initial_train_period (int): Index where out-of-sample predictions begin.
        ticker (str): Ticker label for the report header. Optional.
        significance_level (float): Acc p-val threshold for significance marker. Default 0.05.

    Returns:
        DataFrame: One row per strategy with all metrics. Sorted by Sharpe descending.
    """
    hold_oos = strat_bds['Hold'].iloc[initial_train_period:].dropna(subset=['Target'])
    base_rate = hold_oos['Target'].mean()
    n = len(hold_oos)
    date_start = hold_oos['Date'].iloc[0].date()
    date_end = hold_oos['Date'].iloc[-1].date()

    label = f"{ticker + ' | ' if ticker else ''}{date_start} to {date_end} | {n} days"
    print(f"{'=' * (len(label) + 8)}")
    print(f"=== {label} ===")
    print(f"{'=' * (len(label) + 8)}")
    print(f"Base rate (% up days): {base_rate:.1%}\n")

    rows = []

    all_results = list(strat_bds.items())
    if mod_mod_dic:
        all_results += list(mod_mod_dic.items())

    for name, df in all_results:
        oos = df.iloc[initial_train_period:].dropna(subset=['Signal', 'Target'])
        if len(oos) == 0:
            continue

        n_correct = int((oos['Signal'] == oos['Target']).sum())
        accuracy = n_correct / len(oos)
        trades = int(oos['Signal'].diff().abs().sum())

        trade_days = oos[oos['Signal'].diff().abs() > 0]
        trade_acc = (trade_days['Signal'] == trade_days['Target']).mean() if len(trade_days) > 1 else np.nan

        p_value = stats.binomtest(n_correct, len(oos), base_rate, alternative='greater').pvalue if trades > 1 else np.nan

        sharpe = _sharpe(oos['Strategy_Return']) if 'Strategy_Return' in oos.columns else np.nan
        max_dd = _max_drawdown(oos['Strategy_Return']) if 'Strategy_Return' in oos.columns else np.nan
        wl = _win_loss_ratio(oos) if 'Strategy_Return' in oos.columns else np.nan

        rows.append({
            'Strategy':  name,
            'Accuracy':  accuracy,
            'vs Base':   accuracy - base_rate,
            'Trades':    trades,
            'Trade Acc': trade_acc,
            'Sharpe':    sharpe,
            'Max DD':    max_dd,
            'Win/Loss':  wl,
            'Acc p-val':   p_value,
            'Sig':       '*' if (not np.isnan(p_value) and p_value < significance_level) else '-',
        })

    result = pd.DataFrame(rows).sort_values('Sharpe', ascending=False).reset_index(drop=True)

    # Print formatted table
    col_w = 20
    header = (
        f"{'Strategy':<{col_w}} {'Accuracy':>9} {'vs Base':>8} {'Trades':>7} "
        f"{'Trade Acc':>10} {'Sharpe':>7} {'Max DD':>8} {'Win/Loss':>9} {'Acc p-val':>8} {'':>3}"
    )
    print(header)
    print('-' * len(header))
    for _, row in result.iterrows():
        def fmt(v, width, fmt_str):
            s = format(v, fmt_str) if not (isinstance(v, float) and np.isnan(v)) else 'nan'
            return s.rjust(width)

        print(
            f"{row['Strategy']:<{col_w}} "
            f"{fmt(row['Accuracy'],   8, '.1%')} "
            f"{fmt(row['vs Base'],    7, '+.1%')}  "
            f"{int(row['Trades']):>6}  "
            f"{fmt(row['Trade Acc'], 8, '.1%')}  "
            f"{fmt(row['Sharpe'],    6, '.2f')}  "
            f"{fmt(row['Max DD'],    7, '.1%')}  "
            f"{fmt(row['Win/Loss'],  8, '.2f')}  "
            f"{fmt(row['Acc p-val'],   7, '.3f')} "
            f"{'*' if row['Sig'] == '*' else '-'}"
        )

    return result
