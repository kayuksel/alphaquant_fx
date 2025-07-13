import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import torch.nn.functional as F

# === User-defined technical_indicator ===
def technical_indicator(ohlcv: torch.Tensor, eps = 1e-06) -> torch.Tensor:
    (n, T, f) = ohlcv.shape
    pad = lambda x, L: x if x.size(1) >= L else F.pad(x, (L - x.size(1), 0), mode='replicate')
    log_close = ohlcv[..., 3].clamp_min(eps).log()
    rts = log_close - torch.cat([log_close[:, :1], log_close[:, :-1]], dim=1)
    base_win = 20
    vol_recent = pad(rts, base_win)[:, -base_win:].std(dim=1)
    adapt = (1 + 0.2 * ((vol_recent - vol_recent.median()) / (vol_recent.median() + eps))).clamp(0.8, 1.2)
    multi_fd = torch.stack([rts[:, -1] - pad(rts, w)[:, -w] for w in [5, 10, 20]], dim=0).mean(dim=0).tanh()
    entropy = (1 + pad(rts, base_win).unfold(1, base_win, 1).std(dim=2)).log().mean(dim=1).sigmoid()
    vs = pad(ohlcv[..., 4], 60)
    dv = vs[:, 1:] - vs[:, :-1]
    avg_gain = dv.clamp_min(0).unfold(1, 14, 1).mean(dim=2)[:, -1]
    avg_loss = (-dv).clamp_min(0).unfold(1, 14, 1).mean(dim=2)[:, -1].clamp_min(eps)
    vol_exp = (vs.std(dim=1) / (vs.mean(dim=1) + eps)).clamp(0.8, 1.2)
    rsi_norm = ((100 - 100 / (1 + avg_gain / avg_loss)) / 100) ** vol_exp
    vol_adj = (vs[:, -1] / (vs.mean(dim=1) + eps)).sigmoid() * pad(vs, 14).unfold(1, 14, 1).std(dim=2)[:, -1].sigmoid()
    ens_wins = torch.tensor([10, 30, 60], device=rts.device, dtype=torch.long)
    rp = pad(rts, int(ens_wins.max().item()))
    blend = lambda x: 0.3 * x.quantile(0.25, dim=1) + 0.4 * x.quantile(0.5, dim=1) + 0.3 * x.quantile(0.75, dim=1)
    (S_list, W_list) = ([], [])
    for w in ens_wins.tolist():
        last_roll = rp.unfold(1, w, 1)[:, -1, :]
        wk = torch.exp(-0.5 * ((torch.arange(w, device=rts.device, dtype=rts.dtype) - (w - 1)) / (w * 0.5 + eps)) ** 2)
        moment = (last_roll * wk / (wk.sum() + eps)).sum(dim=1)
        m_last = last_roll.mean(dim=1)
        std_last = last_roll.std(dim=1).clamp_min(eps)
        skew = ((last_roll - m_last.unsqueeze(1)) ** 3).mean(dim=1) / std_last ** 3
        kurt = ((last_roll - m_last.unsqueeze(1)) ** 4).mean(dim=1) / std_last ** 4 - 3
        q_val = last_roll.quantile(0.05, dim=1)
        msk = last_roll < q_val.unsqueeze(1)
        cvar_est = last_roll.masked_fill(~msk, 0).sum(dim=1) / msk.sum(dim=1).clamp_min(1)
        tail = (-(cvar_est - q_val) * (1 + torch.relu(-skew))).sigmoid()
        comb = entropy * moment + (1 - entropy) * blend(last_roll) 
        comb += 0.1 * (skew.tanh() - kurt.tanh() + rsi_norm + multi_fd.tanh())
        S_list.append((-(comb * vol_adj / (1 + 0.5 * vol_recent)) ** 2).exp() * tail * multi_fd.tanh().sigmoid())
        W_list.append(moment.sigmoid() / (std_last ** 2 + eps))
    dyn_w = (-torch.stack(W_list, dim=0)).softmax(dim=0)
    multi_ens = (torch.stack(S_list, dim=0) * dyn_w).sum(dim=0)
    weights = torch.arange(rp.size(1), device=rts.device, dtype=rts.dtype).unsqueeze(0)
    wei_num = 0.7 * (-adapt.unsqueeze(1) * weights).exp() + 0.3 * (-0.05 * weights).exp()
    wei = wei_num / (wei_num.sum(dim=1, keepdim=True) + eps)
    trend = (-10 * ((rp * wei).sum(dim=1) - rp[:, -1]) ** 2).exp() 
    cross = (-(8 * (rp[:, -1] - rp[:, -1].median()) / (rts.std(dim=1) + eps))).sigmoid()
    return (multi_ens * trend * cross + eps).pow(1 / 3) * (1 + 0.1 * multi_fd)

def get_ohlcv(csv_path: str) -> torch.Tensor:
    """
    Load OHLCV from CSV into torch.Tensor shape [n_assets, n_channels, T].
    Expects columns: Gmt time, Open, High, Low, Close, Volume.
    """
    df = pd.read_csv(
        csv_path,
        parse_dates=["Gmt time"],
        date_parser=lambda s: pd.to_datetime(s, format="%d.%m.%Y %H:%M:%S.%f", utc=True)
    ).rename(columns={"Gmt time": "Date"}).set_index("Date")
    arr = df[["Open","High","Low","Close","Volume"]].values.T
    tensor = torch.from_numpy(arr).float()
    return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor


def plot_best_asset_returns(
    ohlcv,
    window_size: int = 200,
    filename: str = "strategy_returns.png"
):

    if ohlcv.ndim == 2:
        ohlcv = ohlcv.unsqueeze(0)
    n_assets, _, T = ohlcv.shape

    close = ohlcv[:, 3, :]
    log_close = torch.log(close + 1e-6)
    log_returns = torch.zeros_like(close)
    log_returns[:, 1:] = log_close[:, 1:] - log_close[:, :-1]
    simple_returns = (torch.exp(log_returns) - 1.0).cpu().numpy()

    # Slide windows
    if T < window_size:
        windows = ohlcv.permute(0, 2, 1).unsqueeze(1)
    else:
        w = ohlcv.unfold(dimension=2, size=window_size, step=1)
        windows = w.permute(0, 2, 3, 1).contiguous()

    flat_win = windows.view(-1, window_size, ohlcv.shape[1])
    flat_ind = technical_indicator(flat_win).view(-1)
    n_windows = windows.size(1)

    raw_signal = torch.full((n_assets, T), 0.5, device=flat_ind.device)
    idx = torch.arange(n_windows, device=flat_ind.device) + window_size - 1
    raw_signal[:, idx] = flat_ind.view(n_assets, n_windows)

    # Quantile transform
    sig = raw_signal[0].cpu().numpy()
    sig_q = rankdata(sig) / len(sig)
    q1, q3 = np.quantile(sig_q, 0.25), np.quantile(sig_q, 0.75)

    # Position computation
    pos = np.zeros(T, dtype=int)
    for t in range(1, T):
        if sig_q[t] > q3:
            pos[t] = 1
        elif sig_q[t] < q1:
            pos[t] = 0
        else:
            pos[t] = pos[t-1]

    pos_shifted = pos[:-1]
    rets = simple_returns[0, 1:]
    strat_rets = pos_shifted * rets
    strat_cum = np.cumprod(1 + strat_rets)
    bench_cum = np.cumprod(1 + rets)

    # === Metrics ===
    annual_factor = 252 * 24  # hourly bars
    excess_rets = strat_rets - rets
    mean_ret = np.mean(strat_rets)
    std_ret = np.std(strat_rets)
    sharpe = mean_ret / (std_ret + 1e-6) * np.sqrt(annual_factor)
    sortino = mean_ret / (np.std(strat_rets[strat_rets < 0]) + 1e-6) * np.sqrt(annual_factor)
    info_ratio = np.mean(excess_rets) / (np.std(excess_rets) + 1e-6) * np.sqrt(annual_factor)
    net_profit = strat_cum[-1] - 1.0
    max_dd = np.max(np.maximum.accumulate(strat_cum) - strat_cum) / np.max(strat_cum)
    annual_ret = (strat_cum[-1]) ** (annual_factor / len(strat_cum)) - 1

    # === Trade-Based Statistics ===
    trades = np.diff(pos)
    entry_points = np.where(trades == 1)[0] + 1
    exit_points = np.where(trades == -1)[0] + 1
    if len(exit_points) < len(entry_points):
        exit_points = np.append(exit_points, len(pos) - 1)

    trade_returns = []
    for entry, exit in zip(entry_points, exit_points):
        trade_return = np.prod(1 + strat_rets[entry:exit]) - 1
        trade_returns.append(trade_return)

    trade_returns = np.array(trade_returns)
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else -1e-6
    pl_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0
    win_rate = 100 * len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0

    # Turnover, Trades, Runtime
    num_trades = len(trade_returns)
    runtime_days = T / 24
    trades_per_day = num_trades / runtime_days
    turnover = 100 * num_trades / T

    # === Plotting ===
    times = np.arange(T)
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(24, 6),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 2]}
    )

    ax1.plot(times, sig_q, label='Quantile-Transformed Indicator', color='steelblue', linewidth=1)
    ax1.axhline(q1, color='red', linestyle='--', label='25% threshold')
    ax1.axhline(q3, color='green', linestyle='--', label='75% threshold')
    start = None
    for t in range(1, T):
        if pos[t] == 1 and pos[t-1] == 0:
            start = t
        if pos[t] == 0 and pos[t-1] == 1 and start is not None:
            ax1.axvspan(start, t, color='purple', alpha=0.1)
            start = None
    if pos[-1] == 1 and start is not None:
        ax1.axvspan(start, T, color='purple', alpha=0.1)

    ax1.set_ylabel('Indicator (Quantile)')
    ax1.legend(loc='upper right', fontsize='small')
    ax1.set_title("Signal and Position (shaded = in market)")

    ax2.plot(times[1:], bench_cum, label='Buy-and-Hold', color='gray', linewidth=1.5)
    ax2.plot(times[1:], strat_cum, label='Strategy', color='orange', linewidth=1.5)
    start = None
    for t in range(1, T):
        if pos[t] == 1 and pos[t-1] == 0:
            start = t
        if pos[t] == 0 and pos[t-1] == 1 and start is not None:
            ax2.axvspan(start, t, color='purple', alpha=0.1)
            start = None
    if pos[-1] == 1 and start is not None:
        ax2.axvspan(start, T, color='purple', alpha=0.1)

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Return')
    ax2.legend(loc='upper left', fontsize='small')
    ax2.set_title(f"Final Excess Return = {strat_cum[-1] - bench_cum[-1]:.4f}")

    # Rich metrics text
    metrics_text = (
        f"Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f} | Info Ratio: {info_ratio:.2f} | "
        f"P/L Ratio: {pl_ratio:.2f} | Win Rate: {win_rate:.1f}%\n"
        f"Net Profit: {100 * net_profit:.2f}% | Annual Return: {100 * annual_ret:.2f}% | "
        f"Max DD: {100 * max_dd:.1f}% | Avg Win: {avg_win:.4f} | Avg Loss: {avg_loss:.4f}\n"
        f"Trades/Day: {trades_per_day:.1f} | Turnover: {turnover:.1f}% | Runtime: {runtime_days:.1f} days"
    )
    fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"✅ Plot saved to {filename}")

import pdb
pdb.set_trace()
if __name__ == '__main__':
    # === Configuration: set data path and output filename here ===
    data_csv = 'AUDNZD.csv'
    output_png = 'indicator_plot.png'

    ohlcv = get_ohlcv(data_csv)
    ind_tensor = technical_indicator(ohlcv)
    plot_best_asset_returns(ohlcv)
