import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import torch.nn.functional as F

# ===============================================================
#  PLACEHOLDER: YOU WILL PASTE YOUR TWO INDICATORS HERE
# ===============================================================

def technical_indicator_crypto(ohlcv: torch.Tensor, eps = 1e-06) -> torch.Tensor:
    (n, T, _) = ohlcv.shape
    logc = ohlcv[..., 3].clamp_min(eps).log()
    rts = logc - torch.cat([logc[:, :1], logc[:, :-1]], dim=1)
    win = lambda x, k: x.unfold(1, k, 1) if x.size(1) >= k else F.pad(x, (k - x.size(1), 0)).unfold(1, k, 1)
    w20 = torch.exp(-0.05 * torch.arange(19, -1, -1, device=rts.device))
    vol_recent = torch.sqrt((w20 * rts[:, -20:] ** 2).sum(1) / w20.sum())
    vr_long = rts.std(1)
    vol_exp = 1 + 0.5 * vol_recent / (vr_long + eps)
    adapt_base = (1 + 0.2 * (vol_recent - vr_long) / (vr_long + eps)).clamp(0.8, 1.2)
    w20_std = win(rts, 20)
    entropy = w20_std.std(2).mean(1).sigmoid()
    regime = (entropy > 0.5).float() * 1.2 + (entropy <= 0.5).float() * 0.6
    adapt = adapt_base * regime
    last_roll = w20_std[:, -1]
    trim = torch.sort(last_roll, 1)[0][:, 5:-5]
    skew = ((trim - trim.mean(1, True)) ** 3).mean(1) / (trim.std(1, False) + eps) ** 3
    dv = ohlcv[..., 4][:, 1:] - ohlcv[..., 4][:, :-1]
    avg_gain = win(dv.clamp_min(0), 14).mean(2)[:, -1]
    avg_loss = win((-dv).clamp_min(0), 14).mean(2)[:, -1].clamp_min(eps)
    rsi_norm = ((100 - 100 / (1 + avg_gain / avg_loss)) / 100) ** (vol_exp * (1 + 0.4 * skew.tanh()))
    vwap3 = (ohlcv[..., 3] * ohlcv[..., 4]).unfold(1, 3, 1).sum(2) / (ohlcv[..., 4].unfold(1, 3, 1).sum(2) + eps)
    vwap_dist = torch.tanh((rts[:, -1] - vwap3[:, -1]) / (vol_recent + eps)).sigmoid()
    wk = torch.exp(-0.5 * ((torch.arange(20, device=rts.device) - 19) / (20 * (0.5 + entropy[:, None]))) ** 2)
    moment = (last_roll * wk / wk.sum(1, True)).sum(1)
    blend = 0.3 * last_roll.quantile(0.25, dim=1) + 0.4 * last_roll.quantile(0.5, dim=1) + 0.3 * last_roll.quantile(0.75, dim=1)
    multi = entropy * moment + (1 - entropy) * blend
    ql = last_roll.quantile(0.05, dim=1)
    qh = last_roll.quantile(0.95, dim=1)
    tail = (0.6 * (-ql * (1 + F.relu(-skew))).sigmoid() + 0.4 * ((ql < -2 * vol_recent) & (qh > 2 * vol_recent)).float()).sigmoid() * (qh - ql).sigmoid()
    micro = win(rts, 3).std(2)[:, -1]
    vol_adj = (ohlcv[..., 4][:, -1] / (ohlcv[..., 4].mean(1) + eps)).sigmoid() * win(dv, 14).std(2)[:, -1].sigmoid() * (micro * 5).sigmoid()
    S = (-(multi * vol_adj / (1 + 0.5 * vol_recent)) ** 2).exp() * tail
    widx = torch.arange(20, device=rts.device)
    wei = 0.7 * (-adapt[:, None] * widx).exp() + 0.3 * (-0.05 * widx).exp()
    wei = wei / wei.sum(1, True)
    vwap10 = (ohlcv[..., 3] * ohlcv[..., 4]).unfold(1, 10, 1).sum(2) / (ohlcv[..., 4].unfold(1, 10, 1).sum(2) + eps)
    trend = (-10 * ((last_roll * wei).sum(1) - last_roll[:, -1]) ** 2).exp() * vwap_dist * (1 + 0.15 * ((ohlcv[..., 3][:, -1] - vwap10[:, -1]) / (vwap10[:, -1] + eps)).sign())
    cross = (-(8 * last_roll[:, -1] / (rts.std(1) + eps))).sigmoid()
    return S * trend * cross

def technical_indicator_fx(ohlcv: torch.Tensor, eps = 1e-06) -> torch.Tensor:
    (n, T, f) = ohlcv.shape
    pad = lambda x, L: x if x.size(1) >= L else F.pad(x, (L - x.size(1), 0), mode='replicate')
    log_close = ohlcv[..., 3].clamp_min(eps).log()
    rts = log_close - torch.cat([log_close[:, :1], log_close[:, :-1]], dim=1)
    base_win = 20
    rts_pad = pad(rts, base_win)
    mad = lambda x: (x - x.median(dim=1, keepdim=True)[0]).abs().median(dim=1)[0]
    vol_recent = mad(rts_pad[:, -base_win:]).clamp_min(eps)
    vol_med = vol_recent.median()
    adapt = 0.8 + 0.4 * ((vol_recent - vol_med) / (vol_med + eps)).sigmoid()
    multi_fd = torch.stack([rts[:, -1] - pad(rts, w)[:, -w] for w in [5, 10, 20]], dim=0).mean(dim=0).tanh()
    entropy = pad(rts, base_win).unfold(1, base_win, 1).std(dim=2).add(1).log().mean(dim=1).sigmoid()
    vs = pad(ohlcv[..., 4], 60)
    dv = vs[:, 1:] - vs[:, :-1]
    avg_gain = dv.clamp_min(0).unfold(1, 14, 1).mean(dim=2)[:, -1]
    avg_loss = dv.clamp_max(0).abs().unfold(1, 14, 1).mean(dim=2)[:, -1].clamp_min(eps)
    vol_mean = vs.mean(dim=1).add(eps)
    vol_std = vs.std(dim=1)
    vol_exp = (vol_std / vol_mean).clamp(0.8, 1.2)
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    rsi_norm = (rsi / 100) ** vol_exp
    vol_adj = vs[:, -1].div(vol_mean).sigmoid() * pad(vs, 14).unfold(1, 14, 1).std(dim=2)[:, -1].sigmoid()
    multi_scale_vol = torch.stack([pad(rts, w).unfold(1, w, 1).std(dim=2)[:, -1] for w in [20, 50, 100]], dim=0).mean(dim=0).tanh()
    ens_wins = torch.tensor([10, 30, 60], device=rts.device, dtype=torch.long)
    rp = pad(rts, int(ens_wins.max().item()))
    def adaptive_blend(x):
        (q25, q50, q75) = (x.quantile(0.25, dim=1), x.quantile(0.5, dim=1), x.quantile(0.75, dim=1))
        qs = torch.stack([q25, q50, q75], dim=0)
        diff = (qs - q50.unsqueeze(0)).abs()
        att = (-5 * diff).softmax(dim=0)
        return (att * qs).sum(dim=0)
    (S_list, W_list) = ([], [])
    for w in ens_wins.tolist():
        last_roll = rp.unfold(1, w, 1)[:, -1, :]
        idx = torch.arange(w, device=rts.device, dtype=rts.dtype).unsqueeze(0)
        wk = (-0.5 * ((idx - (w - 1)) / (w * 0.5 * adapt.unsqueeze(1) + eps)) ** 2).exp()
        moment = (last_roll * wk).sum(dim=1) / (wk.sum(dim=1, keepdim=False) + eps)
        m_last = last_roll.mean(dim=1)
        std_last = last_roll.std(dim=1).clamp_min(eps)
        skew = ((last_roll - m_last.unsqueeze(1)) ** 3).mean(dim=1) / std_last ** 3
        kurt = ((last_roll - m_last.unsqueeze(1)) ** 4).mean(dim=1) / std_last ** 4 - 3
        q_adj = 0.05 + 0.02 * ((adapt - 0.8) / 0.4)
        q_val = last_roll.quantile(q_adj.mean().item(), dim=1)
        msk = last_roll < q_val.unsqueeze(1)
        cvar_est = last_roll.masked_fill(~msk, 0).sum(dim=1) / msk.sum(dim=1).clamp_min(1)
        tail = (-(cvar_est - q_val) * (1 + (-skew).clamp_min(0))).sigmoid()
        comb = entropy * moment + (1 - entropy) * adaptive_blend(last_roll)
        comb = comb + 0.1 * (skew.tanh() - kurt.tanh() + rsi_norm + multi_fd)
        S = (-(comb * vol_adj / (1 + 0.5 * vol_recent)) ** 2).exp() * tail * multi_fd.tanh().sigmoid()
        err = (last_roll[:, -1] - moment).abs()
        att = (-10 * err).sigmoid()
        W = moment.sigmoid() / (std_last ** 2 + eps) * att
        S_list.append(S)
        W_list.append(W)
    dyn_w = (-torch.stack(W_list, dim=0)).softmax(dim=0)
    multi_ens = (torch.stack(S_list, dim=0) * dyn_w).sum(dim=0)
    weights = torch.arange(rp.size(1), device=rts.device, dtype=rts.dtype).unsqueeze(0)
    wei_num = 0.7 * (-adapt.unsqueeze(1) * weights).exp() + 0.3 * (-0.05 * weights).exp()
    wei = wei_num / (wei_num.sum(dim=1, keepdim=True) + eps)
    trend = (-10 * ((rp * wei).sum(dim=1) - rp[:, -1]) ** 2).exp()
    cross = (-(8 * (rp[:, -1] - rp[:, -1].median()) / rts.std(dim=1).clamp_min(eps))).sigmoid()
    return (multi_ens * trend * cross + eps).pow(1 / 3) * (1 + 0.1 * multi_fd + 0.05 * multi_scale_vol)



# ===============================================================
#  LOAD OHLCV
# ===============================================================

def get_ohlcv(csv_path: str) -> torch.Tensor:
    """
    Load OHLCV from CSV into torch.Tensor shape [1, 5, T].
    Columns required: Gmt time, Open, High, Low, Close, Volume.
    """
    df = pd.read_csv(
        csv_path,
        parse_dates=["Gmt time"],
        date_parser=lambda s: pd.to_datetime(s, format="%d.%m.%Y %H:%M:%S.%f", utc=True)
    )

    df = df.rename(columns={"Gmt time": "Date"}).set_index("Date")
    arr = df[["Open", "High", "Low", "Close", "Volume"]].values.T
    tensor = torch.from_numpy(arr).float()
    return tensor.unsqueeze(0)



# ===============================================================
#  DUAL-INDICATOR PLOT FUNCTION
# ===============================================================

def plot_two_indicators(
    ohlcv,
    indicator_crypto,
    indicator_fx,
    window_size=200,
    filename="dual_indicator_plot.png"
):
    if ohlcv.ndim == 2:
        ohlcv = ohlcv.unsqueeze(0)

    n_assets, _, T = ohlcv.shape

    # === Compute return series ===
    close = ohlcv[:, 3, :]
    log_close = torch.log(close + 1e-6)
    log_returns = torch.zeros_like(close)
    log_returns[:, 1:] = log_close[:, 1:] - log_close[:, :-1]
    simple_returns = (torch.exp(log_returns) - 1.0).cpu().numpy()[0]

    # === Sliding windows ===
    if T < window_size:
        windows = ohlcv.permute(0, 2, 1).unsqueeze(1)
    else:
        w = ohlcv.unfold(dimension=2, size=window_size, step=1)
        windows = w.permute(0, 2, 3, 1).contiguous()

    flat_win = windows.reshape(-1, window_size, ohlcv.shape[1])
    n_windows = windows.shape[1]
    idx = torch.arange(n_windows) + window_size - 1

    # === Evaluate both indicators ===
    sig_crypto = indicator_crypto(flat_win).view(-1)
    sig_fx     = indicator_fx(flat_win).view(-1)

    raw_crypto = torch.full((T,), 0.5, device=sig_crypto.device)
    raw_fx     = torch.full((T,), 0.5, device=sig_fx.device)

    raw_crypto[idx] = sig_crypto
    raw_fx[idx] = sig_fx

    # Quantile transforms
    sig_crypto_np = rankdata(raw_crypto.cpu().numpy()) / T
    sig_fx_np     = rankdata(raw_fx.cpu().numpy()) / T

    # thresholds
    q1_c, q3_c = np.quantile(sig_crypto_np, [0.25, 0.75])
    q1_f, q3_f = np.quantile(sig_fx_np, [0.25, 0.75])

    # === Position rules ===
    def compute_positions(sig, q1, q3):
        pos = np.zeros_like(sig, int)
        for t in range(1, len(sig)):
            if sig[t] > q3:
                pos[t] = 1
            elif sig[t] < q1:
                pos[t] = 0
            else:
                pos[t] = pos[t-1]
        return pos

    pos_crypto = compute_positions(sig_crypto_np, q1_c, q3_c)
    pos_fx     = compute_positions(sig_fx_np, q1_f, q3_f)

    # === Strategy returns ===
    rets = simple_returns[1:]
    strat_c = np.cumprod(1 + pos_crypto[:-1] * rets)
    strat_f = np.cumprod(1 + pos_fx[:-1]     * rets)
    bench   = np.cumprod(1 + rets)

    # ===============================================================
    #  PLOTTING
    # ===============================================================

    fig, axes = plt.subplots(
        3, 1,
        figsize=(26, 12),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 2, 3]}
    )

    # ------------------------------
    # 1️⃣ CRYPTO SIGNAL
    # ------------------------------
    ax = axes[0]
    ax.plot(sig_crypto_np, color="blue", linewidth=1.2, alpha=0.5)
    ax.axhline(q1_c, color="red",   linestyle="--")
    ax.axhline(q3_c, color="green", linestyle="--")
    ax.set_title("EvoSignal-Crypto: Signal & Positions")

    start = None
    for t in range(1, T):
        if pos_crypto[t] == 1 and pos_crypto[t-1] == 0:
            start = t
        if pos_crypto[t] == 0 and pos_crypto[t-1] == 1 and start is not None:
            ax.axvspan(start, t, color="purple", alpha=0.1)
            start = None
    if pos_crypto[-1] == 1 and start is not None:
        ax.axvspan(start, T, color="purple", alpha=0.1)

    ax.set_ylabel("Signal (Quantile)")

    # ------------------------------
    # 2️⃣ FX SIGNAL
    # ------------------------------
    ax = axes[1]
    ax.plot(sig_fx_np, color="orange", linewidth=1.2, alpha=0.5)
    ax.axhline(q1_f, color="red",   linestyle="--")
    ax.axhline(q3_f, color="green", linestyle="--")
    ax.set_title("EvoSignal-FX: Signal & Positions")

    start = None
    for t in range(1, T):
        if pos_fx[t] == 1 and pos_fx[t-1] == 0:
            start = t
        if pos_fx[t] == 0 and pos_fx[t-1] == 1 and start is not None:
            ax.axvspan(start, t, color="purple", alpha=0.1)
            start = None
    if pos_fx[-1] == 1 and start is not None:
        ax.axvspan(start, T, color="purple", alpha=0.1)

    ax.set_ylabel("Signal (Quantile)")

    # ------------------------------
    # 3️⃣ SHARED PERFORMANCE
    # ------------------------------
    ax = axes[2]
    ax.plot(bench,      label="Buy-and-Hold",     linestyle="--", color="gray")
    ax.plot(strat_c,    label="EvoSignal-Crypto", color="blue",   linewidth=1.4)
    ax.plot(strat_f,    label="EvoSignal-FX",     color="orange", linewidth=1.4)

    ax.set_title("Cumulative Returns (Shared Benchmark)")
    ax.legend(loc="upper left")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

    print(f"✅ Saved: {filename}")

# ===============================================================
#  MAIN
# ===============================================================

if __name__ == "__main__":
    data_csv = "AUDNZD.csv"

    ohlcv = get_ohlcv(data_csv)

    plot_two_indicators(
        ohlcv,
        indicator_crypto=technical_indicator_crypto,
        indicator_fx=technical_indicator_fx,
        window_size=200,
        filename="crypto_vs_fx.png"
    )
