import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
N = 2000  # synthetic rows that mirror the Kaggle dataset structure

# ── Synthetic F1 Pit‑Stop dataset ─────────────────────────────────────────────
df = pd.DataFrame({
    # Pit‑stop duration (seconds) – right‑skewed, like real F1 data
    'pit_duration':   np.random.exponential(scale=3.2, size=N) + 1.8,
    # Lap number when the stop occurred (roughly normally distributed mid‑race)
    'lap_number':     np.random.normal(loc=32, scale=10, size=N).clip(1, 65).astype(int),
    # Number of pit stops per race (Poisson λ≈1.7)
    'stops_per_race': np.random.poisson(lam=1.7, size=N),
    # Binary: 1 = stop under safety car, 0 = normal  (p ≈ 0.25)
    'safety_car_stop':np.random.binomial(n=1, p=0.25, size=N),
    # Tyre compound chosen (uniform across 5 compounds for simplicity)
    'tyre_compound':  np.random.randint(low=1, high=6, size=N),
})

# ── Global style ──────────────────────────────────────────────────────────────
DARK_BG   = '#0F0F1A'
PANEL_BG  = '#1A1A2E'
ACCENT    = ['#E8143C', '#00D2FF', '#F5C518', '#7FFF00', '#FF6B35']
GRID_CLR  = '#2A2A45'
TEXT_CLR  = '#E8E8F0'

plt.rcParams.update({
    'figure.facecolor':  DARK_BG,
    'axes.facecolor':    PANEL_BG,
    'axes.edgecolor':    GRID_CLR,
    'axes.labelcolor':   TEXT_CLR,
    'axes.titlecolor':   TEXT_CLR,
    'xtick.color':       TEXT_CLR,
    'ytick.color':       TEXT_CLR,
    'text.color':        TEXT_CLR,
    'grid.color':        GRID_CLR,
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'font.family':       'DejaVu Sans',
})

# ── Canvas ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 15))
fig.suptitle(
    'F1 Pit‑Stop Statistical Distributions\n'
    'Synthetic dataset based on: Predicting F1 Pit Stops (Kaggle)',
    fontsize=18, fontweight='bold', color=TEXT_CLR, y=0.98
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32,
                       left=0.06, right=0.97, top=0.91, bottom=0.07)

axes = [fig.add_subplot(gs[r, c]) for r, c in
        [(0,0),(0,1),(0,2),(1,0),(1,1)]]


# ══════════════════════════════════════════════════════════════════════════════
# 1. EXPONENTIAL  –  pit‑stop duration
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]
data = df['pit_duration'].values
color = ACCENT[0]

bins = np.linspace(data.min(), min(data.max(), 18), 55)
ax.hist(data, bins=bins, density=True, color=color, alpha=0.70,
        edgecolor='white', linewidth=0.3, label='Observed')

loc_fit, scale_fit = stats.expon.fit(data, floc=data.min())
x = np.linspace(data.min(), 18, 400)
ax.plot(x, stats.expon.pdf(x, loc_fit, scale_fit),
        color='white', lw=2.2, label=f'Exp fit  (λ={1/scale_fit:.3f})')

ax.set_xlim(data.min(), 18)
ax.set_title('① Exponential Distribution\nPit‑Stop Duration (seconds)',
             fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('Duration (s)')
ax.set_ylabel('Density')
ax.legend(fontsize=8)
ax.grid(True)

mean_val = data.mean()
ax.axvline(mean_val, color='yellow', lw=1.4, ls=':', alpha=0.9)
ax.text(mean_val + 0.3, ax.get_ylim()[1]*0.85,
        f'μ={mean_val:.2f}s', color='yellow', fontsize=8)

stat, p = stats.kstest(data - data.min(), 'expon',
                       args=(0, scale_fit))
ax.text(0.97, 0.97, f'KS p={p:.3f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=7.5, color=TEXT_CLR,
        bbox=dict(fc=DARK_BG, ec=GRID_CLR, alpha=0.7, pad=2))


# ══════════════════════════════════════════════════════════════════════════════
# 2. NORMAL  –  lap number of pit stop
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]
data = df['lap_number'].values.astype(float)
color = ACCENT[1]

mu, sigma = data.mean(), data.std()
ax.hist(data, bins=35, density=True, color=color, alpha=0.70,
        edgecolor='white', linewidth=0.3, label='Observed')

x = np.linspace(data.min(), data.max(), 400)
ax.plot(x, stats.norm.pdf(x, mu, sigma),
        color='white', lw=2.2, label=f'N(μ={mu:.1f}, σ={sigma:.1f})')

ax.set_title('② Normal Distribution\nLap Number of Pit Stop',
             fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('Lap Number')
ax.set_ylabel('Density')
ax.legend(fontsize=8)
ax.grid(True)

for s, ls in [(1,'--'),(2,':')]:
    ax.axvspan(mu - s*sigma, mu + s*sigma,
               alpha=0.08 if s==2 else 0.15, color='white')

ax.axvline(mu, color='yellow', lw=1.4, ls=':')
ax.text(mu + 0.5, ax.get_ylim()[1]*0.88,
        f'μ={mu:.1f}', color='yellow', fontsize=8)

_, p = stats.normaltest(data)
ax.text(0.97, 0.97, f'Norm‑test p={p:.3f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=7.5, color=TEXT_CLR,
        bbox=dict(fc=DARK_BG, ec=GRID_CLR, alpha=0.7, pad=2))


# ══════════════════════════════════════════════════════════════════════════════
# 3. POISSON  –  number of pit stops per race
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[2]
data = df['stops_per_race'].values
color = ACCENT[2]

counts = pd.Series(data).value_counts().sort_index()
lam_est = data.mean()
k_vals = np.arange(0, counts.index.max() + 1)
pmf = stats.poisson.pmf(k_vals, lam_est)

bar_w = 0.55
ax.bar(counts.index, counts / counts.sum(), width=bar_w,
       color=color, alpha=0.80, edgecolor='white', linewidth=0.5,
       label='Observed freq.')
ax.scatter(k_vals, pmf, color='white', zorder=5, s=55, label=f'Poisson(λ={lam_est:.2f})')
ax.vlines(k_vals, 0, pmf, color='white', lw=1.4, alpha=0.6)

ax.set_title('③ Poisson Distribution\nStops per Race Count',
             fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('Number of Pit Stops')
ax.set_ylabel('Probability')
ax.legend(fontsize=8)
ax.grid(True, axis='y')
ax.set_xticks(k_vals)

ax.text(0.97, 0.97, f'λ (mean)={lam_est:.3f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=7.5, color=TEXT_CLR,
        bbox=dict(fc=DARK_BG, ec=GRID_CLR, alpha=0.7, pad=2))


# ══════════════════════════════════════════════════════════════════════════════
# 4. BINOMIAL  –  safety‑car stop flag (Bernoulli / Binomial n=1)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[3]
data = df['safety_car_stop'].values
color = ACCENT[3]

p_est = data.mean()
labels = ['Normal Stop\n(0)', 'Safety‑Car Stop\n(1)']
obs_freq = [np.mean(data == 0), np.mean(data == 1)]
binom_pmf = [stats.binom.pmf(k, 1, p_est) for k in [0, 1]]

x = np.array([0, 1])
bar_w = 0.3
ax.bar(x - bar_w/2, obs_freq,  width=bar_w, color=color,  alpha=0.80,
       edgecolor='white', linewidth=0.5, label='Observed')
ax.bar(x + bar_w/2, binom_pmf, width=bar_w, color='white', alpha=0.30,
       edgecolor='white', linewidth=0.8, label=f'Binomial(n=1, p={p_est:.2f})')

ax.set_xticks([0, 1])
ax.set_xticklabels(labels, fontsize=9)
ax.set_title('④ Binomial Distribution\nSafety‑Car Stop Probability',
             fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('Outcome')
ax.set_ylabel('Probability')
ax.legend(fontsize=8)
ax.grid(True, axis='y')

for xi, obs, theo in zip(x, obs_freq, binom_pmf):
    ax.text(xi - bar_w/2, obs + 0.008, f'{obs:.3f}',
            ha='center', fontsize=8, color=TEXT_CLR)
    ax.text(xi + bar_w/2, theo + 0.008, f'{theo:.3f}',
            ha='center', fontsize=8, color=TEXT_CLR)

ax.text(0.97, 0.97, f'p={p_est:.3f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=7.5, color=TEXT_CLR,
        bbox=dict(fc=DARK_BG, ec=GRID_CLR, alpha=0.7, pad=2))


# ══════════════════════════════════════════════════════════════════════════════
# 5. UNIFORM  –  tyre compound selection
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[4]
data = df['tyre_compound'].values
color = ACCENT[4]

compounds = {1:'Soft', 2:'Medium', 3:'Hard', 4:'Inter', 5:'Wet'}
counts = pd.Series(data).value_counts().sort_index()
expected = len(data) / len(compounds)

bar_colors = [color] * 5
bars = ax.bar(counts.index, counts, color=bar_colors, alpha=0.80,
              edgecolor='white', linewidth=0.5, label='Observed count')

# Uniform expected line
ax.axhline(expected, color='white', lw=2, ls='--',
           label=f'Uniform expected ({expected:.0f})')

ax.set_xticks(list(compounds.keys()))
ax.set_xticklabels([f"{k}\n{v}" for k, v in compounds.items()], fontsize=9)
ax.set_title('⑤ Uniform Distribution\nTyre Compound Selection',
             fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('Compound')
ax.set_ylabel('Count')
ax.legend(fontsize=8)
ax.grid(True, axis='y')

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 8,
            str(int(h)), ha='center', fontsize=8, color=TEXT_CLR)

chi2, p = stats.chisquare(counts)
ax.text(0.97, 0.97, f'χ² p={p:.3f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=7.5, color=TEXT_CLR,
        bbox=dict(fc=DARK_BG, ec=GRID_CLR, alpha=0.7, pad=2))


# ── Watermark / footer ────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         'Synthetic dataset modelled on Kaggle: "Predicting F1 Pit Stops" | '
         'Generated with Python · pandas · matplotlib · scipy',
         ha='center', fontsize=8, color='#888899')

# ── Save ──────────────────────────────────────────────────────────────────────
out = '/mnt/user-data/outputs/f1_distribution_charts.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f"Saved → {out}")
