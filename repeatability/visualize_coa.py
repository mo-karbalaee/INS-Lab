import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec


# ──────────────────────────────────────────────────────────────────────────────
# Electrode layout
# ──────────────────────────────────────────────────────────────────────────────

N_CH_PER_COL = 16
WRISTBAND_LEN_CM = 28.0
COL_SEP_CM = 2.0

Y_POSITIONS = np.linspace(
    -WRISTBAND_LEN_CM / 2,
    +WRISTBAND_LEN_CM / 2,
    N_CH_PER_COL
)

BAD_EMG_CHANNELS = [9]

def electrode_positions():
    positions = []
    for i in range(N_CH_PER_COL):
        positions.append((i,       +COL_SEP_CM / 2, Y_POSITIONS[i]))
    for i in range(N_CH_PER_COL):
        positions.append((i + 16,  -COL_SEP_CM / 2, Y_POSITIONS[i]))
    return np.array(positions)

def electrode_positions_active():
    all_pos = electrode_positions()
    keep_mask = np.ones(32, dtype=bool)
    for ch in BAD_EMG_CHANNELS:
        keep_mask[ch] = False
    return all_pos[keep_mask]

ELECTRODE_POS        = electrode_positions()
ELECTRODE_POS_ACTIVE = electrode_positions_active()


# ──────────────────────────────────────────────────────────────────────────────
# Core computation helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_rms_per_channel(signal: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(signal ** 2, axis=1))


def compute_center_of_activation(rms: np.ndarray) -> tuple[float, float]:
    weights = rms / (rms.sum() + 1e-12)
    coa_x = np.sum(weights * ELECTRODE_POS[:, 1])
    coa_y = np.sum(weights * ELECTRODE_POS[:, 2])
    return float(coa_x), float(coa_y)


# ──────────────────────────────────────────────────────────────────────────────
# Build a summary DataFrame of CoA and RMS per segment
# ──────────────────────────────────────────────────────────────────────────────

def get_coa_df(segments: dict) -> pd.DataFrame:
    """
    Iterate over all segments and compute RMS + CoA.
    Uses precomputed RMS from analyse_repeatability.get_features() if available,
    otherwise computes fresh from the biosignal.
    """
    rows = []
    for gesture, seg_list in segments.items():
        for seg in seg_list:
            fc = seg.get('features_ch')
            rms = fc['rms'] if (fc is not None and 'rms' in fc) else compute_rms_per_channel(seg['biosignal'])
            n_ch = len(rms)
            pos = ELECTRODE_POS_ACTIVE[:n_ch]  # correct physical positions (ch 9 excluded)

            weights = rms / (rms.sum() + 1e-12)
            coa_x = float(np.sum(weights * pos[:, 1]))
            coa_y = float(np.sum(weights * pos[:, 2]))

            rows.append({
                'subject': seg['subject'],
                'gesture': gesture,
                'recording_index': seg['recording_index'],
                'segment_index': seg['segment_index'],
                'coa_x': coa_x,
                'coa_y': coa_y,
                'rms_mean': float(rms.mean()),
                'rms_per_ch': rms,
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# CoA shift computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_coa_shift(coa_df: pd.DataFrame):
    """
    Adds within_shift (cm) to coa_df and returns pairwise between-gesture distances.
    Returns (enriched_coa_df, between_df).
    """
    df = coa_df.copy()
    group_means = (
        df.groupby(['subject', 'gesture'])[['coa_x', 'coa_y']]
        .mean()
        .rename(columns={'coa_x': 'mean_coa_x', 'coa_y': 'mean_coa_y'})
        .reset_index()
    )
    df = df.merge(group_means, on=['subject', 'gesture'], how='left')
    df['within_shift'] = np.sqrt(
        (df['coa_x'] - df['mean_coa_x']) ** 2 +
        (df['coa_y'] - df['mean_coa_y']) ** 2
    )
    between_rows = []
    for subject, sub_df in group_means.groupby('subject'):
        gestures = sub_df['gesture'].values
        xs = sub_df.set_index('gesture')['mean_coa_x']
        ys = sub_df.set_index('gesture')['mean_coa_y']
        for i, ga in enumerate(gestures):
            for gb in gestures[i + 1:]:
                dist = np.sqrt((xs[ga] - xs[gb]) ** 2 + (ys[ga] - ys[gb]) ** 2)
                between_rows.append({'subject': subject, 'gesture_a': ga,
                                     'gesture_b': gb, 'shift_cm': float(dist)})
    return df, pd.DataFrame(between_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Visualization class
# ──────────────────────────────────────────────────────────────────────────────

# Each entry: (scatter_color pastel, mean_marker_color vivid) — same hue
GESTURE_COLOR_PAIRS = [
    ('#F4A0A7', '#E63946'),
    ('#A8C8E8', '#457B9D'),
    ('#A8DDD7', '#2A9D8F'),
    ('#F5E0A0', '#E9C46A'),
    ('#FAD1B0', '#F4A261'),
    ('#D4A8E0', '#9B5DB5'),
    ('#A8C4B8', '#52B788'),
]


class visualize_spatial:

    def __init__(self, segments: dict, coa_df: pd.DataFrame, save_path: str,
                 between_df: pd.DataFrame | None = None):
        self.segments = segments
        self.coa_df = coa_df
        self.between_df = between_df
        self.save_path = save_path
        self.gestures = sorted(coa_df['gesture'].unique())
        self.subjects = sorted(coa_df['subject'].unique())
        self._color_map      = {g: GESTURE_COLOR_PAIRS[i % len(GESTURE_COLOR_PAIRS)][0]
                                for i, g in enumerate(self.gestures)}
        self._mean_color_map = {g: GESTURE_COLOR_PAIRS[i % len(GESTURE_COLOR_PAIRS)][1]
                                for i, g in enumerate(self.gestures)}

    def _draw_wristband_background(self, ax, alpha=0.12):
        rect = plt.Rectangle(
            (-COL_SEP_CM / 2 - 0.3, -WRISTBAND_LEN_CM / 2 - 0.5),
            COL_SEP_CM + 0.6,
            WRISTBAND_LEN_CM + 1.0,
            linewidth=1.5, edgecolor='#555555', facecolor='#F0EDE8', alpha=alpha * 5,
            zorder=0
        )
        ax.add_patch(rect)

        for row in ELECTRODE_POS:
            ch_idx = int(row[0])
            if ch_idx in BAD_EMG_CHANNELS:
                ax.scatter(row[1], row[2], s=40, c='#CC0000', zorder=2,
                           marker='x', linewidths=1.2)
            else:
                ax.scatter(row[1], row[2], s=30, c='#AAAAAA', zorder=1, marker='s')

        ax.set_xlim(-COL_SEP_CM / 2 - 0.6, COL_SEP_CM / 2 + 0.6)
        ax.set_ylim(-WRISTBAND_LEN_CM / 2 - 1, +WRISTBAND_LEN_CM / 2 + 2.5)
        ax.set_xlabel('Lateral position (cm)', fontsize=9)
        ax.set_ylabel('Position along wristband (cm, centered)', fontsize=9)
        ax.set_xticks([-COL_SEP_CM / 2, COL_SEP_CM / 2])
        ax.set_xticklabels(['Left col (ch 17-32)', 'Right col (ch 1-16)'], fontsize=8)
        ax.grid(False)

    def plot_coa_scatter(self, per_subject=True, save=False):
        subjects_to_plot = self.subjects if per_subject else ['ALL']

        for subject in subjects_to_plot:
            if subject == 'ALL':
                df_sub = self.coa_df
            else:
                df_sub = self.coa_df[self.coa_df['subject'] == subject]

            fig, ax = plt.subplots(figsize=(4.5, 9))
            self._draw_wristband_background(ax)

            for gesture in self.gestures:
                df_g = df_sub[df_sub['gesture'] == gesture]
                if df_g.empty:
                    continue
                mean_x = df_g['coa_x'].mean()
                mean_y = df_g['coa_y'].mean()
                label = f"{gesture} (μ: {mean_x:.2f}, {mean_y:.2f})"

                ax.scatter(df_g['coa_x'], df_g['coa_y'],
                           s=80, alpha=0.75, zorder=4,
                           color=self._color_map[gesture],
                           edgecolors='white', linewidths=0.5,
                           label=label)
                ax.scatter(mean_x, mean_y, s=90, zorder=5,
                           color=self._mean_color_map[gesture],
                           marker='x', linewidths=2.2)

            ax.set_title('Center of Activation', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
            plt.tight_layout()
            if save:
                plt.savefig(f"{self.save_path}/coa_scatter_{subject}.png",
                            dpi=200, bbox_inches='tight')
            plt.show()
            plt.close(fig)

    def plot_rms_heatmap(self, subject: str | None = None, save=False):
        df_filt = self.coa_df if subject is None else self.coa_df[self.coa_df['subject'] == subject]

        n_gest = len(self.gestures)
        fig, axes = plt.subplots(1, n_gest, figsize=(3.2 * n_gest, 9),
                                  sharey=True, constrained_layout=True)
        if n_gest == 1:
            axes = [axes]

        global_max = 0.0
        global_min = np.inf
        rms_grids = {}

        for gesture in self.gestures:
            df_g = df_filt[df_filt['gesture'] == gesture]
            if df_g.empty:
                continue
            stack = np.vstack(df_g['rms_per_ch'].values)
            mean_rms = stack.mean(axis=0)
            rms_grids[gesture] = mean_rms
            global_max = max(global_max, mean_rms.max())
            global_min = min(global_min, mean_rms.min())

        norm = Normalize(vmin=global_min, vmax=global_max)
        cmap = plt.cm.inferno

        for ax, gesture in zip(axes, self.gestures):
            if gesture not in rms_grids:
                ax.set_visible(False)
                continue
            mean_rms = rms_grids[gesture]

            ax.set_facecolor('#1a1a1a')
            ax.set_xlim(-COL_SEP_CM / 2 - 0.5, COL_SEP_CM / 2 + 0.5)
            ax.set_ylim(-WRISTBAND_LEN_CM / 2 - 0.5, +WRISTBAND_LEN_CM / 2 + 0.5)
            ax.set_title(gesture, fontsize=10, fontweight='bold',
                         color=self._color_map[gesture])
            ax.set_xticks([-COL_SEP_CM / 2, COL_SEP_CM / 2])
            ax.set_xticklabels(['L', 'R'], fontsize=8)

            active_idx = 0
            for full_idx in range(32):
                x = ELECTRODE_POS[full_idx, 1]
                y = ELECTRODE_POS[full_idx, 2]
                if full_idx in BAD_EMG_CHANNELS:
                    ax.scatter(x, y, s=50, c='#444444', zorder=3,
                               marker='x', linewidths=1.0)
                    ax.text(x, y - 0.55, 'dead', ha='center', va='top',
                            fontsize=4, color='#888888', zorder=4)
                    continue
                if active_idx >= len(mean_rms):
                    break
                rms_val = mean_rms[active_idx]
                color = cmap(norm(rms_val))
                circle = plt.Circle((x, y), radius=0.55, color=color, zorder=3)
                ax.add_patch(circle)
                ax.text(x, y, f'{full_idx + 1}', ha='center', va='center',
                        fontsize=5.5, color='white', zorder=4)
                active_idx += 1

            pos = ELECTRODE_POS_ACTIVE[:len(mean_rms)]
            weights = mean_rms / (mean_rms.sum() + 1e-12)
            coa_x = float(np.sum(weights * pos[:, 1]))
            coa_y = float(np.sum(weights * pos[:, 2]))
            ax.scatter(coa_x, coa_y, s=160, marker='*',
                       color='cyan', edgecolors='white', linewidths=0.8,
                       zorder=6, label='CoA')
            ax.legend(loc='lower right', fontsize=6, framealpha=0.6)
            ax.grid(False)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
        cbar.set_label('Mean RMS (a.u.)', fontsize=9)
        fig.suptitle('Mean RMS Heatmap on Wristband',
                     fontsize=13, fontweight='bold', y=1.01)

        if save:
            tag = subject if subject else 'all'
            plt.savefig(f"{self.save_path}/rms_heatmap_wristband_{tag}.png",
                        dpi=200, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    def plot_coa_shift(self, save=False):
        if 'within_shift' not in self.coa_df.columns:
            raise ValueError("coa_df lacks 'within_shift'. Call compute_coa_shift() first.")

        for subject in self.subjects:
            df_sub = self.coa_df[self.coa_df['subject'] == subject]
            fig, (ax_within, ax_between) = plt.subplots(
                1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.4, 1]}
            )
            fig.suptitle('CoA Shift', fontsize=13, fontweight='bold', y=1.02)

            gestures_ordered = sorted(df_sub['gesture'].unique())
            positions = np.arange(len(gestures_ordered))

            bp = ax_within.boxplot(
                [df_sub[df_sub['gesture'] == g]['within_shift'].values
                 for g in gestures_ordered],
                positions=positions, widths=0.35, patch_artist=True,
                showfliers=False,
                medianprops=dict(color='black', linewidth=1.8),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                boxprops=dict(linewidth=1.2)
            )
            for patch, g in zip(bp['boxes'], gestures_ordered):
                patch.set_facecolor(self._color_map[g])
                patch.set_alpha(0.35)

            rng = np.random.default_rng(seed=0)
            for pos, g in zip(positions, gestures_ordered):
                vals = df_sub[df_sub['gesture'] == g]['within_shift'].values
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax_within.scatter(pos + jitter, vals, s=40, alpha=0.8, zorder=4,
                                  color=self._color_map[g],
                                  edgecolors='white', linewidths=0.4)
                ax_within.scatter(pos, vals.mean(), s=80, zorder=5,
                                  marker='x', linewidths=2.2,
                                  color=self._mean_color_map[g])

            ax_within.set_xticks(positions)
            ax_within.set_xticklabels(gestures_ordered, rotation=30, ha='right', fontsize=9)
            ax_within.set_ylabel('Within-gesture CoA shift (cm)', fontsize=10)
            ax_within.set_title('Within-gesture spatial variability', fontsize=11)
            ax_within.grid(axis='y', alpha=0.3)
            ax_within.set_xlim(-0.6, len(gestures_ordered) - 0.4)

            if self.between_df is not None and not self.between_df.empty:
                sub_between = self.between_df[self.between_df['subject'] == subject]
                g_list = gestures_ordered
                n = len(g_list)
                mat = np.full((n, n), np.nan)
                idx_map = {g: i for i, g in enumerate(g_list)}
                for _, row in sub_between.iterrows():
                    i, j = idx_map.get(row['gesture_a']), idx_map.get(row['gesture_b'])
                    if i is not None and j is not None:
                        mat[i, j] = row['shift_cm']
                        mat[j, i] = row['shift_cm']
                np.fill_diagonal(mat, 0.0)
                vmax = float(np.nanmax(mat)) if not np.all(np.isnan(mat)) else 1.0
                im = ax_between.imshow(mat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
                ax_between.set_xticks(range(n))
                ax_between.set_xticklabels(g_list, rotation=35, ha='right', fontsize=8)
                ax_between.set_yticks(range(n))
                ax_between.set_yticklabels(g_list, fontsize=8)
                for i in range(n):
                    for j in range(n):
                        if not np.isnan(mat[i, j]):
                            txt_col = 'black' if mat[i, j] < vmax * 0.65 else 'white'
                            ax_between.text(j, i, f'{mat[i, j]:.2f}',
                                            ha='center', va='center',
                                            fontsize=8, color=txt_col)
                fig.colorbar(im, ax=ax_between, shrink=0.75, label='Distance (cm)')
                ax_between.set_title('Between-gesture CoA distance', fontsize=11)
            else:
                ax_between.text(0.5, 0.5, 'No between-gesture data',
                                ha='center', va='center', transform=ax_between.transAxes)

            plt.tight_layout()
            if save:
                plt.savefig(f"{self.save_path}/coa_shift_{subject}.png",
                            dpi=200, bbox_inches='tight')
            plt.show()
            plt.close(fig)

    def plot_coa_cluster_overview(self, save=False):
        fig, ax = plt.subplots(figsize=(5, 10))
        self._draw_wristband_background(ax)

        for gesture in self.gestures:
            df_g = self.coa_df[self.coa_df['gesture'] == gesture]
            if df_g.empty:
                continue
            rng = np.random.default_rng(seed=42)
            jitter = rng.normal(0, 0.04, size=(len(df_g), 2))
            ax.scatter(
                df_g['coa_x'].values + jitter[:, 0],
                df_g['coa_y'].values + jitter[:, 1],
                s=50, alpha=0.55, zorder=3,
                color=self._color_map[gesture], edgecolors='none'
            )
            mean_x = df_g['coa_x'].mean()
            mean_y = df_g['coa_y'].mean()
            label = f"{gesture} (μ: {mean_x:.2f}, {mean_y:.2f})"
            ax.scatter(mean_x, mean_y, s=120, zorder=5,
                       color=self._mean_color_map[gesture],
                       marker='x', linewidths=2.2, label=label)
            if len(df_g) >= 3:
                from matplotlib.patches import Ellipse
                std_x = df_g['coa_x'].std()
                std_y = df_g['coa_y'].std()
                ellipse = Ellipse(
                    xy=(mean_x, mean_y),
                    width=2 * std_x, height=2 * std_y,
                    angle=0,
                    edgecolor=self._color_map[gesture],
                    facecolor=self._color_map[gesture],
                    alpha=0.15, lw=1.5, zorder=2
                )
                ax.add_patch(ellipse)

        ax.set_title('CoA Cluster Overview – All Subjects & Gestures',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.text(-COL_SEP_CM / 2, WRISTBAND_LEN_CM / 2 + 1.2, 'ch 17–32',
                ha='center', va='bottom', fontsize=7, color='#444444')
        ax.text(+COL_SEP_CM / 2, WRISTBAND_LEN_CM / 2 + 1.2, 'ch 1–16',
                ha='center', va='bottom', fontsize=7, color='#444444')
        plt.tight_layout()
        if save:
            plt.savefig(f"{self.save_path}/coa_cluster_overview.png",
                        dpi=200, bbox_inches='tight')
        plt.show()
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run_spatial_analysis(segments: dict, save_path: str, save: bool = False):
    print("Computing RMS and Center-of-Activation per segment \u2026")
    coa_df = get_coa_df(segments)
    print(f"  \u2192 {len(coa_df)} segments processed.\n")

    print("Computing CoA shift metrics \u2026")
    coa_df, between_df = compute_coa_shift(coa_df)
    print(f"  \u2192 within_shift added; {len(between_df)} gesture-pair distances computed.\n")

    viz = visualize_spatial(segments, coa_df, save_path, between_df=between_df)

    print("Plotting per-subject CoA scatter \u2026")
    viz.plot_coa_scatter(per_subject=True, save=save)

    print("Plotting CoA shift \u2026")
    viz.plot_coa_shift(save=save)

    print("Plotting cluster overview \u2026")
    viz.plot_coa_cluster_overview(save=save)

    print("Plotting RMS heatmaps \u2026")
    for subject in sorted(coa_df['subject'].unique()):
        viz.plot_rms_heatmap(subject=subject, save=save)

    return coa_df, viz