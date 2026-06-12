"""
umap_analysis.py
────────────────
UMAP dimensionality-reduction analysis for sEMG gesture data.

Provides:
  • build_feature_matrix   – assemble per-segment feature vectors from segments dict
  • run_umap               – fit UMAP and return embeddings + metadata DataFrame
  • visualize_umap         – scatter plots coloured by gesture / subject
  • UMAPAnalysis           – convenience class wrapping all of the above
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional

# ── optional imports with helpful error messages ───────────────────────────────

try:
    import umap  # umap-learn
except ImportError:
    raise ImportError(
        "umap-learn is required.  Install it with:\n"
        "    pip install umap-learn"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (matches visualize_spatial.py)
# ──────────────────────────────────────────────────────────────────────────────

GESTURE_COLORS = [
    '#E63946', '#457B9D', '#2A9D8F',
    '#E9C46A', '#F4A261', '#6A0572', '#264653',
]

SUBJECT_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*']


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_features(signal: np.ndarray, fs: int = 2000) -> np.ndarray:
    """
    Extract a compact per-segment feature vector concatenating time- and
    frequency-domain features across all channels.

    Features per channel (6 total):
        MAV, RMS, WL, peak_freq, MNF, MDF
    Returns a 1D vector of length n_channels * 6.
    """
    # ── time domain ──────────────────────────────────────────────────────────
    mav       = np.mean(np.abs(signal), axis=1)
    rms       = np.sqrt(np.mean(signal ** 2, axis=1))
    wl        = np.sum(np.abs(np.diff(signal, axis=1)), axis=1)

    # ── frequency domain ─────────────────────────────────────────────────────
    fft_vals  = np.fft.rfft(signal, axis=1)
    freqs     = np.fft.rfftfreq(signal.shape[1], d=1 / fs)
    power     = np.abs(fft_vals) ** 2

    peak_freq = freqs[np.argmax(power, axis=1)]
    mnf       = np.sum(freqs * power, axis=1) / (np.sum(power, axis=1) + 1e-12)

    def _mdf(p_row: np.ndarray, f: np.ndarray) -> float:
        cum = np.cumsum(p_row)
        return float(f[np.searchsorted(cum, cum[-1] / 2)])

    mdf = np.array([_mdf(power[ch], freqs) for ch in range(power.shape[0])])

    return np.concatenate([mav, rms, wl, peak_freq, mnf, mdf])  # (n_ch * 6,)


def build_feature_matrix(segments: dict, fs: int = 2000) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build a feature matrix from the segments dict.

    Returns
    -------
    X   : (n_segments, n_features)  float32 array
    meta: (n_segments, 4) DataFrame with columns [subject, gesture,
                                                   recording_index, segment_index]
    """
    rows_X, rows_meta = [], []

    for gesture, seg_list in segments.items():
        for seg in seg_list:
            feat = _extract_features(seg['biosignal'], fs=fs)
            rows_X.append(feat)
            rows_meta.append({
                'subject':          seg['subject'],
                'gesture':          gesture,
                'recording_index':  seg['recording_index'],
                'segment_index':    seg['segment_index'],
            })

    X    = np.array(rows_X, dtype=np.float32)
    meta = pd.DataFrame(rows_meta)
    return X, meta


# ──────────────────────────────────────────────────────────────────────────────
# UMAP fitting
# ──────────────────────────────────────────────────────────────────────────────

def run_umap(
    X: np.ndarray,
    meta: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = 'euclidean',
    random_state: int = 42,
    scale: bool = True,
) -> pd.DataFrame:
    """
    Fit UMAP on X and return a DataFrame of embeddings + metadata.

    Parameters
    ----------
    X            : feature matrix (n_samples, n_features)
    meta         : metadata DataFrame (n_samples, ≥ subject/gesture cols)
    n_neighbors  : UMAP n_neighbors
    min_dist     : UMAP min_dist
    n_components : embedding dimensionality (2 or 3)
    metric       : distance metric
    random_state : reproducibility seed
    scale        : z-score features before embedding

    Returns
    -------
    df_embed: DataFrame with columns UMAP1, UMAP2 (and UMAP3 if 3D),
              plus subject, gesture, recording_index, segment_index
    """
    if scale:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

    print(f"Fitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, "
          f"n_components={n_components}, metric={metric}) …")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    embedding = reducer.fit_transform(X)
    print("  → UMAP done.\n")

    col_names = [f'UMAP{i + 1}' for i in range(n_components)]
    df_embed  = pd.DataFrame(embedding, columns=col_names)
    df_embed  = pd.concat([df_embed, meta.reset_index(drop=True)], axis=1)
    return df_embed, reducer


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

class visualize_umap:
    """All UMAP visualization methods."""

    def __init__(self, df_embed: pd.DataFrame, save_path: str):
        self.df       = df_embed
        self.save_path = save_path
        self.gestures  = sorted(df_embed['gesture'].unique())
        self.subjects  = sorted(df_embed['subject'].unique())
        self._c_gest   = {g: GESTURE_COLORS[i % len(GESTURE_COLORS)]
                         for i, g in enumerate(self.gestures)}
        self._m_subj   = {s: SUBJECT_MARKERS[i % len(SUBJECT_MARKERS)]
                         for i, s in enumerate(self.subjects)}

    # ── internal helpers ──────────────────────────────────────────────────────

    def _legend_patches(self, mapping: dict, title: str):
        return [mpatches.Patch(color=v, label=k) for k, v in mapping.items()], title

    # ── 1. coloured by gesture ────────────────────────────────────────────────

    def plot_by_gesture(self, save: bool = False, alpha: float = 0.7):
        """
        2-D UMAP scatter, one colour per gesture.
        All subjects overlaid.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        for gesture in self.gestures:
            sub = self.df[self.df['gesture'] == gesture]
            ax.scatter(sub['UMAP1'], sub['UMAP2'],
                       s=40, alpha=alpha, c=self._c_gest[gesture],
                       edgecolors='none', label=gesture, zorder=3)

        ax.set_title('UMAP – coloured by Gesture', fontsize=13, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=11)
        ax.set_ylabel('UMAP 2', fontsize=11)
        ax.legend(loc='best', fontsize=9, framealpha=0.85, title='Gesture')
        ax.grid(alpha=0.2)

        plt.tight_layout()
        if save:
            plt.savefig(f"{self.save_path}/umap_by_gesture.png", dpi=200, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    # ── 2. coloured by subject ────────────────────────────────────────────────

    def plot_by_subject(self, save: bool = False, alpha: float = 0.7):
        """2-D UMAP scatter, one colour per subject."""
        subject_colors = {s: GESTURE_COLORS[i % len(GESTURE_COLORS)]
                          for i, s in enumerate(self.subjects)}

        fig, ax = plt.subplots(figsize=(8, 6))

        for subject in self.subjects:
            sub = self.df[self.df['subject'] == subject]
            ax.scatter(sub['UMAP1'], sub['UMAP2'],
                       s=40, alpha=alpha, c=subject_colors[subject],
                       edgecolors='none', label=subject, zorder=3)

        ax.set_title('UMAP – coloured by Subject', fontsize=13, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=11)
        ax.set_ylabel('UMAP 2', fontsize=11)
        ax.legend(loc='best', fontsize=9, framealpha=0.85, title='Subject')
        ax.grid(alpha=0.2)

        plt.tight_layout()
        if save:
            plt.savefig(f"{self.save_path}/umap_by_subject.png", dpi=200, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    # ── 3. gesture × subject facet grid ──────────────────────────────────────

    def plot_faceted_by_subject(self, save: bool = False, alpha: float = 0.75):
        """
        One subplot per subject.  Points coloured by gesture.
        Background (grey) = all other subjects' points for spatial reference.
        """
        n_sub = len(self.subjects)
        ncols = min(3, n_sub)
        nrows = int(np.ceil(n_sub / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(5.5 * ncols, 4.5 * nrows),
                                  sharex=True, sharey=True,
                                  constrained_layout=True)
        axes_flat = np.array(axes).flatten()

        for ax, subject in zip(axes_flat[:n_sub], self.subjects):
            # grey background: all other subjects
            bg = self.df[self.df['subject'] != subject]
            ax.scatter(bg['UMAP1'], bg['UMAP2'],
                       s=12, alpha=0.15, c='#AAAAAA', edgecolors='none', zorder=1)

            # foreground: this subject, coloured by gesture
            fg = self.df[self.df['subject'] == subject]
            for gesture in self.gestures:
                sub = fg[fg['gesture'] == gesture]
                ax.scatter(sub['UMAP1'], sub['UMAP2'],
                           s=50, alpha=alpha, c=self._c_gest[gesture],
                           edgecolors='none', label=gesture, zorder=3)

            ax.set_title(f'Subject: {subject}', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.2)

        # hide unused axes
        for ax in axes_flat[n_sub:]:
            ax.set_visible(False)

        # shared legend
        patches = [mpatches.Patch(color=self._c_gest[g], label=g) for g in self.gestures]
        fig.legend(handles=patches, loc='lower center',
                   ncol=min(len(self.gestures), 5), fontsize=9,
                   title='Gesture', framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.02))

        fig.suptitle('UMAP – Per-Subject Facets (gestures coloured)',
                     fontsize=14, fontweight='bold')

        if save:
            plt.savefig(f"{self.save_path}/umap_faceted_by_subject.png",
                        dpi=200, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    # ── 4. gesture × subject dual encoding ───────────────────────────────────

    def plot_gesture_x_subject(self, save: bool = False, alpha: float = 0.75):
        """
        Colour = gesture, marker = subject.
        Useful for checking cross-subject separability.
        """
        fig, ax = plt.subplots(figsize=(9, 7))

        for subject in self.subjects:
            for gesture in self.gestures:
                sub = self.df[(self.df['subject'] == subject) &
                              (self.df['gesture'] == gesture)]
                if sub.empty:
                    continue
                ax.scatter(
                    sub['UMAP1'], sub['UMAP2'],
                    s=55, alpha=alpha,
                    c=self._c_gest[gesture],
                    marker=self._m_subj[subject],
                    edgecolors='white', linewidths=0.4,
                    zorder=3,
                )

        # legend: gestures (colour)
        g_patches = [mpatches.Patch(color=self._c_gest[g], label=g) for g in self.gestures]
        leg1 = ax.legend(handles=g_patches, loc='upper left',
                         fontsize=8, framealpha=0.85, title='Gesture')
        ax.add_artist(leg1)

        # legend: subjects (marker)
        from matplotlib.lines import Line2D
        s_handles = [Line2D([0], [0], linestyle='none',
                            marker=self._m_subj[s], color='#555555',
                            markersize=7, label=s)
                     for s in self.subjects]
        ax.legend(handles=s_handles, loc='upper right',
                  fontsize=8, framealpha=0.85, title='Subject')

        ax.set_title('UMAP – Gesture (colour) × Subject (marker)',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=11)
        ax.set_ylabel('UMAP 2', fontsize=11)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        if save:
            plt.savefig(f"{self.save_path}/umap_gesture_x_subject.png",
                        dpi=200, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    # ── 5. convex-hull cluster overlay ───────────────────────────────────────

    def plot_gesture_hulls(self, save: bool = False, alpha: float = 0.65):
        """
        Scatter + convex-hull shading per gesture cluster.
        """
        from scipy.spatial import ConvexHull

        fig, ax = plt.subplots(figsize=(8, 6))

        for gesture in self.gestures:
            sub = self.df[self.df['gesture'] == gesture]
            if len(sub) < 3:
                continue
            pts = sub[['UMAP1', 'UMAP2']].values
            ax.scatter(pts[:, 0], pts[:, 1],
                       s=35, alpha=0.6, c=self._c_gest[gesture],
                       edgecolors='none', zorder=3, label=gesture)
            try:
                hull = ConvexHull(pts)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                ax.fill(pts[hull_pts, 0], pts[hull_pts, 1],
                        color=self._c_gest[gesture], alpha=0.12, zorder=2)
                ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                        color=self._c_gest[gesture], lw=1.4,
                        alpha=0.7, zorder=2)
            except Exception:
                pass

        ax.set_title('UMAP – Gesture Clusters (convex hull)',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=11)
        ax.set_ylabel('UMAP 2', fontsize=11)
        ax.legend(loc='best', fontsize=9, framealpha=0.85, title='Gesture')
        ax.grid(alpha=0.2)

        plt.tight_layout()
        if save:
            plt.savefig(f"{self.save_path}/umap_gesture_hulls.png",
                        dpi=200, bbox_inches='tight')
        plt.show()
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper class
# ──────────────────────────────────────────────────────────────────────────────

class UMAPAnalysis:
    """
    All-in-one entry point.

    Usage
    -----
    ua = UMAPAnalysis(segments, save_path)
    ua.run()           # fit UMAP and produce all plots
    ua.df_embed        # access the embedding DataFrame afterwards
    """

    def __init__(
        self,
        segments: dict,
        save_path: str,
        fs: int = 2000,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        scale: bool = True,
    ):
        self.segments     = segments
        self.save_path    = save_path
        self.fs           = fs
        self.n_neighbors  = n_neighbors
        self.min_dist     = min_dist
        self.metric       = metric
        self.scale        = scale

        self.X: Optional[np.ndarray]       = None
        self.meta: Optional[pd.DataFrame]  = None
        self.df_embed: Optional[pd.DataFrame] = None
        self.reducer  = None
        self.viz: Optional[visualize_umap] = None

    def build_features(self):
        """Extract features fresh from raw biosignal."""
        print("Extracting features …")
        self.X, self.meta = build_feature_matrix(self.segments, fs=self.fs)
        print(f"  → Feature matrix shape: {self.X.shape}\n")
        return self

    def build_features_from_precomputed(self):
        """
        Use features already stored in seg['features_ch'] by
        analyse_repeatability.get_features() — avoids redundant recomputation.
        Concatenates mav, rms, wl, peak_freq, mnf, mdf per channel (same order
        as _extract_features). Subject names are mapped to integers.
        """
        print("Using precomputed features from segments …")
        rows_X, rows_meta = [], []
        for gesture, seg_list in self.segments.items():
            for seg in seg_list:
                fc = seg.get('features_ch')
                if fc is None:
                    raise ValueError(
                        "seg['features_ch'] not found. "
                        "Run analyse_repeatability.get_features() first."
                    )
                feat = np.concatenate([
                    fc['mav'], fc['rms'], fc['wl'],
                    fc['peak_freq'], fc['mnf'], fc['mdf']
                ])
                rows_X.append(feat)
                rows_meta.append({
                    'subject':         seg['subject'],
                    'gesture':         gesture,
                    'recording_index': seg['recording_index'],
                    'segment_index':   seg['segment_index'],
                })
        self.X = np.array(rows_X, dtype=np.float32)
        meta   = pd.DataFrame(rows_meta)
        unique_subjects = sorted(meta['subject'].unique())
        meta['subject'] = meta['subject'].map({s: i + 1 for i, s in enumerate(unique_subjects)})
        self.meta = meta
        print(f"  → Feature matrix shape: {self.X.shape}\n")
        return self

    def fit(self):
        if self.X is None:
            self.build_features()
        self.df_embed, self.reducer = run_umap(
            self.X, self.meta,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            scale=self.scale,
        )
        self.viz = visualize_umap(self.df_embed, self.save_path)
        return self

    def plot_all(self, save: bool = False):
        if self.viz is None:
            self.fit()
        self.viz.plot_by_gesture(save=save)
        self.viz.plot_by_subject(save=save)
        self.viz.plot_faceted_by_subject(save=save)
        self.viz.plot_gesture_x_subject(save=save)
        self.viz.plot_gesture_hulls(save=save)
        return self

    def run(self, save: bool = False):
        """Build features → fit UMAP → plot all."""
        return self.build_features().fit().plot_all(save=save)