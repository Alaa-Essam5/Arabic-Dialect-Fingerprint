import os, io, base64, json, tempfile, logging, re
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import soundfile as sf
import joblib
import requests
from groq import Groq
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')


def _noemoji(s):
    """Strip emoji/flag characters that DejaVu Sans cannot render."""
    return re.sub(r'[^-؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿ ]', '', s).strip()


# =============================================================================
# API KEYS — never sent to the browser
# =============================================================================
GROQ_API_KEY       = "put your api key from groq here"
ELEVENLABS_API_KEY = "put your api key from elevenlabs here"

# =============================================================================
# ELEVENLABS VOICE CATALOGUE
# =============================================================================
ELEVENLABS_VOICES = {
    "Sarah":   "EXAVITQu4vr4xnSDxMaL",
    "Jessica": "cgSgspJ2msm6clMCkdW9",
    "Adam":    "pNInz6obpgDQGcFmaJgB",
    "Charlie": "IKne3meq5aSn9XLyUdCD",
}
ELEVENLABS_MODEL    = "eleven_multilingual_v2"
ELEVENLABS_SETTINGS = {
    "stability": 0.50, "similarity_boost": 0.75,
    "style": 0.35, "use_speaker_boost": True,
}

# =============================================================================
# FLASK APP
# =============================================================================
app = Flask(__name__)
CORS(app)

# =============================================================================
# DIALECTS — 4-class model from notebook (Algerian · Gulf · Levantine · Sudanese)
# =============================================================================
DIALECTS = {
    'Algerian':  {'label': 'Algerian',  'arabic': 'جزائري',  'flag': '🇩🇿', 'color': '#1DD1A1', 'code': 'ALG'},
    'Gulf':      {'label': 'Gulf',      'arabic': 'خليجي',   'flag': '🇸🇦', 'color': '#00D2D3', 'code': 'GLF'},
    'Levantine': {'label': 'Levantine', 'arabic': 'شامي',    'flag': '🇱🇧', 'color': '#F8B739', 'code': 'LEV'},
    'Sudanese':  {'label': 'Sudanese',  'arabic': 'سوداني',  'flag': '🇸🇩', 'color': '#EE5A24', 'code': 'SUD'},
}
DIALECT_LIST = ['Algerian', 'Gulf', 'Levantine', 'Sudanese']

# =============================================================================
# AUDIO / MODEL CONSTANTS — must match notebook training settings exactly
# =============================================================================
SR              = 16000      # notebook uses 16 kHz (TARGET_SR)
DURATION        = 30         # seconds to load
N_FFT           = 1024
HOP_LENGTH      = 256
N_MFCC          = 13
N_MELS          = 8

# =============================================================================
# SELECTED FEATURES — the 11 features the SVM classifier uses
# =============================================================================
# Feature names removed — only 11 selected features are computed directly

# =============================================================================
# TOP-11 SELECTED FEATURES (from model_bundle['feature_names'])
# Real per-class centroids are now stored inside model_bundle.pkl by Cell 22
# of the training notebook (keys: 'feature_centroids', 'lda_centroids',
# 'lda_model', 'lda_X_train', 'lda_y_train', 'lda_var_exp').
# App.py reads them at runtime — no hardcoded approximations.
# =============================================================================
SELECTED_FEATURES = [
    'mel_band_03_mean', 'spectral_flatness_std', 'mfcc_02_std',
    'delta2_mfcc_09_std', 'delta2_mfcc_04_std', 'delta_mfcc_02_std',
    'mfcc_03_mean', 'spectral_rolloff_mean', 'delta_mfcc_06_std',
    'spectral_entropy', 'f0_std',
]

# Human-readable short labels for the 11 selected features (for charts)
FEATURE_SHORT_LABELS = [
    'Mel-3\nEnergy', 'Spec\nFlatStd', 'MFCC-2\nStd',
    'ΔΔ-9\nStd', 'ΔΔ-4\nStd', 'Δ-2\nStd',
    'MFCC-3\nMean', 'Rolloff\nMean', 'Δ-6\nStd',
    'Spec\nEntropy', 'F0\nStd',
]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_bundle.pkl")
_model_bundle_cache = None


# =============================================================================
# 11-FEATURE EXTRACTION  — computes only the features the classifier needs
# SELECTED_FEATURES order must match model_bundle['feature_names'] exactly
# =============================================================================
def extract_features(y, sr):
    """
    Compute only the 11 selected features the SVM classifier uses.
    Returns:
        feat_dict  : dict  — {feature_name: float value}  (for explanation/UI)
        model_vec  : np.ndarray shape (11,) in SELECTED_FEATURES order
    """
    y = np.ascontiguousarray(y, dtype=np.float32)

    # Shared STFT — computed once, reused by all features
    D     = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    D_mag = np.abs(D)
    D_pow = D_mag ** 2

    feat_dict = {}

    # ── mel_band_03_mean ─────────────────────────────────────────────
    mel_spec = librosa.feature.melspectrogram(S=D_pow, sr=sr, n_mels=N_MELS)
    mel_db   = librosa.power_to_db(mel_spec)
    feat_dict['mel_band_03_mean'] = float(mel_db[2].mean())   # 0-indexed → band 3

    # ── spectral_flatness_std ────────────────────────────────────────
    flat = librosa.feature.spectral_flatness(S=D_mag)[0]
    feat_dict['spectral_flatness_std'] = float(flat.std())

    # ── MFCC 1-13 (need coeffs 2 & 3) ───────────────────────────────
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(D_pow), n_mfcc=N_MFCC, sr=sr)
    feat_dict['mfcc_02_std']  = float(mfcc[1].std())   # coeff index 1 = MFCC-2
    feat_dict['mfcc_03_mean'] = float(mfcc[2].mean())  # coeff index 2 = MFCC-3

    # ── delta + delta2 (need delta-2-std, delta-6-std, dd-4-std, dd-9-std) ──
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat_dict['delta_mfcc_02_std']   = float(delta[1].std())   # Δ-MFCC-2
    feat_dict['delta_mfcc_06_std']   = float(delta[5].std())   # Δ-MFCC-6
    feat_dict['delta2_mfcc_04_std']  = float(delta2[3].std())  # ΔΔ-MFCC-4
    feat_dict['delta2_mfcc_09_std']  = float(delta2[8].std())  # ΔΔ-MFCC-9

    # ── spectral_rolloff_mean ────────────────────────────────────────
    roll = librosa.feature.spectral_rolloff(S=D_mag, sr=sr)[0]
    feat_dict['spectral_rolloff_mean'] = float(roll.mean())

    # ── spectral_entropy ─────────────────────────────────────────────
    ps_norm = D_pow / (D_pow.sum(axis=0, keepdims=True) + 1e-10)
    feat_dict['spectral_entropy'] = float(
        -np.sum(ps_norm * np.log2(ps_norm + 1e-10), axis=0).mean()
    )

    # ── f0_std via YIN ───────────────────────────────────────────────
    try:
        f0 = librosa.yin(y, fmin=60.0, fmax=400.0,
                         sr=sr, frame_length=512, hop_length=HOP_LENGTH)
        f0_voiced = f0[f0 > 60.0]
        feat_dict['f0_std'] = float(f0_voiced.std()) if len(f0_voiced) > 0 else 0.0
    except Exception:
        feat_dict['f0_std'] = 0.0

    # ── Build model vector in exact SELECTED_FEATURES order ──────────
    model_vec = np.array([feat_dict[f] for f in SELECTED_FEATURES], dtype=np.float32)
    return feat_dict, model_vec


# =============================================================================
# MODEL LOADING
# =============================================================================
def get_model():
    """
    Load model_bundle.pkl — structure:
        { 'model': estimator,  'scaler': StandardScaler,
          'label_encoder': LabelEncoder,  'feature_names': [11 names],
          'model_name': str,  'target_sr': 16000,  'n_features': 11 }
    Falls back to a synthetic SVM if the file is missing.
    """
    global _model_bundle_cache
    if _model_bundle_cache is not None:
        return _model_bundle_cache

    if os.path.exists(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)
            if isinstance(bundle, dict) and 'model' in bundle and 'scaler' in bundle:
                # Quick sanity check
                test_input = bundle['scaler'].transform(np.zeros((1, bundle['n_features'])))
                bundle['model'].predict_proba(test_input)
                _model_bundle_cache = bundle
                logger.info("✅ Loaded model_bundle.pkl  model=%s  features=%d  classes=%s",
                            bundle.get('model_name'), bundle.get('n_features'),
                            list(bundle['label_encoder'].classes_))
                return _model_bundle_cache
        except Exception as e:
            logger.warning("Could not load model_bundle.pkl (%s) — building fallback.", e)

    logger.warning("⚠  model_bundle.pkl not found — building synthetic fallback model.")
    _model_bundle_cache = _build_fallback_bundle()
    return _model_bundle_cache


def _build_fallback_bundle():
    """Synthetic SVM — demo only, used when model_bundle.pkl is missing."""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.svm import SVC
    import numpy as np

    # Use simple offsets in a 11-dim unit space — no real centroids here
    np.random.seed(42)
    n_per   = 400
    n_feat  = len(SELECTED_FEATURES)
    offsets = np.eye(4, n_feat) * 2.0   # each dialect shifted along one axis
    X_parts, y_parts = [], []
    for i, d in enumerate(DIALECT_LIST):
        mu  = offsets[i]
        X   = np.random.randn(n_per, n_feat).astype(np.float32) * 0.8 + mu
        X_parts.append(X)
        y_parts.extend([d] * n_per)

    X_all = np.vstack(X_parts)
    le    = LabelEncoder().fit(y_parts)
    y_enc = le.transform(y_parts)

    sc  = StandardScaler().fit(X_all)
    clf = SVC(kernel='rbf', C=10, gamma='scale',
              probability=True, class_weight='balanced', random_state=42)
    clf.fit(sc.transform(X_all), y_enc)

    return {
        'model': clf, 'scaler': sc, 'label_encoder': le,
        'feature_names': SELECTED_FEATURES, 'model_name': 'fallback_SVM',
        'target_sr': SR, 'n_features': len(SELECTED_FEATURES),
    }


# =============================================================================
# CLASSIFICATION
# =============================================================================
def classify_audio(y, sr):
    """
    Returns:
        pred_key    : str  — e.g. 'Levantine'
        proba_dict  : dict — { 'Algerian': 0.12, 'Gulf': 0.05, ... }
        explanation : list[str]
        feat_dict   : dict — {feature_name: float}
        model_vec   : np.ndarray shape (11,)
    """
    bundle   = get_model()
    scaler   = bundle['scaler']
    clf      = bundle['model']
    le       = bundle['label_encoder']
    classes  = list(le.classes_)          # ['Algerian', 'Gulf', 'Levantine', 'Sudanese']

    feat_dict, model_vec = extract_features(y, sr)

    X_scaled = scaler.transform(model_vec.reshape(1, -1))
    proba    = clf.predict_proba(X_scaled)[0]
    pred_idx = int(np.argmax(proba))
    pred_key = classes[pred_idx]

    # Build proba dict covering all 4 dialects
    proba_dict = {d: 0.0 for d in DIALECT_LIST}
    for i, cls in enumerate(classes):
        if cls in DIALECTS:
            proba_dict[cls] = float(proba[i])

    # Build explanation
    info = DIALECTS[pred_key]
    explanation = [
        f"Predicted <strong>{info['label']}</strong> "
        f"({info['arabic']}) with "
        f"<strong>{proba_dict[pred_key] * 100:.1f}%</strong> confidence."
    ]

    # Key feature values (directly from feat_dict)
    kv = {
        'Mel-band 3 energy':  f"{feat_dict['mel_band_03_mean']:.2f} dB",
        'Spectral rolloff':   f"{feat_dict['spectral_rolloff_mean']:.0f} Hz",
        'MFCC-2 std':         f"{feat_dict['mfcc_02_std']:.2f}",
        'MFCC-3 mean':        f"{feat_dict['mfcc_03_mean']:.2f}",
        'Spectral entropy':   f"{feat_dict['spectral_entropy']:.3f}",
        'F0 std':             f"{feat_dict['f0_std']:.1f} Hz",
    }
    explanation.append("Key features — " + ", ".join(f"{k}: {v}" for k, v in kv.items()) + ".")

    dialect_notes = {
        'Algerian':  "Elevated mel-band energy and moderate spectral entropy match Algerian Arabic — "
                     "characterised by compressed vowel clusters, /dz/ affricates, and French code-switching.",
        'Gulf':      "Lower spectral rolloff and conservative F0 variation align with Gulf Arabic — "
                     "marked by preserved /q/, pharyngeal articulation, and measured prosodic contours.",
        'Levantine': "High F0 standard deviation and elevated Δ-MFCC std reflect Levantine Arabic — "
                     "noted for rising-falling intonation, /q/→/ʔ/ shift, and expressive prosody.",
        'Sudanese':  "Reduced mel-band energy and low spectral entropy are consistent with Sudanese Arabic — "
                     "influenced by Nilo-Saharan substrate, slower pace, and Classical Arabic retention.",
    }
    explanation.append(dialect_notes.get(pred_key, ""))

    return pred_key, proba_dict, explanation, feat_dict, model_vec


# =============================================================================
# AUDIO UTILITIES
# =============================================================================
def b64_to_audio(b64_str):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(base64.b64decode(b64_str))
        tmp = f.name
    try:
        y, sr = librosa.load(tmp, sr=SR, mono=True, duration=DURATION)
    finally:
        os.unlink(tmp)
    return y, sr


def audio_to_b64(y, sr):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#0d0d1a', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# =============================================================================
# VISUALIZATIONS
# =============================================================================
def make_spectrogram(y, sr, dialect_key):
    """Mel spectrogram with spectral centroid overlay."""
    fig, ax = plt.subplots(figsize=(11, 3.5), facecolor='#0d0d1a')
    ax.set_facecolor('#141420')

    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    img  = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel',
                                    ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB', shrink=0.9)

    sc_vals = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sc_t    = librosa.frames_to_time(np.arange(len(sc_vals)), sr=sr)
    ax.plot(sc_t, sc_vals, color='cyan', linewidth=1.5, alpha=0.75,
            label=f'Spectral Centroid  μ={sc_vals.mean():.0f} Hz')
    ax.legend(fontsize=7, facecolor='#1e1e30', labelcolor='#ddd8cc', loc='upper right')

    info  = DIALECTS[dialect_key]
    color = info['color']
    ax.set_title(
        f"Mel Spectrogram  —  {info['label']} ({info['arabic']}) Detected",
        color=color, fontsize=10, fontweight='bold'
    )
    ax.tick_params(colors='#9090aa', labelsize=8)
    ax.xaxis.label.set_color('#9090aa')
    ax.yaxis.label.set_color('#9090aa')
    for sp in ax.spines.values():
        sp.set_edgecolor('#2e2e48')
    plt.tight_layout(pad=0.5)
    return fig_to_b64(fig)


def make_feature_chart(model_vec, detected_key, proba_dict):
    """
    Two-panel decision visualization — both panels use REAL model data.

      Left  — 3D LDA scatter with improved readability.
      Right — Grouped bar chart per feature:
                • 4 dialect centroid bars (coloured, semi-transparent)
                • 1 white bar for the test sample
              Features sorted by |test - predicted centroid| (most diagnostic first).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    bundle = get_model()
    le     = bundle['label_encoder']
    scaler = bundle['scaler']
    colors = {d: DIALECTS[d]['color'] for d in DIALECT_LIST}

    has_real_lda            = ('lda_model' in bundle and 'lda_X_train' in bundle and 'lda_y_train' in bundle)
    has_real_feat_centroids = 'feature_centroids' in bundle

    test_scaled_vec = scaler.transform(model_vec.reshape(1, -1))[0]

    if has_real_lda:
        lda           = bundle['lda_model']
        X_lda_train   = bundle['lda_X_train']
        y_lda_int     = bundle['lda_y_train']
        var_exp       = bundle['lda_var_exp']
        centroids_lda = bundle['lda_centroids']
        y_lda_str     = np.array([le.classes_[i] for i in y_lda_int])
        test_lda      = lda.transform(test_scaled_vec.reshape(1, -1))[0]
        lda_title_suffix = f"Real training data  ({len(X_lda_train)} samples)"
    else:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        np.random.seed(42)
        N_PER  = 300
        n_feat = len(SELECTED_FEATURES)
        X_fb, y_fb = [], []
        offsets = np.eye(4, n_feat) * 1.5
        for i, d in enumerate(DIALECT_LIST):
            pts = np.random.randn(N_PER, n_feat) * 0.6 + offsets[i]
            X_fb.append(pts)
            y_fb.extend([d] * N_PER)
        X_fb     = np.vstack(X_fb)
        y_fb_arr = np.array(y_fb)
        _lda = LinearDiscriminantAnalysis(n_components=3)
        X_lda_train   = _lda.fit_transform(X_fb, y_fb_arr)
        var_exp       = _lda.explained_variance_ratio_
        test_lda      = _lda.transform(test_scaled_vec.reshape(1, -1))[0]
        y_lda_str     = y_fb_arr
        centroids_lda = {d: X_lda_train[y_lda_str == d].mean(axis=0) for d in DIALECT_LIST}
        lda_title_suffix = "Approximate — run Cell 22 for real data"

    if has_real_feat_centroids:
        feat_cents         = bundle['feature_centroids']
        right_panel_source = "real training centroids"
    else:
        feat_cents         = {d: np.zeros(len(SELECTED_FEATURES)) for d in DIALECT_LIST}
        right_panel_source = "approximate — run Cell 22"

    # ── Figure layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 7), facecolor='#0d0d1a')
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38,
                            left=0.03, right=0.97, top=0.91, bottom=0.09)

    # ══════════════════════════════════════════════════════════════════
    # PANEL 1 — 3D LDA scatter (improved readability)
    # ══════════════════════════════════════════════════════════════════
    ax3d = fig.add_subplot(gs[0], projection='3d')
    ax3d.set_facecolor('#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')

    for d in DIALECT_LIST:
        mask = y_lda_str == d
        pts  = X_lda_train[mask]
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                     c=colors[d], s=5, alpha=0.20, linewidths=0)

    for d in DIALECT_LIST:
        cx, cy, cz = centroids_lda[d]
        ax3d.scatter(cx, cy, cz, c='none', s=420, marker='o',
                     edgecolors=colors[d], linewidths=1.4, zorder=5, alpha=0.5)
        ax3d.scatter(cx, cy, cz, c=colors[d], s=240, marker='*',
                     edgecolors='white', linewidths=0.5, zorder=6, alpha=1.0)

    tx, ty, tz = test_lda
    for d in DIALECT_LIST:
        cx, cy, cz = centroids_lda[d]
        is_pred = (d == detected_key)
        ax3d.plot([tx, cx], [ty, cy], [tz, cz],
                  color=colors[d],
                  linewidth=2.5 if is_pred else 0.8,
                  alpha=0.95 if is_pred else 0.20,
                  linestyle='-' if is_pred else '--',
                  zorder=4)

    ax3d.scatter(tx, ty, tz, c='#ffffff', s=380, marker='o',
                 edgecolors=DIALECTS[detected_key]['color'],
                 linewidths=3.5, zorder=10)
    ax3d.text(tx, ty, tz + 0.55, '▶ Your Audio',
              color='#ffffff', fontsize=8.5, fontweight='bold',
              ha='center', va='bottom', zorder=11)

    pct = [f'{v*100:.0f}%' for v in var_exp[:3]]
    ax3d.set_xlabel(f'LDA-1  ({pct[0]})', color='#7070aa', fontsize=7.5, labelpad=6)
    ax3d.set_ylabel(f'LDA-2  ({pct[1]})', color='#7070aa', fontsize=7.5, labelpad=6)
    ax3d.set_zlabel(f'LDA-3  ({pct[2]})', color='#7070aa', fontsize=7.5, labelpad=6)
    ax3d.tick_params(colors='#444466', labelsize=6.5, pad=2)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor('#181830')
    ax3d.yaxis.pane.set_edgecolor('#181830')
    ax3d.zaxis.pane.set_edgecolor('#181830')
    ax3d.grid(True, color='#181830', linewidth=0.6)
    ax3d.view_init(elev=22, azim=130)

    for i, d in enumerate(DIALECT_LIST):
        ax3d.text2D(0.02, 0.96 - i * 0.078,
                    f'★  {DIALECTS[d]["flag"]} {d}',
                    transform=ax3d.transAxes,
                    color=colors[d], fontsize=7.5, fontweight='bold', va='top')

    ax3d.set_title(
        f'3D LDA Decision Space  |  {lda_title_suffix}\n'
        f'Predicted: {DIALECTS[detected_key]["flag"]} {detected_key} '
        f'({proba_dict[detected_key]*100:.1f}% confidence)',
        color='#e8e4d9', fontsize=9, fontweight='bold', pad=10
    )

    # ══════════════════════════════════════════════════════════════════
    # PANEL 2 — Grouped horizontal bar chart
    #
    # Each feature band contains 5 bars stacked vertically:
    #   4 dialect centroid bars (their colour, semi-transparent)
    #   1 white bar  = your test sample value
    #
    # All values are z-scores (StandardScaler space).
    # Features sorted by |test - predicted centroid| (most diagnostic first).
    # ══════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#0e0e1c')

    n_feat      = len(SELECTED_FEATURES)
    n_dialects  = len(DIALECT_LIST)
    n_bars      = n_dialects + 1          # 4 dialects + 1 test sample

    pred_centroid = np.array(feat_cents[detected_key], dtype=float)
    sort_order    = np.argsort(np.abs(test_scaled_vec - pred_centroid))[::-1]

    bar_h   = 0.62                        # height of a single bar
    gap     = 0.06                        # gap between bars within a band
    band_h  = n_bars * (bar_h + gap) + 1.4   # total height per feature band

    y_ticks_feat  = []
    y_ticks_label = []

    # bar order within each band: dialects first, test sample last (top)
    bar_order   = DIALECT_LIST + ['__test__']
    bar_colors_map = {d: colors[d] for d in DIALECT_LIST}
    bar_colors_map['__test__'] = '#ffffff'

    for row_idx, feat_idx in enumerate(sort_order):
        feat_label  = FEATURE_SHORT_LABELS[feat_idx]
        test_val    = test_scaled_vec[feat_idx]
        band_centre = -(row_idx * band_h)

        # vertical positions of bars within this band
        total_bar_span = n_bars * (bar_h + gap) - gap
        bar_bottoms = np.linspace(
            band_centre - total_bar_span / 2,
            band_centre + total_bar_span / 2,
            n_bars
        )

        y_ticks_feat.append(band_centre)
        y_ticks_label.append(feat_label)

        # faint band separator
        ax2.axhline(band_centre - band_h / 2 + 0.3,
                    color='#1a1a38', linewidth=0.7, zorder=0)

        for b_idx, bar_key in enumerate(bar_order):
            y_bar   = bar_bottoms[b_idx]
            b_color = bar_colors_map[bar_key]

            if bar_key == '__test__':
                val   = test_val
                alpha = 0.95
                ec    = DIALECTS[detected_key]['color']
                ew    = 1.0
                lw_bar = 0      # no edge-line (filled white)
            else:
                val   = float(feat_cents[bar_key][feat_idx])
                alpha = 0.55
                ec    = 'none'
                ew    = 0
                lw_bar = 0

            ax2.barh(y_bar, val, height=bar_h,
                     color=b_color, alpha=alpha,
                     edgecolor=ec, linewidth=ew, zorder=2)

            # value label
            off = 0.07 if val >= 0 else -0.07
            ha  = 'left' if val >= 0 else 'right'
            label_color = '#ffffff' if bar_key == '__test__' else b_color
            label_alpha = 1.0      if bar_key == '__test__' else 0.82
            fontweight  = 'bold'   if bar_key == '__test__' else 'normal'
            fontsize    = 6.2      if bar_key == '__test__' else 5.5
            ax2.text(val + off, y_bar,
                     f'{val:+.2f}\u03c3',
                     va='center', ha=ha,
                     color=label_color, alpha=label_alpha,
                     fontweight=fontweight, fontsize=fontsize, zorder=5)

    # ── Axes dressing ──────────────────────────────────────────────────
    ax2.axvline(0, color='#3a3a88', linewidth=1.0, zorder=1)
    ax2.set_yticks(y_ticks_feat)
    ax2.set_yticklabels(y_ticks_label, fontsize=7.5, color='#9090bb')
    ax2.set_xlabel('Feature value  (z-score, StandardScaler space)',
                   color='#8888aa', fontsize=8)
    ax2.tick_params(axis='x', colors='#555577', labelsize=7)
    ax2.tick_params(axis='y', length=0)
    for sp in ax2.spines.values():
        sp.set_edgecolor('#1e1e3a')

    # ── Legend ─────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=colors[d], alpha=0.6,
              label=f'{DIALECTS[d]["flag"]} {d}  centroid')
        for d in DIALECT_LIST
    ]
    legend_items.append(
        Patch(facecolor='#ffffff', alpha=0.95,
              edgecolor=DIALECTS[detected_key]['color'], linewidth=1.0,
              label='Your audio')
    )
    ax2.legend(handles=legend_items,
               loc='lower right', fontsize=6.5,
               facecolor='#0d0d1a', edgecolor='#2e2e48',
               labelcolor='#ccccdd', framealpha=0.92,
               handlelength=1.6)

    ax2.set_title(
        f'Feature Values: Dialect Centroids vs Your Sample\n'
        f'Coloured bars = class centroid  |  White bar = your audio  |  Source: {right_panel_source}',
        color='#e8e4d9', fontsize=8.5, fontweight='bold'
    )

    return fig_to_b64(fig)

def bundle_name():
    try:
        return get_model().get('model_name', 'model_bundle')
    except Exception:
        return 'model_bundle'


def make_mixed_spectrogram(y, sr, label="Mixed Audio"):
    fig, ax = plt.subplots(figsize=(11, 3.5), facecolor='#0d0d1a')
    ax.set_facecolor('#141420')
    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    img  = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel',
                                    ax=ax, cmap='plasma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB', shrink=0.9)
    ax.set_title(label, color='#e8e4d9', fontsize=10)
    ax.tick_params(colors='#9090aa', labelsize=8)
    ax.xaxis.label.set_color('#9090aa')
    ax.yaxis.label.set_color('#9090aa')
    for sp in ax.spines.values():
        sp.set_edgecolor('#2e2e48')
    plt.tight_layout(pad=0.5)
    return fig_to_b64(fig)


# =============================================================================
# GROQ TRANSLATION
# =============================================================================
def groq_translate(source_text, target_dialect):
    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        raise ValueError("Groq API key not set.")

    client = Groq(api_key=GROQ_API_KEY)

    dialect_guides = {
        "Algerian": """
DIALECT: Algerian Arabic (الدارجة الجزائرية) — Algiers register
PHONOLOGY: /q/ → /g/ or stays /q/. Heavy French loanword integration. Vowel reduction.
GRAMMAR: ما...ش negation. راه/رانا + verb for progressive. غادي/راح for future.
LEXICON: واش؟ (what?) | وين؟ (where?) | علاش؟ (why?) | بزاف (a lot) | درك (now)
مليح / بيهي (good/fine) | واكا (okay) | صاحبي (friend) | كيراك؟ (how are you?)
STYLE: Fast, contracted, Arabic + French + Berber mix. Very colloquial.""",

        "Gulf": """
DIALECT: Gulf Arabic (الخليجي) — Saudi/UAE/Kuwait register
PHONOLOGY: /q/ preserved as /q/ or /g/. /k/→/tʃ/ before front vowels.
GRAMMAR: ما before verb for negation. بـ or رح for future.
LEXICON: أبي / ما أبي (want/don't want) | وين؟ | إيش/شو؟ | ليش؟
زين / تمام (good/okay) | واجد/هواية (a lot) | الحين/دحين (now) | خوي (bro)
STYLE: Conservative, measured. Gulf hospitality expressions frequent.""",

        "Levantine": """
DIALECT: Levantine Arabic (الشامي) — Lebanon/Syria/Jordan/Palestine
PHONOLOGY: /q/→/ʔ/ urban. /ʒ/ maintained. Distinctive stress patterns.
GRAMMAR: ما + verb OR مش for negation. عم بـ progressive. رح for future.
LEXICON: بدي / ما بدي | شو؟ | وين؟ | ليش؟ | هيدا/هيدي (this)
كتير (very) | هلق (now) | بعدين (later) | يعني | حكى (talked)
STYLE: Musical, fast-paced, warm. Rich diminutives: حبيبي، يا قلبي.""",

        "Sudanese": """
DIALECT: Sudanese Arabic (السوداني) — Khartoum register
PHONOLOGY: /q/ preserved. Slower pace. Nuba/Nilotic substrate influence.
GRAMMAR: Similar to Classical Arabic. ما negation.
LEXICON: شنو (what?) | وين (where?) | ليه (why?) | كتير (a lot) | دقيقة (now/wait)
كويس/حلو (good/nice) | أيوه (yes) | يا زول (hey man) | والله
STYLE: Measured, dignified. Classical Arabic retention strong.""",

        "Egyptian": """
DIALECT: Egyptian Arabic (المصري) — Cairo register
PHONOLOGY: /q/→/ʔ/. /ǧ/→/g/.
GRAMMAR: مش / ما...ش negation. بـ progressive. هـ future.
LEXICON: إيه? | فين? | ليه? | كده | تمام | أيوه | أوي (very) | عايز (want)
STYLE: Warm, expressive. Widely understood across Arab world.""",

        "Iraqi": """
DIALECT: Iraqi Arabic (العراقي) — Baghdad/Basra register
PHONOLOGY: /q/→/g/ universally. /k/→/tʃ/ regionally.
GRAMMAR: ما negation. بـ progressive. رح future.
LEXICON: شنو؟/شگو؟ | وين؟ | زين (good) | هواية (a lot) | هسه (now) | يمعود (friend)
STYLE: Direct, rhythmic. Strong consonants.""",

        "Moroccan": """
DIALECT: Moroccan Arabic (الدارجة المغربية) — Casablanca/Rabat
PHONOLOGY: Heavy vowel dropping. French & Spanish loanwords.
GRAMMAR: ما...ش negation. كا+يـ progressive. غادي future.
LEXICON: بغيت/ما بغيتش | آش؟/شنو؟ | فين؟ | علاش؟ | مزيان (good) | واخا (okay) | بزاف (a lot) | دابا (now)
STYLE: Rapid, heavy consonant clusters. Arabic + French + Spanish + Tamazight.""",
    }

    guide = dialect_guides.get(
        target_dialect,
        f"Translate authentically into {target_dialect} Arabic dialect using native vocabulary and rhythm."
    )

    system_prompt = f"""You are Dr. Layla Al-Amine, PhD in Arabic dialectology (SOAS). 
Translate the given Arabic text into perfectly authentic {target_dialect} Arabic dialect.

{guide}

RULES:
1. Replace every MSA word with its authentic dialect equivalent.
2. Use the dialect's exact negation, progressive, and future constructions.
3. Write naturally, as a native would text a friend.
4. Preserve all meaning — nothing added, nothing lost.
5. Return ONLY the translated text. No explanation, no transliteration, no preamble."""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Translate into {target_dialect} Arabic:\n\n{source_text}"}
        ],
        temperature=0.45, max_tokens=2048, top_p=0.9, stream=False,
    )
    result = completion.choices[0].message.content.strip()
    lines  = result.split('\n')
    arabic_lines = [l for l in lines if l.strip() and any('\u0600' <= c <= '\u06FF' for c in l)]
    return '\n'.join(arabic_lines) if arabic_lines else result


# =============================================================================
# ELEVENLABS TTS
# =============================================================================
def elevenlabs_tts(text, voice_id):
    if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "YOUR_ELEVENLABS_API_KEY_HERE":
        raise ValueError("ElevenLabs API key not set.")
    url     = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json",
               "xi-api-key": ELEVENLABS_API_KEY}
    payload = {
        "text": text, "model_id": ELEVENLABS_MODEL,
        "voice_settings": {
            "stability":       ELEVENLABS_SETTINGS["stability"],
            "similarity_boost": ELEVENLABS_SETTINGS["similarity_boost"],
        },
    }
    response = requests.post(url, json=payload, headers=headers, timeout=30, verify=False)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    logger.error("ElevenLabs TTS Error %d: %s", response.status_code, response.text)
    raise RuntimeError(f"ElevenLabs TTS failed ({response.status_code}): {response.text}")


# =============================================================================
# ELEVENLABS STT
# =============================================================================
def elevenlabs_stt(y, sr):
    if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "YOUR_ELEVENLABS_API_KEY_HERE":
        raise ValueError("ElevenLabs API key not set.")
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV')
    buf.seek(0)
    response = requests.post(
        "https://api.elevenlabs.io/v1/speech-to-text",
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        files={"file": ("audio.wav", buf, "audio/wav")},
        verify=False,
        data={"model_id": "scribe_v2", "language_code": "ar",
              "timestamps_granularity": "word", "tag_audio_events": "false"},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs STT failed ({response.status_code}): {response.text}")
    data      = response.json()
    full_text = data.get("text", "").strip()
    words_data = [
        {"word": item.get("text","").strip(),
         "start": round(float(item.get("start", 0)), 2),
         "end":   round(float(item.get("end",   0)), 2)}
        for item in data.get("words", []) if item.get("type") == "word"
    ]
    return full_text, words_data


# =============================================================================
# ROUTES
# =============================================================================
@app.route('/health', methods=['GET'])
def health():
    bundle = get_model()
    return jsonify({
        "status":           "ok",
        "groq_configured":  bool(GROQ_API_KEY and GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE"),
        "eleven_configured":bool(ELEVENLABS_API_KEY and ELEVENLABS_API_KEY != "YOUR_ELEVENLABS_API_KEY_HERE"),
        "model_loaded":     os.path.exists(MODEL_PATH),
        "model_name":       bundle.get("model_name", "unknown"),
        "n_features":       bundle.get("n_features", 0),
        "dialects":         DIALECT_LIST,
        "stt_provider":     "ElevenLabs Scribe v2",
        "tts_provider":     "ElevenLabs Multilingual v2",
    })


@app.route('/voices', methods=['GET'])
def get_voices():
    safe = [
        {"id": k, "label": k, "language": "Arabic",
         "gender": "Female" if k in ("Sarah", "Jessica") else "Male",
         "accent": "Multilingual"}
        for k in ELEVENLABS_VOICES
    ]
    return jsonify({"voices": safe})


@app.route('/analyze_and_transcribe', methods=['POST'])
def analyze_and_transcribe():
    try:
        data = request.json
        b64  = data.get('audio', '')
        if not b64:
            return jsonify({"error": "No audio provided"}), 400

        y, sr = b64_to_audio(b64)
        dialect, proba_dict, explanation, feat_dict, model_vec = classify_audio(y, sr)
        spec_b64    = make_spectrogram(y, sr, dialect)
        feature_b64 = make_feature_chart(model_vec, dialect, proba_dict)

        text, words_data = "", []
        try:
            text, words_data = elevenlabs_stt(y, sr)
            # Dialect keyword hints
            kw_map = {
                'Algerian':  ['واش', 'بزاف', 'غدوا', 'راك', 'كيراك', 'درك'],
                'Gulf':      ['وش', 'كيفك', 'زين', 'وين', 'إيش', 'الحين', 'دحين'],
                'Levantine': ['شو', 'هيدا', 'متل', 'كتير', 'هلق', 'بدي', 'عم'],
                'Sudanese':  ['شنو', 'يا زول', 'كويس', 'دقيقة'],
            }
            found_kws = [w for w in text.split()
                         if any(k in w for k in kw_map.get(dialect, []))]
            if found_kws:
                explanation.append("Dialect keywords detected: " + ", ".join(set(found_kws)) + ".")
        except Exception as stt_err:
            logger.warning("ElevenLabs STT failed: %s", stt_err)

        # Expose the 11 model features in raw form for the frontend
        features_raw = {
            "mel_band_03":        feat_dict['mel_band_03_mean'],
            "spectral_rolloff":   feat_dict['spectral_rolloff_mean'],
            "mfcc_02_std":        feat_dict['mfcc_02_std'],
            "mfcc_03_mean":       feat_dict['mfcc_03_mean'],
            "spectral_entropy":   feat_dict['spectral_entropy'],
            "f0_std":             feat_dict['f0_std'],
        }

        return jsonify({
            "dialect":        dialect,
            "dialect_label":  DIALECTS[dialect]["label"],
            "dialect_arabic": DIALECTS[dialect]["arabic"],
            "dialect_flag":   DIALECTS[dialect]["flag"],
            "dialect_color":  DIALECTS[dialect]["color"],
            "explanation":    explanation,
            "proba":          proba_dict,
            "spectrogram":    spec_b64,
            "feature_chart":  feature_b64,
            "text":           text.strip(),
            "words":          words_data,
            "features_raw":   features_raw,
        })
    except Exception as e:
        logger.error("analyze_and_transcribe error: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/mix_and_analyze', methods=['POST'])
def mix_and_analyze():
    try:
        data   = request.json
        b64_1  = data.get('audio1', '')
        b64_2  = data.get('audio2', '')
        weight = float(data.get('weight', 0.5))
        mode   = data.get('mode', 'overlap')

        if not b64_1 or not b64_2:
            return jsonify({"error": "Two audio files required"}), 400

        y1, sr  = b64_to_audio(b64_1)
        y2, sr2 = b64_to_audio(b64_2)
        if sr2 != sr:
            y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr)

        max_len = max(len(y1), len(y2))

        if mode == 'overlap':
            y1_pad = np.pad(y1, (0, max_len - len(y1)))
            y2_pad = np.pad(y2, (0, max_len - len(y2)))
            y_mix  = weight * y1_pad + (1 - weight) * y2_pad
            label  = (f"Overlap Mix  ({weight*100:.0f}% Audio-A  +  "
                      f"{(1-weight)*100:.0f}% Audio-B)  —  {max_len/sr:.1f}s")
        else:
            n1  = int(len(y1) * weight)
            n2  = int(len(y2) * (1 - weight))
            seq = np.concatenate([y1[:n1], y2[:n2]])
            y_mix  = np.pad(seq, (0, max(0, max_len - len(seq))))[:max_len]
            label  = (f"Sequential  ({weight*100:.0f}% A  →  "
                      f"{(1-weight)*100:.0f}% B)  —  {max_len/sr:.1f}s")

        peak = np.max(np.abs(y_mix))
        if peak > 0:
            y_mix = y_mix / peak * 0.95

        dialect, proba_dict, explanation, feat_dict, model_vec = classify_audio(y_mix, sr)
        spec_b64     = make_mixed_spectrogram(y_mix, sr, label)
        feature_b64  = make_feature_chart(model_vec, dialect, proba_dict)
        mix_audio_b64 = audio_to_b64(y_mix, sr)

        return jsonify({
            "dialect":        dialect,
            "dialect_label":  DIALECTS[dialect]["label"],
            "dialect_arabic": DIALECTS[dialect]["arabic"],
            "dialect_flag":   DIALECTS[dialect]["flag"],
            "dialect_color":  DIALECTS[dialect]["color"],
            "proba":          proba_dict,
            "explanation":    explanation,
            "spectrogram":    spec_b64,
            "feature_chart":  feature_b64,
            "mixed_audio":    mix_audio_b64,
            "mode":           mode,
            "label":          label,
        })
    except Exception as e:
        logger.error("mix_and_analyze error: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/translate', methods=['POST'])
def translate():
    try:
        data          = request.json
        source_text   = (data.get('source_text')   or '').strip()
        target_dialect= (data.get('target_dialect') or '').strip()
        voice_id_key  = (data.get('voice_id')       or 'Sarah').strip()

        if not source_text:
            return jsonify({"error": "source_text is required"}), 400
        if not target_dialect:
            return jsonify({"error": "target_dialect is required"}), 400

        el_voice_id = ELEVENLABS_VOICES.get(voice_id_key)
        if not el_voice_id:
            return jsonify({"error": f"Unknown voice '{voice_id_key}'. Call GET /voices."}), 400

        translated_text = groq_translate(source_text, target_dialect)
        audio_b64       = elevenlabs_tts(translated_text, el_voice_id)

        return jsonify({
            "translated_text": translated_text,
            "audio":           audio_b64,
            "audio_format":    "mp3",
            "voice_used":      voice_id_key,
            "target_dialect":  target_dialect,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 503
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        logger.error("Unexpected error in /translate: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


# =============================================================================
# ENTRYPOINT
# =============================================================================
if __name__ == '__main__':
    get_model()   # pre-load on startup
    app.run(host='0.0.0.0', port=5000, debug=False)