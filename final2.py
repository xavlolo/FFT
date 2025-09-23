# app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ===========================
# Core linear-algebra helpers
# ===========================
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)
id2 = np.eye(2, dtype=complex)

def kronN(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def two_qubit(op1, q1, op2, q2, Ntot):
    ops = [id2] * Ntot
    ops[q1] = op1
    ops[q2] = op2
    return kronN(ops)

# ===========================
# Hamiltonians
# ===========================
def build_H_full_general(L, K, J):
    """
    Full 2^(2L) x 2^(2L) Hamiltonian. Use only for small L (<=3 recommended).
    H = sum_i K[i] sz_i sz_{L+i} + sum_i (J[i]/2)(sx_i sx_{i+1} + sy_i sy_{i+1})
    """
    Ntot = 2 * L
    dim = 2 ** Ntot
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(L):
        H += K[i] * two_qubit(sz, i, sz, L + i, Ntot)
    for i in range(L - 1):
        H += 0.5 * J[i] * (
            two_qubit(sx, i, sx, i + 1, Ntot) + two_qubit(sy, i, sy, i + 1, Ntot)
        )
    return H

def Heff_block_general(K, J, m):
    """
    Effective LxL block for one-top + fixed bottom=m.
    Diagonals: d_i^{(m)} = S if i==m; else S - 2*(K[i] + K[m]), S=sum(K).
    Off-diagonals: J[i] on (i,i+1) and (i+1,i).
    """
    K = np.asarray(K, float)
    J = np.asarray(J, float)
    L = len(K)
    S = float(np.sum(K))
    d = np.full(L, S, float)
    for i in range(L):
        if i != m:
            d[i] = S - 2 * (K[i] + K[m])
    Heff = np.diag(d)
    for i in range(L - 1):
        Heff[i, i + 1] = Heff[i + 1, i] = J[i]
    return Heff

def build_H_topbottom_blockdiag(K, J):
    """Block-diagonal Hamiltonian on the one-top+one-bottom subspace (shape L^2 x L^2).
    Block m occupies rows/cols [m*L : (m+1)*L] and equals Heff_block_general(K,J,m)."""
    K = np.asarray(K, float); J = np.asarray(J, float)
    L = len(K)
    Htb = np.zeros((L*L, L*L), float)
    for m in range(L):
        Hm = Heff_block_general(K, J, m)
        s = slice(m*L, (m+1)*L)
        Htb[s, s] = Hm
    return Htb

# ===========================
# Dynamics on a block (LxL)
# ===========================
def eigh_data(H):
    w, V = np.linalg.eigh(H)
    return w.real, V

def evolve_state_in_H(H, psi0_vec, times, hbar=1.0):
    w, V = eigh_data(H)
    c0 = V.conj().T @ psi0_vec
    kets = np.empty((len(times), H.shape[0]), dtype=complex)
    for i, t in enumerate(times):
        kets[i] = V @ (np.exp(-1j * w * t / hbar) * c0)
    return kets, (w, V, c0)

def return_amplitude_block_general(K, J, bottom_m, site, times, hbar=1.0):
    H = Heff_block_general(K, J, bottom_m)
    psi0 = np.zeros(H.shape[0], complex); psi0[site] = 1.0
    kets, _ = evolve_state_in_H(H, psi0, times, hbar=hbar)
    return kets[:, site]

# ===========================
# FFT utilities
# ===========================
def _parabolic_refine(x, y, i):
    if i <= 0 or i >= len(y) - 1:
        return x[i]
    y0 = np.log(max(y[i - 1], 1e-300))
    y1 = np.log(max(y[i], 1e-300))
    y2 = np.log(max(y[i + 1], 1e-300))
    denom = 2.0 * (y0 - 2.0 * y1 + y2)
    if denom == 0:
        return x[i]
    delta = (y0 - y2) / denom
    dx = 0.5 * (x[i + 1] - x[i - 1])
    return x[i] + delta * dx

def fft_from_timeseries(
    f_t,
    times,
    n_levels=3,
    pad_factor=16,
    use_hann=True,
    include_dc=False,
    dc_guard=1.5,
    post_prune=True,
    min_weight=1e-6,
    keep_top=None,
    guard_mult=1.2,
):
    """
    Estimate discrete spectral lines E_k and nonnegative weights p_k from a return amplitude f(t).
    We use FFT to find candidate peaks at angular frequencies ω, then set E = -ω
    because f(t) ≈ Σ p_k e^{-i E_k t} with p_k ≥ 0 for return amplitudes.
    """
    times = np.asarray(times, float)
    f_t = np.asarray(f_t)

    if times.ndim != 1 or times.size < 2:
        raise ValueError("`times` must be a 1D array with length >= 2.")
    if f_t.shape[0] != times.shape[0]:
        raise ValueError("`f_t` and `times` must have the same length.")
    dts = np.diff(times)
    if not np.allclose(dts, dts[0]):
        raise ValueError("`times` must be uniformly spaced for FFT-based estimation.")
    dt = float(dts[0])

    win = np.hanning(len(f_t)) if use_hann else np.ones_like(f_t)
    xw = f_t * win

    Nfft = int(max(1, pad_factor) * len(xw))
    F = dt * np.fft.fft(xw, n=Nfft)
    omega = 2.0 * np.pi * np.fft.fftfreq(Nfft, d=dt)
    S = np.abs(F) ** 2
    S_norm = S / (S.max() if S.max() > 0 else 1.0)

    T = times[-1] - times[0]
    domega = (2 * np.pi) / (T + 1e-12)

    loc = np.where((S[1:-1] > S[:-2]) & (S[1:-1] >= S[2:]))[0] + 1
    if include_dc:
        if S[0] > 0:
            loc = np.unique(np.concatenate(([0], loc)))
    else:
        loc = loc[np.abs(omega[loc]) > dc_guard * domega]

    if loc.size == 0:
        return np.array([]), np.array([]), -omega, S_norm

    cand = loc[np.argsort(S[loc])[::-1]]
    picked = []
    guard = guard_mult * domega
    for k in cand:
        if all(abs(omega[k] - omega[j]) > guard for j in picked):
            picked.append(k)
        if len(picked) >= max(n_levels, 1):
            break
    loc = np.array(picked, dtype=int)

    omega_hat = np.array([_parabolic_refine(omega, S, i) for i in loc])
    E_hat = -omega_hat

    A = np.column_stack([np.exp(-1j * E * times) * win for E in E_hat])
    c, *_ = np.linalg.lstsq(A, xw, rcond=None)
    p = np.clip(np.real(c), 0.0, None)
    if p.sum() > 0: p /= p.sum()

    if post_prune:
        keep = p > min_weight
        E_hat, p = E_hat[keep], p[keep]
        if keep_top is not None and E_hat.size > keep_top:
            idx = np.argsort(p)[::-1][:keep_top]
            E_hat, p = E_hat[idx], p[idx]
        if p.sum() > 0: p /= p.sum()

    order = np.argsort(E_hat)
    return E_hat[order], p[order], -omega, S_norm

# ===========================
# Lanczos (Stieltjes) recovery
# ===========================
def jacobi_from_spectral(E, p, enforce_nonneg_b=True, verbose=False, fmt=6, label=""):
    """
    Recover Jacobi coefficients (a,b) from spectral lines (E, p).
    Returns: a (len N), b (len N-1)
    """
    E = np.asarray(E, float)
    p = np.asarray(p, float)
    if p.sum() <= 0:
        raise ValueError("Weights p must have positive sum.")
    p = p / p.sum()
    N = len(E)

    h_prev = np.zeros_like(E)         # h_-1
    h_curr = np.sqrt(np.clip(p, 0, None))  # h_0
    delta_prev = 0.0
    a, b = [], []
    log_lines = []
    if verbose:
        hdr = f"[Lanczos{' ' + label if label else ''}]"
        log_lines.append(f"{hdr} Start")
        for k in range(N):
            log_lines.append(f"{hdr}   k={k}: E={E[k]:.{fmt}f}, p={p[k]:.{fmt}f}")

    for n in range(N):
        an = float(np.sum(E * h_curr**2))
        a.append(an)
        r = E*h_curr - an*h_curr - delta_prev*h_prev
        if n < N-1:
            bn = float(np.linalg.norm(r))
            b.append(abs(bn) if enforce_nonneg_b else bn)
            h_next = r / (bn if bn > 1e-15 else 1.0)
        else:
            h_next = h_curr
        if verbose:
            log_lines.append(f"[Lanczos] n={n}: a[{n}]={an:.{fmt}f}" +
                             ("" if n == 0 else f", uses b[{n-1}]={delta_prev:.{fmt}f}"))
        h_prev, h_curr = h_curr, h_next
        delta_prev = b[-1] if n < N-1 else 0.0

    if verbose:
        log_lines.append(f"[Lanczos] Done. a={np.round(a, fmt)}, b={np.round(b, fmt)}")
    return np.array(a), np.array(b), "\n".join(log_lines)

# ===========================
# Parameter recovery
# ===========================
def recover_J_from_blocks(b_lists, robust="median"):
    B = np.vstack(b_lists)  # (#m, L-1)
    J_est = np.median(B, axis=0) if robust == "median" else np.mean(B, axis=0)
    spread = np.std(B, axis=0)
    return J_est, spread

def recover_K_from_blocks_with_m(a_lists, m_list):
    A = np.vstack(a_lists)      # (M, L)
    m_list = list(m_list)
    M, L = A.shape
    S_est = float(np.mean([A[r, m] for r, m in enumerate(m_list)]))  # use a_mm

    rows = []
    rhs  = []
    for r, m in enumerate(m_list):
        for i in range(L):
            if i == m:
                continue
            row = np.zeros(L, float); row[i] = 1.0; row[m] = 1.0
            c_im = 0.5 * (S_est - A[r, i])
            rows.append(row); rhs.append(c_im)
    rows.append(np.ones(L, float)); rhs.append(S_est)  # sum constraint

    Msys = np.vstack(rows)
    bsys = np.asarray(rhs, float)
    K_est, *_ = np.linalg.lstsq(Msys, bsys, rcond=None)
    residual = float(np.linalg.norm(Msys @ K_est - bsys))
    return K_est, S_est, residual

# ===========================
# PST preset
# ===========================
def pst_J(L, J0=1.0):
    # J_i = J0 * sqrt((i+1)*(L-(i+1))) for i=0..L-2
    return np.array([J0 * np.sqrt((i+1)*(L-(i+1))) for i in range(L-1)], float)

# ===========================
# Plot helpers (Streamlit-safe)
# ===========================
def plot_spin_ladder(K, J, title="Spin ladder diagram"):
    """Schematic: top and bottom chains with vertical K_i and top J_i edges."""
    K = np.asarray(K, float); J = np.asarray(J, float)
    L = len(K)
    y_top, y_bot = 1.0, 0.0
    xs = np.arange(L)

    fig, ax = plt.subplots(figsize=(7, 2.6))
    # nodes
    ax.scatter(xs, [y_top]*L, s=80)
    ax.scatter(xs, [y_bot]*L, s=80)
    # top edges (J)
    for i in range(L-1):
        ax.plot([xs[i], xs[i+1]], [y_top, y_top])
        ax.text((xs[i]+xs[i+1])/2, y_top+0.05, f"J{i}={J[i]:.3g}", ha='center', fontsize=8)
    # vertical edges (K)
    for i in range(L):
        ax.plot([xs[i], xs[i]], [y_bot, y_top])
        ax.text(xs[i], 0.5, f"K{i}={K[i]:.3g}", ha='center', va='center', fontsize=8, rotation=90)
    ax.set_yticks([y_bot, y_top]); ax.set_yticklabels(["bottom", "top"])
    ax.set_xticks(xs); ax.set_xlim(-0.5, L-0.5)
    ax.set_title(title); ax.set_frame_on(False)
    ax.get_yaxis().set_visible(True)
    ax.get_xaxis().set_visible(True)
    plt.tight_layout()
    return fig

def plot_matrix_abs(H, title, xlabel="", ylabel="", figsize=(5.2,4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.abs(H))
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

def plot_top_populations_all(K, J, alpha, times, hbar=1.0, title="Top-site occupations"):
    K = np.asarray(K, float); J = np.asarray(J, float); alpha = np.asarray(alpha, complex)
    L = len(K)
    if np.linalg.norm(alpha) == 0:
        raise ValueError("alpha cannot be all zeros.")
    alpha = alpha / np.linalg.norm(alpha)
    P_top = [np.zeros_like(times, dtype=float) for _ in range(L)]
    psiA = np.zeros(L, complex); psiA[0] = 1.0
    for m in range(L):
        w = float(np.abs(alpha[m])**2)
        if w == 0: 
            continue
        Hm = Heff_block_general(K, J, m)
        kets_m, _ = evolve_state_in_H(Hm, psiA, times, hbar=hbar)
        for i in range(L):
            P_top[i] += w * (np.abs(kets_m[:, i])**2)
    # plot
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for i in range(L):
        lab = f"top {i}" if L > 3 else ["$P_A$","$P_B$","$P_C$"][i] if i < 3 else f"top {i}"
        ax.plot(times, P_top[i], label=lab)
    ax.set_xlabel("time"); ax.set_ylabel("occupation probability")
    ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=8, framealpha=0.8)
    fig.tight_layout()
    return fig, P_top

def plot_fft_panel(E_axis, S_norm, E_est, p_est, title):
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.plot(E_axis, S_norm, lw=1.0, label="FFT |F|^2 (norm)")
    if E_est.size:
        ax.stem(E_est, p_est, linefmt='-', markerfmt='o', basefmt=' ', label="Recovered sticks")
    ax.set_xlabel("Energy E"); ax.set_ylabel("normalized power / p_j")
    ax.set_title(title); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    return fig

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="Spin-Ladder FFT → Lanczos", layout="wide")
st.title("Two-excitation spin-ladder: FFT → Lanczos recovery")
st.caption("Interactively explore blocks $H_{\\mathrm{eff}}^{(m)}$, spectra, Lanczos steps, PST couplings, and parameter recovery.")

with st.sidebar:
    st.header("Setup")
    L = st.number_input("Chain length L (top & bottom)", min_value=2, max_value=10, value=6, step=1)
    dt = st.number_input("Time step dt", min_value=1e-4, max_value=1.0, value=0.01, step=0.01, format="%.5f")
    tmax = st.number_input("Max time tmax (larger → sharper FFT peaks)", min_value=0.1, max_value=200.0, value=80.0, step=1.0)
    times = np.arange(0.0, tmax + dt, dt)

    st.markdown("---")
    use_pst = st.checkbox("Use PST preset (K=0, engineered J)", value=False)
    if use_pst:
        J0 = st.number_input("PST scale J0 (t_pst = π/(2J0))", min_value=1e-6, value=1.0, step=0.1, format="%.6f")
        K_true = np.zeros(L, float)
        J_true = pst_J(L, J0)
    else:
        default_K = [1.0]*L
        default_J = [1.0]*(L-1)
        K_true = np.array(st.text_input(f"Enter K (L={L} reals, comma/space sep)", " ".join(map(str, default_K))).replace(",", " ").split(), float)[:L]
        J_true = np.array(st.text_input(f"Enter J (L-1={L-1} reals, comma/space sep)", " ".join(map(str, default_J))).replace(",", " ").split(), float)[:max(L-1,0)]

    st.markdown("---")
    st.subheader("Initial bottom state α")
    use_single = st.checkbox("Single bottom site (α=1 at one m, others 0)", value=True)
    if use_single:
        m0 = st.number_input("Choose bottom site m0", min_value=0, max_value=L-1, value=0, step=1)
        alpha = np.zeros(L, complex); alpha[m0] = 1.0
    else:
        alpha_vals = np.array(st.text_input(f"α (L={L} reals; imag=0 for simplicity)", "1 " + "0 "*(L-1)).split(), float)[:L]
        alpha = alpha_vals.astype(complex)

    st.markdown("---")
    st.subheader("Recovery blocks and FFT")
    m_list = st.multiselect("Choose bottom m indices", options=list(range(L)), default=list(range(min(L,3))))
    clamp_axis = st.checkbox("Clamp FFT energy axis", value=True)
    if clamp_axis:
        Emin = st.number_input("E_min", value=-10.0, step=0.5)
        Emax = st.number_input("E_max", value=10.0, step=0.5)
        xlim = (Emin, Emax)
    else:
        xlim = None

    n_levels_fft = st.slider("Max spectral lines to pick (per block)", 1, min(6, L), min(6, L))
    pad_factor = st.select_slider("FFT zero-padding factor", options=[1, 4, 8, 16, 32, 64], value=32)
    include_dc = st.checkbox("Include DC candidate", value=True)
    guard_mult = st.slider("Peak guard width multiplier", 0.5, 4.0, 2.0, 0.1)
    min_weight = st.number_input("Post-prune min weight", min_value=0.0, value=1e-5, step=1e-5, format="%.6f")

    st.markdown("---")
    st.subheader("Verbose Lanczos")
    enable_verbose = st.checkbox("Print detailed Lanczos recursion for one block", value=False)
    if enable_verbose:
        m_dbg = st.selectbox("Verbose block m", options=m_list or [0], index=0)
        decimals = st.number_input("Decimal places", min_value=1, max_value=12, value=6, step=1)

# ===========================
# Main layout
# ===========================
tabs = st.tabs(["Overview", "Hamiltonians", "Dynamics", "FFT & Lanczos", "Parameter recovery", "Eigenvalues"])

# --- Overview ---
with tabs[0]:
    st.subheader("Spin chain diagram")
    fig_diag = plot_spin_ladder(K_true, J_true, title="Spin ladder (top J, vertical K)")
    st.pyplot(fig_diag)
    if use_pst:
        st.info(f"PST preset: J_i = J0·sqrt((i+1)(L-(i+1))) with J0={J0:g}. Transfer time t_pst = π/(2J0) ≈ {np.pi/(2*J0):.4g}")

    st.markdown("""
**What this app does:**  
1) For each selected bottom index \(m\), build the effective block \(H_{\\mathrm{eff}}^{(m)}\\in\\mathbb{R}^{L\\times L}\).  
2) Compute the return amplitude \(f_{00}^{(m)}(t)\) at top site 0, FFT it to get candidate spectral lines \(\\{E^{(m)}, p^{(m)}\\}\).  
3) Run **Lanczos/Stieltjes** on \(\\{E,p\\}\) to recover Jacobi coefficients \(a,b\\), from which off-diagonals \(b\\) estimate **\(J\)** and diagonals \(a\\) across blocks estimate **\(K\)**.
""")

# --- Hamiltonians ---
with tabs[1]:
    cols = st.columns(3)
    # Single block heatmap
    with cols[0]:
        st.markdown("**Single block**")
        m_for_block = st.number_input("Block m for heatmap", min_value=0, max_value=L-1, value=0, step=1, key="blk")
        Hm = Heff_block_general(K_true, J_true, m_for_block)
        st.pyplot(plot_matrix_abs(Hm, f"|H_eff^(m={m_for_block})|", "top index", "top index"))
    # Block-diagonal top-bottom subspace
    with cols[1]:
        st.markdown("**Top–bottom subspace (block-diagonal)**")
        Htb = build_H_topbottom_blockdiag(K_true, J_true)
        st.pyplot(plot_matrix_abs(Htb, f"|H_top-bottom| (L^2×L^2)", "index", "index", figsize=(5.6,5.2)))
    # Full H (only if small)
    with cols[2]:
        st.markdown("**Full H (2^(2L)×2^(2L))**")
        if L <= 3:
            Hfull = build_H_full_general(L, K_true, J_true)
            st.pyplot(plot_matrix_abs(Hfull.real, f"|H_full| (L={L})", "index", "index", figsize=(6.2,6.0)))
        else:
            st.warning("Full H is huge; enable when L ≤ 3.")

# --- Dynamics ---
with tabs[2]:
    st.subheader("Top-site occupation probabilities (exact block-sum)")
    figP, P_top = plot_top_populations_all(K_true, J_true, alpha, times, title="Top-site occupations")
    st.pyplot(figP)

# --- FFT & Lanczos ---
with tabs[3]:
    if not m_list:
        st.info("Select at least one block m in the sidebar.")
    else:
        dfs = []
        for m in m_list:
            f_t = return_amplitude_block_general(K_true, J_true, m, site=0, times=times)
            E_est, p_est, E_axis, S_norm = fft_from_timeseries(
                f_t, times,
                n_levels=n_levels_fft, pad_factor=pad_factor,
                use_hann=True, include_dc=include_dc, guard_mult=guard_mult,
                post_prune=True, min_weight=min_weight, keep_top=n_levels_fft
            )
            title = f"FFT return at top site 0 | bottom m={m}"
            fig_fft = plot_fft_panel(E_axis, S_norm, E_est, p_est, title)
            if xlim is not None: plt.xlim(*xlim)
            st.pyplot(fig_fft)

            if E_est.size == 0:
                st.warning(f"No peaks found for m={m}. Increase tmax or pad_factor.")
                continue

            a_m, b_m, _ = jacobi_from_spectral(E_est, p_est, enforce_nonneg_b=True, verbose=False)
            df = pd.DataFrame({
                "m": m,
                "a_n": list(np.round(a_m, 10)) + [np.nan]*(L - len(a_m)),
                "b_n": list(np.round(b_m, 10)) + [np.nan]*(L-1 - len(b_m))
            })
            dfs.append(df)

        if dfs:
            st.markdown("**Recovered Jacobi coefficients per block**")
            st.dataframe(pd.concat(dfs, ignore_index=True))

        if enable_verbose and len(m_list) > 0:
            st.markdown("---")
            st.subheader(f"Verbose Lanczos log (m={m_dbg})")
            f_t = return_amplitude_block_general(K_true, J_true, m_dbg, site=0, times=times)
            E_est, p_est, *_ = fft_from_timeseries(
                f_t, times,
                n_levels=n_levels_fft, pad_factor=pad_factor,
                use_hann=True, include_dc=include_dc, guard_mult=guard_mult,
                post_prune=True, min_weight=min_weight, keep_top=n_levels_fft
            )
            if E_est.size == 0:
                st.warning("No peaks for verbose block; adjust tmax/padding.")
            else:
                a_v, b_v, log_text = jacobi_from_spectral(E_est, p_est, verbose=True, fmt=decimals, label=f"m={m_dbg}, site=0")
                st.code(log_text, language="text")

# --- Parameter recovery ---
with tabs[4]:
    if not m_list:
        st.info("Select at least one block m in the sidebar.")
    else:
        a_lists, b_lists = [], []
        for m in m_list:
            f_t = return_amplitude_block_general(K_true, J_true, m, site=0, times=times)
            E_est, p_est, *_ = fft_from_timeseries(
                f_t, times,
                n_levels=n_levels_fft, pad_factor=pad_factor,
                use_hann=True, include_dc=include_dc, guard_mult=guard_mult,
                post_prune=True, min_weight=min_weight, keep_top=n_levels_fft
            )
            if E_est.size == 0: 
                continue
            a_m, b_m, _ = jacobi_from_spectral(E_est, p_est, enforce_nonneg_b=True)
            if len(a_m) == L and len(b_m) == L-1:
                a_lists.append(a_m); b_lists.append(b_m)

        if not a_lists or not b_lists:
            st.warning("Couldn’t assemble complete (a,b) for any block. Increase tmax or padding.")
        else:
            J_est, J_spread = recover_J_from_blocks(b_lists, robust="median")
            K_est, S_est, resK = recover_K_from_blocks_with_m(a_lists, m_list)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**J: true vs estimated**")
                st.dataframe(pd.DataFrame({"J_true": np.round(J_true,6), "J_est": np.round(J_est,6), "spread": np.round(J_spread,6)}))
            with col2:
                st.markdown("**K: true vs estimated**")
                st.dataframe(pd.DataFrame({"K_true": np.round(K_true,6), "K_est": np.round(K_est,6)}))
                st.caption(f"sum(K)≈{S_est:.6f}, residual {resK:.2e}")

# --- Eigenvalues ---
with tabs[5]:
    st.subheader("Eigenvalues of blocks")
    if not m_list:
        st.info("Select blocks m.")
    else:
        rows = []
        for m in m_list:
            Hm = Heff_block_general(K_true, J_true, m)
            evals, evecs = np.linalg.eigh(Hm)
            rows.append(pd.DataFrame({"m": m, "eigval": evals}))
        if rows:
            st.dataframe(pd.concat(rows, ignore_index=True))
    st.markdown("---")
    if L <= 3:
        st.subheader("Eigenvalues of full H")
        Hfull = build_H_full_general(L, K_true, J_true)
        evals_full = np.linalg.eigvalsh(Hfull)
        st.write(pd.DataFrame({"eigval(H_full)": evals_full.real}))
    else:
        st.info("Full H eigenvalues only when L ≤ 3 (size grows as 2^(2L)).")
