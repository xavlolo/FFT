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
    times = np.asarray(times, float)
    f_t = np.asarray(f_t)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("`times` must be 1D with length >= 2.")
    if f_t.shape[0] != times.shape[0]:
        raise ValueError("`f_t` and `times` must have the same length.")
    dts = np.diff(times)
    if not np.allclose(dts, dts[0]):
        raise ValueError("`times` must be uniformly spaced.")
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
# Lanczos (detailed) + T
# ===========================
def build_tridiagonal(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    N = len(a)
    T = np.zeros((N, N), float)
    np.fill_diagonal(T, a)
    for i in range(N - 1):
        T[i, i + 1] = T[i + 1, i] = b[i]
    return T

def lanczos_detailed(E, p, enforce_nonneg_b=True):
    E = np.asarray(E, float)
    p = np.asarray(p, float)
    if p.sum() <= 0:
        raise ValueError("Weights p must have positive sum.")
    p = p / p.sum()
    N = len(E)

    h_prev = np.zeros_like(E)              # h_-1
    h_curr = np.sqrt(np.clip(p, 0, None))  # h_0
    b_prev = 0.0

    steps = []
    hs = []
    a_list, b_list = [], []
    for n in range(N):
        a_n = float(np.sum(E * (h_curr ** 2)))
        r = E * h_curr - a_n * h_curr - b_prev * h_prev
        if n < N - 1:
            b_n = float(np.linalg.norm(r))
            b_list.append(abs(b_n) if enforce_nonneg_b else b_n)
            h_next = r / (b_n if b_n > 1e-15 else 1.0)
        else:
            b_n = 0.0
            h_next = h_curr

        preview = np.array2string(np.real(h_curr[:min(5, N)]), precision=6, suppress_small=True)
        steps.append(dict(n=n, a_n=a_n, b_prev=b_prev if n > 0 else 0.0, b_n=b_n, h_preview=preview))
        hs.append(h_curr.copy())
        a_list.append(a_n)

        h_prev, h_curr = h_curr, h_next
        b_prev = b_list[-1] if n < N - 1 else 0.0

    a = np.array(a_list, float)
    b = np.array(b_list, float)
    steps_df = pd.DataFrame(steps)

    T = build_tridiagonal(a, b)
    H = np.column_stack(hs)
    G = H.T @ H
    evals_T, Q = np.linalg.eigh(T)
    weights_from_T = (Q[0, :] ** 2)
    return a, b, steps_df, T, G, evals_T, weights_from_T

# ===========================
# Parameter recovery
# ===========================
def recover_J_from_blocks(b_lists, robust="median"):
    B = np.vstack(b_lists)
    J_est = np.median(B, axis=0) if robust == "median" else np.mean(B, axis=0)
    spread = np.std(B, axis=0)
    return J_est, spread

def recover_K_from_blocks_with_m(a_lists, m_list):
    A = np.vstack(a_lists)
    m_list = list(m_list)
    M, L = A.shape
    S_est = float(np.mean([A[r, m] for r, m in enumerate(m_list)]))
    rows, rhs = [], []
    for r, m in enumerate(m_list):
        for i in range(L):
            if i == m: continue
            row = np.zeros(L, float); row[i] = 1.0; row[m] = 1.0
            c_im = 0.5 * (S_est - A[r, i])
            rows.append(row); rhs.append(c_im)
    rows.append(np.ones(L, float)); rhs.append(S_est)
    Msys = np.vstack(rows); bsys = np.asarray(rhs, float)
    K_est, *_ = np.linalg.lstsq(Msys, bsys, rcond=None)
    residual = float(np.linalg.norm(Msys @ K_est - bsys))
    return K_est, S_est, residual

# ===========================
# PST preset
# ===========================
def pst_J(L, J0=1.0):
    return np.array([J0 * np.sqrt((i+1)*(L-(i+1))) for i in range(L-1)], float)

# ===========================
# Parsing + display helpers
# ===========================
def parse_float_list(text: str, n: int, default_fill: float = 1.0):
    text = (text or "").strip()
    if not text:
        arr = np.array([], float)
    else:
        arr = np.array(text.replace(",", " ").split(), float)
    if arr.size < n:
        arr = np.concatenate([arr, np.full(n - arr.size, default_fill)])
    else:
        arr = arr[:n]
    return arr

def parse_complex_list(text: str, n: int, default_fill: complex = 0.0+0.0j):
    """Accept tokens like 0.5, 0.5+0.5i, 1j, -i; commas/spaces; pads/truncates to n."""
    txt = (text or "").strip()
    if not txt:
        arr = np.array([], complex)
    else:
        # normalize forms: allow 'i' as imaginary unit and Unicode minus
        norm = txt.replace("i", "j").replace("−", "-")
        tokens = norm.replace(",", " ").split()
        vals = []
        for t in tokens:
            try:
                vals.append(complex(t))
            except Exception:
                # last resort: strip spaces inside e.g. "0.5 + 0.5j"
                vals.append(complex(t.replace(" ", "")))
        arr = np.array(vals, complex)
    if arr.size < n:
        arr = np.concatenate([arr, np.full(n - arr.size, default_fill, dtype=complex)])
    else:
        arr = arr[:n]
    return arr

def format_bitstring(bits):
    return "".join("1" if b else "0" for b in bits)

def expanded_state_terms(L, alpha):
    """Return list of (coef, six_bit_string) terms for L<=3 with top excitation at 0."""
    if L > 3:
        return []
    terms = []
    for m, amp in enumerate(alpha):
        if np.abs(amp) < 1e-12:
            continue
        bits = [0]*(2*L)
        bits[0] = 1          # top site 0 excited
        bits[L + m] = 1      # bottom site m excited
        terms.append((amp, format_bitstring(bits)))
    return terms

# ===========================
# Plot helpers (Streamlit-safe)
# ===========================
def plot_spin_ladder(K, J, title="Spin ladder diagram"):
    K = np.asarray(K, float); J = np.asarray(J, float)
    L = len(K)
    y_top, y_bot = 1.0, 0.0
    xs = np.arange(L)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.scatter(xs, [y_top]*L, s=80)
    ax.scatter(xs, [y_bot]*L, s=80)
    for i in range(L-1):
        ax.plot([xs[i], xs[i+1]], [y_top, y_top])
        ax.text((xs[i]+xs[i+1])/2, y_top+0.05, f"J{i}={J[i]:.3g}", ha='center', fontsize=8)
    for i in range(L):
        ax.plot([xs[i], xs[i]], [y_bot, y_top])
        ax.text(xs[i], 0.5, f"K{i}={K[i]:.3g}", ha='center', va='center', fontsize=8, rotation=90)
    ax.set_yticks([y_bot, y_top]); ax.set_yticklabels(["bottom", "top"])
    ax.set_xticks(xs); ax.set_xlim(-0.5, L-0.5)
    ax.set_title(title); ax.set_frame_on(False)
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
        if w == 0: continue
        Hm = Heff_block_general(K, J, m)
        kets_m, _ = evolve_state_in_H(Hm, psiA, times, hbar=hbar)
        for i in range(L):
            P_top[i] += w * (np.abs(kets_m[:, i])**2)
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
    ax.set_xlabel("Energy E"); ax.set_ylabel("normalized weight / p_j")
    ax.set_title(title); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    return fig


def lanczos_math_lines(E, p, prec=6, max_show=6):
    """
    Build LaTeX strings that show, step-by-step, how a_n and b_n are computed
    with numeric substitution.

    E, p: arrays of same length N (real); sum(p)=1 recommended.
    prec: decimals to show
    max_show: truncate term-by-term sums if N is large (to keep LaTeX readable)
    """
    E = np.asarray(E, float)
    p = np.asarray(p, float)
    if p.sum() <= 0:
        return [r"\text{No positive weights }p_k."]
    p = p / p.sum()
    N = len(E)

    h_prev = np.zeros_like(E)              # h_{-1}
    h_curr = np.sqrt(np.clip(p, 0, None))  # h_0
    b_prev = 0.0

    def fmt(x):  # compact formatting
        return f"{x:.{prec}f}"

    # Header with inputs
    lines = []
    lines.append(r"\textbf{Inputs: } "
                 + r"E=\{" + ",\,".join(fmt(x) for x in E) + r"\},\ "
                 + r"p=\{" + ",\,".join(fmt(x) for x in p) + r"\}.")
    lines.append(r"h_{-1}=0,\quad h_0(k)=\sqrt{p_k}=\{"
                 + ",\,".join(fmt(x) for x in h_curr) + r"\}.")

    for n in range(N):
        # a_n = sum_k E_k h_n(k)^2
        terms = [rf"{fmt(E[k])}\cdot({fmt(h_curr[k])})^2" for k in range(N)]
        if N > max_show:
            shown = " + ".join(terms[:max_show]) + r" + \cdots"
        else:
            shown = " + ".join(terms)
        a_n = float(np.sum(E * (h_curr ** 2)))
        lines.append(rf"a_{n}=\sum_k E_k\,h_{n}(k)^2 \;=\; {shown} \;=\; {fmt(a_n)}.")

        # r^(n) = E*h_n - a_n*h_n - b_{n-1}*h_{n-1}
        r = E*h_curr - a_n*h_curr - b_prev*h_prev
        # show first few component formulas
        comp_lines = []
        showK = min(N, max_show)
        for k in range(showK):
            comp_lines.append(
                rf"r^{{({n})}}({k})={fmt(E[k])}\cdot{fmt(h_curr[k])}"
                rf"-{fmt(a_n)}\cdot{fmt(h_curr[k])}"
                rf"-{fmt(b_prev)}\cdot{fmt(h_prev[k])}"
                rf"={fmt(r[k])}"
            )
        r_block = r" \;,\; ".join(comp_lines)
        if N > showK:
            r_block += r" \;,\; \ldots"
        lines.append(r_block)

        # b_n = ||r^(n)||_2
        if N > max_show:
            lines.append(rf"b_{n}=\|r^{{({n})}}\|_2=\sqrt{{\sum_k (r_k)^2}} = {fmt(np.linalg.norm(r))}.")
        else:
            sq_terms = " + ".join(rf"({fmt(rv)})^2" for rv in r)
            lines.append(rf"b_{n}=\|r^{{({n})}}\|_2=\sqrt{{{sq_terms}}} = {fmt(np.linalg.norm(r))}.")

        # next h
        if n < N-1 and np.linalg.norm(r) > 1e-15:
            h_next = r / np.linalg.norm(r)
            preview = ",\,".join(fmt(x) for x in h_next[:showK])
            if N > showK:
                preview += r",\,\ldots"
            lines.append(rf"h_{{{n+1}}}=\frac{{r^{{({n})}}}}{{b_{n}}}\ \Rightarrow\ "
                         rf"h_{{{n+1}}}(k)=\{{{preview}\}}.")
        else:
            h_next = h_curr
            lines.append(rf"h_{{{n+1}}}=h_{{{n}}}\ \text{{(termination)}}.")

        # shift for next step
        h_prev, h_curr = h_curr, h_next
        b_prev = float(np.linalg.norm(r)) if n < N-1 else 0.0

    return lines


def rung_zz_table(K, i, m):
    """
    Build a rung-by-rung table explaining the ZZ contribution to d_i^(m):
      row j: top|0/1>, z_top ∈ {+1,-1}, bottom|0/1>, z_bot ∈ {+1,-1},
              product = z_top*z_bot, numeric contribution = K_j * product
    Returns (df, diag_value, S).
    """
    K = np.asarray(K, float)
    L = len(K)
    rows = []
    for j in range(L):
        top_state = "|1⟩" if j == i else "|0⟩"
        z_top = -1 if j == i else +1
        bot_state = "|1⟩" if j == m else "|0⟩"
        z_bot = -1 if j == m else +1
        prod = z_top * z_bot
        contrib_val = K[j] * prod
        # symbolic ±K_j
        sign = "＋" if prod > 0 else "−"
        contrib_sym = f"{sign}K_{j}"
        rows.append({
            "Rung j": j,
            "Top State": top_state,
            "z_top": z_top,
            "Bottom State": bot_state,
            "z_bot": z_bot,
            "Product z_top·z_bot": prod,
            "ZZ Contribution (symbolic)": contrib_sym,
            "ZZ Contribution (value)": contrib_val,
        })
    df = pd.DataFrame(rows)
    diag = float(df["ZZ Contribution (value)"].sum())
    S = float(np.sum(K))
    return df, diag, S


# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="Spin-Ladder FFT → Lanczos", layout="wide")
st.title("Two-excitation spin-ladder: FFT → Lanczos recovery")
st.caption("Interactively explore blocks $H_{\\mathrm{eff}}^{(m)}$, spectra, Lanczos steps, PST couplings, and parameter recovery.")

with st.sidebar:
    st.header("Setup")
    # Default L = 3 → 6 qubits total (3 top, 3 bottom)
    L = st.number_input("Chain length L (top & bottom)", min_value=2, max_value=10, value=3, step=1)
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
        default_K = " ".join(["1"] * L)
        default_J = " ".join(["1"] * max(L-1, 0))
        K_true = parse_float_list(
            st.text_input(f"Enter K (L={L} reals, comma/space sep)", default_K),
            n=L, default_fill=1.0
        )
        J_true = parse_float_list(
            st.text_input(f"Enter J (L-1={L-1} reals, comma/space sep)", default_J),
            n=max(L-1, 0), default_fill=1.0
        )

    st.markdown("---")
    st.subheader("Bottom initial state α (complex)")
    bottom_mode = st.radio(
        "Choose how to set the bottom excitation",
        ["Single site m0", "Custom amplitudes a,b,c,… (complex)"],
        index=0
    )
    if bottom_mode == "Single site m0":
        m0 = st.number_input("Choose bottom site m0", min_value=0, max_value=L-1, value=0, step=1)
        alpha = np.zeros(L, complex); alpha[m0] = 1.0
    else:
        example = "1, 0, 0" if L == 3 else ("1, " + ", ".join(["0"]*(L-1)))
        txt = st.text_input(
            f"Enter α as comma/space separated complex numbers (length L={L}). "
            "You may use i or j for √−1 (examples: 1, 0.5+0.5i, -i).",
            example
        )
        alpha_raw = parse_complex_list(txt, n=L, default_fill=0.0+0.0j)
        norm = np.linalg.norm(alpha_raw)
        if norm == 0:
            st.warning("All-zero amplitudes — defaulting to α = (1, 0, …, 0).")
            alpha = np.zeros(L, complex); alpha[0] = 1.0
        else:
            alpha = alpha_raw / norm
        st.caption(f"Normalized ||α||₂ = 1 (original norm was {norm:.6g}).")

    # Show α table
    st.dataframe(pd.DataFrame({"m": np.arange(L), "α_m": alpha}), use_container_width=True)

    st.markdown("---")
    st.subheader("Recovery blocks and FFT")
    m_list = st.multiselect("Choose bottom m indices", options=list(range(L)), default=list(range(min(L,3))))
    m_list = [m for m in m_list if 0 <= m < L]

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
    st.pyplot(plot_spin_ladder(K_true, J_true, title="Spin ladder (top J couplings, vertical K couplings)"))

    st.markdown("### Model Hamiltonian")
    st.markdown("Total qubits: $2L$ (top sites $0..L-1$, bottom sites $L..2L-1$).")
    st.latex(r"""
    H \;=\; \sum_{i=0}^{L-2} \frac{J_i}{2}\!\left(\sigma_i^x \sigma_{i+1}^x + \sigma_i^y \sigma_{i+1}^y\right)
          \;+\; \sum_{i=0}^{L-1} K_i \,\sigma_i^z \sigma_{L+i}^z
    """)
    st.caption("Top chain has nearest-neighbour XX+YY (strengths $J_i$). Each top site $i$ couples via ZZ to its bottom partner $L+i$ with strength $K_i$.")

    st.markdown("### Effective block Hamiltonian used in FFT→Lanczos")
    st.latex(r"""
    H_{\mathrm{eff}}^{(m)} \;=\; \mathrm{diag}\!\left(d^{(m)}_0,\dots,d^{(m)}_{L-1}\right)
    \;+\; \sum_{i=0}^{L-2} J_i\left(\,|i\rangle\langle i{+}1| + |i{+}1\rangle\langle i|\,\right),
    """)
    st.latex(r"""
    d^{(m)}_i \;=\; \begin{cases}
    S, & i=m,\\[2pt]
    S - 2\big(K_i + K_m\big), & i\neq m,
    \end{cases}
    \qquad S \;=\; \sum_{j=0}^{L-1} K_j.
    """)
    

# --- Worked rung-by-rung ZZ table (under Effective block Hamiltonian) ---
with st.expander("Worked example: ZZ contributions per rung (builds dᵢ^(m))", expanded=True):
    st.markdown(
        "Use this table to see exactly how the diagonal entry "
        r"$d_i^{(m)}=\sum_j K_j\,z^{\text{top}}_j z^{\text{bot}}_j$ "
        "is formed from the ZZ terms on each rung."
    )
    # Default to your example when possible: L=3, i=m=1
    default_im = 1 if L >= 2 else 0
    i_ex = st.number_input("Top excitation site i", min_value=0, max_value=L-1,
                           value=min(default_im, L-1), step=1, key="zz_i")
    m_ex = st.number_input("Bottom excitation site m", min_value=0, max_value=L-1,
                           value=min(default_im, L-1), step=1, key="zz_m")

    df_zz, diag_val, S_val = rung_zz_table(K_true, i_ex, m_ex)
    st.dataframe(df_zz.style.format({
        "z_top": "{:+d}",
        "z_bot": "{:+d}",
        "Product z_top·z_bot": "{:+d}",
        "ZZ Contribution (value)": "{:+.6g}",
    }), use_container_width=True)

    # Show the algebra in LaTeX (robust to braces)
    st.markdown("### Diagonal entry calculation")
    
    # 1) S line: S = sum_{j=0}^{L-1} K_j = <number>
    tex_S = r"\displaystyle S \;=\; \sum_{j=0}^{%d} K_j \;=\; %.6g" % (L-1, S_val)
    st.latex(tex_S)
    
    # 2) d_i^(m) line
    if i_ex == m_ex:
        tex_d = r"\displaystyle d_{%d}^{(%d)} \;=\; S \;=\; %.6g" % (i_ex, m_ex, S_val)
        st.latex(tex_d)
        st.caption("When top and bottom excitations are on the same rung (i = m), there is no shift from S.")
    else:
        tex_d = (
            r"\displaystyle d_{%d}^{(%d)} \;=\; S - 2\big(K_{%d}+K_{%d}\big)"
            r"\;=\; %.6g - 2\big(%.6g+%.6g\big) \;=\; %.6g"
        ) % (i_ex, m_ex, i_ex, m_ex, S_val, K_true[i_ex], K_true[m_ex], diag_val)
        st.latex(tex_d)
        st.caption("Two rungs (i and m) are misaligned, each contributing a −2K shift relative to the S baseline.")



    # Initial state explanation
    st.markdown("### Initial state")
    st.markdown("We **fix one excitation on the top at site 0**, and set the bottom as a superposition with amplitudes $\\alpha_m$:")
    st.latex(r"""
    |\psi(0)\rangle
    \;=\; |1 0 0 \ldots 0\rangle_{\text{top}}
          \;\otimes\; \sum_{m=0}^{L-1} \alpha_m\,|m\rangle_{\text{bottom}},
    \qquad \sum_{m}|\alpha_m|^2=1 .
    """)
    if L <= 3:
        # Product form for L=3:
        st.latex(r"""
        |\psi(0)\rangle
        \;=\; |100\rangle_{\text{top}} \;\otimes\;
               \big(\alpha_0|100\rangle_{\text{bottom}} + \alpha_1|010\rangle_{\text{bottom}} + \alpha_2|001\rangle_{\text{bottom}}\big).
        """)
        # Expanded 6-bit strings
        terms = expanded_state_terms(L, alpha)
        if terms:
            st.markdown("Expanded in the 6-qubit computational basis:")
            latex_terms = " \\, + \\, ".join([f"({amp:.3g})\\,|{bitstr}\\rangle" for amp, bitstr in terms])
            st.latex(rf"|\psi(0)\rangle \;=\; {latex_terms}")

    if use_pst:
        st.info(f"PST preset: $J_i = J_0\\sqrt{{(i+1)(L-(i+1))}}$ with $J_0={J0:g}$.  Transfer time $t_{{\\rm pst}}=\\pi/(2J_0)\\approx{np.pi/(2*J0):.4g}$")

    st.markdown("""
**Pipeline:**  
1) For each selected bottom index \(m\), build \(H_{\\mathrm{eff}}^{(m)}\\in\\mathbb{R}^{L\\times L}\).  
2) Compute the return amplitude \(f_{00}^{(m)}(t)\) at top site 0, FFT it to get spectral lines \(\\{E^{(m)}, p^{(m)}\\}\).  
3) Run **Lanczos/Stieltjes** on \(\\{E,p\\}\) to recover Jacobi \(a,b\\); use \(b\\) to estimate **\(J\)** and cross-block \(a\\) to estimate **\(K\)**.
""")

# --- Hamiltonians ---
with tabs[1]:
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Single block**")
        m_for_block = st.number_input("Block m for heatmap", min_value=0, max_value=L-1, value=0, step=1, key="blk")
        Hm = Heff_block_general(K_true, J_true, m_for_block)
        st.pyplot(plot_matrix_abs(Hm, f"|H_eff^(m={m_for_block})|", "top index", "top index"))
    with cols[1]:
        st.markdown("**Top–bottom subspace (block-diagonal)**")
        Htb = build_H_topbottom_blockdiag(K_true, J_true)
        st.pyplot(plot_matrix_abs(Htb, f"|H_top-bottom| (L^2×L^2)", "index", "index", figsize=(5.6,5.2)))
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
                f_t, times, n_levels=n_levels_fft, pad_factor=pad_factor,
                use_hann=True, include_dc=include_dc, guard_mult=guard_mult,
                post_prune=True, min_weight=min_weight, keep_top=n_levels_fft
            )
            title = f"FFT return at top site 0 | bottom m={m}"
            fig_fft = plot_fft_panel(E_axis, S_norm, E_est, p_est, title)
            if xlim is not None:
                ax = fig_fft.axes[0]; ax.set_xlim(*xlim)
            st.pyplot(fig_fft)

            if E_est.size == 0:
                st.warning(f"No peaks found for m={m}. Increase tmax or pad_factor.")
                continue

            # Compact (a,b) for summary table
            a_m, b_m, _, *_ = lanczos_detailed(E_est, p_est, enforce_nonneg_b=True)
            a_col = np.concatenate([np.round(a_m, 10), np.full(L - len(a_m), np.nan)])
            b_col = np.concatenate([np.round(b_m, 10), np.full(L - len(b_m), np.nan)])
            df = pd.DataFrame({"m": np.full(L, m), "n": np.arange(L), "a_n": a_col, "b_n": b_col})
            dfs.append(df)

            # Detailed panel
            with st.expander(f"Lanczos details for m={m}", expanded=False):
                a, b, steps_df, T, G, evals_T, weights_T = lanczos_detailed(E_est, p_est, enforce_nonneg_b=True)
                st.markdown("**Step-by-step iterations**"); st.dataframe(steps_df.style.format(precision=6), use_container_width=True)
                st.markdown("**Jacobi (tridiagonal) matrix $T$**")
                dfT = pd.DataFrame(np.round(T, 10)); st.dataframe(dfT, use_container_width=True)
                st.download_button("Download T as CSV", data=dfT.to_csv(index=False).encode("utf-8"), file_name=f"T_m{m}.csv", mime="text/csv")
                st.markdown("**Orthonormality check**: $G = H^\\top H$ (should be $I$)")
                figG, axG = plt.subplots(figsize=(4.2, 3.6)); im = axG.imshow(G); axG.set_title("Inner products of h_n")
                figG.colorbar(im, ax=axG, fraction=0.046, pad=0.04); figG.tight_layout(); st.pyplot(figG)
                st.markdown("**Consistency check** (eigs of $T$ vs FFT sticks)")
                comp_df = pd.DataFrame({"E_from_FFT": np.round(E_est, 10), "p_from_FFT": np.round(p_est, 10),
                                        "E_from_T": np.round(evals_T, 10), "p_from_T": np.round(weights_T, 10)})
                st.dataframe(comp_df, use_container_width=True)
                figS, axS = plt.subplots(figsize=(6.6, 3.4))
                axS.stem(E_est, p_est, linefmt='-', markerfmt='o', basefmt=' ', label="FFT sticks")
                axS.stem(evals_T, weights_T, linefmt='--', markerfmt='s', basefmt=' ', label="From T")
                axS.set_xlabel("Energy E"); axS.set_ylabel("weight"); axS.set_title("Spectrum: FFT vs Jacobi-matrix reconstruction")
                axS.grid(alpha=0.25); axS.legend(); figS.tight_layout(); st.pyplot(figS)
                st.markdown("**Lanczos math**")
                st.latex(r"""
                \text{Init:}\quad h_{-1}=0,\qquad h_0(k)=\sqrt{p_k}.
                """)
                st.latex(r"""
                a_n \;=\; \sum_k E_k\,h_n(k)^2,\qquad
                r^{(n)} \;=\; E\,h_n - a_n h_n - b_{n-1}h_{n-1},\qquad
                b_n \;=\; \|r^{(n)}\|_2,\qquad
                h_{n+1} \;=\; \frac{r^{(n)}}{b_n}.
                """)
                st.latex(r"""
                T \;=\;
                \begin{pmatrix}
                a_0 & b_0 & 0   & \cdots & 0 \\
                b_0 & a_1 & b_1 & \ddots & \vdots \\
                0   & b_1 & a_2 & \ddots & 0 \\
                \vdots & \ddots & \ddots & \ddots & b_{N-2}\\
                0 & \cdots & 0 & b_{N-2} & a_{N-1}
                \end{pmatrix},\quad
                G_{00}(z)=\sum_k \frac{p_k}{z-E_k}
                \;=\;
                \cfrac{1}{z-a_0-\cfrac{b_0^2}{z-a_1-\cfrac{b_1^2}{\ddots}}}.
                """)
                st.markdown("**Math derivation for $a_n$ and $b_n$ (numbers plugged into the sums)**")
                prec_ui = st.slider(f"Precision (block m={m})", 1, 8, 4, key=f"prec_{m}")
                for eq in lanczos_math_lines(E_est, p_est, prec=prec_ui, max_show=6):
                    st.latex(eq)


        if dfs:
            st.markdown("**Recovered Jacobi coefficients per block**")
            st.dataframe(pd.concat(dfs, ignore_index=True))

        if enable_verbose and m_list:
            st.markdown("---"); st.subheader(f"Verbose Lanczos log (m={m_dbg})")
            f_t = return_amplitude_block_general(K_true, J_true, m_dbg, site=0, times=times)
            E_est, p_est, *_ = fft_from_timeseries(
                f_t, times, n_levels=n_levels_fft, pad_factor=pad_factor,
                use_hann=True, include_dc=include_dc, guard_mult=guard_mult,
                post_prune=True, min_weight=min_weight, keep_top=n_levels_fft
            )
            if E_est.size == 0:
                st.warning("No peaks for verbose block; adjust tmax/padding.")
            else:
                a, b, steps_df, *_ = lanczos_detailed(E_est, p_est, enforce_nonneg_b=True)
                lines = []
                for _, row in steps_df.iterrows():
                    n = int(row["n"])
                    lines.append(f"[Lanczos] n={n}: a[{n}]={row['a_n']:.{decimals}f}" + ("" if n == 0 else f", uses b[{n-1}]={row['b_prev']:.{decimals}f}"))
                lines.append(f"[Lanczos] Done. a={np.round(a, decimals)}, b={np.round(b, decimals)}")
                st.code("\n".join(lines), language="text")

# --- Parameter recovery ---
with tabs[4]:
    if not m_list:
        st.info("Select at least one block m in the sidebar.")
    else:
        a_lists, b_lists, used_ms = [], [], []
        for m in m_list:
            f_t = return_amplitude_block_general(K_true, J_true, m, site=0, times=times)
            E_est, p_est, *_ = fft_from_timeseries(
                f_t, times, n_levels=n_levels_fft, pad_factor=pad_factor,
                use_hann=True, include_dc=include_dc, guard_mult=guard_mult,
                post_prune=True, min_weight=min_weight, keep_top=n_levels_fft
            )
            if E_est.size == 0: continue
            a_m, b_m, _, *_ = lanczos_detailed(E_est, p_est, enforce_nonneg_b=True)
            if len(a_m) == L and len(b_m) == L-1:
                a_lists.append(a_m); b_lists.append(b_m); used_ms.append(m)

        if not a_lists or not b_lists:
            st.warning("Couldn’t assemble complete (a,b) for any block. Increase tmax or padding.")
        else:
            J_est, J_spread = recover_J_from_blocks(b_lists, robust="median")
            K_est, S_est, resK = recover_K_from_blocks_with_m(a_lists, used_ms)
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
            evals, _ = np.linalg.eigh(Hm)
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
