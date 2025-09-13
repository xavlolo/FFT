import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="3-Site Spectral Recovery — Vector App (Run Button)", layout="wide")
st.title("3-Site Spectral Recovery — Vector App")
st.caption("Vector-crisp plots (SVG) + results. Updates only when you press **Run / Update**.")

# ---------------- Core math utilities ----------------
N = 6
hbar = 1.0
sx = np.array([[0,1],[1,0]], complex)
sy = np.array([[0,-1j],[1j,0]], complex)
sz = np.array([[1,0],[0,-1]], complex)
id2 = np.eye(2, dtype=complex)
n_single = 0.5 * (id2 - sz)

Q1, Q2, Q3, Q4, Q5, Q6 = range(6)
SINGLE_EXC_LABELS = ['|100000>','|010000>','|001000>','|000100>','|000010>','|000001>']
SINGLE_EXC_IDXS   = [32, 16, 8, 4, 2, 1]

def kronN(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def one_qubit(op, q):
    ops = [id2]*N
    ops[q] = op
    return kronN(ops)

def two_qubit(op1, q1, op2, q2):
    ops = [id2]*N
    ops[q1] = op1
    ops[q2] = op2
    return kronN(ops)

def build_H_full(K14, K25, K36, J12, J23):
    H = np.zeros((2**N, 2**N), complex)
    H += K14 * two_qubit(sz, Q1, sz, Q4)
    H += K25 * two_qubit(sz, Q2, sz, Q5)
    H += K36 * two_qubit(sz, Q3, sz, Q6)
    H += 0.5 * J12 * (two_qubit(sx, Q1, sx, Q2) + two_qubit(sy, Q1, sy, Q2))
    H += 0.5 * J23 * (two_qubit(sx, Q2, sx, Q3) + two_qubit(sy, Q2, sy, Q3))
    return H

def extract_1exc(H):
    idx = SINGLE_EXC_IDXS
    return H[np.ix_(idx, idx)], SINGLE_EXC_LABELS

def extract_top(H1):
    return H1[:3,:3].copy(), ['|A>','|B>','|C>'], [0,1,2]

def build_Heff_from_paper(K14, K25, K36, J12, J23):
    d1 = -K14 + K25 + K36
    d2 =  K14 - K25 + K36
    d3 =  K14 + K25 - K36
    Heff = np.array([[d1,  J12,   0 ],
                     [J12,  d2,  J23],
                     [ 0 ,  J23,  d3]], complex)
    return Heff

def psi0_top(a=1, b=0, c=0):
    v = np.array([a,b,c], complex)
    return v / np.linalg.norm(v)

def psi0_full(a=1, b=0, c=0):
    psi = np.zeros(2**N, complex)
    psi[32] = a; psi[16] = b; psi[8] = c
    psi /= np.linalg.norm(psi)
    return psi

def eigh_data(H):
    w, V = np.linalg.eigh(H)
    return w.real, V

def evolve_state_in_H(H, psi0_vec, times, hbar=1.0):
    w, V = eigh_data(H)
    c0 = V.conj().T @ psi0_vec
    kets = []
    for t in times:
        phase = np.exp(-1j * w * t / hbar)
        kets.append(V @ (phase * c0))
    return np.array(kets), (w, V, c0)

def populations_full_from_density(H, psi0, times, dt):
    def comm(H, rho): return -1j * (H @ rho - rho @ H)
    rho = np.outer(psi0, psi0.conj())
    n_ops = [one_qubit(n_single, q) for q in range(N)]
    pops = np.zeros((len(times), N))
    for ti, _ in enumerate(times):
        for q in range(N):
            pops[ti,q] = float(np.real(np.trace(rho @ n_ops[q])))
        if ti < len(times)-1:
            k1 = dt * comm(H, rho)
            k2 = dt * comm(H, rho + 0.5*k1)
            k3 = dt * comm(H, rho + 0.5*k2)
            k4 = dt * comm(H, rho + k3)
            rho = rho + (k1 + 2*k2 + 2*k3 + k4)/6.0
    return pops

def omega_from_params(J12, J23, hbar=1.0):
    return (np.sqrt(J12**2 + J23**2)) / hbar

def milestone_times(Omega, hbar=1.0):
    pi = np.pi
    labels = ["t0", "t1/4", "t1/2", "t3/4", "t1"]
    values = [0.0, (pi/(2*Omega))*hbar, (pi/Omega)*hbar,
              (3*pi/(2*Omega))*hbar, (2*pi/Omega)*hbar]
    return labels, values

def analytic_pc_eq6(Htop, psi_top, times, hbar=1.0):
    w, V = eigh_data(Htop)
    proj0 = V.T @ psi_top
    d = (V[2, :] * proj0).real
    base = np.sum(d**2)
    w21 = abs(w[1]-w[0]) / hbar
    w31 = abs(w[2]-w[0]) / hbar
    w32 = abs(w[2]-w[1]) / hbar
    PC = base + 2.0*d[0]*d[1]*np.cos(w21*times) + 2.0*d[0]*d[2]*np.cos(w31*times) + 2.0*d[1]*d[2]*np.cos(w32*times)
    return PC, (w, V, d, (w21, w31, w32))

def eigvals_and_site_weights(Htop, site=2):
    E, V = np.linalg.eigh(Htop)
    p = (V[site, :]**2).real
    p = p / p.sum()
    return E, p

def jacobi_from_spectral(E, p, fmt=3):
    E = np.asarray(E, float)
    p = np.asarray(p, float); p = p / p.sum()
    N = len(E)
    h_prev = np.zeros_like(E)
    h_curr = np.sqrt(p)
    delta_prev = 0.0
    a, b, logs = [], [], []
    for n in range(N):
        an = float(np.sum(E * h_curr**2)); a.append(an)
        logs.append(f"Step {n+1}: D{n+1} = {an:.{fmt}f}")
        r = E*h_curr - an*h_curr - delta_prev*h_prev
        if n < N-1:
            bn = float(np.linalg.norm(r)); b.append(bn)
            h_next = r / (bn if bn > 0 else 1.0)
            logs.append(f"         δ{n+1} = {bn:.{fmt}f}")
            logs.append(f"         h{n+2} = {np.round(h_next, fmt)}")
        else:
            h_next = h_curr
        h_prev, h_curr = h_curr, h_next
        delta_prev = b[-1] if n < N-1 else 0.0
    T = np.zeros((N, N), float)
    np.fill_diagonal(T, a); np.fill_diagonal(T[1:], b); np.fill_diagonal(T[:,1:], b)
    return np.array(a), np.array(b), T, "\\n".join(logs)

def return_amplitude(Htop, site, times, hbar=1.0):
    psi0 = np.zeros(3, complex); psi0[site] = 1.0
    kets, _ = evolve_state_in_H(Htop, psi0, times, hbar=hbar)
    return kets[:, site]

def _parabolic_refine(x, y, i):
    if i <= 0 or i >= len(y)-1: return x[i]
    y0, y1, y2 = np.log(np.maximum(y[i-1], 1e-300)), np.log(np.maximum(y[i], 1e-300)), np.log(np.maximum(y[i+1], 1e-300))
    denom = 2*(y0 - 2*y1 + y2)
    if denom == 0: return x[i]
    delta = (y0 - y2)/denom
    dx = 0.5*(x[i+1] - x[i-1])
    return x[i] + delta*dx

def fft_extract_E_p(Htop, site, times, n_peaks=3, pad=16, use_hann=True):
    f_t = return_amplitude(Htop, site=site, times=times)
    dt = float(times[1] - times[0])
    w = np.hanning(len(f_t)) if use_hann else np.ones_like(f_t)
    xw = f_t * w
    N = len(xw); Nfft = int(pad*N)
    F = dt * np.fft.fft(xw, n=Nfft)
    omega = 2.0*np.pi*np.fft.fftfreq(Nfft, d=dt)
    S = np.abs(F)**2
    loc = np.where((S[1:-1] > S[:-2]) & (S[1:-1] >= S[2:]))[0] + 1
    candidates = np.unique(np.concatenate(([0], loc)))
    if len(candidates) > n_peaks:
        idx = candidates[np.argsort(S[candidates])[-n_peaks:]]
    else:
        idx = candidates
    E_est = np.array([_parabolic_refine(omega, S, i) for i in np.sort(idx)], float)
    A = np.column_stack([np.exp(-1j*E_est[j]*times) * w for j in range(len(E_est))])
    c, *_ = np.linalg.lstsq(A, xw, rcond=None)
    p_est = np.real(c); p_est = np.clip(p_est, 0.0, None)
    if p_est.sum() > 0: p_est /= p_est.sum()
    order = np.argsort(E_est)
    return E_est[order], p_est[order], omega, S

# ---------------- Sidebar form (Run/Update) ----------------
with st.sidebar:
    st.header("Configuration")
    with st.form("controls"):
        preset = st.selectbox("Preset", ["Case 1 (diff K, diff J)", "Case 2 (same K, same J)", "Case 3 (same K, diff J)", "Custom"], index=2)
        if preset == "Case 1 (diff K, diff J)":
            K14, K25, K36, J12, J23 = 1.0, 1.5, 0.0, 1.0, 2.0
        elif preset == "Case 2 (same K, same J)":
            K14, K25, K36, J12, J23 = 1.0, 1.0, 1.0, 1.0, 1.0
        elif preset == "Case 3 (same K, diff J)":
            K14, K25, K36, J12, J23 = 0.0, 0.0, 0.0, 1.0, 1.5
        else:
            K14 = st.number_input("K14", value=0.0, step=0.1, format="%.3f")
            K25 = st.number_input("K25", value=0.0, step=0.1, format="%.3f")
            K36 = st.number_input("K36", value=0.0, step=0.1, format="%.3f")
            J12 = st.number_input("J12", value=1.0, step=0.1, format="%.3f")
            J23 = st.number_input("J23", value=1.5, step=0.1, format="%.3f")

        st.markdown("---")
        st.subheader("Initial state on top sites")
        init_choice = st.selectbox("Choose |ψ(0)⟩", ["|A⟩", "|B⟩", "|C⟩", "Custom"])
        if init_choice == "|A⟩":
            a, b, c = 1.0, 0.0, 0.0
        elif init_choice == "|B⟩":
            a, b, c = 0.0, 1.0, 0.0
        elif init_choice == "|C⟩":
            a, b, c = 0.0, 0.0, 1.0
        else:
            a = st.number_input("a (real)", value=1.0, step=0.1)
            b = st.number_input("b (real)", value=0.0, step=0.1)
            c = st.number_input("c (real)", value=0.0, step=0.1)

        st.markdown("---")
        st.subheader("Sampling")
        dt   = st.number_input("dt", min_value=1e-4, max_value=0.1, value=0.01, step=0.001, format="%.4f")
        tmax = st.number_input("tmax", min_value=0.5, max_value=60.0, value=15.0, step=0.5, format="%.2f")

        st.markdown("---")
        st.subheader("FFT options")
        site_map = {"A (0)":0, "B (1)":1, "C (2)":2}
        site = site_map[st.selectbox("Anchor site for f_ss(t)", list(site_map.keys()), index=2)]
        pad = st.selectbox("Zero-padding factor", [1,2,4,8,16,32], index=4)
        use_hann = st.checkbox("Apply Hann window", value=True)
        n_peaks = st.number_input("Number of peaks", min_value=1, max_value=6, value=3, step=1)

        run = st.form_submit_button("Run / Update")

def compute_all(K14,K25,K36,J12,J23,a,b,c,dt,tmax,site,pad,use_hann,n_peaks):
    times = np.arange(0.0, tmax + dt, dt)
    Hfull = build_H_full(K14, K25, K36, J12, J23)
    H1, _  = extract_1exc(Hfull)
    Htop, _, _ = extract_top(H1)

    psi_top  = psi0_top(a,b,c); psi_full = psi0_full(a,b,c)

    kets_top, _ = evolve_state_in_H(Htop, psi_top, times, hbar=hbar)
    P_A_eff, P_B_eff, P_C_eff = (np.abs(kets_top[:,0])**2, np.abs(kets_top[:,1])**2, np.abs(kets_top[:,2])**2)

    P_C_analytic, (w, V, d, omegas) = analytic_pc_eq6(Htop, psi_top, times, hbar=hbar)

    pops_full = populations_full_from_density(Hfull, psi_full, times, dt)
    P_q1_full, P_q2_full, P_q3_full = pops_full[:,Q1], pops_full[:,Q2], pops_full[:,Q3]

    Omega = omega_from_params(J12, J23, hbar=hbar)
    m_labels, m_times = milestone_times(Omega, hbar=hbar)
    kets_marks, _ = evolve_state_in_H(Htop, psi_top, m_times, hbar=hbar)
    P_C_eff_marks = np.abs(kets_marks[:,2])**2
    P_C_full_marks = np.interp(m_times, times, P_q3_full)
    PC_analytic_marks, _ = analytic_pc_eq6(Htop, psi_top, np.array(m_times), hbar=hbar)

    E, p = eigvals_and_site_weights(Htop, site=site)
    a_j, b_j, T, logs = jacobi_from_spectral(E, p)

    E_est, p_est, omega, S = fft_extract_E_p(Htop, site=site, times=times, n_peaks=n_peaks, pad=pad, use_hann=use_hann)

    return dict(times=times, Htop=Htop, Heff=build_Heff_from_paper(K14,K25,K36,J12,J23),
                P_A_eff=P_A_eff, P_B_eff=P_B_eff, P_C_eff=P_C_eff,
                P_q1_full=P_q1_full, P_q2_full=P_q2_full, P_q3_full=P_q3_full,
                P_C_analytic=P_C_analytic,
                m_labels=m_labels, m_times=m_times,
                P_C_eff_marks=P_C_eff_marks, P_C_full_marks=P_C_full_marks, P_C_analytic_marks=PC_analytic_marks,
                eigvals=w, eigvecs=V, d=d, omegas=omegas, Omega=Omega,
                jacobi_a=a_j, jacobi_b=b_j, T=T, jacobi_logs=logs,
                E_true=E, p_true=p, E_est=E_est, p_est=p_est, omega=omega, S=S,
                params=(K14,K25,K36,J12,J23), site=site, dt=dt, tmax=tmax)

# --- Session-state gating: compute only on button press (or on first load) ---
cfg = dict(K14=K14,K25=K25,K36=K36,J12=J12,J23=J23,a=a,b=b,c=c,dt=dt,tmax=tmax,site=site,pad=pad,use_hann=use_hann,n_peaks=n_peaks)
if "res" not in st.session_state:
    # First load: compute once with current defaults
    with st.spinner("Computing initial results..."):
        st.session_state.res = compute_all(**cfg)
        st.session_state.cfg = cfg
elif run:
    with st.spinner("Running..."):
        st.session_state.res = compute_all(**cfg)
        st.session_state.cfg = cfg
else:
    # No run: keep previous results
    pass

res = st.session_state.res

# ---------------- Tabs (Vector only) ----------------
tab_dyn, tab_spec, tab_lanczos, tab_results = st.tabs(
    ["Dynamics (SVG)", "Spectrum (SVG)", "Lanczos", "Results"]
)

with tab_dyn:
    t = res["times"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=res['P_A_eff'], mode='lines', name='P_A eff'))
    fig.add_trace(go.Scatter(x=t, y=res['P_B_eff'], mode='lines', name='P_B eff'))
    fig.add_trace(go.Scatter(x=t, y=res['P_C_eff'], mode='lines', name='P_C eff'))
    fig.add_trace(go.Scatter(x=t, y=res['P_q1_full'], mode='lines', line=dict(dash='dash'), name='P_q1 full'))
    fig.add_trace(go.Scatter(x=t, y=res['P_q2_full'], mode='lines', line=dict(dash='dash'), name='P_q2 full'))
    fig.add_trace(go.Scatter(x=t, y=res['P_q3_full'], mode='lines', line=dict(dash='dash'), name='P_q3 full'))
    for mt in res['m_times']:
        fig.add_vline(x=mt, line_width=1, line_dash="dot", opacity=0.3)
    fig.add_trace(go.Scatter(x=res['m_times'], y=res['P_C_analytic_marks'], mode='markers', marker_symbol='triangle-up', name='Ω milestones analytic'))
    fig.add_trace(go.Scatter(x=res['m_times'], y=res['P_C_full_marks'], mode='markers', marker_symbol='x', name='Ω milestones full'))
    fig.update_layout(xaxis_title="time", yaxis_title="Probabilities (q1,q2,q3)",
                      legend_orientation='h', legend_y=1.12, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab_spec:
    omega = res["omega"]; S = res["S"]
    y = (S/np.max(S)) if S.max()>0 else S
    figS = go.Figure()
    figS.add_trace(go.Scatter(x=omega, y=y, mode='lines', name='FFT |F(ω)|² (norm)'))
    def add_stems(fig, xs, ys, name, dash=None):
        X, Y = [], []
        for x, y in zip(xs, ys):
            X += [x, x, None]; Y += [0, y, None]
        fig.add_trace(go.Scatter(x=X, y=Y, mode='lines+markers', name=name,
                                 line=dict(dash=dash) if dash else None))
    add_stems(figS, res["E_est"], res["p_est"], "FFT sticks")
    add_stems(figS, res["E_true"], res["p_true"], "True sticks", dash="dash")
    figS.update_layout(xaxis_title="E (ω)", yaxis_title="p_j", margin=dict(l=10,r=10,t=10,b=10))
    figS.update_xaxes(range=[-3,3])
    st.plotly_chart(figS, use_container_width=True)

with tab_lanczos:
    st.markdown("**Jacobi/Lanczos reconstruction from $(E,p)$ at anchor site**")
    st.code(res["jacobi_logs"])
    st.write("a =", np.round(res["jacobi_a"], 6))
    st.write("b =", np.round(res["jacobi_b"], 6))
    st.dataframe(pd.DataFrame(res["T"]).style.format(precision=6), use_container_width=True)

with tab_results:
    K14,K25,K36,J12,J23 = res["params"]
    st.subheader("Parameters")
    st.write(f"K14={K14}, K25={K25}, K36={K36}, J12={J12}, J23={J23}")
    st.subheader("Eigenvalues and site weights (anchor site index {})".format(res["site"] if "site" in res else ""))
    c1, c2 = st.columns(2)
    c1.dataframe(pd.DataFrame({"E_true": res["E_true"], "p_true": res["p_true"]}).style.format(precision=6), use_container_width=True)
    c2.dataframe(pd.DataFrame({"E_est": res["E_est"], "p_est": res["p_est"]}).style.format(precision=6), use_container_width=True)

    st.subheader("Sampling diagnostics")
    E_absmax = float(np.max(np.abs(res["E_true"])))
    E_sorted = np.sort(res["E_true"])
    if len(E_sorted) >= 2:
        d_omegas = np.diff(E_sorted)
        delta_omega_min = float(np.min(np.abs(d_omegas))) if np.any(d_omegas != 0) else E_absmax
    else:
        delta_omega_min = E_absmax
    dt_nyq = np.pi/max(E_absmax, 1e-9)
    T_rayleigh = 2*np.pi/max(delta_omega_min, 1e-9)
    st.write(f"Recommended dt < π/ω_max ≈ {dt_nyq:.4f}; your dt = {res['dt']:.4f}")
    st.write(f"Recommended T ≳ 2π/Δω_min ≈ {T_rayleigh:.4f}; your T ≈ {res['tmax']:.4f}")

