#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Planetary μ–ε Monte-Carlo -> OC/SC/CPR with frequency dependence

Features
- Material classes spanning planetary surfaces (basalt, anorthosite, regolith, ice, salts,
  metal-rich, ferrimagnetic soils).
- Frequency sweep 0.3–12 GHz with simple dispersive proxies for ε*(f), μ*(f).
- Fresnel with magnetic media, RHCP incidence -> OC/SC coefficients -> CPR.
- Pluggable roughness models: toy, facet, iem-lite, none
- Stability filter for CPR (avoid OC≈0 blow-ups), optional capping for visualization.
- CSV outputs (raw + cleaned) and several summary plots.

Notes
- This is a first-order facet model; no multiple scattering / dihedrals.
- Dispersion terms are proxies; swap in lab-fit curves when available.

Run
    python full_test.py --outdir results

Author: Dany Waller
"""
# Imports:
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- Utilities -----------------------------

def complex_sqrt(z):
    return np.sqrt(z + 0j)

def snell_cos_theta_t(n1, n2, cos_theta_i):
    # sin^2(theta_t) = (n1/n2)^2 * (1 - cos^2 theta_i)
    sin2_t = (n1 / n2)**2 * (1.0 - cos_theta_i**2)
    cos2_t = 1.0 - sin2_t
    return complex_sqrt(cos2_t)

def fresnel_coeffs_magnetic(eta1, eta2, n1, n2, cos_theta_i):
    # Fresnel reflection with magnetic media
    ct = snell_cos_theta_t(n1, n2, cos_theta_i)
    rs = (eta2 * cos_theta_i - eta1 * ct) / (eta2 * cos_theta_i + eta1 * ct)
    rp = (eta1 * cos_theta_i - eta2 * ct) / (eta1 * cos_theta_i + eta2 * ct)
    return rs, rp

def circular_components_from_rs_rp(rs, rp):
    oc = 0.5 * (rs + rp)
    sc = 0.5 * (rs - rp)
    return np.abs(oc)**2, np.abs(sc)**2

def nearest(arr, target):
    arr = np.asarray(arr)
    return arr[np.argmin(np.abs(arr - target))]

# ----------------------------- Physics ------------------------------

EPS0 = 8.854187817e-12
MU0  = 4*np.pi*1e-7
C0   = 299_792_458.0

def eps_complex_dispersion(eps_real0, tan_e0, sigma, f, f0=1.0e9):
    """
    ε*(f) = ε' - i( tanδe(f)*ε' + σ/(ωε0) ), with tanδe ~ baseline * (1 + a*(f/f0)^b)
    """
    a, b = 0.5, 0.3
    tan_e_f = tan_e0 * (1.0 + a * (f/f0)**b)
    eps_im_sigma = sigma / (2*np.pi*f*EPS0)
    return eps_real0 - 1j*(tan_e_f*eps_real0 + eps_im_sigma)

def mu_complex_dispersion(mu_real0, tan_m0, f, f0=1.0e9, ferri=False, rng=None):
    """
    μ*(f) = μ' - i tanδm(f) μ', with tanδm ~ baseline * (1 + c*(f/f0)^d)
    Optionally add a weak Lorentzian magnetic loss bump for ferrimagnetic soils.
    """
    c, d = 0.4, 0.2
    tan_m_f = tan_m0 * (1.0 + c * (f/f0)**d)
    mu_c = mu_real0 - 1j*tan_m_f*mu_real0
    if ferri:
        if rng is None:
            rng = np.random.default_rng(0)
        fr = 1.5e9 + 1.5e9 * rng.random(mu_real0.shape)  # 1.5–3.0 GHz
        gamma = 0.6e9
        # Lorentz term (very mild) – applied to μ'' only
        lorentz = (gamma**2) / ((2*np.pi*(f - fr))**2 + gamma**2)
        mu_c = (mu_c.real) - 1j*(mu_c.real * (tan_m_f + 0.02*lorentz))
    return mu_c

# -------------------------- Roughness models -------------------------

def roughness_apply(OC, SC, model, *, theta_rad, rms_slope,
                    freq_Hz, sigma_h_m, dihedral_k,
                    rng, use_random_sigma_h=False, sigma_h_range=(0.003, 0.03)):
    """
    Returns (OCr, SCr) given chosen roughness model.

    Models:
      - 'none'   : return OC, SC as-is
      - 'facet'  : no extra OC->SC transfer; facet slopes already sampled
      - 'simple'    : k = 0.5 * rms_slope^2; transfer k*OC to SC
      - 'iem-lite': F = exp(-(4π σh cosθ / λ)^2); transfer (1-F)*OC to SC + dihedral term
    if model == "none" or model == "facet":
        return OC, SC

    if model == "simple":
        k = 0.5 * np.square(rms_slope)
        k = np.minimum(k, 0.8)
        OCn = (1 - k) * OC
        SCn = SC + k * OC
        return OCn, SCn

    if model == "iem-lite":
        lam = C0 / freq_Hz
        if use_random_sigma_h:
            sigma_h = rng.uniform(sigma_h_range[0], sigma_h_range[1], size=OC.shape)
        else:
            sigma_h = sigma_h_m

        # Smooth-surface power reduction (scalar), classic exp(-(4πσh cosθ / λ)^2)
        red = np.exp(-np.square(4*np.pi*sigma_h*np.cos(theta_rad)/lam))
        # Transfer a fraction (1 - red) of OC into SC (single-bounce depolarization)
        trans = np.clip(1.0 - red, 0.0, 0.95)
        OC1 = (1 - trans) * OC
        SC1 = SC + trans * OC

        # Add a simple dihedral boost scaling with sin^2(theta)
        SC2 = SC1 + dihedral_k * np.square(np.sin(theta_rad)) * OC1
        OC2 = OC1 * (1 - 0.25*dihedral_k)  # tiny energy bookkeeping
        return OC2, SC2

    raise ValueError(f"Unknown roughness model: {model}")

# ----------------------------- Materials -----------------------------

MATERIAL_CLASSES = {
    "Basaltic rock": {
        "eps_real": (6.0, 9.0),
        "tan_e": (5e-3, 5e-2),
        "sigma": (0.0, 1e-3),
        "mu_real": (0.98, 1.08),
        "tan_m": (1e-4, 5e-3),
        "ferrimag": False,
    },
    "Anorthositic rock": {
        "eps_real": (3.0, 5.0),
        "tan_e": (2e-3, 2e-2),
        "sigma": (0.0, 5e-4),
        "mu_real": (0.98, 1.05),
        "tan_m": (1e-4, 2e-3),
        "ferrimag": False,
    },
    "Porous regolith/soil": {
        "eps_real": (1.6, 3.5),
        "tan_e": (1e-3, 1e-2),
        "sigma": (0.0, 2e-4),
        "mu_real": (0.98, 1.10),
        "tan_m": (5e-5, 2e-3),
        "ferrimag": False,
    },
    "Water ice (clean/cold)": {
        "eps_real": (3.05, 3.25),
        "tan_e": (3e-4, 2e-3),
        "sigma": (0.0, 1e-6),
        "mu_real": (0.995, 1.01),
        "tan_m": (5e-5, 5e-4),
        "ferrimag": False,
    },
    "Salts/evaporites": {
        "eps_real": (4.5, 7.5),
        "tan_e": (1e-3, 5e-2),
        "sigma": (0.0, 2e-3),
        "mu_real": (0.98, 1.05),
        "tan_m": (1e-4, 2e-3),
        "ferrimag": False,
    },
    "Metal-rich regolith": {
        "eps_real": (5.0, 12.0),
        "tan_e": (5e-3, 5e-2),
        "sigma": (1e-3, 5e-2),
        "mu_real": (0.98, 1.20),
        "tan_m": (5e-4, 1e-2),
        "ferrimag": False,
    },
    "Ferrimagnetic soil": {
        "eps_real": (4.0, 9.0),
        "tan_e": (2e-3, 3e-2),
        "sigma": (0.0, 5e-3),
        "mu_real": (1.05, 1.50),
        "tan_m": (1e-3, 2e-2),
        "ferrimag": True, #  (Fe-oxide/npFe)
    },
}

# ----------------------------- CLI & Main ----------------------------

def build_arguments():
    p = argparse.ArgumentParser(description="Monte-Carlo μ–ε tradeoffs in CPR/OC/SC with frequency dependence.")
    p.add_argument("--n_per_class", type=int, default=2000, help="Samples per material class.")
    p.add_argument("--fmin", type=float, default=0.05, help="Min frequency (GHz).")
    p.add_argument("--fmax", type=float, default=20.0, help="Max frequency (GHz).")
    p.add_argument("--nf", type=int, default=100, help="Number of frequency samples.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")
    p.add_argument("--oc_min", type=float, default=1e-6, help="OC floor for stable CPR.")
    p.add_argument("--cpr_cap", type=float, default=10.0, help="Visualization cap for CPR (<=0 disables capping).")
    p.add_argument("--outdir", type=str, default=".", help="Output directory for CSVs and plots.")
    p.add_argument("--no_plots", action="store_true", help="Skip plotting (still writes CSVs).")

    # Roughness control
    p.add_argument("--roughness-model", type=str, default="none",
                   choices=["simple", "facet", "iem-lite", "none"],
                   help="Roughness/depole model.")
    p.add_argument("--sigma-h-m", type=float, default=0.01,
                   help="IEM-lite RMS height σh in meters (used if not randomized).")
    p.add_argument("--dihedral-k", type=float, default=0.15,
                   help="IEM-lite dihedral fraction (0–0.5 is reasonable).")
    p.add_argument("--use-random-sigma-h", action="store_true",
                   help="Randomize σh per sample in IEM-lite.")
    p.add_argument("--sigma-h-range", type=float, nargs=2, default=[0.003, 0.03],
                   help="Range for randomized σh (meters) in IEM-lite.")
    return p.parse_args()


def main():
    args = build_arguments()
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Frequencies
    freqs = np.linspace(args.fmin*1e9, args.fmax*1e9, args.nf)

    # Angle & roughness distributions (shared per class)
    theta_deg = np.clip(rng.normal(35.0, 12.0, size=args.n_per_class), 0.0, 80.0)
    theta_rad = np.deg2rad(theta_deg)
    cos_theta = np.cos(theta_rad)
    rms_slope = np.clip(rng.normal(0.25, 0.12, size=args.n_per_class), 0.0, 0.8)

    if args.roughness_model == "none":
        theta_rad = np.zeros_like(theta_rad)
        cos_theta = np.ones_like(cos_theta)

    # Medium 1 (vacuum, relative units)
    eps1 = 1.0 + 0j
    mu1  = 1.0 + 0j
    eta1 = np.sqrt(mu1/eps1)
    n1   = np.sqrt(mu1*eps1)

    records = []

    # --- Sample and simulate ---
    for cls_name, par in MATERIAL_CLASSES.items():
        # Draw base params
        eps_r0 = rng.uniform(*par["eps_real"], size=args.n_per_class)
        tan_e0 = 10**rng.uniform(np.log10(par["tan_e"][0]),
                                 np.log10(par["tan_e"][1]),
                                 size=args.n_per_class)
        # Avoid log10(0)
        sigma_lo = max(par["sigma"][0], 1e-9)
        sigma0 = 10**rng.uniform(np.log10(sigma_lo),
                                 np.log10(par["sigma"][1]),
                                 size=args.n_per_class)
        mu_r0  = rng.uniform(*par["mu_real"], size=args.n_per_class)
        tan_m0 = 10**rng.uniform(np.log10(par["tan_m"][0]),
                                 np.log10(par["tan_m"][1]),
                                 size=args.n_per_class)

        for f in freqs:
            # Dispersive ε*(f) and μ*(f)
            eps2 = eps_complex_dispersion(eps_r0, tan_e0, sigma0, f)
            mu2  = mu_complex_dispersion(mu_r0, tan_m0, f, ferri=par["ferrimag"], rng=rng)

            eta2 = np.sqrt(mu2/eps2)
            n2   = np.sqrt(mu2*eps2)

            rs, rp = fresnel_coeffs_magnetic(eta1, eta2, n1, n2, cos_theta)
            OC, SC = circular_components_from_rs_rp(rs, rp)

            # --- Apply chosen roughness model (frequency-aware if needed) ---
            OCr, SCr = roughness_apply(
                OC, SC, args.roughness_model,
                theta_rad=theta_rad,
                rms_slope=rms_slope,
                freq_Hz=f,
                sigma_h_m=args.sigma_h_m,
                dihedral_k=args.dihedral_k,
                rng=rng,
                use_random_sigma_h=args.use_random_sigma_h,
                sigma_h_range=tuple(args.sigma_h_range),
            )

            CPR = np.divide(SCr, OCr, out=np.full_like(SCr, np.nan), where=OCr > 1e-18)

            block = pd.DataFrame({
                "class": cls_name,
                "freq_Hz": f,
                "theta_deg": theta_deg,
                "rms_slope": rms_slope,
                "eps_real0": eps_r0,
                "tan_delta_e0": tan_e0,
                "sigma_Spm": sigma0,
                "mu_real0": mu_r0,
                "tan_delta_m0": tan_m0,
                "eps2_real": np.real(eps2),
                "eps2_imag": -np.imag(eps2),
                "mu2_real": np.real(mu2),
                "mu2_imag": -np.imag(mu2),
                "OC": OCr,
                "SC": SCr,
                "CPR": CPR
            })
            records.append(block)

    df = pd.concat(records, ignore_index=True)

    if args.roughness_model == 'none':
        args.roughness_model = 'No Roughness Model'
    elif args.roughness_model == 'simple':
        args.roughness_model = 'Simple Roughness Model'
    elif args.roughness_model == 'facet':
        args.roughness_model = 'Facet-only Roughness Model'
    elif args.roughness_model == 'iem-lite':
        args.roughness_model = 'IEM-Lite Roughness Model'

    # Save raw
    raw_csv = os.path.join(args.outdir, f"planetary_mu_epsilon_cpr_{str(args.roughness_model).lower().replace(' ', '_')}_RAW.csv")
    df.to_csv(raw_csv, index=False)
    print(f"Wrote raw samples: {raw_csv}  (rows={len(df):,})")

    # Stable CPR (avoid OC→0 blow-ups)
    df["CPR_stable"] = np.where(df["OC"] >= args.oc_min, df["CPR"], np.nan)
    if args.cpr_cap and args.cpr_cap > 0:
        df["CPR_stable"] = np.where(df["CPR_stable"] <= args.cpr_cap, df["CPR_stable"], args.cpr_cap)

    clean_csv = os.path.join(args.outdir, f"planetary_mu_epsilon_cpr_{str(args.roughness_model).lower().replace(' ', '_')}_CLEAN.csv")
    df.to_csv(clean_csv, index=False)
    print(f"Wrote cleaned samples: {clean_csv}")

    if args.no_plots:
        return

    # ----------- Plots -----------
    g = df.groupby(["class", "freq_Hz"], as_index=False).agg(
        CPR_med=("CPR_stable", "median"),
        OC_med=("OC", "median"),
        SC_med=("SC", "median"),
        n=("CPR_stable", "count"),
    )

    # 1) Median CPR vs frequency by class
    plt.figure()
    for cls in g["class"].unique():
        d = g[g["class"] == cls]
        plt.plot(d["freq_Hz"]/1e9, d["CPR_med"], marker="o", label=cls)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Median CPR (stable)")
    plt.title(f"Median CPR vs Frequency by Material Class\n{args.roughness_model}")
    plt.legend(fontsize="small", ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"CPR_vs_freq_{str(args.roughness_model).lower().replace(' ', '_')}.png"), dpi=180)

    # 2) Median OC vs frequency
    plt.figure()
    for cls in g["class"].unique():
        d = g[g["class"] == cls]
        plt.plot(d["freq_Hz"]/1e9, d["OC_med"], marker="o", label=cls)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Median OC")
    plt.title(f"Median OC vs Frequency\n{args.roughness_model}")
    plt.legend(fontsize="small", ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"OC_vs_freq_{str(args.roughness_model).lower().replace(' ', '_')}.png"), dpi=180)

    # 3) Median SC vs frequency
    plt.figure()
    for cls in g["class"].unique():
        d = g[g["class"] == cls]
        plt.plot(d["freq_Hz"]/1e9, d["SC_med"], marker="o", label=cls)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Median SC")
    plt.title(f"Median SC vs Frequency\n{args.roughness_model}")
    plt.legend(fontsize="small", ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"SC_vs_freq_{str(args.roughness_model).lower().replace(' ', '_')}.png"), dpi=180)

    # 4) CPR heatmaps across (μ', ε') at ~0.85, 2.37, 7,14 GHz (use nearest frequencies)
    def median_cpr_grid(dfin, f_Hz, nbins=28):
        sub = dfin[np.isclose(dfin["freq_Hz"], f_Hz)]
        if sub.empty:
            return None, None, None
        mu = sub["mu2_real"].to_numpy()
        epsr = sub["eps2_real"].to_numpy()
        cpr = sub["CPR_stable"].to_numpy()
        if len(mu) == 0 or np.all(np.isnan(cpr)):
            return None, None, None
        mu_edges = np.linspace(np.nanmin(mu), np.nanmax(mu), nbins+1)
        eps_edges = np.linspace(np.nanmin(epsr), np.nanmax(epsr), nbins+1)
        grid = np.full((nbins, nbins), np.nan)
        for i in range(nbins):
            for j in range(nbins):
                m = (mu >= mu_edges[i]) & (mu < mu_edges[i+1]) & \
                    (epsr >= eps_edges[j]) & (epsr < eps_edges[j+1])
                if np.any(m):
                    grid[i, j] = np.nanmedian(cpr[m])
        mu_cent = 0.5*(mu_edges[:-1] + mu_edges[1:])
        eps_cent = 0.5*(eps_edges[:-1] + eps_edges[1:])
        return mu_cent, eps_cent, grid

    unique_freqs = np.sort(df["freq_Hz"].unique())
    for fGHz in [0.85, 2.37, 7.14]:
        f_sel = nearest(unique_freqs, fGHz*1e9)
        mu_cent, eps_cent, grid = median_cpr_grid(df, f_sel, nbins=28)
        if grid is None:
            continue
        plt.figure()
        extent = [eps_cent.min(), eps_cent.max(), mu_cent.min(), mu_cent.max()]
        plt.imshow(np.flipud(grid), aspect="auto", extent=extent, origin="lower")
        plt.xlabel("ε' (real permittivity)")
        plt.ylabel("μ' (real permeability)")
        plt.title(f"Median CPR (stable) across (μ', ε') at ~{f_sel/1e9:.2f} GHz\n{args.roughness_model}")
        cbar = plt.colorbar()
        lab = "Median CPR"
        if args.cpr_cap and args.cpr_cap > 0:
            lab += f" (capped at {args.cpr_cap:g})"
        cbar.set_label(lab)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"CPR_heatmap_{str(args.roughness_model).lower().replace(' ', '_')}_{f_sel/1e9:.2f}GHz.png"), dpi=180)

    # 5) S-band scatter CPR vs μ' (marker size ~ ε')
    s_freq = nearest(unique_freqs, 2.38e9)
    sband = df[np.isclose(df["freq_Hz"], s_freq)].copy()
    plt.figure()
    ms = 6 + 3*(sband["eps2_real"] - sband["eps2_real"].min()) / \
            (sband["eps2_real"].max() - sband["eps2_real"].min() + 1e-12)
    plt.scatter(sband["mu2_real"], sband["CPR_stable"], s=ms, alpha=0.25)
    plt.xlabel("μ' (~S-band)")
    plt.ylabel("CPR (stable)")
    plt.title(f"CPR vs μ' at ~{s_freq/1e9:.2f} GHz\n{args.roughness_model} (marker size ~ ε')")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"CPR_vs_mu_Sband_{str(args.roughness_model).lower().replace(' ', '_')}.png"), dpi=180)

    print(f"Plots saved to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()