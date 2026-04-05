# Science Case and IRF Plan

## Motivation

The primary use case for spore is **multi-detector realtime point source searches** — estimating the joint sensitivity of two or more neutrino detectors to a neutrino point source, with the specific goal of informing and supporting realtime multimessenger follow-up programs.

When an external trigger arrives (a gravitational wave, a gamma-ray flare, a fast radio burst), each detector independently searches its event stream for a coincident neutrino excess. The sensitivity of that search depends on the source declination, the observation time window, the detector's effective area at the source position, and the angular and energy resolution of the event selection. Nedflix is designed to simulate this process realistically for any combination of detectors, allowing joint sensitivity studies that are not possible from within a single collaboration.

External sensitivity studies in the literature (e.g., Kimura et al. 2023, arXiv:2302.04130; Jiang et al. 2023, arXiv:2310.16875; Lan et al. 2024, arXiv:2412.16868) uniformly use the IceCube 10-year point-source effective area (arXiv:2101.09836) as a proxy for IceCube's realtime response. No external group has published a study using the GFU-specific IRF directly, because the effective area has only been shown in figures rather than released as tabulated data. Nedflix aims to close this gap.

## Target Event Selection: IceCube GFU

The **Gamma-ray Follow-up (GFU)** selection is IceCube's primary realtime muon neutrino track stream. It was originally designed for triggered follow-up of gamma-ray sources (e.g., blazar flares), and has since become the backbone of IceCube's broader realtime multimessenger program.

### What GFU is used for

- **Realtime GOLD/BRONZE track alerts** sent via GCN within ~1 minute of detection, triggering follow-up by optical, X-ray, and gamma-ray telescopes
- **Fast Response Analysis** (arXiv:1909.05834): time-windowed searches around external triggers (GW events, GRBs, blazar flares) on timescales from seconds to weeks
- **Multiplet searches**: looking for clusters of neutrinos from the same direction in short time windows, without requiring an external trigger
- **Standard point source analysis**: at high energies GFU and the PS track sample share the same SplineMPE reconstruction; the 10-year PS release is a reprocessed version of the same stream

Key properties: runs at ~2 mHz (~6,000 events/day, mostly atmospheric neutrinos), low-latency online processing, per-event directional uncertainty estimate suitable for unbinned likelihood analyses.

## IRF Plan

The GFU IRF will be assembled from three sources, each requiring digitization or borrowing from public data releases.

### 1. Effective area

**Source:** arXiv:1610.01814 (Aartsen et al. 2017, JINST), Figure 3 — effective area vs. neutrino energy for several declination bands. Cross-check against Figure 3 of arXiv:1612.06028.

**Method:** Digitize the curves using a plot digitizer. Fit or interpolate to produce a 2D table A_eff(E, sin δ) in the HDF5 format spore expects.

**Notes:** The effective area is for the GFU muon neutrino track selection. It is the quantity that no external paper has previously tabulated; this digitization would be a novel contribution.

### 2. Angular resolution (PSF)

**Source:** arXiv:1610.01814, Figure 2 — median angular resolution vs. true neutrino energy (left panel) and vs. sin(declination) (right panel), for an E^-2 spectrum. The text states a summary value of **0.5° median** for E^-2 weighted events.

**Cross-check:** arXiv:1908.04884, Figure 3 — after the next-generation realtime upgrade, the through-going track resolution is ~0.15–0.40° over 100 GeV to 100 PeV, with a **0.2° systematic floor** applied to all per-event uncertainties.

**Method:** Digitize the median PSF curve from Figure 2 of 1610.01814. Model the full PSF as a Kent distribution (or Rayleigh distribution in the small-angle approximation) parameterized by the energy-dependent median, with a 0.2° floor. Store as an inverse-CDF table over (E, u) in spore's angular response format.

### 3. Energy resolution

**Source:** arXiv:2101.09836 (IceCube 10-year PS data release), Figure 4 and the accompanying data files. The GFU and PS selections use the same SplineMPE reconstruction, so the energy proxy distribution P(E_reco | E_true, δ) from the PS release is a reasonable proxy for GFU.

**Method:** Load the smearing matrix directly from the 10-year PS data release files (publicly available at icecube.wisc.edu/data-releases). Marginalize or rebin to produce the inverse-CDF format spore expects for energy response.

**Notes:** This is the weakest part of the GFU IRF — no dedicated energy resolution figure exists for the GFU selection in the public literature. The PS release smearing matrix is the best available substitute.

## Key References

| Paper | arXiv ID | Relevance |
|---|---|---|
| GFU selection characterization | 1610.01814 | Primary source for GFU A_eff and PSF |
| IceCube Realtime Alert System | 1612.06028 | GFU effective area and angular error figures |
| Next Generation Realtime Alerts | 1908.04884 | Updated GFU+EHE resolution, 0.2° floor |
| Fast Response Analysis | 1909.05834 | GFU-based time-windowed search methodology |
| 10-year PS data release | 2101.09836 | Public IRF files; energy proxy smearing matrix |
| External sensitivity study (GW) | 2302.04130 | Example of how external groups use public IRFs |
| External sensitivity study (GW) | 2310.16875 | Same; uses 10-yr PS A_eff directly |
