# IceCube Data Releases

This directory holds IceCube public data releases used by the SPORE paper
plot scripts.  The data files are large and are not distributed with the
package; they must be downloaded separately and placed here.

The paper plot scripts in `scripts/paper_plots/` look for data under
`resources/data_releases/<release-name>/`.

---

## PS-10yr Point-Source Data Release

**Directory:** `resources/data_releases/ps10yr_data_release/`

**Used by:** `scripts/paper_plots/pstracks_roundtrip.py`,
`scripts/paper_plots/point_source_demo.py`

**Description:** IceCube 10-year time-integrated point-source sample
(IC40 through IC86-2017).  Contains per-season event files, detector uptime
lists, effective area tables, and 5D smearing matrices.

**Download:**
- Data release page: https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/
- Reference: Abbasi et al. (IceCube), arXiv:1609.04981; Aartsen et al. (IceCube), Phys. Rev. Lett. 124, 051103 (2020)

---

## HESE-7yr Data Release

**Directory:** `resources/data_releases/hese_7yr_data_release/`

**Used by:** `scripts/build_hese_detector_response.py`,
`scripts/build_hese_atmospheric_flux.py`

**Description:** IceCube 7.5-year High-Energy Starting Events (HESE) data
release.  Contains event data, instrument response functions, and
likelihood analysis code from the IceCube collaboration.

**Download:**
- Data release page: https://icecube.wisc.edu/data-releases/2020/09/icecube-7-5-year-hese-data-release/
- Reference: Abbasi et al. (IceCube), arXiv:2011.03545

---

## Adding a New Data Release

Place the unpacked release directory under `resources/data_releases/` and
update the relevant script's `DATA` path variable.  The `data_releases/`
subdirectory contents are gitignored; only this README is tracked.
