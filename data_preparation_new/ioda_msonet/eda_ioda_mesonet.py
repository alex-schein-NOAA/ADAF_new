#!/usr/bin/env python3
"""EDA on the prepBUFR->IODA mesonet (MSONET) output vs same-cycle ADPSFC (METAR/SYNOP).

Cycle 2024-05-27 00Z, 1-h window. Produces:
  - console + markdown summary (per-variable valid counts, unique CONUS stations,
    density vs ADPSFC, 5-channel completeness spectrum)
  - ioda_pressure_elev.png   : stationPressure vs stationElevation (key physics check:
    should be the clean barometric elevation line, NOT the flat altimeter band)
  - ioda_channel_combos.png  : 2^5 presence/absence spectrum of the 5 ADAF channels

Run in NNJA_AI_environment (h5py + matplotlib).
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product

HERE = "/scratch3/BMC/wrfruc/Micah.Craine/ADAF_new/data_preparation_new/ioda_msonet"
MSONET = f"{HERE}/run_2024052700/ioda_msonet.nc"
ADPSFC = f"{HERE}/run_2024052700_adpsfc/ioda_adpsfc.nc"
FILL = 1e30  # IODA float fill is 3.4e38

# IODA var name -> ADAF station channel
ADAF_CHANNELS = {
    "stationPressure": "sp",
    "airTemperature":  "t",
    "specificHumidity": "q",
    "windEastward":    "u10",
    "windNorthward":   "v10",
}
GOOD_QM = {0, 1, 2, 3}  # prepbufr quality markers considered usable

# CONUS bbox (IODA longitude is 0-360)
LON0, LON1 = 235.0, 293.0
LAT0, LAT1 = 24.0, 50.0


def load(path):
    d = {}
    with h5py.File(path, "r") as f:
        d["lat"] = f["MetaData/latitude"][:]
        d["lon"] = f["MetaData/longitude"][:]
        d["elev"] = f["MetaData/stationElevation"][:].astype("float64")
        d["sid"] = f["MetaData/stationIdentification"][:]
        for v in ADAF_CHANNELS:
            d[v] = f[f"ObsValue/{v}"][:].astype("float64")
            qmkey = f"QualityMarker/{v}"
            d[v + "_qm"] = f[qmkey][:] if qmkey in f else np.zeros(d[v].shape, "int32")
    return d


def valid_mask(d, v):
    """value present (not fill/nan) AND quality marker usable."""
    val = d[v]
    m = np.isfinite(val) & (np.abs(val) < FILL)
    qm = d[v + "_qm"]
    m &= np.isin(qm, list(GOOD_QM))
    return m


def conus_mask(d):
    return (d["lon"] >= LON0) & (d["lon"] <= LON1) & (d["lat"] >= LAT0) & (d["lat"] <= LAT1)


def summarize(name, d):
    cm = conus_mask(d)
    n = d["lat"].shape[0]
    out = [f"### {name}", "",
           f"- rows (obs, global): **{n:,}**, CONUS: **{cm.sum():,}**"]
    # unique stations (CONUS, any valid ADAF channel)
    anyvalid = np.zeros(n, bool)
    for v in ADAF_CHANNELS:
        anyvalid |= valid_mask(d, v)
    sid_conus = d["sid"][cm & anyvalid]
    nstn = len(np.unique(sid_conus))
    out.append(f"- unique CONUS stations (>=1 usable ADAF channel): **{nstn:,}**")
    out.append("")
    out.append("| ADAF ch | IODA var | CONUS obs w/ usable value | % of CONUS rows | unique CONUS stns |")
    out.append("|---|---|---|---|---|")
    per_stn = {}
    for v, ch in ADAF_CHANNELS.items():
        m = valid_mask(d, v) & cm
        stns = len(np.unique(d["sid"][m]))
        per_stn[ch] = stns
        out.append(f"| {ch} | {v} | {m.sum():,} | {100*m.sum()/max(cm.sum(),1):.1f}% | {stns:,} |")
    return "\n".join(out), nstn, per_stn, cm


def channel_spectrum(d, cm, ax, title):
    """Per-station 2^5 presence/absence of the 5 ADAF channels (CONUS)."""
    sids = d["sid"][cm]
    uniq, inv = np.unique(sids, return_inverse=True)
    # station has channel if any CONUS ob for it is valid
    chans = list(ADAF_CHANNELS)
    has = np.zeros((len(uniq), 5), bool)
    for j, v in enumerate(chans):
        m = (valid_mask(d, v) & cm)[cm]
        np.logical_or.at(has[:, j], inv, m)
    # encode each station's 5-bit pattern
    codes = (has * (1 << np.arange(5))).sum(1)
    counts = np.bincount(codes, minlength=32)
    labels = []
    for c in range(32):
        bits = [(ADAF_CHANNELS[chans[j]]) for j in range(5) if (c >> j) & 1]
        labels.append("+".join(bits) if bits else "(none)")
    order = np.argsort(counts)[::-1]
    order = order[counts[order] > 0]
    ax.barh([labels[i] for i in order][::-1], [counts[i] for i in order][::-1], color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("unique stations")
    return counts, labels


def main():
    md, _, mstn = MSONET, ADPSFC, None  # placeholder
    dm = load(MSONET)
    da = load(ADPSFC)

    sec_m, nstn_m, perstn_m, cm_m = summarize("MSONET (mesonet)", dm)
    sec_a, nstn_a, perstn_a, cm_a = summarize("ADPSFC (METAR + SYNOP land)", da)

    ratio = nstn_m / max(nstn_a, 1)
    header = [
        "# IODA mesonet EDA — cycle 2024-05-27 00Z (1-h window)",
        "",
        f"Source: prepBUFR `rap.t00z.prepbufr.tm00` -> `bufr2ioda.x` + `prepbufr_<subset>.yaml`.",
        "Both subsets are from the SAME prepBUFR file (apples-to-apples, same cycle/source).",
        "",
        f"**Headline: MSONET gives {nstn_m:,} unique CONUS stations vs ADPSFC {nstn_a:,} "
        f"= {ratio:.1f}x denser.**",
        "",
    ]

    # ---- pressure vs elevation (key physics) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, (nm, d, cm) in zip(axes, [("MSONET", dm, cm_m), ("ADPSFC", da, cm_a)]):
        m = valid_mask(d, "stationPressure") & cm & np.isfinite(d["elev"]) & (np.abs(d["elev"]) < FILL)
        elev = d["elev"][m]
        sp = d["stationPressure"][m] / 100.0  # Pa -> hPa
        ax.scatter(elev, sp, s=2, alpha=0.15, color="darkred")
        # ISA barometric reference line
        ez = np.linspace(max(elev.min(), -100), elev.max(), 100)
        isa = 1013.25 * (1 - 2.25577e-5 * ez) ** 5.25588
        ax.plot(ez, isa, "k--", lw=1.2, label="ISA barometric")
        ax.set_title(f"{nm}: stationPressure vs elevation (n={m.sum():,})")
        ax.set_xlabel("station elevation (m)")
        ax.set_ylabel("station pressure (hPa)")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{HERE}/ioda_pressure_elev.png", dpi=110)
    plt.close(fig)

    # ---- channel completeness spectrum ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    channel_spectrum(dm, cm_m, axes[0], "MSONET: 5-channel completeness (CONUS stations)")
    channel_spectrum(da, cm_a, axes[1], "ADPSFC: 5-channel completeness (CONUS stations)")
    fig.tight_layout()
    fig.savefig(f"{HERE}/ioda_channel_combos.png", dpi=110)
    plt.close(fig)

    report = "\n".join(header) + "\n" + sec_m + "\n\n" + sec_a + "\n\n"
    report += ("## Per-channel CONUS station density (MSONET / ADPSFC)\n\n"
               "| ADAF ch | MSONET stns | ADPSFC stns | ratio |\n|---|---|---|---|\n")
    for ch in ADAF_CHANNELS.values():
        report += f"| {ch} | {perstn_m[ch]:,} | {perstn_a[ch]:,} | {perstn_m[ch]/max(perstn_a[ch],1):.1f}x |\n"
    report += ("\n## Plots\n- `ioda_pressure_elev.png` — pressure-vs-elevation; MSONET should "
               "trace the ISA barometric line (true station pressure), confirming it fixes the "
               "old METAR altimeter bug.\n- `ioda_channel_combos.png` — 5-channel completeness.\n")

    with open(f"{HERE}/ioda_mesonet_eda.md", "w") as f:
        f.write(report)
    print(report)
    print("\nWrote ioda_mesonet_eda.md, ioda_pressure_elev.png, ioda_channel_combos.png")


if __name__ == "__main__":
    main()
