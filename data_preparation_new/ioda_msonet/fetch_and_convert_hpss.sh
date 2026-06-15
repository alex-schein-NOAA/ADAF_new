#!/bin/bash
#SBATCH -A wrfruc
#SBATCH -p u1-service
#SBATCH --job-name=ioda_hpss
#SBATCH --output=logs/ioda_hpss_%j.out
#SBATCH --error=logs/ioda_hpss_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G
#
# ============================================================================
# Fetch RAP prepBUFR from the BMC FDR HPSS archive and convert each hourly cycle
# into an ADAF-ready IODA mesonet file, in the exact layout the reader expects:
#     <IODA_ROOT>/run_<YYYYMMDDHH>/ioda_msonet.nc
# i.e. ready for:  sample_generate_new.py --obs_source ioda --ioda_root <IODA_ROOT>
#
# This is the "HISTORY for training" step from the handover, now UNBLOCKED
# (HPSS keytab provisioned 2026-06-12; hsi/htar authenticate).
#
# FDR archive layout (CONFIRMED 2026-06-12 by walking the tree):
#   /BMC/fdr/Permanent/YYYY/MM/DD/data/grids/rap/obs/YYYYMMDDHH00.zip
#     * 6-HOURLY zips, base cycle 00/06/12/18, each ~7 GB.
#     * "each cycle zip contains data for 6 cycles starting from this cycle"
#       (per Guoqing.Ge/mytools/raphrrr/dataFromHPSS/rapobs.sh), e.g.
#       202405270000.zip holds hours 00Z..05Z of 2024-05-27.
#     * Inside, the file we want is:  <YYYYMMDDHH>.rap.t<HH>z.prepbufr.tm00
#       (alongside many satellite *.bufr_d and sub-hourly rtma_ru.*.prepbufr.tm00
#        which we SKIP via selective unzip -- saves unpacking ~6 GB per zip).
#
# Usage:
#   sbatch fetch_and_convert_hpss.sh START_YYYYMMDD END_YYYYMMDD [HOURS]
#     HOURS : comma-list of cycle hours to keep (e.g. "00,12") or "all" (default).
#
#   Dry run (no hsi/unzip/convert -- just print the plan; runs anywhere):
#     DRY_RUN=1 bash fetch_and_convert_hpss.sh 20240527 20240528 all
#
# Env knobs:
#   IODA_ROOT  output root (default below; matches funcs_modified.IODA_ROOT)
#   STAGE_DIR  scratch space for zip download + prepbufr extraction
#   CONVERT    1 (default) run the converter; 0 = fetch+stage prepbufr only
#   KEEP_PREPBUFR 0 (default) delete prepbufr after convert; 1 = keep it
# ============================================================================
set -uo pipefail

# Under sbatch, Slurm copies this script to a node-local spool dir, so
# dirname "${BASH_SOURCE[0]}" points to /var/spool/... (not writable, wrong paths).
# SLURM_SUBMIT_DIR is the dir we sbatch'd from (== this script's dir). Fall back to
# BASH_SOURCE for direct `bash` / DRY_RUN invocation.
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RUNNER="${HERE}/run_bufr2ioda.sh"

START="${1:?need START_YYYYMMDD}"
END="${2:?need END_YYYYMMDD}"
HOURS="${3:-all}"

IODA_ROOT="${IODA_ROOT:-${HERE}}"
STAGE_DIR="${STAGE_DIR:-${HERE}/_hpss_stage}"
CONVERT="${CONVERT:-1}"
KEEP_PREPBUFR="${KEEP_PREPBUFR:-0}"
DRY_RUN="${DRY_RUN:-0}"
SUBSET="msonet"

mkdir -p "${HERE}/logs" "${STAGE_DIR}"

# want_hour HH -> 0 if this cycle hour is requested, 1 otherwise
want_hour() {
  [[ "${HOURS}" == "all" ]] && return 0
  [[ ",${HOURS}," == *",$1,"* ]] && return 0
  return 1
}

run() {  # echo + execute, or just echo under DRY_RUN
  echo "+ $*"
  [[ "${DRY_RUN}" == "1" ]] && return 0
  "$@"
}

n_fetched=0; n_converted=0; n_skipped_exist=0; n_missing=0

day="${START}"
while [[ "${day}" -le "${END}" ]]; do
  yy="${day:0:4}"; mm="${day:4:2}"; dd="${day:6:2}"

  for base in 00 06 12 18; do
    # which of the 6 hours in this zip do we still need (requested AND not done)?
    needed_cdates=()
    for off in 0 1 2 3 4 5; do
      HH=$(printf "%02d" $((10#${base} + off)))
      cdate="${yy}${mm}${dd}${HH}"
      want_hour "${HH}" || continue
      if [[ -s "${IODA_ROOT}/run_${cdate}/ioda_${SUBSET}.nc" ]]; then
        n_skipped_exist=$((n_skipped_exist + 1)); continue
      fi
      needed_cdates+=("${cdate}")
    done
    [[ ${#needed_cdates[@]} -eq 0 ]] && continue   # nothing to do -> don't download

    zip="${yy}${mm}${dd}${base}00.zip"
    hpss="/BMC/fdr/Permanent/${yy}/${mm}/${dd}/data/grids/rap/obs/${zip}"
    echo "==== ${zip}  (need cycles: ${needed_cdates[*]}) ===="

    # 1) pull the 6-hour zip from HPSS
    run hsi -q "get ${STAGE_DIR}/${zip} : ${hpss}"
    if [[ "${DRY_RUN}" != "1" && ! -s "${STAGE_DIR}/${zip}" ]]; then
      echo "WARNING: ${hpss} not retrieved (missing/empty) -- skipping"
      n_missing=$((n_missing + 1)); continue
    fi
    n_fetched=$((n_fetched + 1))

    # 2) selectively extract ONLY the hourly rap prepbufr files (skip satellite +
    #    rtma_ru), flattening any path prefix.
    run unzip -o -j "${STAGE_DIR}/${zip}" "*.rap.t??z.prepbufr.tm00" -d "${STAGE_DIR}/extract"
    run rm -f "${STAGE_DIR}/${zip}"

    # 3) convert each needed cycle
    for cdate in "${needed_cdates[@]}"; do
      HH="${cdate:8:2}"
      pb="${STAGE_DIR}/extract/${cdate}.rap.t${HH}z.prepbufr.tm00"
      if [[ "${DRY_RUN}" != "1" && ! -s "${pb}" ]]; then
        echo "WARNING: ${pb##*/} not found in zip -- skipping ${cdate}"
        n_missing=$((n_missing + 1)); continue
      fi
      if [[ "${CONVERT}" == "1" ]]; then
        run bash "${RUNNER}" "${cdate}" "${pb}" "${IODA_ROOT}/run_${cdate}" "${SUBSET}"
        if [[ "${DRY_RUN}" == "1" || -s "${IODA_ROOT}/run_${cdate}/ioda_${SUBSET}.nc" ]]; then
          n_converted=$((n_converted + 1))
          [[ "${KEEP_PREPBUFR}" == "1" ]] || run rm -f "${pb}"
        else
          echo "WARNING: converter produced no output for ${cdate}"
        fi
      else
        echo "  (CONVERT=0) staged ${pb}"
      fi
    done
  done

  day="$(date -d "${day} +1 day" +%Y%m%d)"
done

echo "=========================================================="
echo "DONE  fetched_zips=${n_fetched}  converted=${n_converted}  "\
     "already_existed=${n_skipped_exist}  missing=${n_missing}"
echo "Output IODA cycles under: ${IODA_ROOT}/run_<YYYYMMDDHH>/ioda_${SUBSET}.nc"
echo "Next: sample_generate_new.py --obs_source ioda --ioda_root ${IODA_ROOT}"
