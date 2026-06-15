#!/usr/bin/env bash
# Stand up the prepBUFR -> IODA path end-to-end for ONE cycle and ONE subset, using
# the C++ bufr2ioda.x engine + a prepbufr_<subset>.yaml (the same toolchain that
# produced Angie's example ioda_msonet.nc). Reuses the RDASApp build at aburke/RDASApp.
#
# Usage: run_bufr2ioda.sh <CDATE_YYYYMMDDHH> <prepbufr_path> <workdir> [subset=msonet]
#   subset is the prepbufr_<subset>.yaml stem (e.g. msonet, adpsfc).
set -eu

CDATE="${1:?need CDATE YYYYMMDDHH}"
PREPBUFR="${2:?need prepbufr path}"
WORKDIR="${3:?need workdir}"
SUBSET="${4:-msonet}"

RDASAPP=/scratch3/BMC/wrfruc/aburke/RDASApp
BUFR2IODA="${RDASAPP}/build/bin/bufr2ioda.x"
YAML_TEMPLATE="${RDASAPP}/rrfs-test/IODA/yaml/prepbufr_${SUBSET}.yaml"

REFERENCE_TIME="${CDATE:0:4}-${CDATE:4:2}-${CDATE:6:2}T${CDATE:8:2}:00:00Z"

mkdir -p "${WORKDIR}/bufr"
ln -sf "${PREPBUFR}" "${WORKDIR}/bufr/prepbufr"

# Materialize the yaml with the cycle reference time substituted.
sed -e "s/@REFERENCETIME@/${REFERENCE_TIME}/" "${YAML_TEMPLATE}" > "${WORKDIR}/prepbufr_${SUBSET}.yaml"

# Load the RDAS spack-stack runtime (provides bufr2ioda.x shared libs).
source "${RDASAPP}/ush/detect_machine.sh"
echo "MACHINE_ID=${MACHINE_ID}"
source /apps/lmod/lmod/init/bash
module purge
module use "${RDASAPP}/modulefiles"
module load "RDAS/${MACHINE_ID}.intel"

# The aburke/RDASApp build's runtime env has drifted from its modulefile: the
# oneapi openblas-0.3.24 it was linked against was garbage-collected from the spack
# tree, and it needs libstdc++ GLIBCXX_3.4.32 (gcc>=13.2) which this spack-stack's
# compilers don't ship. Stage minimal symlinks to ABI-compatible substitutes and
# prepend ONLY that dir, so we don't shadow other spack-stack libs.
LIBFIX="${WORKDIR}/_libfix"
mkdir -p "${LIBFIX}"
ln -sf /contrib/spack-stack/spack-stack-1.9.3/envs/ue-gcc-12.4.0/install/gcc/12.4.0/openblas-0.3.24-5x25sj2/lib/libopenblas.so.0 "${LIBFIX}/libopenblas.so.0"
ln -sf /apps/rdhpcs-conda/lib/libstdc++.so.6 "${LIBFIX}/libstdc++.so.6"
export LD_LIBRARY_PATH="${LIBFIX}:${LD_LIBRARY_PATH}"

cd "${WORKDIR}"
echo "REFERENCE_TIME=${REFERENCE_TIME}"
echo "Running: ${BUFR2IODA} prepbufr_${SUBSET}.yaml"
time "${BUFR2IODA}" "prepbufr_${SUBSET}.yaml"

echo "=== output ==="
ls -la "${WORKDIR}"/ioda_${SUBSET}.nc
