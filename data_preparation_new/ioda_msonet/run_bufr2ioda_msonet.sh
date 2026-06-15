#!/usr/bin/env bash
# Stand up the prepBUFR -> IODA "MSONET" path end-to-end for ONE cycle, using the
# C++ bufr2ioda.x engine + prepbufr_msonet.yaml (the same toolchain that produced
# Angie's example ioda_msonet.nc). Reuses the RDASApp build at aburke/RDASApp.
#
# Usage: run_bufr2ioda_msonet.sh <CDATE_YYYYMMDDHH> <prepbufr_path> <workdir>
set -eu

CDATE="${1:?need CDATE YYYYMMDDHH}"
PREPBUFR="${2:?need prepbufr path}"
WORKDIR="${3:?need workdir}"

RDASAPP=/scratch3/BMC/wrfruc/aburke/RDASApp
BUFR2IODA="${RDASAPP}/build/bin/bufr2ioda.x"
YAML_TEMPLATE="${RDASAPP}/rrfs-test/IODA/yaml/prepbufr_msonet.yaml"

REFERENCE_TIME="${CDATE:0:4}-${CDATE:4:2}-${CDATE:6:2}T${CDATE:8:2}:00:00Z"

mkdir -p "${WORKDIR}/bufr"
ln -sf "${PREPBUFR}" "${WORKDIR}/bufr/prepbufr"

# Materialize the yaml with the cycle reference time substituted.
sed -e "s/@REFERENCETIME@/${REFERENCE_TIME}/" "${YAML_TEMPLATE}" > "${WORKDIR}/prepbufr_msonet.yaml"

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
echo "Running: ${BUFR2IODA} prepbufr_msonet.yaml"
time "${BUFR2IODA}" prepbufr_msonet.yaml

echo "=== output ==="
ls -la "${WORKDIR}"/ioda_msonet.nc
