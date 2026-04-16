#!/bin/bash
set -euo pipefail

file_counter=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/" && pwd)"
TEMPLATE_DIR="${SCRIPT_DIR}/templates_G_UV"
mkdir -p "${TEMPLATE_DIR}" "${SCRIPT_DIR}/logs_G_UV" "${SCRIPT_DIR}/submitted_G_UV"

KXKY_FILE="${CODE_DIR}/kxkyset.txt"
UV_FILE="${CODE_DIR}/UVset.txt"
RUN_DIR="/home/9pm/ED_qbasis/qbasis/bin/ML/HF/Gcalpath"
SUBMIT_NEXT_SCRIPT="${RUN_DIR}/job_submit/submit_next_job_G_UV.sh"

NUP=8
NDN=8
QX=0
QY=0
TP=0

job_header='#!/bin/bash
#SBATCH -A __ACCOUNT__
#SBATCH -p burst
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -c 1
#SBATCH -J GUV
#SBATCH --mem=100G
#SBATCH -t 4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cosdis123@gmail.com

cd /home/9pm/ED_qbasis/qbasis/bin/ML/HF/Gcalpath
export OMP_NUM_THREADS=32
'

if [[ ! -f "${KXKY_FILE}" ]]; then
  echo "Missing ${KXKY_FILE}"
  exit 1
fi

if [[ ! -f "${UV_FILE}" ]]; then
  echo "Missing ${UV_FILE}"
  exit 1
fi

mapfile -t k_points < <(grep -oE '\([0-9]+,[0-9]+\)' "${KXKY_FILE}")
mapfile -t uv_points < <(grep -oE '\([0-9.]+,-?[0-9.]+\)' "${UV_FILE}")

if (( ${#k_points[@]} == 0 )); then
  echo "No (kx,ky) pairs found in ${KXKY_FILE}"
  exit 1
fi

if (( ${#uv_points[@]} == 0 )); then
  echo "No (U,V) pairs found in ${UV_FILE}"
  exit 1
fi

for k_pair in "${k_points[@]}"; do
  k_vals="${k_pair#(}"
  k_vals="${k_vals%)}"
  IFS=',' read -r kx ky <<< "${k_vals}"

  for uv_pair in "${uv_points[@]}"; do
    uv_vals="${uv_pair#(}"
    uv_vals="${uv_vals%)}"
    IFS=',' read -r U V <<< "${uv_vals}"

    result_gp="./L4x4Nup${NUP}Ndn${NDN}U${U}V${V}tp${TP}f${kx}x${ky}Q${QX}x${QY}_Gp.dat"
    result_gh="./L4x4Nup${NUP}Ndn${NDN}U${U}V${V}tp${TP}f${kx}x${ky}Q${QX}x${QY}_Gh.dat"

    if [[ -f "${result_gp}" && -f "${result_gh}" ]]; then
      echo "Skipping: k=(${kx},${ky}) U=${U} V=${V} (results exist)."
      continue
    fi

    job_file="${TEMPLATE_DIR}/job$(printf '%06d' "${file_counter}").sh"
    {
      printf "%s\n" "${job_header}"
      echo "time ./Square_Hubbard_tbc_G_calculation_UV.x ${NUP} ${NDN} ${kx} ${ky} ${QX} ${QY} ${U} ${V} ${TP}"
      echo "job_rc=\$?"
      echo "bash \"${SUBMIT_NEXT_SCRIPT}\" \"\${SLURM_JOB_ACCOUNT:-__ACCOUNT__}\""
      echo "exit \${job_rc}"
    } > "${job_file}"

    chmod +x "${job_file}"
    ((file_counter++))
  done
done

echo "Generated $((file_counter - 1)) G-UV template jobs in ${TEMPLATE_DIR}."
