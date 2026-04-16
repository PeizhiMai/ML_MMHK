#!/usr/bin/env bash
set -euo pipefail

for i in {1..15}; do
  for j in $(seq "$i" "$((i+1))"); do
    mkdir -p "Nup${i}Ndn${j}"
  done
done

for i in {1..15}; do
  for j in $(seq "$i" "$((i+1))"); do
    cp L4x4Nup${i}Ndn${j}* "Nup${i}Ndn${j}/"
  done
done

rm -r Nup15Ndn16

tar -czvf tpn25U6obs.tar.gz Nup*

mkdir old

mv L* old/

