#!/usr/bin/env bash
# Delete Merlin scans that are not in merlin_keep_list.txt.
#
# Usage:
#   bash scripts/prune_merlin_scans.sh /path/to/merlin_data
#
# Dry-run (print what would be deleted without deleting):
#   bash scripts/prune_merlin_scans.sh /path/to/merlin_data --dry-run

set -euo pipefail

SCAN_DIR="${1:-}"
DRY_RUN=0
[ "${2:-}" = "--dry-run" ] && DRY_RUN=1

if [ -z "${SCAN_DIR}" ]; then
    echo "Usage: $0 /path/to/merlin_data [--dry-run]"
    exit 1
fi

if [ ! -d "${SCAN_DIR}" ]; then
    echo "Error: directory not found: ${SCAN_DIR}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEEP_LIST="${SCRIPT_DIR}/../merlin_keep_list.txt"

if [ ! -f "${KEEP_LIST}" ]; then
    echo "Error: keep list not found: ${KEEP_LIST}"
    exit 1
fi

keep_count=$(wc -l < "${KEEP_LIST}")
echo "Keep list: ${KEEP_LIST} (${keep_count} IDs)"
echo "Scan dir:  ${SCAN_DIR}"
[ "${DRY_RUN}" = "1" ] && echo "Mode: DRY RUN — nothing will be deleted"
echo ""

deleted=0
kept=0

while IFS= read -r -d '' scan_file; do
    stem="${scan_file%.nii.gz}"
    stem="${stem##*/}"
    if grep -qxF "${stem}" "${KEEP_LIST}"; then
        kept=$((kept + 1))
    else
        if [ "${DRY_RUN}" = "1" ]; then
            echo "  would delete: $(basename "${scan_file}")"
        else
            rm "${scan_file}"
        fi
        deleted=$((deleted + 1))
    fi
done < <(find "${SCAN_DIR}" -maxdepth 1 -name "*.nii.gz" -print0)

echo ""
echo "Kept:    ${kept}"
if [ "${DRY_RUN}" = "1" ]; then
    echo "Would delete: ${deleted}"
else
    echo "Deleted: ${deleted}"
fi
