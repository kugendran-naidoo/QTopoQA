#!/usr/bin/env bash
set -euo pipefail
set -E

export LC_ALL=C
export LC_NUMERIC=C

usage() {
  cat <<'EOF'
Usage:
  mac_mk_safe_ramdisk.sh --list
  mac_mk_safe_ramdisk.sh --mount <device>
  mac_mk_safe_ramdisk.sh --unmount <device|mountpoint>
  mac_mk_safe_ramdisk.sh --detach <device|mountpoint> [--force]
  mac_mk_safe_ramdisk.sh --detach-all [--force]
  mac_mk_safe_ramdisk.sh --size <value> [--headroom <value>] [--name <vol>]
                  [--filesystem <APFS|JHFS+>] [--force]

Creates and manages RAM-backed disks on macOS in a defensive way.

Options:
  --list        Show currently attached RAM disks and exit.
  --mount       Mount a known RAM disk by device node (e.g. /dev/disk6).
  --unmount     Unmount a known RAM disk by device or mount point.
  --detach      Detach a known RAM disk safely (prompts unless --force).
  --detach-all  Detach every RAM disk discovered (prompts for each unless --force).
  --size        Requested RAM disk size (required for creation). Supports suffixes
                B, K, M, G, T with optional "iB"; no suffix defaults to GiB. Decimals allowed.
  --headroom    Additional free memory to keep beyond the disk size.
                Accepts absolute sizes (e.g. 1G) or percentages (e.g. 25%).
                Defaults to 25% of the requested size.
  --name        Volume name to assign (default RAMDisk).
  --filesystem  Filesystem format for the volume: APFS (default) or JHFS+.
  --force       Skip swap-in-use checks and detach confirmations (use with care).
  -h, --help    Show this help.

Headroom keeps breathing room for macOS so other processes stay responsive.
25% is a good starting point; increase it (e.g. 40–50%) if you expect other
memory-hungry workloads to run alongside the RAM disk. Example: --size 4G --headroom 25%
means the script will only proceed if at least 5 GiB of free RAM is available.

Safety tips:
  - Set SAFE_RAMDISK_TEST_MODE=1 to convert all actions into dry runs (no system changes).
  - Use small trial sizes (e.g. --size 256M) before provisioning larger disks.

Suggested workflow:
  1. Inspect current RAM disks:  mac_mk_safe_ramdisk.sh --list
  2. Dry-run with a small disk to confirm headroom:  mac_mk_safe_ramdisk.sh --size 512M --headroom 25% --name ScratchTest
  3. Create the final disk once satisfied:  mac_mk_safe_ramdisk.sh --size 12G --headroom 4G --name BuildCache
  4. Unmount when finished:    mac_mk_safe_ramdisk.sh --unmount /dev/diskN
  5. Detach after unmounting:  mac_mk_safe_ramdisk.sh --detach /dev/diskN --force

Examples:
  mac_mk_safe_ramdisk.sh --list
  mac_mk_safe_ramdisk.sh --mount /dev/disk6
  mac_mk_safe_ramdisk.sh --unmount /Volumes/RAMDisk
  mac_mk_safe_ramdisk.sh --size 750MiB --filesystem JHFS+
  mac_mk_safe_ramdisk.sh --size 0.5 --headroom 1 --name Scratch   # values without suffix are GiB
  mac_mk_safe_ramdisk.sh --detach-all --force
EOF
}

ramdisk_inventory() {
  local info
  if ! info=$(hdiutil info 2>/dev/null); then
    return 0
  fi

  python3 - "$info" <<'PY'
import sys

raw = sys.argv[1]
records = raw.split("================================================")
for block in records:
    if "<ram>" not in block and "ram://" not in block:
        continue
    dev = None
    mount = None
    blockcount = None
    blocksize = None
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("dev-entry:"):
            dev = line.split(":", 1)[1].strip()
        elif line.startswith("blockcount"):
            try:
                blockcount = int(line.split(":", 1)[1].strip())
            except Exception:
                blockcount = None
        elif line.startswith("blocksize"):
            try:
                blocksize = int(line.split(":", 1)[1].strip())
            except Exception:
                blocksize = None
        elif line.startswith("/dev/disk"):
            parts = line.split()
            if not parts:
                continue
            candidate = parts[0]
            if dev is None:
                dev = candidate
            tail = parts[-1]
            if tail.startswith("/Volumes/"):
                mount = tail
        elif ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in {"mount-point", "mounted volume"} and value:
                mount = value
    if not dev:
        continue
    total_bytes = 0
    if blockcount is not None and blocksize is not None:
        try:
            total_bytes = blockcount * blocksize
        except Exception:
            total_bytes = 0
    if mount is None:
        mount = ""
    print(f"{dev}\x1f{mount}\x1f{total_bytes}")
PY
}

list_ramdisks() {
  local found=0
  while IFS=$'\x1f' read -r dev mount bytes; do
    [[ -z "$dev" ]] && continue
    if (( found == 0 )); then
      echo "RAM disks currently attached:"
    fi
    found=1
    local bytes_safe="${bytes:-0}"
    local size_gib
    size_gib=$(format_gib "$bytes_safe")
    local mount_disp="$mount"
    if [[ -z "$mount_disp" ]]; then
      mount_disp="(not mounted)"
    fi
    echo "  $dev  ${size_gib} GiB  $mount_disp"
  done < <(ramdisk_inventory)
  if (( found == 0 )); then
    echo "No RAM disks currently attached."
  fi
}

normalize_device_identifier() {
  local candidate="$1"
  if [[ "$candidate" =~ ^/dev/disk[0-9]+$ ]]; then
    printf "%s\n" "$candidate"
  elif [[ "$candidate" =~ ^disk[0-9]+$ ]]; then
    printf "/dev/%s\n" "$candidate"
  else
    printf "%s\n" ""
  fi
}

find_ramdisk_entry() {
  local target="$1"
  local canonical
  canonical=$(normalize_device_identifier "$target")
  while IFS=$'\x1f' read -r dev mount bytes; do
    [[ -z "$dev" ]] && continue
    if [[ -n "$canonical" && "$dev" == "$canonical" ]]; then
      printf "%s\x1f%s\x1f%s\n" "$dev" "$mount" "$bytes"
      return 0
    fi
    if [[ -n "$mount" && "$target" == "$mount" ]]; then
      printf "%s\x1f%s\x1f%s\n" "$dev" "$mount" "$bytes"
      return 0
    fi
  done < <(ramdisk_inventory)
  return 1
}

run_cmd() {
  local description="$1"
  shift
  if (( TEST_MODE_ACTIVE )); then
    echo "[TEST MODE] $description: $*"
    return 0
  fi
  "$@"
}

confirm_detach() {
  local dev="$1"
  local mount="$2"
  if (( TEST_MODE_ACTIVE )) || (( FORCE_MODE )); then
    return 0
  fi
  local prompt="Detach RAM disk $dev"
  if [[ -n "$mount" ]]; then
    prompt+=" ($mount)"
  fi
  prompt+="? [y/N] "
  if [[ -t 0 ]]; then
    local reply
    read -r -p "$prompt" reply
    case "$reply" in
      [Yy]|[Yy][Ee][Ss])
        return 0
        ;;
      *)
        fail "Detach cancelled."
        ;;
    esac
  else
    fail "Detach of $dev requires confirmation; re-run interactively or add --force."
  fi
}

detach_ramdisk_device() {
  local dev="$1"
  local mount="$2"
  local verify
  verify=$(find_ramdisk_entry "$dev" || true)
  if [[ -z "$verify" ]]; then
    fail "Refusing: $dev is not currently reported as a RAM disk."
  fi
  confirm_detach "$dev" "$mount"
  if [[ -n "$mount" ]]; then
    echo "Unmounting $dev ($mount) before detach..."
    run_cmd "diskutil unmountDisk $dev" diskutil unmountDisk "$dev"
  else
    echo "Ensuring $dev is unmounted before detach..."
    if ! run_cmd "diskutil unmountDisk $dev" diskutil unmountDisk "$dev"; then
      echo "diskutil reported $dev as already unmounted; continuing with detach."
    fi
  fi
  echo "Detaching $dev..."
  run_cmd "hdiutil detach $dev" hdiutil detach "$dev"
}

perform_mount() {
  local entry
  entry=$(find_ramdisk_entry "$MOUNT_TARGET") || fail "Target '$MOUNT_TARGET' is not a known RAM disk."
  IFS=$'\x1f' read -r dev mount bytes <<<"$entry"
  if [[ -n "$mount" ]]; then
    echo "$dev is already mounted at $mount."
    return 0
  fi
  echo "Mounting $dev..."
  run_cmd "diskutil mountDisk $dev" diskutil mountDisk "$dev"
  local updated
  updated=$(find_ramdisk_entry "$dev" || true)
  if [[ -n "$updated" ]]; then
    IFS=$'\x1f' read -r _dev new_mount _bytes <<<"$updated"
    if [[ -n "$new_mount" ]]; then
      echo "$dev now mounted at $new_mount"
    else
      echo "$dev mounted successfully (mount point unavailable; check 'mac_mk_safe_ramdisk.sh --list')."
    fi
  fi
}

perform_unmount() {
  local entry
  entry=$(find_ramdisk_entry "$UNMOUNT_TARGET") || fail "Target '$UNMOUNT_TARGET' is not a known RAM disk."
  IFS=$'\x1f' read -r dev mount bytes <<<"$entry"
  if [[ -z "$mount" ]]; then
    echo "$dev appears to be already unmounted."
  else
    echo "Unmounting $dev ($mount)..."
  fi
  run_cmd "diskutil unmountDisk $dev" diskutil unmountDisk "$dev"
}

perform_detach() {
  local entry
  entry=$(find_ramdisk_entry "$DETACH_TARGET") || fail "Target '$DETACH_TARGET' is not a known RAM disk."
  IFS=$'\x1f' read -r dev mount bytes <<<"$entry"
  detach_ramdisk_device "$dev" "$mount"
}

perform_detach_all() {
  local any=0
  while IFS=$'\x1f' read -r dev mount bytes; do
    [[ -z "$dev" ]] && continue
    any=1
    echo
    echo "Preparing to detach $dev..."
    detach_ramdisk_device "$dev" "$mount"
  done < <(ramdisk_inventory)
  if (( any == 0 )); then
    echo "No RAM disks currently attached."
  fi
}

parse_headroom_percentage() {
  local spec="${1:-}"
  local base_bytes="${2:-0}"

  python3 - "$spec" "$base_bytes" <<'PY'
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

raw_spec = sys.argv[1].strip()
base_bytes = Decimal(sys.argv[2])

if not raw_spec.endswith('%'):
    print("Internal error: parse_headroom_percentage expects a % value.", file=sys.stderr)
    sys.exit(1)

percent_text = raw_spec[:-1].strip()
if not percent_text:
    print("Headroom percentage must include a numeric value before '%'.", file=sys.stderr)
    sys.exit(1)

try:
    percent = Decimal(percent_text)
except InvalidOperation:
    print(f"Invalid percentage value '{raw_spec}'.", file=sys.stderr)
    sys.exit(1)

if percent < 0:
    print("Headroom percentage must not be negative.", file=sys.stderr)
    sys.exit(1)

bytes_value = (base_bytes * percent) / Decimal(100)
bytes_int = int(bytes_value.to_integral_value(rounding=ROUND_HALF_UP))
print(bytes_int)
PY
}

parse_size() {
  local spec="${1:-}"
  local allow_zero="${2:-0}"

  python3 - "$spec" "$allow_zero" <<'PY'
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re

spec = sys.argv[1].strip()
allow_zero = sys.argv[2] == "1"

if not spec:
    print("Size value may not be empty.", file=sys.stderr)
    sys.exit(1)

pattern = re.compile(r'^(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[KMGTP]?i?[Bb]?|[Bb])?$', re.IGNORECASE)
match = pattern.fullmatch(spec)
if not match:
    print(f"Invalid size specification: '{spec}'", file=sys.stderr)
    sys.exit(1)

num_text = match.group('num')
unit_text = (match.group('unit') or '').lower()

unit_aliases = {
    '': 'gib',  # default to GiB when omitted
    'b': 'b',
    'k': 'kib',
    'kb': 'kb',
    'ki': 'kib',
    'kib': 'kib',
    'm': 'mib',
    'mb': 'mb',
    'mi': 'mib',
    'mib': 'mib',
    'g': 'gib',
    'gb': 'gb',
    'gi': 'gib',
    'gib': 'gib',
    't': 'tib',
    'tb': 'tb',
    'ti': 'tib',
    'tib': 'tib',
    'p': 'pib',
    'pb': 'pb',
    'pi': 'pib',
    'pib': 'pib',
}

if unit_text not in unit_aliases:
    print(f"Unsupported size unit in '{spec}'", file=sys.stderr)
    sys.exit(1)

unit = unit_aliases[unit_text]

factors = {
    'b': Decimal(1),
    'kb': Decimal(1000) ** 1,
    'mb': Decimal(1000) ** 2,
    'gb': Decimal(1000) ** 3,
    'tb': Decimal(1000) ** 4,
    'pb': Decimal(1000) ** 5,
    'kib': Decimal(1024) ** 1,
    'mib': Decimal(1024) ** 2,
    'gib': Decimal(1024) ** 3,
    'tib': Decimal(1024) ** 4,
    'pib': Decimal(1024) ** 5,
}

try:
    number = Decimal(num_text)
except InvalidOperation:
    print(f"Invalid numeric value in '{spec}'", file=sys.stderr)
    sys.exit(1)

bytes_value = number * factors[unit]

if bytes_value < 0 or (bytes_value == 0 and not allow_zero):
    cmp_text = "greater" if not allow_zero else "not negative"
    print(f"Size must be {cmp_text} than 0.", file=sys.stderr)
    sys.exit(1)

# Round to the nearest whole byte.
bytes_int = int(bytes_value.to_integral_value(rounding=ROUND_HALF_UP))
print(bytes_int)
PY
}

format_gib() {
  local bytes="${1:-0}"
  python3 - "$bytes" <<'PY'
import sys
from decimal import Decimal, ROUND_HALF_UP

value = Decimal(sys.argv[1])
gib = value / (Decimal(1024) ** 3)
print(f"{gib.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}")
PY
}

fail() {
  echo "Error: $*" >&2
  exit 1
}

TEST_MODE_RAW="${SAFE_RAMDISK_TEST_MODE:-${TEST_MODE:-0}}"
if [[ -n "$TEST_MODE_RAW" && "$TEST_MODE_RAW" != "0" ]]; then
  TEST_MODE_ACTIVE=1
else
  TEST_MODE_ACTIVE=0
fi
unset TEST_MODE_RAW

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

SIZE_SPEC=""
HEADROOM_SPEC="25%"
VOL_NAME="RAMDisk"
FILESYSTEM="APFS"
FORCE_MODE=0
SIZE_SET=0
HEADROOM_SET=0
NAME_SET=0
FILESYSTEM_SET=0
LIST_MODE=0
MOUNT_TARGET=""
UNMOUNT_TARGET=""
DETACH_TARGET=""
DETACH_ALL=0
ACTION_COUNT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size)
      [[ $# -ge 2 ]] || fail "Missing argument for --size"
      SIZE_SPEC="$2"
      SIZE_SET=1
      shift 2
      ;;
    --headroom)
      [[ $# -ge 2 ]] || fail "Missing argument for --headroom"
      HEADROOM_SPEC="$2"
      HEADROOM_SET=1
      shift 2
      ;;
    --name)
      [[ $# -ge 2 ]] || fail "Missing argument for --name"
      VOL_NAME="$2"
      NAME_SET=1
      shift 2
      ;;
    --filesystem)
      [[ $# -ge 2 ]] || fail "Missing argument for --filesystem"
      FILESYSTEM="$2"
      FILESYSTEM_SET=1
      shift 2
      ;;
    --force)
      FORCE_MODE=1
      shift
      ;;
    --mount)
      [[ $# -ge 2 ]] || fail "Missing argument for --mount (examples: --mount /dev/disk6  or  --mount /Volumes/RAMDisk)"
      MOUNT_TARGET="$2"
      ACTION_COUNT=$((ACTION_COUNT + 1))
      shift 2
      ;;
    --unmount)
      [[ $# -ge 2 ]] || fail "Missing argument for --unmount (examples: --unmount /dev/disk6  or  --unmount /Volumes/RAMDisk)"
      UNMOUNT_TARGET="$2"
      ACTION_COUNT=$((ACTION_COUNT + 1))
      shift 2
      ;;
    --detach)
      [[ $# -ge 2 ]] || fail "Missing argument for --detach (examples: --detach /dev/disk6  or  --detach /Volumes/RAMDisk)"
      DETACH_TARGET="$2"
      ACTION_COUNT=$((ACTION_COUNT + 1))
      shift 2
      ;;
    --detach-all)
      DETACH_ALL=1
      ACTION_COUNT=$((ACTION_COUNT + 1))
      shift
      ;;
    --list)
      LIST_MODE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if (( ACTION_COUNT > 1 )); then
  fail "Only one of --mount, --unmount, --detach, or --detach-all may be specified at a time."
fi

if (( LIST_MODE )); then
  if (( ACTION_COUNT > 0 )); then
    fail "--list cannot be combined with mount/unmount/detach actions."
  fi
  if (( SIZE_SET || HEADROOM_SET || NAME_SET || FILESYSTEM_SET )); then
    fail "--list cannot be combined with creation options."
  fi
  list_ramdisks
  exit 0
fi

if (( ACTION_COUNT > 0 )); then
  if (( SIZE_SET || HEADROOM_SET || NAME_SET || FILESYSTEM_SET )); then
    fail "Mount/unmount/detach actions cannot be combined with creation options."
  fi
  if [[ -n "$MOUNT_TARGET" ]]; then
    perform_mount
    exit 0
  elif [[ -n "$UNMOUNT_TARGET" ]]; then
    perform_unmount
    exit 0
  elif [[ -n "$DETACH_TARGET" ]]; then
    perform_detach
    exit 0
  elif (( DETACH_ALL )); then
    perform_detach_all
    exit 0
  else
    fail "No management action specified."
  fi
fi

if [[ -z "$SIZE_SPEC" ]]; then
  echo "Error: --size is required when creating a RAM disk." >&2
  usage >&2
  exit 1
fi

[[ -n "$VOL_NAME" ]] || fail "Volume name may not be empty."
[[ "$VOL_NAME" != */* ]] || fail "Volume name may not contain '/'."

lower_fs=$(printf '%s' "$FILESYSTEM" | tr '[:upper:]' '[:lower:]')
case "$lower_fs" in
  apfs)
    FILESYSTEM="APFS"
    ;;
  jhfs+|journaled-hfs+|journaled_hfs+|journaledhfs+)
    FILESYSTEM="JHFS+"
    ;;
  *)
    fail "Unsupported filesystem '$FILESYSTEM'. Use APFS or JHFS+."
    ;;
esac

SIZE_BYTES=$(parse_size "$SIZE_SPEC")
if [[ "$HEADROOM_SPEC" == *% ]]; then
  HEADROOM_BYTES=$(parse_headroom_percentage "$HEADROOM_SPEC" "$SIZE_BYTES")
else
  HEADROOM_BYTES=$(parse_size "$HEADROOM_SPEC" 1)
fi

(( SIZE_BYTES > 0 )) || fail "Size must be greater than 0."
(( HEADROOM_BYTES >= 0 )) || fail "Headroom must be zero or greater."

pagesize=$(pagesize)

read -r P_FREE P_SPEC P_PURG <<<"$(
  vm_stat | awk '
    /Pages free:/        {gsub("\\.","",$3); f=$3}
    /Pages speculative:/ {gsub("\\.","",$3); s=$3}
    /Pages purgeable:/   {gsub("\\.","",$3); p=$3}
    END {
      if (f=="") f=0
      if (s=="") s=0
      if (p=="") p=0
      printf "%s %s %s\n", f+0, s+0, p+0
    }'
)"

FREE_BYTES=$(( (P_FREE + P_SPEC + P_PURG) * pagesize ))
NEED_BYTES=$(( SIZE_BYTES + HEADROOM_BYTES ))

swap_line=$(sysctl -n vm.swapusage 2>/dev/null || true)
swap_used_value=0
if [[ -n "$swap_line" ]]; then
  swap_value=$(printf "%s\n" "$swap_line" | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p')
  if [[ -n "$swap_value" ]]; then
    swap_used_value=$(parse_size "${swap_value}M" 1)
  fi
fi

free_gib=$(format_gib "$FREE_BYTES")
size_gib=$(format_gib "$SIZE_BYTES")
headroom_gib=$(format_gib "$HEADROOM_BYTES")
swap_used_gib=$(format_gib "$swap_used_value")

echo "== Memory check =="
echo "Requested RAM disk: ${size_gib} GiB"
echo "Required headroom : ${headroom_gib} GiB"
echo "Free (conservative): ${free_gib} GiB  (Free+Speculative+Purgeable)"
echo "Swap used         : ${swap_used_gib} GiB"
echo

if (( FORCE_MODE == 0 )) && (( swap_used_value > 0 )); then
  fail "Refusing: system is already swapping (${swap_used_gib} GiB used). Use --force to override."
fi

if (( FREE_BYTES < NEED_BYTES )); then
  fail "Refusing: need at least $(format_gib "$NEED_BYTES") GiB free; only ${free_gib} GiB available."
fi

if command -v memory_pressure >/dev/null 2>&1; then
  level=$(memory_pressure -Q | awk '
    /memory free percentage/ {
      sub(/.*: /, "");
      gsub(/%/, "");
      p=$0 + 0;
      if (p < 10) { print 2; exit }
      if (p < 20) { print 1; exit }
      print 0;
      exit
    }
  ')
  case "${level:-0}" in
    2) fail "Refusing: memory pressure is CRITICAL." ;;
    1) echo "Warning: memory pressure elevated (proceeding cautiously)." ;;
  esac
fi

if hdiutil info | awk '
  /^image-path: <ram>/ {inram=1}
  inram && /^dev-entry:/ {found=1}
  END {exit !found}
'; then
  fail "A RAM disk device is already attached; not creating another."
fi

if mount | awk -v vol="/Volumes/${VOL_NAME}" '$3 == vol {found=1} END {exit !found}'; then
  fail "Volume /Volumes/${VOL_NAME} already exists; not creating."
fi

sectors=$(( (SIZE_BYTES + 511) / 512 ))
(( sectors > 0 )) || fail "Computed sector count invalid."

DEV=""
cleanup_on_error() {
  local exit_code=$?
  if [[ -n "$DEV" ]]; then
    hdiutil detach "$DEV" >/dev/null 2>&1 || true
  fi
  exit "$exit_code"
}

trap cleanup_on_error ERR INT TERM

echo "All checks passed. Creating $(format_gib "$SIZE_BYTES") GiB RAM disk with $(format_gib "$HEADROOM_BYTES") GiB headroom..."
if (( TEST_MODE_ACTIVE )); then
  echo "[TEST MODE] Would run: hdiutil attach -nomount \"ram://${sectors}\""
  echo "[TEST MODE] Would run: diskutil eraseVolume \"$FILESYSTEM\" \"$VOL_NAME\" <device>"
  echo "[TEST MODE] RAM disk creation skipped."
  exit 0
fi
DEV=$(hdiutil attach -nomount "ram://${sectors}" | awk 'NR==1 {print $1; exit}')
[[ -n "$DEV" ]] || fail "Failed to attach RAM disk device."

diskutil eraseVolume "$FILESYSTEM" "$VOL_NAME" "$DEV"

trap - ERR INT TERM

echo "Created: $DEV mounted at /Volumes/${VOL_NAME}"
