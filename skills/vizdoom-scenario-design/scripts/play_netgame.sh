#!/bin/bash
# N human-controlled ViZDoom instances on one machine (1 host + N-1 joins), for playtesting a scenario.
# Generalised from the repository's play4.sh (Simple Tag, 4 players). Works with macOS bash 3.2.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: play_netgame.sh [options] scenario.wad [-- extra engine args]

Options:
  --players N          number of instances, 1-16 (default 2)
  --mode MODE          coop | deathmatch (default deathmatch)
  --names A,B,...      player names, one per player (default Player0,...,PlayerN-1);
                       scenarios that bind roles to names need the exact names
  --colors C1,C2,...   player colours as "rr gg bb" hex triplets, one per player or one for all
                       (default "ff 00 00" for everyone)
  --audio-player P     instance that plays sound: 0..N-1, all or none (default 0);
                       one listener avoids overlapping spatial mixes
  --map MAP            map to start (default: the engine's first map, MAP01)
  --port PORT          host port (default $VIZDOOM_PORT or 5029)
  --respawn            allow respawning (default: +sv_norespawn 1)
  --dry-run            print the commands and exit without starting anything
  -h, --help           show this help

Environment: VIZDOOM_BIN (engine binary), VIZDOOM_IWAD (default: freedoom2.wad of the
vizdoom package), VIZDOOM_PYTHON (Python used to locate the vizdoom package).

Example (Simple Tag: three red hunters, one green prey, host plays sound):
  play_netgame.sh --players 4 --mode deathmatch \
      --colors "ff 00 00,ff 00 00,ff 00 00,00 ff 00" scenarios/simple_tag_audio.wad
EOF
}

die() {
    printf 'play_netgame.sh: %s\n' "$1" >&2
    exit 2
}

players=2
mode=deathmatch
names=""
colors=""
audio_player=0
map=""
port="${VIZDOOM_PORT:-5029}"
respawn=0
dry_run=0
wad=""
extra=()

while (($#)); do
    case "$1" in
        --players) players="${2-}"; shift 2 || die "--players needs a value" ;;
        --mode) mode="${2-}"; shift 2 || die "--mode needs a value" ;;
        --names) names="${2-}"; shift 2 || die "--names needs a value" ;;
        --colors) colors="${2-}"; shift 2 || die "--colors needs a value" ;;
        --audio-player) audio_player="${2-}"; shift 2 || die "--audio-player needs a value" ;;
        --map) map="${2-}"; shift 2 || die "--map needs a value" ;;
        --port) port="${2-}"; shift 2 || die "--port needs a value" ;;
        --respawn) respawn=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; extra=("$@"); break ;;
        -*) die "unknown option: $1 (see --help)" ;;
        *) [[ -z "$wad" ]] || die "only one WAD can be given (extra: $1)"; wad="$1"; shift ;;
    esac
done

# ---- validate arguments -------------------------------------------------------
[[ -n "$wad" ]] || { usage >&2; exit 2; }
[[ -f "$wad" ]] || die "WAD not found: $wad"
wad="$(cd "$(dirname "$wad")" && pwd)/$(basename "$wad")"
[[ "$players" =~ ^[0-9]+$ ]] && ((players >= 1 && players <= 16)) || die "--players must be an integer from 1 to 16, got '$players'"
case "$mode" in coop|deathmatch) ;; *) die "--mode must be coop or deathmatch, got '$mode'" ;; esac
[[ "$port" =~ ^[0-9]+$ ]] || die "--port must be a number, got '$port'"

if [[ -n "$names" ]]; then
    IFS=',' read -r -a name_list <<< "$names"
    ((${#name_list[@]} == players)) || die "--names has ${#name_list[@]} entries for $players players"
else
    name_list=()
    for ((i = 0; i < players; i++)); do name_list+=("Player$i"); done
fi
IFS=',' read -r -a color_list <<< "${colors:-ff 00 00}"
if ((${#color_list[@]} == 1 && players > 1)); then
    for ((i = 1; i < players; i++)); do color_list+=("${color_list[0]}"); done
fi
((${#color_list[@]} == players)) || die "--colors has ${#color_list[@]} entries for $players players"
for color in "${color_list[@]}"; do
    [[ "$color" =~ ^[0-9a-fA-F]{2}\ [0-9a-fA-F]{2}\ [0-9a-fA-F]{2}$ ]] || die "colour must look like 'ff 00 00', got '$color'"
done
case "$audio_player" in
    all|none) ;;
    *) [[ "$audio_player" =~ ^[0-9]+$ ]] && ((audio_player < players)) \
        || die "--audio-player must be 0..$((players - 1)), all or none, got '$audio_player'" ;;
esac

# ---- locate the engine, IWAD and OpenAL ---------------------------------------
find_python() {
    local candidate
    for candidate in "${VIZDOOM_PYTHON:-}" .venv/bin/python python3 python; do
        if [[ -n "$candidate" ]] && command -v "$candidate" >/dev/null 2>&1 \
            && "$candidate" -c "import vizdoom" >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}
package_dir=""
if [[ -z "${VIZDOOM_BIN:-}" || -z "${VIZDOOM_IWAD:-}" ]]; then
    if python_bin="$(find_python)"; then
        package_dir="$("$python_bin" -c "import os, vizdoom; print(os.path.dirname(vizdoom.__file__))" 2>/dev/null)"
    fi
fi
# Execute the engine binary directly so cleanup owns exact child PIDs; never
# pkill an unrelated ViZDoom application.
if [[ -n "${VIZDOOM_BIN:-}" ]]; then
    executable="$VIZDOOM_BIN"
elif [[ -n "$package_dir" && -x "$package_dir/vizdoom" ]]; then
    executable="$package_dir/vizdoom"
else
    executable=""
fi
iwad="${VIZDOOM_IWAD:-${package_dir:+$package_dir/freedoom2.wad}}"
if ((!dry_run)); then
    [[ -n "$executable" && -x "$executable" ]] || die "ViZDoom engine not found; install the vizdoom package or set VIZDOOM_BIN"
    [[ -n "$iwad" && -f "$iwad" ]] || die "IWAD not found; set VIZDOOM_IWAD"
fi
executable="${executable:-<vizdoom>}"
iwad="${iwad:-<freedoom2.wad>}"

if [[ "$OSTYPE" == darwin* ]]; then
    # OpenAL Soft supports both audible output and ViZDoom loopback capture.
    for libdir in /opt/homebrew/opt/openal-soft/lib /usr/local/opt/openal-soft/lib; do
        if [[ -f "$libdir/libopenal.1.dylib" ]]; then
            export DYLD_LIBRARY_PATH="$libdir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
            break
        fi
    done
fi

# ---- build the commands -------------------------------------------------------
common=(-iwad "$iwad" -file "$wad" -windowed)
if [[ "$mode" == deathmatch ]]; then
    common+=(-deathmatch)
fi
common+=(+timelimit 0 +teamplay 0 +sv_noautoaim 1 +sv_nocrouch 1 +sv_nofreelook 1)
if [[ "$mode" == deathmatch ]]; then
    common+=(+sv_spawnfarthest 1)
fi
common+=(+sv_forcerespawn 0)
if ((!respawn)); then
    common+=(+sv_norespawn 1)
fi
common+=(+snd_sfxvolume 1 +snd_musicvolume 0)
if [[ -n "$map" ]]; then
    common+=(+map "$map")
fi

commands=()
command_for() {  # slot -> one shell-quoted command line
    local slot="$1" line arg
    local args=("$executable" "${common[@]}")
    if [[ "$audio_player" != "$slot" && "$audio_player" != all ]]; then
        args+=(-nosound)
    fi
    if ((players > 1)); then
        if ((slot == 0)); then args+=(-host "$players" -port "$port"); else args+=(-join "127.0.0.1:$port"); fi
    fi
    args+=(+name "${name_list[$slot]}" +color "${color_list[$slot]}")
    if ((${#extra[@]})); then args+=("${extra[@]}"); fi
    line=""
    for arg in "${args[@]}"; do line+="$(printf '%q' "$arg") "; done
    printf '%s' "${line% }"
}
for ((slot = 0; slot < players; slot++)); do
    commands+=("$(command_for "$slot")")
done

if ((dry_run)); then
    [[ "$OSTYPE" == darwin* && -n "${DYLD_LIBRARY_PATH:-}" ]] && printf '# DYLD_LIBRARY_PATH=%s\n' "$DYLD_LIBRARY_PATH"
    for ((slot = 0; slot < players; slot++)); do
        role=join
        ((slot == 0)) && role=host
        ((players == 1)) && role=single
        printf '# %s %s (%s)\n%s\n' "$role" "$slot" "${name_list[$slot]}" "${commands[$slot]}"
    done
    exit 0
fi

# ---- launch -------------------------------------------------------------------
pids=()
cleanup() {
    trap - EXIT INT TERM
    if ((${#pids[@]})); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((slot = 0; slot < players; slot++)); do
    eval "${commands[$slot]}" &
    pids+=("$!")
    sleep 0.5
done
printf '%d instance(s) of %s, mode %s, audio listener: %s. Ctrl+C closes only these instances.\n' \
    "$players" "$(basename "$wad")" "$mode" "$audio_player"
printf 'To restart the map, enter "changemap %s" in the host console.\n' "${map:-map01}"
wait "${pids[0]}"
