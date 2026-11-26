import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from doomgen.builder import ProceduralMapBuilder
from doomgen.logic.acs_builder import ACSBuilder, ScriptType


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
    "text.usetex": False,
})


THEME_BLUE = "#3498db"  # Standard Blue
THEME_NAVY = "#2c3e50"  # Navy


EXAMPLES_DIR = Path(__file__).resolve().parent
PLAY_DIR     = EXAMPLES_DIR / "play"
OUTPUT_DIR   = EXAMPLES_DIR / "output"

VIZDOOM_BIN  = PLAY_DIR / ("vizdoom.exe" if sys.platform == "win32" else "vizdoom")
IWAD         = PLAY_DIR / "freedoom2.wad"
OUTPUT_WAD   = OUTPUT_DIR / "test_acs_random.wad"
VIZDOOM_LOG  = PLAY_DIR / "vizdoom.log"

RNG_LO    = 0
RNG_HI    = 9
N_SAMPLES = 1000000
SPAN      = RNG_HI - RNG_LO + 1
TICS      = 35 * 5


def build_wad():
    acs = ACSBuilder()
    acs.add_include("zcommon.acs")

    script_body = f"""
    int i;
    int val;
    int counts[{SPAN}];

    for (i = 0; i < {N_SAMPLES}; i++) {{
        val = Random({RNG_LO}, {RNG_HI});
        counts[val - {RNG_LO}]++;
        if (i % 10000 == 0) Delay(1);
    }}

    Log(s:"ACS_RANDOM_TEST_START");
    for (i = 0; i < {SPAN}; i++) {{
        Log(d:{RNG_LO} + i, s:":", d:counts[i]);
    }}
    Log(s:"ACS_RANDOM_TEST_END");

    Delay(5);
    Exit_Normal(0);
    """
    acs.add_script(ScriptType.OPEN, script_body, number=1)

    builder = ProceduralMapBuilder(bounds=(-512, -512, 512, 512), seed=42)
    builder.add_area("room", shape=(0, 0, 512, 512))
    x, y = builder.get_random_point_in_area("room")
    builder.add_thing("PLAYER1_START", x, y, angle=90)

    builder.map_data.behavior = acs.compile()
    builder.map_data.scripts  = acs.to_code()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    builder.build(str(OUTPUT_WAD))


def run_vizdoom() -> str:
    if VIZDOOM_LOG.exists():
        VIZDOOM_LOG.unlink()

    cmd = [
        str(VIZDOOM_BIN),
        "-iwad", str(IWAD),
        "-file", str(OUTPUT_WAD),
        "+warp", "01",
        "-window",
        "-width", "320",
        "-height", "240",
        "-nosound",
        "-nomusic",
        "+sv_scriptlimit", "0",
        "+quickexit",
        "-nosfx",
        "+sv_cheats", "1",
        "+logfile", str(VIZDOOM_LOG),
    ]

    proc = subprocess.Popen(cmd, cwd=str(PLAY_DIR))
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    if VIZDOOM_LOG.exists():
        return VIZDOOM_LOG.read_text(errors="replace")
    return ""



def parse_results(log: str) -> dict[int, int]:
    counts = {}
    in_block = False
    for line in log.splitlines():
        if "ACS_RANDOM_TEST_START" in line:
            in_block = True
            continue
        if "ACS_RANDOM_TEST_END" in line:
            break
        if in_block:
            match = re.search(r"(\d+):(\d+)", line)
            if match:
                counts[int(match.group(1))] = int(match.group(2))
    return counts


def report(counts: dict[int, int]):
    total    = sum(counts.values())

    print(f"\nACS Random({RNG_LO}, {RNG_HI}) — {total} samples\n")
    print(f"{'Value':>6}  {'Count':>6}  {'%':>6}  {'Deviation':>10}")
    print("-" * 36)
    for v in range(RNG_LO, RNG_HI + 1):
        c   = counts.get(v, 0)
        pct = 100.0 * c / total if total else 0.0
        dev = pct - 100.0 / SPAN
        print(f"{v:>6}  {c:>6}  {pct:>5.1f}%  {dev:>+9.2f}%")
    print("-" * 36)

    max_dev = max(
        abs(100.0 * counts.get(v, 0) / total - 100.0 / SPAN)
        for v in range(RNG_LO, RNG_HI + 1)
    ) if total else 0.0
    print(f"\nMax deviation from uniform: {max_dev:.2f}%")
    if max_dev < 5.0:
        print("Distribution looks uniform.")
    else:
        print("Distribution is skewed — investigate.")


def plot_results(counts: dict[int, int]):
    total    = sum(counts.values())
    expected_freq = total / SPAN
    values   = np.arange(RNG_LO, RNG_HI + 1)
    observed = np.array([counts.get(v, 0) for v in values])

    ecdf = np.cumsum(observed) / total
    theoretical_cdf = (values - RNG_LO + 1) / SPAN

    d_stat = np.max(np.abs(ecdf - theoretical_cdf))

    chi_stat, p_val = stats.chisquare(observed, f_exp=np.full_like(observed, expected_freq))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"ACS Random({RNG_LO}, {RNG_HI}) Distribution Analysis — {total} samples", fontsize=13)

    # Left: Histogram with STD bands and Relative Deviation labels
    ax = axes[0]
    expected_freq = total / SPAN
    deviations = 100.0 * (observed - expected_freq) / expected_freq

    # Standard deviation for a binomial distribution for each bin
    # sigma = sqrt(n * p * (1-p))
    p = 1.0 / SPAN
    sigma = np.sqrt(total * p * (1 - p))

    ax.bar(values, observed, color=THEME_BLUE, alpha=0.8, edgecolor=THEME_NAVY, linewidth=0.6, label="Observed")
    ax.axhline(expected_freq, color=THEME_NAVY, linestyle="--", linewidth=1.2, label=f"Expected ({expected_freq:.0f})")

    # STD Bands
    ax.axhspan(expected_freq - sigma, expected_freq + sigma, color='#2ecc71', alpha=0.15, label="$\pm 1 \sigma$")
    ax.axhspan(expected_freq - 2*sigma, expected_freq + 2*sigma, color='#f39c12', alpha=0.10, label="$\pm 2 \sigma$")

    # Add relative deviation labels on top of bars
    for i, (v, count, dev) in enumerate(zip(values, observed, deviations)):
        ax.text(v, count + (0.1 * sigma), f"{dev:+.2f}%",
                ha='center', va='bottom', fontsize=7, color=THEME_NAVY, fontweight='bold')

    y_min = expected_freq - 4 * sigma
    y_max = max(observed.max(), expected_freq + 4 * sigma) + (sigma * 0.5)
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(values)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_title(f"Sample Counts ($\sigma={sigma:.1f}$, $\chi^2$ p={p_val:.4f})")
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)

    # Right: ECDF vs Theoretical CDF
    ax2 = axes[1]
    ax2.step(values, ecdf, where='post', label='Empirical (ECDF)', color=THEME_BLUE, linewidth=2)
    ax2.plot(values, theoretical_cdf, '--', label='Theoretical (CDF)', color=THEME_NAVY, alpha=0.8)
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title(f"ECDF vs. Theoretical CDF (Max Diff: {d_stat:.4f})")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    out_pdf = OUTPUT_DIR / "acs_random_analysis.pdf"
    plt.savefig(out_pdf, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_pdf}")
    plt.show()


def main():
    print("Building WAD...")
    build_wad()

    print("\nRunning VizDoom...")
    log = run_vizdoom()

    if not log:
        print(f"No log found at {VIZDOOM_LOG}.")
        return

    counts = parse_results(log)
    if not counts:
        print("ACS output block not found in log. Raw log tail:")
        print("\n".join(log.splitlines()[-30:]))
        return

    report(counts)
    plot_results(counts)


if __name__ == "__main__":
    main()
