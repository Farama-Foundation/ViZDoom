"""MPE-style 4-player tag: three slow red hunters, one faster green prey, two rocks.

Demonstrates -deathmatch arenas, mode="overwrite" obstacles, DECORATE + MAPINFO,
name-bound roles (independent of network join order), colour translations,
CheckSight proximity capture, and latched USER50-52 outcomes reset in OPEN.
Pass --audio for the invisible-prey variant with SNDINFO, a synthetic WAV chirp
and positional PlaySound.
"""

from __future__ import annotations

import argparse
import io
import math
import struct
import wave
from pathlib import Path
from typing import Any

from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.things import ThingType
from doomgen.geometry.shapes import circle, rect
from doomgen.logic.acs_builder import ACSBuilder, ScriptType


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class SimpleTagScenario(Scenario):
    def get_default_config(self) -> dict[str, Any]:
        return {
            "seed": 42,
            "arena_size": 690,
            "hunter_speed": 0.75,
            "prey_speed": 1.0,
            "audio": False,
            "sound_interval_tics": 10,
            "acc_path": None,
        }

    def validate_config(self) -> None:
        cfg = self.config
        if not 500 <= int(cfg["arena_size"]) <= 1024:
            raise ValueError("arena_size must be between 500 and 1024")
        if not 0 < float(cfg["hunter_speed"]) < float(cfg["prey_speed"]):
            raise ValueError("speeds must satisfy 0 < hunter_speed < prey_speed")
        if int(cfg["sound_interval_tics"]) < 4:
            raise ValueError("sound_interval_tics must be >= 4")

    def build_acs(self) -> ACSBuilder:
        cfg = self.config
        audio = bool(cfg["audio"])
        acs = ACSBuilder()
        acs.add_include("zcommon.acs")
        acs.add_define("RESET_SETTLE_TICS", 35)
        acs.add_define("TAG_RADIUS", 48)
        acs.add_map_var("tag_ready", "int", 0)
        acs.add_map_var("tag_rotation", "int", 0)
        acs.add_map_var("tag_spawned", "int", 0)
        acs.add_global_code("int tag_players[4];")
        acs.add_global_var("tag_caught", 50, "int")
        acs.add_global_var("tag_catcher", 51, "int")
        acs.add_global_var("tag_elapsed", 52, "int")

        acs.add_script(
            ScriptType.OPEN,
            """
            tag_caught = 0;
            tag_catcher = 0;
            tag_elapsed = 0;
            int role;
            for (role = 0; role < 4; role++) tag_players[role] = -1;
            tag_rotation = Random(0, 3);
            CreateTranslation(1, 112:127=176:191);
            CreateTranslation(2, 112:127=112:127);
            tag_ready = 1;
            """,
            number=6,
        )

        invisible = (
            """
                SetActorProperty(0, APROP_RenderStyle, STYLE_None);
                SetActorProperty(0, APROP_Alpha, 0);
            """
            if audio
            else ""
        )
        setup = f"""
            int slot = PlayerNumber();
            int pn = -1;
            int role;
            ClearInventory();
            SetPlayerProperty(0, 1, PROP_TOTALLYFROZEN);
            while (!tag_ready) Delay(1);
            for (role = 0; role < 4; role++) {{
                if (StrCmp(StrParam(n:0), StrParam(s:"Player", d:role)) == 0)
                    pn = role;
            }}
            if (pn < 0) terminate;
            if (tag_players[pn] != -1 && tag_players[pn] != slot) terminate;
            if (tag_caught) terminate;
            tag_players[pn] = slot;
            tag_spawned &= ~(1 << pn);
            Thing_ChangeTID(1000 + pn, 0);
            Thing_ChangeTID(0, 1000 + pn);
            SetActorProperty(0, APROP_Invulnerable, 1);
            int dest = 2000 + ((pn + tag_rotation) % 4);
            while (!SetActorPosition(0, GetActorX(dest), GetActorY(dest), 0, FALSE))
                Delay(1);
            SetActorVelocity(0, 0, 0, 0, FALSE, FALSE);
            SetActorAngle(0, VectorAngle(-GetActorX(0), -GetActorY(0)));
            SetActorProperty(0, APROP_Health, 100);
            SetActorProperty(0, APROP_RenderStyle, STYLE_Normal);
            SetActorProperty(0, APROP_Alpha, 1.0);
            if (pn < 3) {{
                SetActorProperty(0, APROP_Speed, {float(cfg['hunter_speed'])});
                Thing_SetTranslation(0, 1);
                HudMessage(s:"HUNTER"; HUDMSG_PLAIN, 1, CR_RED, 0.5, 0.02, 0);
            }} else {{
                SetActorProperty(0, APROP_Speed, {float(cfg['prey_speed'])});
                Thing_SetTranslation(0, 2);
                {invisible}
                HudMessage(s:"PREY"; HUDMSG_PLAIN, 1, CR_GREEN, 0.5, 0.02, 0);
            }}
            tag_spawned |= 1 << pn;
            while (Timer() <= RESET_SETTLE_TICS) Delay(1);
            if (!tag_caught) SetPlayerProperty(0, 0, PROP_TOTALLYFROZEN);
        """
        acs.add_script(ScriptType.ENTER, setup, number=1)
        acs.add_script(ScriptType.RESPAWN, setup, number=2)

        sound = (
            f"""
                if ((tag_elapsed - 1) % {int(cfg['sound_interval_tics'])} == 0)
                    PlaySound(1003, "tag/prey", CHAN_BODY, 1.0, FALSE, ATTN_NORM);
            """
            if audio
            else ""
        )
        acs.add_script(
            ScriptType.OPEN,
            f"""
            int hunter, dx, dy, ready;
            while (Timer() <= RESET_SETTLE_TICS) Delay(1);
            while (!tag_caught) {{
                ready = tag_spawned == 15;
                for (hunter = 0; hunter < 4; hunter++) {{
                    if (tag_players[hunter] < 0 || !PlayerInGame(tag_players[hunter]) ||
                        GetActorProperty(1000 + hunter, APROP_Health) <= 0)
                        ready = 0;
                    Thing_SetTranslation(1000 + hunter, 1 + (hunter == 3));
                }}
                if (ready) {{
                    tag_elapsed++;
                    {sound}
                    for (hunter = 0; hunter < 3; hunter++) {{
                        dx = (GetActorX(1003) - GetActorX(1000 + hunter)) >> 16;
                        dy = (GetActorY(1003) - GetActorY(1000 + hunter)) >> 16;
                        if (dx * dx + dy * dy <= TAG_RADIUS * TAG_RADIUS &&
                            CheckSight(1000 + hunter, 1003, 0)) {{
                            tag_catcher = hunter + 1;
                            tag_caught = 1;
                            SetPlayerProperty(1, 1, PROP_TOTALLYFROZEN);
                            SetActorState(1003, "TagCaught");
                            break;
                        }}
                    }}
                }}
                Delay(1);
            }}
            """,
            number=5,
        )
        return acs

    @staticmethod
    def prey_sound() -> bytes:
        """Short mono chirp spatialized by the engine at the prey's position."""
        rate = 22050
        count = rate // 10
        samples = [
            int(
                22000
                * (math.sin(math.pi * i / count) ** 2)
                * math.sin(2 * math.pi * (650 * (i / rate) + 1800 * (i / rate) ** 2))
            )
            for i in range(count)
        ]
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(struct.pack(f"<{count}h", *samples))
        return output.getvalue()

    def generate(self, output_path: str) -> None:
        cfg = self.config
        size = int(cfg["arena_size"])
        half = size // 2
        builder = ProceduralMapBuilder(
            bounds=(-half - 96, -half - 96, half + 96, half + 96),
            seed=int(cfg["seed"]),
        )
        arena = builder.add_area(
            "Arena",
            rect(0, 0, size, size),
            ceiling_height=192,
            floor_texture="FLOOR0_3",
            ceiling_texture="F_SKY1",
            wall_texture="ROCK1",
            light_level=224,
        )
        for index, (x, y) in enumerate(((-140, 0), (140, 0))):
            rock = builder.add_area(
                f"Rock_{index}",
                circle(x, y, 56, segments=20),
                floor_height=64,
                ceiling_height=192,
                floor_texture="RROCK09",
                ceiling_texture="F_SKY1",
                wall_texture="ROCK1",
                light_level=224,
                tag=200 + index,
                mode="overwrite",
            )
            builder.connect_adjacent(arena, rock)

        spawn = half - 176
        for index, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, 1), (1, -1))):
            builder.add_thing("MAP_SPOT", sx * spawn, sy * spawn, tid=2000 + index)
            builder.add_thing(ThingType.DEATHMATCH_START, sx * (half - 64), sy * (half - 64))
        builder.add_thing(ThingType.PLAYER1_START, -(half - 64), -(half - 64))

        builder.wad_writer.add_lump(
            "DECORATE",
            b"""
            actor TagPlayer : DoomPlayer {
                +NOTELEFRAG
                +INVULNERABLE
                +NODAMAGE
                +NOPAIN
                Player.DisplayName "Tag player"
                States {
                TagCaught:
                    TNT1 A 0 A_ChangeFlag("NODAMAGE", FALSE)
                    TNT1 A 0 A_ChangeFlag("INVULNERABLE", FALSE)
                    TNT1 A 0 A_Die
                    Stop
                }
            }
            """,
        )
        builder.wad_writer.add_lump(
            "MAPINFO",
            b"""
            gameinfo { playerclasses = "TagPlayer" }
            map MAP01 "Simple Tag" {
                sky1 = "SKY1"
                nojump
                nocrouch
                nofreelook
            }
            """,
        )
        if cfg["audio"]:
            builder.wad_writer.add_lump(
                "SNDINFO", b"tag/prey DSTAG\n$rolloff tag/prey log 48 0.25\n"
            )
            builder.wad_writer.add_lump("DSTAG", self.prey_sound())

        acs = self.build_acs()
        builder.map_data.behavior = acs.compile(acc_path=cfg["acc_path"])
        builder.map_data.scripts = acs.to_code()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        builder.build(str(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", action="store_true", help="invisible prey with positional chirp")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acc", default=None)
    args = parser.parse_args()
    default_name = "simple_tag_audio.wad" if args.audio else "simple_tag.wad"
    output = args.output or (DEFAULT_OUTPUT_DIR / default_name)
    SimpleTagScenario({"seed": args.seed, "audio": args.audio, "acc_path": args.acc}).generate(str(output))


if __name__ == "__main__":
    main()
