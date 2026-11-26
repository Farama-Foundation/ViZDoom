"""
Tests for the Doom translator module.
"""

import pytest

from doomgen.doom.wad import DoomMapData
from doomgen.doom.things import ThingType, get_thing_radius, EASY_ENEMIES
from doomgen.doom.tags import TagRegistry


class TestDoomMapData:
    """Tests for DoomMapData class."""

    def test_add_vertex(self):
        """Test adding vertices."""
        map_data = DoomMapData("MAP01")

        idx1 = map_data.add_vertex(100, 200)
        idx2 = map_data.add_vertex(300, 400)

        assert idx1 == 0
        assert idx2 == 1
        assert len(map_data.vertices) == 2

    def test_vertex_deduplication(self):
        """Test that duplicate vertices are reused."""
        map_data = DoomMapData("MAP01")

        idx1 = map_data.add_vertex(100, 200)
        idx2 = map_data.add_vertex(100, 200)  # Same coordinates

        assert idx1 == idx2
        assert len(map_data.vertices) == 1

    def test_add_sector(self):
        """Test adding a sector."""
        map_data = DoomMapData("MAP01")

        idx = map_data.add_sector(
            floor_height=0,
            ceiling_height=128,
            floor_texture='FLOOR4_8',
            light_level=160
        )

        assert idx == 0
        assert len(map_data.sectors) == 1
        assert map_data.sectors[0]['floor_height'] == 0

    def test_add_thing(self):
        """Test adding a thing."""
        map_data = DoomMapData("MAP01")

        idx = map_data.add_thing(
            x=256,
            y=512,
            thing_type=ThingType.PLAYER1_START,
            angle=90
        )

        assert idx == 0
        assert len(map_data.things) == 1
        assert map_data.things[0]['type'] == ThingType.PLAYER1_START



class TestThingType:
    """Tests for ThingType enum."""

    def test_player_starts(self):
        """Test player start thing types."""
        assert ThingType.PLAYER1_START.value == 1
        assert ThingType.PLAYER2_START.value == 2
        assert ThingType.PLAYER3_START.value == 3
        assert ThingType.PLAYER4_START.value == 4

    def test_enemies(self):
        """Test enemy thing types."""
        assert ThingType.ZOMBIEMAN.value == 3004
        assert ThingType.IMP.value == 3001
        assert ThingType.CYBERDEMON.value == 16

    def test_get_thing_radius(self):
        """Test thing radius lookup."""
        assert get_thing_radius(ThingType.PLAYER1_START) == 16
        assert get_thing_radius(ThingType.CYBERDEMON) == 48
        assert get_thing_radius(ThingType.CACODEMON) == 31

    def test_enemy_groups(self):
        """Test enemy groupings."""
        assert ThingType.IMP in EASY_ENEMIES
        assert ThingType.ZOMBIEMAN in EASY_ENEMIES


class TestTagRegistry:
    """Tests for TagRegistry class."""

    def test_allocate(self):
        """Test tag allocation."""
        registry = TagRegistry()

        tag1 = registry.allocate()
        tag2 = registry.allocate()

        assert tag1 == 1
        assert tag2 == 2

    def test_allocate_named(self):
        """Test named tag allocation."""
        registry = TagRegistry()

        tag = registry.allocate(name="door1")

        assert tag == 1
        assert registry.get("door1") == 1

    def test_reserve(self):
        """Test tag reservation."""
        registry = TagRegistry()

        registry.reserve(10, name="special")

        assert registry.is_used(10)
        assert registry.get("special") == 10

    def test_reserve_conflict(self):
        """Test that reserving used tag raises error."""
        registry = TagRegistry()

        registry.allocate()  # Uses tag 1

        with pytest.raises(ValueError):
            registry.reserve(1)

    def test_get_or_allocate(self):
        """Test get or allocate."""
        registry = TagRegistry()

        tag1 = registry.get_or_allocate("door")
        tag2 = registry.get_or_allocate("door")  # Same name

        assert tag1 == tag2

    def test_reset(self):
        """Test registry reset."""
        registry = TagRegistry()

        registry.allocate()
        registry.allocate()
        registry.reset()

        tag = registry.allocate()
        assert tag == 1


class TestACSCompilation:
    """Tests for ACS compilation with bundled compiler."""

    def test_acs_compile_with_bundled_compiler(self, tmp_path):
        """Test that ACS compilation uses bundled compiler by default."""
        from doomgen.logic.acs_builder import ACSBuilder, ScriptType
        from unittest.mock import patch, MagicMock

        builder = ACSBuilder()
        builder.add_script(
            ScriptType.OPEN,
            'Print(s:"Hello World");'
        )

        # Mock subprocess.run to verify the correct path is used
        with patch('doomgen.logic.acs_builder.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Create a fake output file so the compile method thinks it succeeded
            output_file = tmp_path / "script.o"
            output_file.write_bytes(b"fake bytecode")

            # Call compile with no acc_path parameter
            result = builder.compile(output_path=str(output_file), acc_path=None)

            # Verify subprocess.run was called
            assert mock_run.called
            # The first argument should be a list where the first element is a path containing "acc"
            call_args = mock_run.call_args[0][0]
            assert "acc" in call_args[0]

    def test_acs_compile_missing_executable(self, tmp_path):
        """Test that FileNotFoundError is raised when ACC compiler is missing."""
        from doomgen.logic.acs_builder import ACSBuilder, ScriptType

        builder = ACSBuilder()
        builder.add_script(
            ScriptType.OPEN,
            'Print(s:"Hello World");'
        )

        # Try to compile with a nonexistent path
        with pytest.raises(FileNotFoundError):
            builder.compile(
                output_path=str(tmp_path / "output.o"),
                acc_path="/nonexistent/path/to/acc"
            )

    def test_acs_compile_respects_explicit_path(self, tmp_path):
        """Test that explicit acc_path parameter is respected."""
        from doomgen.logic.acs_builder import ACSBuilder, ScriptType
        from unittest.mock import patch, MagicMock

        builder = ACSBuilder()
        builder.add_script(
            ScriptType.OPEN,
            'Print(s:"Hello World");'
        )

        custom_path = "/custom/path/to/acc"

        with patch('doomgen.logic.acs_builder.subprocess.run') as mock_run, \
             patch('doomgen.logic.acs_builder.validate_executable') as mock_validate:
            mock_run.return_value = MagicMock(returncode=0)
            mock_validate.return_value = True  # Pretend the custom path exists

            output_file = tmp_path / "script.o"
            output_file.write_bytes(b"fake bytecode")

            # Call compile with explicit path
            result = builder.compile(
                output_path=str(output_file),
                acc_path=custom_path
            )

            # Verify the custom path was used
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == custom_path


class TestProceduralMapBuilderZDBSP:
    """Tests for ZDBSP integration in ProceduralMapBuilder.build()."""

    def test_build_uses_bundled_zdbsp(self, tmp_path):
        """Test that build() uses bundled ZDBSP by default."""
        from doomgen.builder import ProceduralMapBuilder
        from unittest.mock import patch, MagicMock

        # Create a minimal procedural map
        builder = ProceduralMapBuilder(
            bounds=(-512, -512, 512, 512),
            seed=42
        )
        builder.add_area("room1", (-256, -256, 256, 256))

        output_wad = tmp_path / "test.wad"

        # Mock run_zdbsp to verify the correct path is used
        with patch('doomgen.builder.run_zdbsp') as mock_zdbsp:
            mock_zdbsp.return_value = True

            builder.build(str(output_wad))

            # Verify run_zdbsp was called
            assert mock_zdbsp.called
            # The zdbsp_path argument should contain "zdbsp"
            call_kwargs = mock_zdbsp.call_args[1]
            assert "zdbsp" in call_kwargs["zdbsp_path"]

    def test_build_warns_when_zdbsp_missing(self, tmp_path, capsys):
        """Test that build() issues a warning when ZDBSP is not found."""
        from doomgen.builder import ProceduralMapBuilder
        from unittest.mock import patch, MagicMock

        builder = ProceduralMapBuilder(
            bounds=(-512, -512, 512, 512),
            seed=42
        )
        builder.add_area("room1", (-256, -256, 256, 256))

        output_wad = tmp_path / "test.wad"

        # Mock validate_executable to return False (tool not found)
        with patch('doomgen.builder.validate_executable') as mock_validate:
            mock_validate.return_value = False

            # Build should not raise an error
            builder.build(str(output_wad))

            # Check that a warning was printed
            captured = capsys.readouterr()
            assert "ZDBSP not found" in captured.out or "Warning" in captured.out

    def test_build_respects_explicit_zdbsp_path(self, tmp_path):
        """Test that explicit zdbsp_path parameter is respected."""
        from doomgen.builder import ProceduralMapBuilder
        from unittest.mock import patch, MagicMock

        builder = ProceduralMapBuilder(
            bounds=(-512, -512, 512, 512),
            seed=42
        )
        builder.add_area("room1", (-256, -256, 256, 256))

        output_wad = tmp_path / "test.wad"
        custom_zdbsp_path = "/custom/path/to/zdbsp"

        with patch('doomgen.builder.run_zdbsp') as mock_zdbsp, \
             patch('doomgen.builder.validate_executable') as mock_validate:
            mock_zdbsp.return_value = True
            mock_validate.return_value = True  # Pretend the custom path exists

            builder.build(str(output_wad), zdbsp_path=custom_zdbsp_path)

            # Verify the custom path was used
            call_kwargs = mock_zdbsp.call_args[1]
            assert call_kwargs["zdbsp_path"] == custom_zdbsp_path
