# GameState

```{eval-rst}
.. autoclass:: vizdoom.GameState
   :members:
   :undoc-members:
```

**number** - number of the state in the episode.

**tic** - ingame time, 1 tic is 1/35 of second in the game world. Note: added in 1.1.1.

See also:
- [`DoomGame.get_state`](./doomGame.md#vizdoom.DoomGame.get_state),
- [examples/python/basic.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/basic.py),
- [examples/python/buffers.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py).
- [examples/python/audio_buffer.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py).


## Data types used in GameState

```{eval-rst}
.. autoclass:: vizdoom.Label
   :members:
   :undoc-members:
```

**object_id** - unique object ID, if both Labels and Objects information is enabled, this will be the same as **id** in corresponding **Object**.

**object_name** - ingame object name, many different objects can have the same name (e.g. Medikit, Clip, Zombie).

**value** - value that represents this particular object in **labels_buffer**.

**x**, **y**, **width**, **height** - describes bounding box of this particular object in **labels_buffer**. Note: added in 1.1.5.


See also:
- [`DoomGame.set_labels_buffer_enabled`](./doomGame.md#vizdoom.DoomGame.set_labels_buffer_enabled),
- [examples/python/objects_and_sectors.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/labels_buffer.py).



```{eval-rst}
.. autoclass:: vizdoom.Object
   :members:
   :undoc-members:
```

**id** - unique object ID.

**name** - ingame object name, many different objects can have the same name (e.g. Medikit, Clip, Zombie).

See also:
- [`DoomGame.set_objects_info_enabled`](./doomGame.md#vizdoom.DoomGame.set_sectors_info_enabled),
- [examples/python/objects_and_sectors.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/objects_and_sectors.py).

Note: added in 1.1.8.


```{eval-rst}
.. autoclass:: vizdoom.Line
   :members:
   :undoc-members:
```

**x1**, **y1** - position of the line's first vertex.

**x2**, **y2** - position of the line's second vertex.

**is_blocking** - is true, if line is a wall that can't be passed.

See also:
- [examples/python/objects_and_sectors.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/objects_and_sectors.py).

Note: added in 1.1.8.


```{eval-rst}
.. autoclass:: vizdoom.Sector
   :members:
   :undoc-members:
```

**floor_height** - height of the sector's floor.

**ceiling_height** - height of the sector's ceiling.

**lines** - contains list of line segments, that forms sector.

See also:
- [`DoomGame.set_sectors_info_enabled`](./doomGame.md#vizdoom.DoomGame.set_sectors_info_enabled),
- [examples/python/objects_and_sectors.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/objects_and_sectors.py).

Note: added in 1.1.8.


```{eval-rst}
.. autoclass:: vizdoom.ServerState
   :members:
   :undoc-members:
```

ServerState is the main object returned by [`DoomGame.get_server_state`](./doomGame.md#vizdoom.DoomGame.get_server_state) method, and it purpose is to get more information about the state of the multi-player game.

See also:
- [`DoomGame.get_server_state`](./doomGame.md#vizdoom.DoomGame.get_server_state).

Note: added in 1.1.6.
