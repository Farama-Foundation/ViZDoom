# Utilities
Here we document the helpers functions that are not part of any object.
They mostly help to deal with conversion of Doom's engine types.
The declarations of all the enums can be found in the `include/ViZDoomUtils.h` header file.


## Time conversion functions

```{eval-rst}
.. autofunction:: vizdoom.doom_tics_to_ms
.. autofunction:: vizdoom.ms_to_doom_tics
.. autofunction:: vizdoom.doom_tics_to_sec
.. autofunction:: vizdoom.sec_to_doom_tics
```

## Doom fixed point conversion functions

```{eval-rst}
.. autofunction:: vizdoom.doom_fixed_to_float
```

## Button functions

```{eval-rst}
.. autofunction:: vizdoom.is_binary_button
.. autofunction:: vizdoom.is_delta_button
```

## Category functions

```{eval-rst}
.. autofunction:: vizdoom.get_default_categories
```
