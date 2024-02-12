# Utilities

Here we document the helpers functions that are not part of any object.
They mostly help to deal with conversion of Doom's engine types.
The declarations of all the enums can be found in the `include/ViZDoomUtils.h` header file.


## Time conversion functions

### `doomTicsToMs`

| C++    | `double doomTicsToMs(double tics, unsigned int ticrate = 35)` |
| :--    | :--                                                           |
| Python | `doom_tics_to_ms(tics: float, ticrate: int = 35) -> float`    |

Calculates how many tics will be made during given number of milliseconds.

Note: changed in 1.1.0


---
### `msToDoomTics`

| C++    | `double msToDoomTics(double ms, unsigned int ticrate = 35)` |
| :--    | :--                                                         |
| Python | `ms_to_doom_tics(ms: float, ticrate: int = 35) -> float`    |

Calculates the number of milliseconds that will pass during specified number of tics.

Note: changed in 1.1.0


---
### `doomTicsToSec`

| C++    | `double doomTicsToSec(double tics, unsigned int ticrate = 35)` |
| :--    | :--                                                            |
| Python | `doom_tics_to_sec(tics: float, ticrate: int = 35) -> float`    |

Calculates how many tics will be made during given number of seconds.

Note: added in 1.1.0


---
### `secToDoomTics`

| C++    | `double secToDoomTics(double sec, unsigned int ticrate = 35)` |
| :--    | :--                                                           |
| Python | `sec_to_doom_tics(sec: float, ticrate: int = 35) -> float`    |

Calculates the number of seconds that will pass during specified number of tics.

Note: added in 1.1.0


## Doom fixed point conversion functions

### `doomFixedToDouble`

| C++    | `double doomFixedToDouble(int / double doomFixed)`      |
| :--    | :--                                                     |
| Python | `doom_fixed_to_double(doom_fixed: int | float) -> float` |

Converts fixed point numeral to a floating point value.
Doom engine internally use fixed point numbers.
If you assign fixed point numeral to `USER1` - `USER60` GameVariables,
you can convert them to floating point by using this function.

Python alias (added in 1.1.0): `doom_fixed_to_float(doomFixed: int | float) -> float`


## Button functions

### `isBinaryButton`

| C++    | `bool isBinaryButton(Button button)`     |
| :--    | :--                                      |
| Python | `is_binary_button(button: Button): bool` |

Returns true if [`Button`](./enums.md#button) is binary button.


---
### `isDeltaButton`

| C++    | `bool isDeltaButton(Button button)`       |
| :--    | :--                                       |
| Python | `is_delta_button(button: Button) -> bool` |

Returns true if [`Button`](./enums.md#button) is delta button.
