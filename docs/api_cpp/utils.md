# Utilities


## `doomTicsToMs`

| C++    | `double doomTicsToMs(double tics, unsigned int ticrate = 35)` |
| :--    | :--                                                           |
| Python | `doom_tics_to_ms(tics: float, ticrate: int = 35) -> float`    |

Changed in 1.1

Calculates how many tics will be made during given number of milliseconds.


---
## `msToDoomTics`

| C++    | `double msToDoomTics(double ms, unsigned int ticrate = 35)` |
| :--    | :--                                                         |
| Python | `ms_to_doom_tics(ms: float, ticrate: int = 35) -> float`    |

Changed in 1.1

Calculates the number of milliseconds that will pass during specified number of tics.


---
## `doomTicsToSec`

| C++    | `double doomTicsToSec(double tics, unsigned int ticrate = 35)` |
| :--    | :--                                                            |
| Python | `doom_tics_to_sec(tics: float, ticrate: int = 35) -> float`    |

Added in 1.1

Calculates how many tics will be made during given number of seconds.


---
## `secToDoomTics`

| C++    | `double secToDoomTics(double sec, unsigned int ticrate = 35)` |
| :--    | :--                                                           |
| Python | `sec_to_doom_tics(sec: float, ticrate: int = 35) -> float`    |

Added in 1.1

Calculates the number of seconds that will pass during specified number of tics.


---
## `doomFixedToDouble`

| C++    | `double doomFixedToDouble(int / double doomFixed)`      |
| :--    | :--                                                     |
| Python | `doom_fixed_to_double(doomFixed: int | float) -> float` |

Converts Doom's fixed point numeral to a floating point value.

See also:
- [`Enums: User variables` in `GameVariables`](./enums.md#user-acs-variables)

Python aliases (added in 1.1):
`doom_fixed_to_float(doomFixed: int | float) -> float`


---
## `isBinaryButton`

| C++    | `bool isBinaryButton(Button button)`     |
| :--    | :--                                      |
| Python | `is_binary_button(button: Button): bool` |

Returns true if button is binary button.

See also:
- [`Enums: Button`](./enums.md#button)


---
## `isDeltaButton`

| C++    | `bool isDeltaButton(Button button)`       |
| :--    | :--                                       |
| Python | `is_delta_button(button: Button) -> bool` |

Returns true if button is delta button.

See also:
- [`Enums: Button`](./enums.md#button)
