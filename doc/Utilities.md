# Utilities

In Java utilities functions are static methods in DoomGame class.

* [doomTicsToMs](#doomTicsToMs)
* [msToDoomTics](#msToDoomTics)
* [doomTicsToSec](#doomTicsToSec)
* [secToDoomTics](#secToDoomTics)
* [doomFixedToDouble](#doomFixedToDouble)
* [isBinaryButton](#isBinaryButton)
* [isDeltaButton](#isDeltaButton)


---
### <a name="doomTicsToMs"></a> `doomTicsToMs`

| C++    | `double doomTicsToMs(double tics, unsigned int ticrate = 35)` |
| :--    | :--                                                           |
| Python | `float doom_tics_to_ms(float tics, int ticrate = 35)`         |

Changed in 1.1

Calculates how many tics will be made during given number of milliseconds.


---
### <a name="msToDoomTics"></a>`msToDoomTics`

| C++    | `double msToDoomTics(double ms, unsigned int ticrate = 35)` |
| :--    | :--                                                         |
| Python | `float ms_to_doom_tics(float ms, int ticrate = 35)`         |

Changed in 1.1

Calculates the number of milliseconds that will pass during specified number of tics.


---
### <a name="doomTicsToSec"></a>`doomTicsToSec`

| C++    | `double doomTicsToSec(double tics, unsigned int ticrate = 35)` |
| :--    | :--                                                            |
| Python | `float doom_tics_to_sec(float tics, int ticrate = 35)`         |

Added in 1.1

Calculates how many tics will be made during given number of seconds.


---
### <a name="secToDoomTics"></a>`secToDoomTics`

| C++    | `double secToDoomTics(double sec, unsigned int ticrate = 35)` |
| :--    | :--                                                           |
| Python | `float sec_to_doom_tics(float sec, int ticrate = 35)`         |

Added in 1.1

Calculates the number of seconds that will pass during specified number of tics.


---
### <a name="doomFixedToDouble"></a>`doomFixedToDouble`

| C++    | `double doomFixedToDouble(int / double doomFixed)`  |
| :--    | :--                                                 |
| Python | `float doom_fixed_to_double(int / float doomFixed)` |

Converts Doom's fixed point numeral to a floating point value.

See also: 
- Types: `User variables` in `GameVariables`

Python aliases (added in 1.1):
`float doom_fixed_to_float(int / float doomFixed)`


---
### <a name="isBinaryButton"></a>`isBinaryButton`

| C++    | `bool isBinaryButton(Button button)`    |
| :--    | :--                                     |
| Python | `bool is_binary_button(Button button)`  |

Returns true if button is binary button.


---
### <a name="isDeltaButton"></a>`isDeltaButton`

| C++    | `bool isDeltaButton(Button button)`    |
| :--    | :--                                    |
| Python | `bool is_delta_button(Button button)`  |

Returns true if button is delta button.

