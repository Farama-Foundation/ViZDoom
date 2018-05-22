# FAQ

This file contains a list of questions that ViZDoom users may ask at some point.
You can find more questions and answers by search in issues page with "question" tag
[is:issue label:question](https://github.com/mwydmuch/ViZDoom/issues?utf8=%E2%9C%93&q=is%3Aissue+label%3Aquestion)

Did not find answer for your question? Post an [issue](https://github.com/mwydmuch/ViZDoom/issues)


### How to use Windows binaries?

For Windows we are providing a compiled environment that can be download from [releases](https://github.com/mwydmuch/ViZDoom/releases) page.
To install it see [Building: Installation of Windows binaries](Building.md#windows_bin).


**Original issue and answer:**
https://github.com/mwydmuch/ViZDoom/issues/190


### How to use original Doom's assets?

We cannot provide original Doom's assets due to licensing issues.
ViZDoom uses [freedoom2.wad](https://freedoom.github.io) as default assets.
However, you can use original Doom's assets by placing doom.wad or doom2.wad.
You can just place it in you working directory,
specify path to it by using [`DoomGame: setDoomGamePath`](DoomGame.md#setDoomGamePath)
or place it in the same directory as vizdoom(.exe).


### How to create/modify scenarios?

You can create or modify existing scenarios using many available Doom map editors.
We recommend using one of these two editors:
- [SLADE3](http://slade.mancubus.net/) - great Doom map (scenario) editor for Linux, MacOS and Windows.
- [Doom Builder 2](http://www.doombuilder.com/) - another great Doom map editor for Windows.

You should select ZDoom as your Doom engine version and UDMF map format (Universal Doom Map Format),
that supports the widest range of features.

**Original issue and answer:**
https://github.com/mwydmuch/ViZDoom/issues/319


### How to stack frames?

ViZDoom does not automatically stacks frames for you.
You have to manually store the states from [`DoomGame: getState`](DoomGame.md#getState). and build up stacked states for your agent.

**Original issue and answer: (contains code an example)**
https://github.com/mwydmuch/ViZDoom/issues/296


### How to change keyboard binding for Spectator Mode?

When you launch an instance of vizdoom, it will create _vizdoom.ini in your working directory (if it does not exist yet).
This file contains all the additional engine settings, including key bindings, that you can edit freely.

You can also load .ini file from different location using [`DoomGame: setDoomConfigPath`](DoomGame.md#setDoomConfigPath).

**Original issue and answer:**
https://github.com/mwydmuch/ViZDoom/issues/253


### Is it possible to generate maze navigation scenario from a text file (like in DeepMind Lab)?

Try [NavDoom](https://github.com/agiantwhale/navdoom).

**Original issue and answer:**
https://github.com/mwydmuch/ViZDoom/issues/308


### How to control game speed in `ASYNC` modes?

See: [`DoomGame: setTicrate`](DoomGame.md#setTicrate) and [examples/python/ticrate.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/ticrate.py)

**Original issue and answer:**
https://github.com/mwydmuch/ViZDoom/issues/209


### How can to make an exact 90 degree turn in one action?

See: [examples/python/delta_buttons.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/delta_buttons.py)

**Original issue and answer:**
https://github.com/mwydmuch/ViZDoom/issues/279

See also:
- [`Types: Button`](Types.md#button)
- [examples/python/DeltaDuttons.cpp](https://github.com/mwydmuch/ViZDoom/tree/master/examples/cpp/DeltaDuttons.cpp)
- [examples/python/DeltaDuttons.java](https://github.com/mwydmuch/ViZDoom/tree/master/examples/java/DeltaDuttons.java)


### Agent does not fire after picking up ammo or weapon?

Some weapons have a `noautofire` flag for weapons to prevent using them by accident when picking up.
Agent needs to release `ATTACK` button and then press it again to start firing after picking one of those weapons or ammo for them.

**Original issue and answer:**
https://github.com/mwydmuch/ViZDoom/issues/289

See also:
- List of weapon flags: [https://zdoom.org/wiki/Weapon_flags](https://zdoom.org/wiki/Weapon_flags)
- List of Doom weapons: [https://zdoom.org/wiki/Classes:DoomWeapon](https://zdoom.org/wiki/Classes:DoomWeapon)


### How to pick up items (medikit, ammo, armour) when inventory is full?

CVARs implemented in ZDoom engine are very helpful in quickly modifying some aspects of the game.
`game.add_game_args("+sv_unlimited_pickup 1")` adding before init will allow picking up unlimited items.

**Original issue and answer:**
https://github.com/mwydmuch/ViZDoom/issues/187

See also:
- List of CVARs: [https://zdoom.org/wiki/CVARs:Configuration](https://zdoom.org/wiki/CVARs:Configuration)
