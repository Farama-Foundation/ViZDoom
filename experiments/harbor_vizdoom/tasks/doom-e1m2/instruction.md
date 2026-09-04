# Complete DOOM E1M2: Nuclear Plant

Reach the exit and complete the second level of DOOM, E1M2 “Nuclear Plant,”
on skill level 1. You must navigate, open doors or activate switches with
`USE`, and fight or evade enemies as needed. The episode succeeds only when
the level exit is reached.

Use the ViZDoom MCP tools to interact with the episode:

- Call `get_current_state` first. It returns episode metadata followed by two
  labeled images: the first-person screen and the automap.
- Call `make_action` with a JSON list named `buttons`. You may press multiple
  buttons together; an empty list waits. Every call advances four frames and
  returns the resulting metadata, screen, and automap.
- There is no automatic history lookup. Before leaving a state you may call
  `store_state` with your own `memory_id`. Retrieve that exact stored state
  later with `get_stored_state`. Reusing an ID replaces its earlier state.

Use only names listed in `available_buttons`. Movement, turning, firing,
running, strafing, using doors/switches, and weapon selection can be combined.
Continue until a tool result reports `done: true`. Death, the decision limit,
or a timeout is failure.

Do not use the internet or access the game by any route other than the
provided ViZDoom MCP tools. Built-in web search and web fetch are disabled.
