# Play ViZDoom BASIC

Kill the red monster before the episode times out.

You are in a rectangular gray room. You start along one long wall, and one red
monster is placed somewhere along the opposite wall. One accurate hit kills
it. Shooting costs reward, and waiting also costs reward, so aim before firing.

Use the ViZDoom MCP tools to interact with the episode:

- Call `get_current_state` first. It returns the current screenshot, valid
  button names, and episode metadata.
- Call `make_action` with a JSON list named `buttons`. You may press multiple
  buttons together. An empty list waits. Each call returns the resulting
  screenshot and state.
- Before leaving a useful state, call `store_state` with your own `memory_id`.
  Retrieve that exact state later with `get_stored_state`. There is no tool for
  browsing arbitrary previous states.

The available buttons are `MOVE_LEFT`, `MOVE_RIGHT`, and `ATTACK`. Continue
acting until a tool result reports `done: true`. Success means the monster was
killed; a timeout is failure.

Do not use the internet or try to access the game by any route other than the
provided ViZDoom MCP tools. Built-in web search and web fetch are disabled.
