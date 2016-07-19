#!/usr/bin/env lua

----------------------------------------------------------------------
-- This script presents how to use the most basic features of the environment.
-- It configures the engine, and makes the agent perform random actions.
-- It also gets current state and reward earned with the action.
-- <episodes> number of episodes are played.
-- Random combination of buttons is chosen for every action.
-- Game variables from state and last reward are printed.
--
-- To see the scenario description go to "../../scenarios/README.md"
----------------------------------------------------------------------

require("vizdoom")

-- Create DoomGame instance. It will run the game and communicate with you.
game = vizdoom.DoomGame()

-- Now it's time for configuration!
-- load_config could be used to load configuration instead of doing it here with code.
-- If load_config is used in-code configuration will work. Note that the most recent changes will add to previous ones.
-- game.load_config("../../examples/config/basic.cfg")

-- Sets path to vizdoom engine executive which will be spawned as a separate process. Default is "./vizdoom".
game:set_vizdoom_path("../../bin/vizdoom")

-- Sets path to doom2 iwad resource file which contains the actual doom game. Default is "./doom2.wad".
game:set_doom_game_path("../../scenarios/freedoom2.wad")
--game.set_doom_game_path("../../scenarios/doom2.wad") -- Not provided with environment due to licences.

-- Sets path to additional resources wad file which is basically your scenario wad.
-- If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
game:set_doom_scenario_path("../../scenarios/basic.wad")

-- Sets map to start (scenario .wad files can contain many maps).
game:set_doom_map("map01")

-- Sets resolution. Default is 320X240
game:set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)

-- Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
game:set_screen_format(vizdoom.ScreenFormat.RGB24)

-- Enables depth buffer.
game:set_depth_buffer_enabled(true)

-- Enables labeling of in game objects labeling.
game:set_labels_buffer_enabled(true)

-- Enables buffer with top down map of he current episode/level .
game:set_map_buffer_enabled(true)

-- Sets other rendering options
game:set_render_hud(false)
game:set_render_crosshair(false)
game:set_render_weapon(true)
game:set_render_decals(false)
game:set_render_particles(false)

-- Adds buttons that will be allowed.
game:add_available_button(vizdoom.Button.MOVE_LEFT)
game:add_available_button(vizdoom.Button.MOVE_RIGHT)
game:add_available_button(vizdoom.Button.ATTACK)

-- Adds game variables that will be included in state.
game:add_available_game_variable(vizdoom.GameVariable.AMMO2)

-- Causes episodes to finish after 200 tics (actions)
game:set_episode_timeout(200)

-- Makes episodes start after 10 tics (~after raising the weapon)
game:set_episode_start_time(10)

-- Makes the window appear (turned on by default)
game:set_window_visible(true)

-- Turns on the sound. (turned off by default)
game:set_sound_enabled(true)

-- Sets the livin reward (for each move) to -1
game:set_living_reward(-1)

-- Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
game:set_mode(vizdoom.Mode.PLAYER)

-- Initialize the game. Further configuration won't take any effect from now on.
game:init()

-- Define some actions. Each list entry corresponds to declared buttons:
-- MOVE_LEFT, MOVE_RIGHT, ATTACK
-- 5 more combinations are naturally possible but only 3 are included for transparency when watching.
actions = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}

-- Run this many episodes
episodes = 10

-- Sets time that will pause the engine after each action.
-- Without this everything would go too fast for you to keep track of what's happening.
sleep_time = 28

for i = 1, episodes do

    print("Episode #" .. i)

    -- Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game:new_episode()
    while not game:is_episode_finished() do

        -- Gets the state
        state = game:get_state()

        -- Which consists of:
        n           = state.number
        vars        = state.game_variables
        screen_buf  = state.screen_buffer
        depth_buf   = state.depth_buffer
        labels_buf  = state.labels_buffer
        map_buf     = state.map_buffer
        labels      = state.labels

        --Makes a random action and get remember reward.
        reward = game:make_action(actions[math.random(1,3)])

        --Makes a "prolonged" action and skip frames:
        --skiprate = 4
        --reward = game.make_action(choice(actions), skiprate)

        --The same could be achieved with:
        --game.set_action(choice(actions))
        --game.advance_action(skiprate)
        --reward = game.get_last_reward()

        -- Prints state's reward.
        print("State # " .. n)
        print("Game variables: " .. vars[1])
        print("Reward: " .. reward)
        print("=====================")

        if sleep_time > 0 then
            vizdoom.sleep(sleep_time)
        end
    end

    -- Check how the episode went.
    print("Episode finished.")
    print("total reward: " .. game:get_total_reward())
    print("************************")

end

game:close()
