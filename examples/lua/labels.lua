#!/usr/bin/env lua

----------------------------------------------------------------------
-- This script presents SPECTATOR mode. In SPECTATOR mode you play and
-- your agent can learn from it.
-- Configuration is loaded from "../../examples/config/<SCENARIO_NAME>.cfg" file.
--
-- To see the scenario description go to "../../scenarios/README.md"
----------------------------------------------------------------------

require("vizdoom")

game = vizdoom.DoomGame()

-- Choose scenario config file you wish to watch.
-- Don't load two configs cause the second will overrite the first one.
-- Multiple config files are ok but combining these ones doesn't make much sense.

--game:load_config("../../examples/config/basic.cfg")
game:load_config("../../examples/config/deadly_corridor.cfg")
--game:load_config("../../examples/config/deathmatch.cfg")
--game:load_config("../../examples/config/defend_the_center.cfg")
--game:load_config("../../examples/config/defend_the_line.cfg")
--game:load_config("../../examples/config/health_gathering.cfg")
--game:load_config("../../examples/config/my_way_home.cfg")
--game:load_config("../../examples/config/predict_position.cfg")
--game:load_config("../../examples/config/take_cover.cfg")

-- Enables freelook in engine
game:add_game_args("+freelook 1")

game:set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
game:set_window_visible(true)

-- Enables labeling of the in game objects.
game:set_labels_buffer_enabled(true)

game:init()

episodes = 10

for i = 1, episodes do

    print("Episode #" .. i)

    game:new_episode()

    while not game:is_episode_finished() do

        -- Gets the state.
        state = game:get_state()

        -- Get labels buffer and labels data.
        labels_buf = state.labels_buffer
        labels = state.labels

        game:advance_action()
        reward = game:get_last_reward()

        print("State #" .. state.number)

        print("Labels:")

        -- Print information about objects visible on the screen.
        -- object_id identifies specific in game object.
        -- object_name contains name of object.
        -- value tells which value represents object in labels_buffer.
        for k, l in pairs(labels) do
            print("#" .. k .. ": object id: " .. l.object_id .. " object name: " .. l.object_name .. " label: " .. l.value)
        end

        print("Reward: " .. reward)
        print("=====================")

    end

    print("Episode finished.")
    print("total reward: " .. game:get_total_reward())
    print("************************")

end

game:close()
