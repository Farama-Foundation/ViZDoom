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
--game:load_config("../../examples/config/deadly_corridor.cfg")
game:load_config("../../examples/config/deathmatch.cfg")
--game:load_config("../../examples/config/defend_the_center.cfg")
--game:load_config("../../examples/config/defend_the_line.cfg")
--game:load_config("../../examples/config/health_gathering.cfg")
--game:load_config("../../examples/config/my_way_home.cfg")
--game:load_config("../../examples/config/predict_position.cfg")
--game:load_config("../../examples/config/take_cover.cfg")

-- Enables freelook in engine
game:add_game_args("+freelook 1")

game:set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)

-- Enables spectator mode, so you can play. Sounds strange but it is agent who is supposed to watch not you.
game:set_window_visible(true)
game:set_mode(vizdoom.Mode.SPECTATOR)

game:init()

episodes = 10

for i = 1, episodes do

    print("Episode #" .. i)

    game:new_episode()

    while not game:is_episode_finished() do

        s = game:get_state()

        game:advance_action()
        -- a = game:get_last_action()
        r = game:get_last_reward()

        print("State # " .. s.number)
        print("Reward: " .. r)
        print("=====================")

    end

    print("Episode finished.")
    print("total reward: " .. game:get_total_reward())
    print("************************")

end

game:close()
