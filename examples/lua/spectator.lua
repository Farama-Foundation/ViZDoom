#!/usr/bin/env lua

----------------------------------------------------------------------
-- This script presents SPECTATOR mode. In SPECTATOR mode you play and
-- your agent can learn from it.
-- Configuration is loaded from "../../examples/config/<SCENARIO_NAME>.cfg" file.
--
-- To see the scenario description go to "../../scenarios/README.md"
----------------------------------------------------------------------

package.path = package.path .. ";./vizdoom/?.lua"
require "vizdoom.init"

local game = vizdoom.DoomGame()

-- Choose scenario config file you wish to watch.
-- Don't load two configs cause the second will overrite the first one.
-- Multiple config files are ok but combining these ones doesn't make much sense.

--game:loadConfig("../../examples/config/basic.cfg")
--game:loadConfig("../../examples/config/deadly_corridor.cfg")
game:loadConfig("../../examples/config/deathmatch.cfg")
--game:loadConfig("../../examples/config/defend_the_center.cfg")
--game:loadConfig("../../examples/config/defend_the_line.cfg")
--game:loadConfig("../../examples/config/health_gathering.cfg")
--game:loadConfig("../../examples/config/my_way_home.cfg")
--game:loadConfig("../../examples/config/predict_position.cfg")
--game:loadConfig("../../examples/config/take_cover.cfg")

-- Enables freelook in engine
game:addGameArgs("+freelook 1")

game:setScreenResolution(vizdoom.ScreenResolution.RES_640X480)

-- Enables spectator mode, so you can play. Sounds strange but it is agent who is supposed to watch not you.
game:setWindowVisible(true)
game:setMode(vizdoom.Mode.SPECTATOR)

game:init()

local episodes = 10

for i = 1, episodes do

    print("Episode #" .. i)

    game:newEpisode()

    while not game:isEpisodeFinished() do

        local state = game:getState()

        game:advanceAction()
        local action = game:getLastAction()
        local reward = game:getLastReward()

        print("State # " .. state.number)
        print("Reward: " .. reward)
        local actionSize = game:getAvailableButtonsSize()
        local actionStr = "Action:"
        for k = 1, actionSize do actionStr = actionStr .. " " .. action[k] end
        print(actionStr)
        print("=====================")

    end

    print("Episode finished.")
    print("total reward: " .. game:getTotalReward())
    print("************************")

end

game:close()
