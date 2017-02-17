#!/usr/bin/env th

----------------------------------------------------------------------
-- This script presents how to run some scenarios.
-- Configuration is loaded from "../../scenarios/<SCENARIO_NAME>.cfg" file.
-- <episodes> number of episodes are played.
-- Random combination of buttons is chosen for every action.
-- Game variables from state and last reward are printed.
--
-- To see the scenario description go to "../../scenarios/README.md"
----------------------------------------------------------------------

require "vizdoom"

local game = vizdoom.DoomGame()

-- Choose scenario config file you wish to watch.
-- Don't load two configs cause the second will overrite the first one.
-- Multiple config files are ok but combining these ones doesn't make much sense.

game:loadConfig("../../scenarios/basic.cfg")
--game:loadConfig("../../scenarios/deadly_corridor.cfg")
--game:loadConfig("../../scenarios/deathmatch.cfg")
--game:loadConfig("../../scenarios/defend_the_center.cfg")
--game:loadConfig("../../scenarios/defend_the_line.cfg")
--game:loadConfig("../../scenarios/health_gathering.cfg")
--game:loadConfig("../../scenarios/my_way_home.cfg")
--game:loadConfig("../../scenarios/predict_position.cfg")
--game:loadConfig("../../scenarios/take_cover.cfg")

-- Makes the screen bigger to see more details.
game:setScreenResolution(vizdoom.ScreenResolution.RES_640X480)
game:init()

-- Creates all possible actions depending on how many buttons there are.
local actionsNum = game:getAvailableButtonsSize()
local actions = {}
for i = 1, actionsNum do
    actions[i] = {}
    for j = 1, actionsNum do
        actions[i][j] = 0
        if i == j then actions[i][j] = 1 end
    end
end


local episodes = 10
local sleepTime = 0.028

for i = 1, episodes do

    print("Episode #" .. i)

    -- Not needed for the first episode but the loop is nicer.
    game:newEpisode()
    while not game:isEpisodeFinished() do

        -- Gets the state and possibly to something with it
        local state = game:getState()
        local vars = state.gameVariables

        -- Makes a random action and save the reward.
        local reward = game:makeAction(actions[math.random(1, actions_num)])

        print("State # " .. state.number)
        local varsStr = "Game variables:"
        for k, a in pairs(vars) do varsStr = varsStr .. " " .. a end
        print(varsStr)
        print("Reward: " .. reward)
        print("=====================")

        -- Sleep some time because processing is too fast to watch.
        if sleepTime > 0 then
            sys.sleep(sleepTime)
        end
    end

    print("Episode finished.")
    print("total reward: " .. game:getTotalReward())
    print("************************")

end

game:close()

