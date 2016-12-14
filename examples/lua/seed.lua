#!/usr/bin/env th

--------------------------------------------------------------------------------
-- Seed example
-- This script presents how to run deterministic episodes by setting
-- seed. After setting the seed every episode will look the same (if
-- agent will behave deterministicly of course).
-- Config is loaded from "../../scenarios/<SCENARIO_NAME>.cfg" file.
-- <episodes> number of episodes are played.
-- Random combination of buttons is chosen for every action.

-- Game variables from state and last reward are printed.
-- To see the scenario description go to "../../scenarios/README.md"
--------------------------------------------------------------------------------

require "vizdoom"
require "torch"

local game = vizdoom.DoomGame()

game:loadConfig("../../scenarios/basic.cfg")
-- game:loadConfig("../../scenarios/deadly_corridor.cfg")
-- game:loadConfig("../../scenarios/defend_the_center.cfg")
-- game:loadConfig("../../scenarios/defend_the_line.cfg")
-- game:loadConfig("../../scenarios/health_gathering.cfg")
-- game:loadConfig("../../scenarios/my_way_home.cfg")
-- game:loadConfig("../../scenarios/predict_position.cfg")

-- Lets make episode shorter and observe starting position of Cacodemon.
game:setEpisodeTimeout(50)
game:setScreenResolution(vizdoom.ScreenResolution.RES_640X480)

local seed = 666 -- Appropriate seed for Doom.
game:setSeed(seed) -- It can be changed after init.

game:init()

-- Three example sample actions
local actions = {
    [1] = torch.IntTensor({1,0,0}),
    [2] = torch.IntTensor({0,1,0}),
    [3] = torch.IntTensor({0,0,1})
}

local episodes = 10
local sleepTime = 0.028

-- To be used by the main game loop
local state, reward

for i=1, episodes do
    print("Episode #"..i)

    -- Seed can be changed anytime. It will take effect from next episodes.
    -- game.set_seed(seed)
    game:newEpisode()

    while not game:isEpisodeFinished() do
        state = game:getState()

        -- Make a random action
        local action = actions[torch.random(#actions)]
        reward = game:makeAction(action)

        print("Seed:", game:getSeed())

        if sleepTime > 0 then
            sys.sleep(sleepTime)
        end
    end
    -- Check how the episode went.
    print("Episode finished.")
    print("total reward:", game:getTotalReward())
    print("************************")
end

game:close()
