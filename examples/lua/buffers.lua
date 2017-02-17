#!/usr/bin/env th

----------------------------------------------------------------------
-- This script presents SPECTATOR mode. In SPECTATOR mode you play and
-- your agent can learn from it.
-- Configuration is loaded from "../../scenarios/<SCENARIO_NAME>.cfg" file.
--
-- To see the scenario description go to "../../scenarios/README.md"
----------------------------------------------------------------------

require "vizdoom"
require "torch"
require "image"
require "sys"

game = vizdoom.DoomGame()

-- Choose scenario config file you wish to watch.
-- Don't load two configs cause the second will overrite the first one.
-- Multiple config files are ok but combining these ones doesn't make much sense.

--game:loadConfig("../../scenarios/basic.cfg")
game:loadConfig("../../scenarios/deadly_corridor.cfg")
--game:loadConfig("../../scenarios/deathmatch.cfg")
--game:loadConfig("../../scenarios/defend_the_center.cfg")
--game:loadConfig("../../scenarios/defend_the_line.cfg")
--game:loadConfig("../../scenarios/health_gathering.cfg")
--game:loadConfig("../../scenarios/my_way_home.cfg")
--game:loadConfig("../../scenarios/predict_position.cfg")
--game:loadConfig("../../scenarios/take_cover.cfg")

-- Enables freelook in engine
game:addGameArgs("+freelook 1")

game:setScreenResolution(vizdoom.ScreenResolution.RES_640X480)
game:setWindowVisible(true)

-- Enables labeling of the in game objects.
game:setLabelsBufferEnabled(true)

-- Enables depth buffer.
game:setDepthBufferEnabled(true)

-- Enables labeling of in game objects labeling.
game:setLabelsBufferEnabled(true)

-- Enables buffer with top down map of he current episode/level .
game:setAutomapBufferEnabled(true)
game:setAutomapMode(vizdoom.AutomapMode.OBJECTS)
game:setAutomapRotate(false)
game:setAutomapRenderTextures(false)

game:setRenderHud(true)
game:setRenderMinimalHud(false)

game:init()

local actions = {
    [1] = torch.IntTensor({1,0,0}),
    [2] = torch.IntTensor({0,1,0}),
    [3] = torch.IntTensor({0,0,1})
}

local episodes = 10
local sleepTime = 0.028

for i = 1, episodes do

    print("Episode #" .. i)

    game:newEpisode()

    while not game:isEpisodeFinished() do

        -- Gets the state.
        local state = game:getState()

        local screen = state.screenBuffer
        screenWin = image.display{image=screen, legend="ViZDoom Screen Buffer", offscreen=false, win=screenWin}

        -- Depth buffer, always in 8-bit gray channel format.
        -- This is most fun. It looks best if you inverse colors.
        local depth = state.depthBuffer
        if depth then
            depthWin = image.display{image=depth, legend="ViZDoom Depth Buffer", offscreen=false, win=depthWin}
        end

        -- Labels buffer, always in 8-bit gray channel format.
        -- Shows only visible game objects (enemies, pickups, exploding barrels etc.), each with unique label.
        -- Labels data are available in state.labels, also see labels.py example.
        local labels = state.labelsBuffer
        if labels then
            labelsWin = image.display{image=labels, legend="ViZDoom Labels Buffer", offscreen=false, win=labelsWin}
        end

        -- Map buffer, in the same format as screen buffer.
        -- Shows top down map of the current episode/level.
        local automap = state.automapBuffer
        if automap then
            automapWin = image.display{image=automap, legend="ViZDoom Automap Buffer", offscreen=false, win=automapWin}
        end

        local action = actions[torch.random(#actions)]
        local reward = game:makeAction(action)

        print("State #" .. state.number)
        print("=====================")

        if sleepTime > 0 then
            sys.sleep(sleepTime)
        end

    end

    print("Episode finished.")
    print("total reward: " .. game:getTotalReward())
    print("************************")

end

game:close()
