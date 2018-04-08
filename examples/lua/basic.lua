#!/usr/bin/env th

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

require "vizdoom"
-- local vizdoom = require "vizdoom" is possible

-- you may want to do this for convenience
local Button = vizdoom.Button
local Mode = vizdoom.Mode
local GameVariable = vizdoom.GameVariable
local ScreenFormat = vizdoom.ScreenFormat
local ScreenResolution = vizdoom.ScreenResolution

require "torch"
require "sys"

-- Create DoomGame instance. It will run the game and communicate with you.
local game = vizdoom.DoomGame()

-- Now it's time for configuration!
-- loadConfig could be used to load configuration instead of doing it here with code.
-- If loadConfig is used in-code configuration will work. Note that the most recent changes will add to previous ones.
-- game.loadConfig("../../scenarios/basic.cfg")

-- Sets path to additional resources wad file which is basically your scenario wad.
-- If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
game:setDoomScenarioPath("../../scenarios/basic.wad")

-- Sets map to start (scenario .wad files can contain many maps).
game:setDoomMap("map01")

-- Sets resolution. Default is 320X240
game:setScreenResolution(vizdoom.ScreenResolution.RES_640X480)

-- Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
game:setScreenFormat(vizdoom.ScreenFormat.RGB24)

-- Enables depth buffer.
game:setDepthBufferEnabled(true)

-- Enables labeling of in game objects labeling.
game:setLabelsBufferEnabled(true)

-- Enables buffer with top down map of he current episode/level .
game:setAutomapBufferEnabled(true)

-- Sets other rendering options
game:setRenderHud(false)
game:setRenderMinimalHud(false) -- If hud is enabled
game:setRenderCrosshair(false)
game:setRenderWeapon(true)
game:setRenderDecals(false)
game:setRenderParticles(false)
game:setRenderEffectsSprites(false)
game:setRenderMessages(false)
game:setRenderCorpses(false)

-- Adds buttons that will be allowed.
game:addAvailableButton(vizdoom.Button.MOVE_LEFT)
game:addAvailableButton(vizdoom.Button.MOVE_RIGHT)
game:addAvailableButton(vizdoom.Button.ATTACK)

-- Adds game variables that will be included in state.
game:addAvailableGameVariable(vizdoom.GameVariable.AMMO2)

-- Causes episodes to finish after 200 tics (actions)
game:setEpisodeTimeout(200)

-- Makes episodes start after 10 tics (~after raising the weapon)
game:setEpisodeStartTime(10)

-- Makes the window appear (turned on by default)
game:setWindowVisible(true)

-- Turns on the sound. (turned off by default)
game:setSoundEnabled(true)

-- Sets the livin reward (for each move) to -1
game:setLivingReward(-1)

-- Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
game:setMode(vizdoom.Mode.PLAYER)

-- Enables engine output to console.
--game:setConsoleEnabled(true)

-- Initialize the game. Further configuration won't take any effect from now on.
game:init()

-- Define some actions. Each list entry corresponds to declared buttons:
-- MOVE_LEFT, MOVE_RIGHT, ATTACK
-- game:getAvailableButtonsSize() can be used to check the number of available buttons.
-- 5 more combinations are naturally possible but only 3 are included for transparency when watching.

local actions = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
-- action can be table or IntTensor
local actions = {
    [1] = torch.IntTensor({1,0,0}),
    [2] = torch.IntTensor({0,1,0}),
    [3] = torch.IntTensor({0,0,1})
}

-- Run this many episodes
local episodes = 10

-- Sets time that will pause the engine after each action.
-- Without this everything would go too fast for you to keep track of what's happening.
local sleepTime = 0.028

for i = 1, episodes do

    print("Episode #" .. i)

    -- Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game:newEpisode()

    while not game:isEpisodeFinished() do

        -- Gets the state
        local state = game:getState()

        -- Which consists of:
        local n           = state.number
        local vars        = state.gameVariables -- IntTensor
        local screenBuf   = state.screenBuffer -- ByteTensor
        local depthBuf    = state.depthBuffer
        local labelsBuf   = state.labelsBuffer
        local automapBuf  = state.automapBuffer
        local labels      = state.labels

        -- Makes a random action and get remember reward.
        --reward = game:makeAction(actions[math.random(1,3)])
        local action = actions[torch.random(#actions)]
        local reward = game:makeAction(action)

        -- Makes a "prolonged" action and skip frames.
        --skiprate = 4
        --reward = game:makeAction(choice(actions), skiprate)

        -- The same could be achieved with:
        --game:setAction(choice(actions))
        --game:advanceAction(skiprate)
        --reward = game:getLastReward()

        -- Prints state's reward.
        print("State # " .. n)
        print("Game variables: " .. vars[1])
        print("Reward: " .. reward)
        print("=====================")

        if sleepTime > 0 then
            sys.sleep(sleepTime)
        end
    end

    -- Check how the episode went.
    print("Episode finished.")
    print("total reward: " .. game:getTotalReward())
    print("************************")

end

game:close()
