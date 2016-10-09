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
-- loadConfig could be used to load configuration instead of doing it here with code.
-- If loadConfig is used in-code configuration will work. Note that the most recent changes will add to previous ones.
-- game.loadConfig("../../examples/config/basic.cfg")

-- Sets path to ViZDoom engine executive which will be spawned as a separate process. Default is "./vizdoom".
game:setViZDoomPath("../../bin/vizdoom")

-- Sets path to iwad resource file which contains the actual doom game. Default is "./doom2.wad".
game:setDoomGamePath("../../scenarios/freedoom2.wad")
--game.setDoomGame_path("../../scenarios/doom2.wad") -- Not provided with environment due to licences.

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
game:setRenderCrosshair(false)
game:setRenderWeapon(true)
game:setRenderDecals(false)
game:setRenderParticles(false)
game:setRenderEffectsSprites(false)

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
sleepTime = 28

for i = 1, episodes do

    print("Episode #" .. i)

    -- Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game:newEpisode()

    while not game:isEpisodeFinished() do

        -- Gets the state
        state = game:getState()

        -- Which consists of:
        n           = state.number
        vars        = state.gameVariables
        screenBuf   = state.screenBuffer
        depthBuf    = state.depthBuffer
        labelsBuf   = state.labelsBuffer
        automapBuf  = state.automapBuffer
        labels      = state.labels

        -- Makes a random action and get remember reward.
        reward = game:makeAction(actions[math.random(1,3)])

        -- Makes a "prolonged" action and skip frames.
        --skiprate = 4
        --reward = game.makeAction(choice(actions), skiprate)

        -- The same could be achieved with:
        --game.setAction(choice(actions))
        --game.advanceAction(skiprate)
        --reward = game.getLastReward()

        -- Prints state's reward.
        print("State # " .. n)
        print("Game variables: " .. vars[1])
        print("Reward: " .. reward)
        print("=====================")

        if sleepTime > 0 then
            vizdoom.sleep(sleepTime)
        end
    end

    -- Check how the episode went.
    print("Episode finished.")
    print("total reward: " .. game:getTotalReward())
    print("************************")

end

game:close()
