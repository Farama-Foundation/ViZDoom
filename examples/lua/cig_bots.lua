#!/usr/bin/env th

require "vizdoom"
require "torch"
require "sys"

-- Create DoomGame instance. It will run the game and communicate with you.
local game = vizdoom.DoomGame()

-- Use CIG example config or your own.
game:loadConfig("../../scenarios/cig.cfg")

-- Select map you want to use.
game:setDoomMap("map01") -- Limited deathmatch.
--game:setDoomMap("map02") -- Full deathmatch.

-- Start multiplayer game only with your AI (with options that will be used in the competition, details in cig_host example).
game:addGameArgs("-host 1 -deathmatch +timelimit 1 " ..
                 "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 " ..
                 "+viz_respawn_delay 10 +viz_nocheat 1")

-- Name your agent and select color
-- colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game:addGameArgs("+name AI +colorset 0")

game:setMode(vizdoom.Mode.SPECTATOR)
--game:setWindowVisible(false)

game:init();

-- Three example sample actions
local actions = {
    [1] = torch.IntTensor({1,0,0,0,0,0,0,0,0}),
    [2] = torch.IntTensor({0,1,0,0,0,0,0,0,0}),
    [3] = torch.IntTensor({0,0,1,0,0,0,0,0,0})
}

-- Play with this many bots
local bots = 7

-- Run this many episodes
local episodes = 10

-- To be used by the main game loop
local state, reward

for i = 1, episodes do

    print("Episode #"..i)
    -- Add specific number of bots
    -- (file examples/bots.cfg must be placed in the same directory as the Doom executable file,
    -- edit this file to adjust bots).
    game:sendGameCommand("removebots")
    for i = 1, bots do
        game:sendGameCommand("addbot")
    end

    -- Play until the game (episode) is over.
    while not game:isEpisodeFinished() do

        -- Analyze the state
        state = game:getState()

        -- Make a random action
        local action = actions[torch.random(#actions)]
        reward = game:makeAction(action)

        -- Check if player is dead
        if game:isPlayerDead() then
            -- Respawn immediately after death, new state will be available.
            game:respawnPlayer()
        end

        --print("Frags:", game:getGameVariable(vizdoom.GameVariable.FRAGCOUNT))
    end

    print("Episode finished.")
    print("************************")

    game:newEpisode()
end

game:close()
