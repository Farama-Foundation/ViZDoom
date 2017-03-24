#!/usr/bin/env th

require "vizdoom"
require "torch"

-- Check arguments for a host's ip.
local ip = "127.0.0.1"
for a = 1, #arg do
    if arg[a] == "-ip" and type(arg[a + 1]) == "string" then
        ip = arg[a + 1]
    end
end

-- Create DoomGame instance. It will run the game and communicate with you.
local game = vizdoom.DoomGame()

-- Use CIG example config or your own.
game:loadConfig("../../scenarios/cig.cfg")

-- Select game and map you want to use.
game:setDoomMap("map01") -- Limited deathmatch.
--game:setDoomMap("map02") -- Full deathmatch.

-- Join existing game.
game:addGameArgs("-join " .. ip) -- Connect to a host for a multiplayer game.

-- Name your agent and select color
-- colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game:addGameArgs("+name AI +colorset 0")

game:setMode(vizdoom.Mode.ASYNC_PLAYER) -- During the competition, async mode will be forced for all agents.
--game:setWindowVisible(false)

game:init();

-- Three example sample actions
local actions = {
    [1] = torch.IntTensor({1,0,0,0,0,0,0,0,0}),
    [2] = torch.IntTensor({0,1,0,0,0,0,0,0,0}),
    [3] = torch.IntTensor({0,0,1,0,0,0,0,0,0})
}

-- To be used by the main game loop
local state, reward

-- Play until the game (episode) is over.
while not game:isEpisodeFinished() do

    -- Analyze the state
    state = game:getState()

    -- Make a random action
    local action = actions[torch.random(#actions)]
    local reward = game:makeAction(action)

    -- Check if player is dead
    if game:isPlayerDead() then
        -- Respawn immediately after death, new state will be available.
        game:respawnPlayer()
    end

    --print("Frags:", game:getGameVariable(vizdoom.GameVariable.FRAGCOUNT))
end

game:close()
