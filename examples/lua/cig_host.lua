#!/usr/bin/env th

require "vizdoom"
require "torch"

-- Create DoomGame instance. It will run the game and communicate with you.
local game = vizdoom.DoomGame()

-- Use CIG example config or your own.
game:loadConfig("../../scenarios/cig.cfg")

-- Select map you want to use.
game:setDoomMap("map01") -- Limited deathmatch.
--game:setDoomMap("map02") -- Full deathmatch.

-- Host game with options that will be used in the competition.
game:addGameArgs("-host 2 " ..              -- This machine will function as a host for a multiplayer game with this many players (including this machine). It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
                 "-deathmatch " ..          -- Deathmatch rules are used for the game.
                 "+timelimit 10.0 " ..      -- The game (episode) will end after this many minutes have elapsed.
                 "+sv_forcerespawn 1 " ..   -- Players will respawn automatically after they die.
                 "+sv_noautoaim 1 " ..      -- Autoaim is disabled for all players.
                 "+sv_respawnprotect 1 " .. -- Players will be invulnerable for two second after spawning.
                 "+sv_spawnfarthest 1 " ..  -- Players will be spawned as far as possible from any other players.
                 "+sv_nocrouch 1" ..        -- Disables crouching.
                 "+viz_respawn_delay 10 " ..-- Sets delay between respanws (in seconds).
                 "+viz_nocheat 1")          -- Disables depth and labels buffer and the ability to use commands that could interfere with multiplayer game.

-- This can be used to host game without taking part in it (can be simply added as argument of vizdoom executable).
--game:add_game_args("viz_spectator 1")

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
    reward = game:makeAction(action)

    -- Check if player is dead
    if game:isPlayerDead() then
        -- Respawn immediately after death, new state will be available.
        game:respawnPlayer()
    end

    --print("Frags:", game:getGameVariable(vizdoom.GameVariable.FRAGCOUNT))
end

game:close()
