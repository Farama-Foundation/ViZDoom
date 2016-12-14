#!/usr/bin/env th

local threads = require 'threads'

local noOfPlayers = 2
local episodes = 10

local pool = threads.Threads(

    -- spawned threads
    noOfPlayers,

    -- Set-up the environment on each thread
    function()
        -- for some reason 'require' has an unexpected behavior when called
        -- from 'threads' to execute a module outside this director.
        require "vizdoom"

        actions = {
            [1] = torch.IntTensor({1,0,0}),
            [2] = torch.IntTensor({0,1,0}),
            [3] = torch.IntTensor({0,0,1})
        }
    end,

    function()
        function player1()
            -- Create DoomGame instance.
            game = vizdoom.DoomGame()

            -- Config
            game:loadConfig("../config/basic.cfg")
            game:setMode(vizdoom.Mode.ASYNC_PLAYER)

            -- Default Doom's ticrate is 35 per second,
            -- so this one will work 2 times faster.
            game:setTicrate(70)

            game:init()

            for i = 1, episodes do
                game:newEpisode()
                print("New episode: ", __threadid)

                while not game:isEpisodeFinished() do
                    -- Make a random action
                    local action = actions[torch.random(#actions)]
                    reward = game:makeAction(action)
                end
            end
            game:close()
        end

        function player2()
            -- Create DoomGame instance.
            game = vizdoom.DoomGame()

            -- Config
            game:loadConfig("../config/basic.cfg")
            game:setMode(vizdoom.Mode.ASYNC_PLAYER)

            -- And this one will work 2 times slower.
            game:setTicrate(17)

            game:init()

            for i = 1, episodes do
                game:newEpisode()
                print("New episode: ", __threadid)

                while not game:isEpisodeFinished() do
                    -- Make a random action
                    local action = actions[torch.random(#actions)]
                    reward = game:makeAction(action)
                end
            end
            game:close()
        end
    end
)

pool:specific(true)

pool:addjob(
    1,
    function()
        player1()
    end
)
pool:addjob(
    2,
    function()
        player2()
    end
)

pool:synchronize()
pool:terminate()
print("Done")
