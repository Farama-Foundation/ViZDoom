local vizdoom = require("../../src/lib_lua/init.lua")

local game = vizdoom.vizdoom_new()

vizdoom.vizdoom_setViZDoomPath(game, "../../bin/vizdoom");
vizdoom.vizdoom_setDoomGamePath(game, "../../scenarios/freedoom2.wad")
vizdoom.vizdoom_setDoomScenarioPath(game, "../../scenarios/basic.wad");
vizdoom.vizdoom_setDoomMap(game, "map01");
vizdoom.vizdoom_setScreenResolution(game,
                                    vizdoom.ScreenResolution.RES_1280X1024);
vizdoom.vizdoom_setScreenFormat(game, vizdoom.ScreenFormat.RGB24);
vizdoom.vizdoom_setRenderHud(game, false);
vizdoom.vizdoom_setRenderCrosshair(game, false);
vizdoom.vizdoom_setRenderWeapon(game, true);
vizdoom.vizdoom_setRenderDecals(game, false);
vizdoom.vizdoom_setRenderParticles(game, false);

vizdoom.vizdoom_addAvailableButton(game, vizdoom.Button.MOVE_LEFT);
vizdoom.vizdoom_addAvailableButton(game, vizdoom.Button.MOVE_RIGHT);
vizdoom.vizdoom_addAvailableButton(game, vizdoom.Button.ATTACK);
vizdoom.vizdoom_addAvailableGameVariable(game, vizdoom.GameVariable.AMMO2);

vizdoom.vizdoom_setEpisodeTimeout(game, 200);
vizdoom.vizdoom_setEpisodeStartTime(game, 10);
vizdoom.vizdoom_setWindowVisible(game, true);
vizdoom.vizdoom_setSoundEnabled(game, true);
--vizdoom.vizdoom_setMode(game, vizdoom.Mode.PLAYER);
vizdoom.vizdoom_init(game);

print("setAction ----------------------------------")
local actions = torch.IntTensor({0,1,1});
print(actions);
vizdoom.vizdoom_setAction(game, actions:cdata())

print("getLastAction -------------------------------")
local lastAction = torch.IntTensor()
vizdoom.vizdoom_getLastAction(game, lastAction:cdata())
print(lastAction)

print("makeAction -------------------------------")
local reward = vizdoom.vizdoom_makeAction(game, actions:cdata())
local reward = vizdoom.vizdoom_makeAction_byTics(game, actions:cdata(), 10)
print("Reward: "..reward)

print("\nScreen Size  -------------------------------")
local screenWidth = vizdoom.vizdoom_getScreenWidth(game)
local screenHeight = vizdoom.vizdoom_getScreenHeight(game)
local channels = vizdoom.vizdoom_getScreenChannels(game)

print("Format: "..vizdoom.vizdoom_getScreenFormat(game))
print("Size:   "..tonumber(vizdoom.vizdoom_getScreenSize(game)))
print("Width:  "..screenWidth)
print("Height: "..screenHeight)
print("Chnnls: "..channels)

local gameVarsTensor = torch.IntTensor()
local imageBufferTensor = torch.ByteTensor()
local gameStateNo = vizdoom.vizdoom_getState(game, gameVarsTensor:cdata(),
                                             imageBufferTensor:cdata())
-- overwrites the buffer, basically it's the same data.
vizdoom.vizdoom_getGameScreen(game, imageBufferTensor:cdata())

print("\nGameState ----------------------------------")
print("GameState.number: "..gameStateNo)
print("GameState.gameVars: ", gameVarsTensor)
print("GameState.imageBuffer sz: "..imageBufferTensor:size(2))

--local image = require 'image'
--local screen = imageBufferTensor:view(channels, screenHeight, screenWidth)
--image.save('test.png', screen:float():div(255))

print("-----------------------------------------------------")
print("REMOVE vizdoom_close(game) to keep the game running!!")
print("-----------------------------------------------------")
vizdoom.vizdoom_close(game)
