local ViZDoomLua = torch.class('vizdoom.ViZDoomLua')

function ViZDoomLua:__init()

   self.lib = vizdoom.lib
   self.game = self.lib.vizdoom_new()

   -- Tensors to be reused
   self.gameVars     = torch.IntTensor()
   self.imageBuff    = torch.ByteTensor()
   self.lastAction   = torch.IntTensor()

end


function ViZDoomLua:gc()
   self.lib.vizdoom_gc(self.game)
end


function ViZDoomLua:init()
   return self.lib.vizdoom_init(self.game)
end


function ViZDoomLua:close()
   self.lib.vizdoom_close(self.game)
end


function ViZDoomLua:isRunning()
   return self.lib.vizdoom_isRunning(self.game)
end


function ViZDoomLua:newEpisode()
   self.lib.vizdoom_newEpisode(self.game)
end


function ViZDoomLua:newEpisodeFromPath(path)
   self.lib.vizdoom_newEpisode_path(self.game, path)
end


function ViZDoomLua:replayEpisode(path)
   self.lib.vizdoom_replayEpisode(self.game, path)
end


function ViZDoomLua:setAction(action)
   self.lib.vizdoom_setAction(self.game, action:cdata())
end


function ViZDoomLua:advanceAction(tics)
   tics = tics or 1
   self.lib.vizdoom_advanceAction(self.game, tics)
end


function ViZDoomLua:advanceActionOneTic()
   self.lib.vizdoom_advanceActionOneTic(self.game)
end


function ViZDoomLua:makeAction(action)
   return self.lib.vizdoom_makeAction(self.game, action:cdata())
end


function ViZDoomLua:makeActionByTics(action, tics)
   return self.lib.vizdoom_makeAction(self.game, action:cdata(), tics)
end


function ViZDoomLua:getState()
   local no = self.lib.vizdoom_getState(self.game,
                                        self.gameVars:cdata(),
                                        self.imageBuff:cdata())
   return {
      ["number"] = no,
      ["gameVariables"] = self.gameVars,
      ["imageBuffer"] = self.imageBuff
   }
   --[[
      In order to display the screen, the tensor must be transformed:

      state.imageBuffer:view(1024,1280,3)
                       :permute(3,1,2)
	               :index(1,torch.LongTensor{3,2,1})
   --]]
end


function ViZDoomLua:getLastAction()
   self.lib.vizdoom_getLastAction(self.game, self.lastAction:cdata())
   return self.lastAction
end


function ViZDoomLua:isNewEpisode()
   return self.lib.vizdoom_isNewEpisode(self.game)
end


function ViZDoomLua:isEpisodeFinished()
   return self.lib.vizdoom_isEpisodeFinished(self.game)
end


function ViZDoomLua:isPlayerDead()
   return self.lib.vizdoom_isPlayerDead(self.game)
end


function ViZDoomLua:respawnPlayer()
   self.lib.vizdoom_respawnPlayer(self.game)
end


function ViZDoomLua:addAvailableButton(button)
   self.lib.vizdoom_addAvailableButton(self.game, button)
end


function ViZDoomLua:addAvailableButtonMaxVal(button, maxValue)
   self.lib.vizdoom_addAvailableButton_bm(self.game, button, maxValue)
end


function ViZDoomLua:clearAvailableButtons()
   self.lib.vizdoom_clearAvailableButtons(self.game)
end


function ViZDoomLua:getAvailableButtonsSize()
   return self.lib.vizdoom_getAvailableButtonsSize(self.game)
end


function ViZDoomLua:setButtonMaxValue(button, maxValue)
   self.lib.vizdoom_setButtonMaxValue(self.game, button, maxValue)
end


function ViZDoomLua:getButtonMaxValue()
   return self.lib.vizdoom_getButtonMaxValue(self.game)
end


function ViZDoomLua:addAvailableGameVariable(var)
   return self.lib.vizdoom_addAvailableGameVariable(self.game, var)
end


function ViZDoomLua:clearAvailableGameVariables()
   self.lib.vizdoom_clearAvailableGameVariables(self.game)
end


function ViZDoomLua:getAvailableGameVariablesSize()
   return self.lib.vizdoom_getAvailableGameVariablesSize(self.game)
end


function ViZDoomLua:addGameArgs(args)
   self.lib.vizdoom_addGameArgs(self.game, args)
end


function ViZDoomLua:clearGameArgs()
   self.lib.vizdoom_clearGameArgs(self.game)
end


function ViZDoomLua:sendGameCommand(cmd)
   self.lib.vizdoom_sendGameCommand(self.game, cmd)
end


function ViZDoomLua:getGameScreen()
   self.lib.vizdoom_getGameScreen(self.game, self.imageBuff:cdata())
   return self.imageBuff
end


function ViZDoomLua:getMode()
   return self.lib.vizdoom_getMode(self.game)
end


function ViZDoomLua:setMode(mode)
   self.lib.vizdoom_setMode(self.game, mode)
end


function ViZDoomLua:getTicrate()
   return self.lib.vizdoom_getTicrate(self.game)
end


function ViZDoomLua:setTicrate(ticrate)
   self.lib.vizdoom_setTicrate(self.game, ticrate)
end


function ViZDoomLua:getGameVariable(var)
   return self.lib.vizdoom_getGameVariable(self.game, var)
end


function ViZDoomLua:setViZDoomPath(path)
   self.lib.vizdoom_setViZDoomPath(self.game, path)
end


function ViZDoomLua:setDoomGamePath(path)
   self.lib.vizdoom_setDoomGamePath(self.game, path)
end


function ViZDoomLua:setDoomScenarioPath(path)
   self.lib.vizdoom_setDoomScenarioPath(self.game, path)
end


function ViZDoomLua:setDoomMap(path)
   self.lib.vizdoom_setDoomMap(self.game, path)
end


function ViZDoomLua:setDoomSkill(skill)
   self.lib.vizdoom_setDoomSkill(self.game, skill)
end


function ViZDoomLua:setDoomConfigPath(path)
   self.lib.vizdoom_setDoomConfigPath(self.game, path)
end


function ViZDoomLua:getSeed()
   return self.lib.vizdoom_getSeed(self.game)
end


function ViZDoomLua:setSeed(seed)
   self.lib.vizdoom_setSeed(self.game, seed)
end


function ViZDoomLua:getEpisodeStartTime()
   return self.lib.vizdoom_getEpisodeStartTime(self.game)
end


function ViZDoomLua:setEpisodeStartTime(tics)
   self.lib.vizdoom_setEpisodeStartTime(self.game, tics)
end


function ViZDoomLua:getEpisodeTimeout()
   return self.lib.vizdoom_getEpisodeTimeout(self.game)
end


function ViZDoomLua:setEpisodeTimeout(tics)
   self.lib.vizdoom_setEpisodeTimeout(self.game, tics)
end


function ViZDoomLua:getEpisodeTime()
   return self.lib.vizdoom_getEpisodeTime(self.game)
end


function ViZDoomLua:getLivingReward()
   return self.lib.vizdoom_getLivingReward(self.game)
end


function ViZDoomLua:setLivingReward(livingReward)
   self.lib.vizdoom_setLivingReward(self.game, livingReward)
end


function ViZDoomLua:getDeathPenalty()
   return self.lib.vizdoom_getDeathPenalty(self.game)
end


function ViZDoomLua:setDeathPenalty(deathPenalty)
   self.lib.vizdoom_setDeathPenalty(self.game, deathPenalty)
end


function ViZDoomLua:getLastReward()
   return self.lib.vizdoom_getLastReward(self.game)
end


function ViZDoomLua:getTotalReward()
   return self.lib.vizdoom_getTotalReward(self.game)
end


function ViZDoomLua:setScreenResolution(resolution)
   self.lib.vizdoom_setScreenResolution(self.game, resolution)
end


function ViZDoomLua:setScreenFormat(format)
   self.lib.vizdoom_setScreenFormat(self.game, format)
end


function ViZDoomLua:getScreenFormat()
   return self.lib.vizdoom_getScreenFormat(self.game)
end


function ViZDoomLua:setRenderHud(hud)
   self.lib.vizdoom_setRenderHud(self.game, hud)
end


function ViZDoomLua:setRenderWeapon(weapon)
   self.lib.vizdoom_setRenderWeapon(self.game, weapon)
end


function ViZDoomLua:setRenderCrosshair(crosshair)
   self.lib.vizdoom_setRenderCrosshair(self.game, crosshair)
end


function ViZDoomLua:setRenderDecals(decals)
   self.lib.vizdoom_setRenderDecals(self.game, decals)
end


function ViZDoomLua:setRenderParticles(particles)
   self.lib.vizdoom_setRenderParticles(self.game, particles)
end


function ViZDoomLua:setRenderParticles(particles)
   self.lib.vizdoom_setRenderParticles(self.game, particles)
end


function ViZDoomLua:setWindowVisible(visibility)
   self.lib.vizdoom_setWindowVisible(self.game, visibility)
end


function ViZDoomLua:setConsoleEnabled(console)
   self.lib.vizdoom_setConsoleEnabled(self.game, console)
end


function ViZDoomLua:setSoundEnabled(sound)
   self.lib.vizdoom_setSoundEnabled(self.game, sound)
end


function ViZDoomLua:getScreenWidth()
   return self.lib.vizdoom_getScreenWidth(self.game)
end


function ViZDoomLua:getScreenHeight()
   return self.lib.vizdoom_getScreenHeight(self.game)
end


function ViZDoomLua:getScreenChannels()
   return self.lib.vizdoom_getScreenChannels(self.game)
end


function ViZDoomLua:getScreenPitch()
   return self.lib.vizdoom_getScreenPitch(self.game)
end


function ViZDoomLua:getScreenSize()
   return self.lib.vizdoom_getScreenSize(self.game)
end


function ViZDoomLua:loadConfig(filename)
   return self.lib.vizdoom_loadConfig(self.game, filename)
end
