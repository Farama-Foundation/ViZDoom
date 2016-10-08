#include "ViZDoomGame_lua.h"


DoomGame* vizdoom_new() {
#if debug
  printf("Game is being initialized...\n");
#endif
  return new DoomGame();
}

void vizdoom_gc(DoomGame* _this) {
  delete _this;
}

bool vizdoom_init(DoomGame* _this) {
  return _this->init();
}

void vizdoom_close(DoomGame* _this) {
  _this->close();
}

bool vizdoom_isRunning(DoomGame* _this) {
  return _this->isRunning();
}

// Episodes
void vizdoom_newEpisode(DoomGame* _this) {
  _this->newEpisode();
}

void vizdoom_newEpisode_path(DoomGame* _this, const char* path) {
  _this->newEpisode(path);
}

void vizdoom_replayEpisode(DoomGame* _this, const char* path) {
  _this->replayEpisode(path);
}

// Actions
void vizdoom_setAction(DoomGame* _this, THIntTensor* actions) {

  int* data = actions->storage->data;
  long int* actionsSize = actions->size;

  std::vector<int> actionsVec(data, data + *actionsSize);

  _this->setAction(actionsVec);
}

void vizdoom_advanceAction(DoomGame* _this, unsigned int tics) {
  _this->advanceAction(tics);
}

void vizdoom_advanceActionOneTic(DoomGame* _this) {
  _this->advanceAction();
}

double vizdoom_makeAction(DoomGame* _this, THIntTensor* actions) {

  int* data = actions->storage->data;
  long int* actionsSize = actions->size;

  std::vector<int> actionsVec(data, data + *actionsSize);

  return _this->makeAction(actionsVec);
}

double vizdoom_makeAction_byTics(DoomGame* _this, THIntTensor* actions,
                                 unsigned int tics) {

  int* data = actions->storage->data;
  long int* actionsSize = actions->size;

  std::vector<int> actionsVec(data, data + *actionsSize);

  return _this->makeAction(actionsVec, tics);
}

// Get State
int vizdoom_getState(DoomGame* _this,
                     THIntTensor* emptyGameVars,
                     THByteTensor* emptyImageBuffer) {

  GameState state = _this->getState();

  // Put the contents of state.GameVariables to tensor
  size_t varsSize = state.gameVariables.size();
  // Couldn't figure our why it dissapears so we are copying it for now
  std::vector<int>& gameVars = *(new std::vector<int>(state.gameVariables));

  THIntStorage* varsStorage =
    THIntStorage_newWithData(gameVars.data(), varsSize);

  if(varsStorage) {
    long sizedata[2]   = { varsSize };
    long stridedata[2] = { 1 };

    THLongStorage* size    = THLongStorage_newWithData(sizedata, 1);
    THLongStorage* stride  = THLongStorage_newWithData(stridedata, 1);
    THIntTensor_setStorage(emptyGameVars, varsStorage, 0LL, size, stride);
  }

  // Put the contents of state.imageBuffer to tensor
  size_t stateSize = _this->getScreenSize();
  uint8_t * imageBuffer = state.imageBuffer;

  THByteStorage* iStorage = THByteStorage_newWithData(imageBuffer, stateSize);

  if(iStorage) {
    long sizedata[1]   = { stateSize };
    long stridedata[1] = { 1 };

    THLongStorage* size    = THLongStorage_newWithData(sizedata, 1);
    THLongStorage* stride  = THLongStorage_newWithData(stridedata, 1);
    THByteTensor_setStorage(emptyImageBuffer, iStorage, 0LL, size, stride);
  }

  return state.number;
}

// Get Last Action
void vizdoom_getLastAction(DoomGame* _this, THIntTensor* emptyActionTensor) {
  std::vector<int>& lastAction=*(new std::vector<int>(_this->getLastAction()));
  size_t lastActionSz = lastAction.size();

  THIntStorage* actStorage =
    THIntStorage_newWithData(lastAction.data(), lastActionSz);

  if(actStorage) {
    long sizedata[1]   = { lastActionSz };
    long stridedata[1] = { 1 };

    THLongStorage* size    = THLongStorage_newWithData(sizedata, 1);
    THLongStorage* stride  = THLongStorage_newWithData(stridedata, 1);
    THIntTensor_setStorage(emptyActionTensor, actStorage, 0LL, size, stride);
  }
}

// Episode info
bool vizdoom_isNewEpisode(DoomGame* _this) {
  return _this->isNewEpisode();
}

bool vizdoom_isEpisodeFinished(DoomGame* _this) {
  return _this->isEpisodeFinished();
}

bool vizdoom_isPlayerDead(DoomGame* _this) {
  return _this->isPlayerDead();
}

void vizdoom_respawnPlayer(DoomGame* _this) {
  _this->respawnPlayer();
}


// Buttons
void vizdoom_addAvailableButton(DoomGame* _this, int button) {
  _this->addAvailableButton((Button)button);
}

void vizdoom_addAvailableButton_bm(DoomGame* _this, int button,
                                   unsigned int maxValue) {
  _this->addAvailableButton((Button)button, maxValue);
}

void vizdoom_clearAvailableButtons(DoomGame* _this) {
  _this->clearAvailableButtons();
}

int vizdoom_getAvailableButtonsSize(DoomGame* _this) {
  return _this->getAvailableButtonsSize();
}

void vizdoom_setButtonMaxValue(DoomGame* _this, int button,
                               unsigned int maxValue) {
  _this->setButtonMaxValue((Button)button, maxValue);
}

int vizdoom_getButtonMaxValue(DoomGame* _this, int button) {
  return _this->getButtonMaxValue((Button)button);
}

// Game Variables
void vizdoom_addAvailableGameVariable(DoomGame* _this, int var) {
  _this->addAvailableGameVariable((GameVariable)var);
}

void vizdoom_clearAvailableGameVariables(DoomGame* _this) {
  _this->clearAvailableGameVariables();
}

int vizdoom_getAvailableGameVariablesSize(DoomGame* _this) {
  return _this->getAvailableGameVariablesSize();
}

void vizdoom_addGameArgs(DoomGame* _this, const char* args) {
  _this->addGameArgs(args);
}

void vizdoom_clearGameArgs(DoomGame* _this) {
  _this->clearGameArgs();
}

void vizdoom_sendGameCommand(DoomGame* _this, const char* cmd) {
  _this->sendGameCommand(cmd);
}

void vizdoom_getGameScreen(DoomGame* _this, THByteTensor* emptyScreenTensor) {

  // Fill the screen from the returned pointer
  uint8_t * imageBuffer = _this->getGameScreen();
  size_t stateSize = _this->getScreenSize();

  THByteStorage* storage = THByteStorage_newWithData(imageBuffer, stateSize);

  if(storage) {
    long sizedata[1]   = { stateSize };
    long stridedata[1] = { 1 };

    THLongStorage* size    = THLongStorage_newWithData(sizedata, 1);
    THLongStorage* stride  = THLongStorage_newWithData(stridedata, 1);
    THByteTensor_setStorage(emptyScreenTensor, storage, 0LL, size, stride);
  }
}


// Mode
int vizdoom_getMode(DoomGame* _this) {
  return _this->getMode();
}

void vizdoom_setMode(DoomGame* _this, int mode) {
  _this->setMode((Mode)mode);
}

// Ticrate
unsigned int vizdoom_getTicrate(DoomGame* _this) {
  return _this->getTicrate();
}

void vizdoom_setTicrate(DoomGame* _this, unsigned int mode) {
  _this->setTicrate((Mode)mode);
}

int vizdoom_getGameVariable(DoomGame* _this, int var) {
  return _this->getGameVariable((GameVariable) var);
}

// Paths
void vizdoom_setViZDoomPath(DoomGame* _this, const char* path){
  _this->setViZDoomPath(path);
}

void vizdoom_setDoomGamePath(DoomGame* _this, const char* path){
  _this->setDoomGamePath(path);
}

void vizdoom_setDoomScenarioPath(DoomGame* _this, const char* path){
  _this->setDoomScenarioPath(path);
}

void vizdoom_setDoomMap(DoomGame* _this, const char* map){
  _this->setDoomMap(map);
}

void vizdoom_setDoomSkill(DoomGame* _this, const int skill){
  _this->setDoomSkill(skill);
}

void vizdoom_setDoomConfigPath(DoomGame* _this, const char* path){
  _this->setDoomConfigPath(path);
}

// Seed
unsigned int vizdoom_getSeed(DoomGame* _this) {
  return _this->getSeed();
}

void vizdoom_setSeed(DoomGame* _this, unsigned int seed) {
  _this->setSeed(seed);
}

// Time
unsigned int vizdoom_getEpisodeStartTime(DoomGame* _this) {
  return _this->getEpisodeStartTime();
}
void vizdoom_setEpisodeTimeout(DoomGame* _this, unsigned int tics) {
  _this->setEpisodeTimeout(tics);
}

unsigned int vizdoom_getEpisodeTimeout(DoomGame* _this) {
  return _this->getEpisodeTimeout();
}

void vizdoom_setEpisodeStartTime(DoomGame* _this, unsigned int tics) {
  _this->setEpisodeStartTime(tics);
}

unsigned int vizdoom_getEpisodeTime(DoomGame* _this) {
  return _this->getEpisodeTime();
}

// Rewards and penalties
double vizdoom_getLivingReward(DoomGame* _this) {
  return _this->getLivingReward();
}

void vizdoom_setLivingReward(DoomGame* _this, double livingReward) {
  _this->setLivingReward(livingReward);
}

double vizdoom_getDeathPenalty(DoomGame* _this) {
  return _this->getDeathPenalty();
}

void vizdoom_setDeathPenalty(DoomGame* _this, double deathPenalty) {
  _this->setDeathPenalty(deathPenalty);
}

double vizdoom_getLastReward(DoomGame* _this) {
  return _this->getLastReward();
}

double vizdoom_getTotalReward(DoomGame* _this) {
  return _this->getTotalReward();
}

// Screen and rendering options
void vizdoom_setScreenResolution(DoomGame* _this, int resolution) {
  _this->setScreenResolution((ScreenResolution)resolution);
}

void vizdoom_setScreenFormat(DoomGame* _this, int format) {
  _this->setScreenFormat((ScreenFormat)format);
}

unsigned int vizdoom_getScreenFormat(DoomGame* _this) {
  return _this->getScreenFormat();
}

void vizdoom_setRenderHud(DoomGame* _this, bool hud) {
  _this->setRenderHud(hud);
}

void vizdoom_setRenderWeapon(DoomGame* _this, bool weapon) {
  _this->setRenderWeapon(weapon);
}

void vizdoom_setRenderCrosshair(DoomGame* _this, bool crosshair) {
  _this->setRenderCrosshair(crosshair);
}

void vizdoom_setRenderDecals(DoomGame* _this, bool decals) {
  _this->setRenderDecals(decals);
}

void vizdoom_setRenderParticles(DoomGame* _this, bool particles) {
  _this->setRenderParticles(particles);
}

void vizdoom_setWindowVisible(DoomGame* _this, bool visibility) {
  _this->setWindowVisible(visibility);
}

void vizdoom_setConsoleEnabled(DoomGame* _this, bool console) {
  _this->setConsoleEnabled(console);
}

void vizdoom_setSoundEnabled(DoomGame* _this, bool sound) {
  _this->setSoundEnabled(sound);
}

int vizdoom_getScreenWidth(DoomGame* _this) {
  return _this->getScreenWidth();
}

int vizdoom_getScreenHeight(DoomGame* _this) {
  return _this->getScreenHeight();
}

int vizdoom_getScreenChannels(DoomGame* _this) {
  return _this->getScreenChannels();
}

size_t vizdoom_getScreenPitch(DoomGame* _this) {
  return _this->getScreenPitch();
}

size_t vizdoom_getScreenSize(DoomGame* _this) {
  return _this->getScreenSize();
}

// Load config
bool vizdoom_loadConfig(DoomGame* _this, const char* filename) {
  return _this->loadConfig(filename);
}
