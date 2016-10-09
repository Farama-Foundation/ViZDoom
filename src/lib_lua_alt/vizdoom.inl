typedef unsigned char uint8_t;

DoomGame* vizdoom_new();

void vizdoom_gc(DoomGame*);
bool vizdoom_init(DoomGame*);

void vizdoom_close(DoomGame*);
bool vizdoom_isRunning(DoomGame*);

void vizdoom_newEpisode(DoomGame*);
void vizdoom_newEpisode_path(DoomGame*, const char*);
void vizdoom_replayEpisode(DoomGame*, const char*);

void vizdoom_setAction(DoomGame*, THIntTensor*);
void vizdoom_advanceAction(DoomGame*, unsigned int);
void vizdoom_advanceActionOneTic(DoomGame*);
double vizdoom_makeAction(DoomGame*, THIntTensor*);
double vizdoom_makeAction_byTics(DoomGame*, THIntTensor*, unsigned int);

// DoomGame::updateState() is protected

int vizdoom_getState(DoomGame*, THIntTensor*, THByteTensor*);
void vizdoom_getLastAction(DoomGame*, THIntTensor*);

bool vizdoom_isNewEpisode(DoomGame*);
bool vizdoom_isEpisodeFinished(DoomGame*);
bool vizdoom_isPlayerDead(DoomGame*);
void vizdoom_respawnPlayer(DoomGame*);

void vizdoom_addAvailableButton(DoomGame*, int);
void vizdoom_addAvailableButton_bm(DoomGame*, int, unsigned int);
void vizdoom_clearAvailableButtons(DoomGame*);
int vizdoom_getAvailableButtonsSize(DoomGame*);
void vizdoom_setButtonMaxValue(DoomGame*, int, unsigned int);
int vizdoom_getButtonMaxValue(DoomGame*, int);

void vizdoom_addAvailableGameVariable(DoomGame*, int);
void vizdoom_clearAvailableGameVariables(DoomGame*);
int vizdoom_getAvailableGameVariablesSize(DoomGame*);
void vizdoom_addGameArgs(DoomGame*, const char*);
void vizdoom_clearGameArgs(DoomGame*);

void vizdoom_sendGameCommand(DoomGame*, const char*);

double vizdoom_getLivingReward(DoomGame*);
void vizdoom_setLivingReward(DoomGame*, double);
double vizdoom_getDeathPenalty(DoomGame*);
void vizdoom_setDeathPenalty(DoomGame*, double);
double vizdoom_getLastReward(DoomGame*);
double vizdoom_getTotalReward(DoomGame*);

bool vizdoom_loadConfig(DoomGame*, const char*);

int vizdoom_getMode(DoomGame*);
void vizdoom_setMode(DoomGame*, int);

unsigned int vizdoom_getTicrate(DoomGame*);
void vizdoom_setTicrate(DoomGame*, unsigned int);
int vizdoom_getGameVariable(DoomGame*, int);

void vizdoom_setViZDoomPath(DoomGame*, const char*);
void vizdoom_setDoomGamePath(DoomGame*, const char*);
void vizdoom_setDoomScenarioPath(DoomGame*, const char*);
void vizdoom_setDoomMap(DoomGame*, const char*);
void vizdoom_setDoomSkill(DoomGame*, int);
void vizdoom_setDoomConfigPath(DoomGame*, const char*);

unsigned int vizdoom_getSeed(DoomGame*);
void vizdoom_setSeed(DoomGame*, unsigned int);

unsigned int vizdoom_getEpisodeStartTime(DoomGame*);
void vizdoom_setEpisodeStartTime(DoomGame*, unsigned int);
unsigned int vizdoom_getEpisodeTimeout(DoomGame*);
void vizdoom_setEpisodeTimeout(DoomGame*, unsigned int);
unsigned int vizdoom_getEpisodeTime(DoomGame*);

void vizdoom_setScreenResolution(DoomGame*, int);
unsigned int vizdoom_getScreenFormat(DoomGame*);
void vizdoom_setScreenFormat(DoomGame*, int);
void vizdoom_setRenderHud(DoomGame*, bool);
void vizdoom_setRenderWeapon(DoomGame*, bool);
void vizdoom_setRenderCrosshair(DoomGame*, bool);
void vizdoom_setRenderDecals(DoomGame*, bool);
void vizdoom_setRenderParticles(DoomGame*, bool);
void vizdoom_setWindowVisible(DoomGame*, bool);
void vizdoom_setConsoleEnabled(DoomGame*, bool);
void vizdoom_setSoundEnabled(DoomGame*, bool);

int vizdoom_getScreenWidth(DoomGame*);
int vizdoom_getScreenHeight(DoomGame*);
int vizdoom_getScreenChannels(DoomGame*);
size_t vizdoom_getScreenPitch(DoomGame*);
size_t vizdoom_getScreenSize(DoomGame*);
void vizdoom_getGameScreen(DoomGame*, THByteTensor*);
