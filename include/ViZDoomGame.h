/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#ifndef __VIZDOOM_GAME_H__
#define __VIZDOOM_GAME_H__

#include "ViZDoomDefines.h"

#include <list>
#include <string>
#include <vector>

namespace vizdoom {

    class DoomController;

    class DoomGame {

    public:

        DoomGame();
        virtual ~DoomGame();


        /* Flow Control */
        /*------------------------------------------------------------------------------------------------------------*/

        /*
         * Initializes the environment and spawns the ViZDoom engine process and starts the game.
         * Configuration cannot be changed after calling method.
         * Init returns true when the game was started properly and false otherwise.
         */
        bool init();

        /*
         * Closes ViZDoom engine and corresponding processes.
         * It is automatically invoked by the destructor.
         * Game can be initialized again after being closed.
         */
        void close();

        /*
         * Initializes a new episode. All rewards, gameVariables and map states are restarted.
         * After calling this method, first state from new episode will be available.
         */
        void newEpisode();

        /*
         * Checks if the ViZDoom game is running.
         */
        bool isRunning();

        /*
         * Sets the player's action for the next tics.
         * Each vector's value corresponds to a button specified with addAvailableButton method
         * or in configuration file (in order of appearance).
         */
        void setAction(std::vector<int> const &actions);

        /*
         * Processes a specified number of tics. If updateState is set the state will be updated after last processed tic
         * and a new reward will be calculated. To get new state use getState and to get the new reward use getLastReward.
         * If updateState is not set but renderOnly is turned on, the state will not be updated but a new frame
         * will be rendered after last processed tic. To get the new frame use getGameScreen.
         */
        void advanceAction(unsigned int tics, bool updateState, bool renderOnly);

        /*
         * Processes a specified number of tics, updates state and calculates a new reward.
         * Short for advanceAction(tics, true, false).
         */
        void advanceAction(unsigned int tics);

        /*
         * Processes one tic, updates the state and calculates a new reward.
         * Short for advanceAction(1, true, false).
         */
        void advanceAction();

        /*
         * Function combining usability of setAction, advanceAction and getLastReward.
         * Sets the player's action for the next tics, processes a specified number of tics,
         * updates the state and calculates a new reward, which is returned.
         */
        double makeAction(std::vector<int> const &actions, unsigned int tics);

        /*
         * Function combining usability of setAction, advanceAction and getLastReward.
         * Sets the player's action for the next tics, processes one tic, updates the state
         * and calculates a new reward, which is returned.
         * Short for makeAction(action, 1).
         */
        double makeAction(std::vector<int> const &actions);

        /*
         * Returns true if the current episode is in the initial state (no actions were performed yet).
         */
        bool isNewEpisode();

        /*
         * Returns true if the current episode is in the terminal state (is finished).
         * MakeAction and advanceAction methods will take no effect after this point (unless newEpisode method is called).
         */
        bool isEpisodeFinished();

        /*
         * Returns true if the player is in the terminal state (is finished).
         */
        bool isPlayerDead();

        /*
         * This method respawns player after death in multiplayer mode.
         * After calling this method, first state after respawn will be available.
         */
        void respawnPlayer();

        /*
         * Sends the command to Doom console. Can be used for cheats, multiplayer etc.
         * For more details consult ZDoom Wiki - http://zdoom.org/wiki/Console
         */
        void sendGameCommand(std::string cmd);

        /*
         * Returns DoomGame::State structure with the current game state.
         */
        GameState getState();

        /*
         * Returns a vector with the last action performed.
         * Each vector's value corresponds to a button added with addAvailableButton (in order of appearance).
         * Most useful in SPECTATOR mode.
         */
        std::vector<int> getLastAction();


        /* Buttons settings */
        /*------------------------------------------------------------------------------------------------------------*/

        /*
         * Makes the specified input type (e.g. TURN_LEFT, MOVE_FORWARD ) available (possible to use).
         * If the given button has already been added, the method has no effect.
         * If the specified button supports non-boolean values, no maximum value constraint is set.
         */
        void addAvailableButton(Button button);

        /*
         * Combines functionalities of addAvailableButton and setButtonMaxValue in one method.
         * Makes the specified input type available and sets the maximum allowed (absolute) value for it.
         * If the button has already been added the maximum value is overridden.
         */
        void addAvailableButton(Button button, int maxValue);

        /*
         * Clears all available buttons added so far.
         */
        void clearAvailableButtons();

        /*
         * Returns the number of available buttons.
         */
        int getAvailableButtonsSize();

        /*
         * Sets the maximum allowed (absolute) value for the specified button.
         * If the button has not been added yet using addAvailableButton, this method does not add it,
         * but the maximum value is set anyway.
         * Setting maximum value equal to 0 results in no constraint at all (infinity).
         * This method makes sense only for delta buttons.
         */
        void setButtonMaxValue(Button button, int maxValue);

        /*
         * Returns the maximum allowed (absolute) value for the specified button.
         */
        int getButtonMaxValue(Button button);


        /* GameVariables getters and setters */
        /*------------------------------------------------------------------------------------------------------------*/

        /*
         * Adds the specified GameVariable to the list of game variables (e.g. AMMO1, HEALTH, ATTACK\_READY)
         * that are included in the game's state (returned by getState method).
         */
        void addAvailableGameVariable(GameVariable var);

        /*
         * Clears the list of available game variables that are included in the game's state (returned by getState method).
         */
        void clearAvailableGameVariables();

        /*
         * Returns the number of available game variables.
         */
        int getAvailableGameVariablesSize();

        /*
         * Returns the current value of the specified game variable (AMMO1, HEALTH etc.).
         * The specified game variable does not need to be among available game variables (included in the state).
         * It could be used for e.g. shaping. Returns 0 in case of not finding given GameVariable.
         */
        int getGameVariable(GameVariable var);


        /* GameArgs getters and setters */
        /*------------------------------------------------------------------------------------------------------------*/

        /*
         * Adds a custom argument that will be passed to vizdoom process during initialization.
         * For more details consult ZDoom Wiki - http://zdoom.org/wiki/Command_line_parameters
         */
        void addGameArgs(std::string args);

        /*
         * Clears all arguments previously added with addGameArg method.
         */
        void clearGameArgs();


        /* Rewards getters and setters */
        /*------------------------------------------------------------------------------------------------------------*/

        /*
         * Returns the reward granted to the player after every tic.
         */
        double getLivingReward();

        /*
         * Sets the reward granted to the player after every tic. A negative value is also allowed.
         */
        void setLivingReward(double livingReward);

        /*
         * Returns the penalty for player's death.
         */
        double getDeathPenalty();

        /*
         * Sets a penalty for player's death. Note that in case of a negative value, the player will be rewarded upon dying.
         */
        void setDeathPenalty(double deathPenalty);

        /*
         * Returns a reward granted after last update of State.
         */
        double getLastReward();

        /*
         * Returns the sum of all rewards gathered in the current episode.
         */
        double getTotalReward();


        /* General game getters and setters */
        /*------------------------------------------------------------------------------------------------------------*/

        /*
         * Loads configuration (resolution, available buttons etc.) from a configuration file.
         * In case of multiple invocations, older configurations will be overwritten by the recent ones.
         * Overwriting does not involve resetting to default values, thus only overlapping parameters will be changed.
         * The method returns true if the whole configuration file was correctly read and applied,
         * false if the file was inaccessible or contained errors.
         */
        bool loadConfig(std::string filename);

        /*
         * Returns current mode.
         */
        Mode getMode();

        /*
         * Sets mode (e.g. PLAYER, SPECTATOR) in which the game will be started.
         */
        void setMode(Mode mode);

        /*
         * Sets path to ViZDoom engine executable.
         * Default value: "./vizdoom"
         */
        void setViZDoomPath(std::string path);

        /*
         * Sets path to the Doom engine based game file (wad format).
         * Default value: "./doom2.wad"
         */
        void setDoomGamePath(std::string path);

        /*
         * Sets path to additional scenario file (wad format).
         * Default value: ""
         */
        void setDoomScenarioPath(std::string path);

        /*
         * Sets the map name to be used.
         * Default value: "map01", if set to empty "map01" will be used.
         */
        void setDoomMap(std::string map);

        /*
         * Sets Doom game difficulty level which is called `skill' in Doom.
         * The higher is the skill the harder the game becomes.
         * Skill level affects monster' aggressiveness, monster's speed, weapon damage, ammunition quantities etc.
         * Default value: 3
         */
        void setDoomSkill(int skill);

        /*
         * Sets path for ViZDoom engine configuration file.
         * The file is responsible for configuration of Doom engine itself.
         * If it doesn't exist, it will be created after vizdoom executable is run.
         * This method is not needed for most of the tasks and is added for convenience of users with hacking tendencies.
         * Default value: "", if leave empty "vizdoom.ini" will be used.
         */
        void setDoomConfigPath(std::string path);

        /*
         * Return ViZDoom's game seed.
         */
        unsigned int getSeed();

        /*
         * Sets the seed of the ViZDoom's randomizing engine.
         */
        void setSeed(unsigned int seed);

        /*
         * Returns start delay of every episode in tics.
         */
        unsigned int getEpisodeStartTime();

        /*
         * Sets start delay of every episode in tics.
         * Every episode will effectively start (from the user's perspective) after given number of tics.
         */
        void setEpisodeStartTime(unsigned int tics);

        /*
         * Returns the number of tics after which the episode will be finished.
         */
        unsigned int getEpisodeTimeout();

        /*
         * Sets the number of tics after which the episode will be finished. 0 will result in no timeout.
         */
        void setEpisodeTimeout(unsigned int tics);

        /*
         * Returns number of current episode tic.
         */
        unsigned int getEpisodeTime();


        /* Output getters and setters */
        /*------------------------------------------------------------------------------------------------------------*/

        /*
         * Sets the screen resolution.
         * Supported resolutions are part of ScreenResolution enumeration (e.g. RES_320X240, RES_1920X1080).
         * The buffer as well as content of ViZDoom's display window will be affected.
         */
        void setScreenResolution(ScreenResolution resolution);

        /*
         * Returns the format of the screen buffer.
         */
        ScreenFormat getScreenFormat();

        /*
         * Sets the format of the screen buffer.
         * Supported formats are defined in ScreenFormat enumeration type (e.g. CRCGCB, CRCGCBDB, RGB24, GRAY8).
         * The format change affects only the buffer so it will not have any effect on
         * the content of ViZDoom's display window.
         */
        void setScreenFormat(ScreenFormat format);

        /*
         * Determine if game's hud will be rendered in game.
         */
        void setRenderHud(bool hud);

        /*
         * Determine if weapon held by player will be rendered in game.
         */
        void setRenderWeapon(bool weapon);

        /*
         * Determine if crosshair will be rendered in game.
         */
        void setRenderCrosshair(bool crosshair);

        /*
         * Determine if decals (marks on the walls) will be rendered in game.
         */
        void setRenderDecals(bool decals);

        /*
         * Determine if particles will be rendered in game.
         */
        void setRenderParticles(bool particles);

        /*
         * Determines if ViZDoom's window will be visible.
         * ViZDoom with window disabled can be used on Linux system without X Server.
         */
        void setWindowVisible(bool visibility);

        /*
         * Determines if ViZDoom's console output will be enabled.
         */
        void setConsoleEnabled(bool console);

        /*
         * Determines if ViZDoom's sound will be played.
         */
        void setSoundEnabled(bool sound);

        /*
         * Returns game's screen width.
         */
        int getScreenWidth();

        /*
         * Returns game's screen height.
         */
        int getScreenHeight();

        /*
         * Returns number of channels in game's screen buffer.
         */
        int getScreenChannels();

        /*
         * Returns size in bytes of one row in game's screen buffer.
         */
        size_t getScreenPitch();

        /*
         * Returns size in bytes of game's screen buffer.
         */
        size_t getScreenSize();

        /*
         * Returns a pointer to the raw screen buffer.
         */
        uint8_t * const getGameScreen();

    protected:

        DoomController *doomController;

        /* Game state and actions */
        /*------------------------------------------------------------------------------------------------------------*/

        bool running;

        Mode mode;

        GameState state;
        void updateState();

        std::vector <GameVariable> availableGameVariables;
        std::vector <Button> availableButtons;
        std::vector<int> lastAction;

        unsigned int nextStateNumber;
        unsigned int lastMapTic;
        unsigned int seed;

        /* Rewards */
        /*------------------------------------------------------------------------------------------------------------*/

        double lastReward;
        double lastMapReward;
        double summaryReward;

        double livingReward;
        double deathPenalty;

    private:

        /* Load config helpers */
        /*------------------------------------------------------------------------------------------------------------*/

        static bool StringToBool(std::string boolString);
        static ScreenResolution StringToResolution(std::string str);
        static ScreenFormat StringToFormat(std::string str);
        static Button StringToButton(std::string str);
        static GameVariable StringToGameVariable(std::string str);
        static unsigned int StringToUint(std::string str);
        static bool ParseListProperty(int &line_number, std::string &value, std::ifstream& input, std::vector<std::string> &output);

    };
}

#endif
