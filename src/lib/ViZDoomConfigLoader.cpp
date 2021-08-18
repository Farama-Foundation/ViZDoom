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

#include "ViZDoomConfigLoader.h"
#include "ViZDoomExceptions.h"
#include "ViZDoomPathHelpers.h"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim_all.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <fstream>
#include <iostream>

// Why there aren't any beautiful macros here...

namespace vizdoom {

    namespace b = boost;
    using namespace b::algorithm;

    ConfigLoader::ConfigLoader(DoomGame *game) : game(game) {

    }

    ConfigLoader::~ConfigLoader() {

    }

    bool ConfigLoader::stringToBool(std::string boolString) {
        if (boolString == "true" || boolString == "1") return true;
        if (boolString == "false" || boolString == "0") return false;

        throw std::exception();
    }

    int ConfigLoader::stringToInt(std::string str) {
        int value = b::lexical_cast<int>(str);
        return value;
    }

    unsigned int ConfigLoader::stringToUint(std::string str) {
        unsigned int value = b::lexical_cast <unsigned int> (str);
        if (str[0] == '-') throw b::bad_lexical_cast();
        return value;
    }

    ScreenResolution ConfigLoader::stringToResolution(std::string str) {

        if (str == "res_160x120") return RES_160X120;

        if (str == "res_200x125") return RES_200X125;
        if (str == "res_200x150") return RES_200X150;

        if (str == "res_256x144") return RES_256X144;
        if (str == "res_256x160") return RES_256X160;
        if (str == "res_256x192") return RES_256X192;

        if (str == "res_320x180") return RES_320X180;
        if (str == "res_320x200") return RES_320X200;
        if (str == "res_320x240") return RES_320X240;
        if (str == "res_320x256") return RES_320X256;

        if (str == "res_400x225") return RES_400X225;
        if (str == "res_400x250") return RES_400X250;
        if (str == "res_400x300") return RES_400X300;

        if (str == "res_512x288") return RES_512X288;
        if (str == "res_512x320") return RES_512X320;
        if (str == "res_512x384") return RES_512X384;

        if (str == "res_640x360") return RES_640X360;
        if (str == "res_640x400") return RES_640X400;
        if (str == "res_640x480") return RES_640X480;

        if (str == "res_800x450") return RES_800X450;
        if (str == "res_800x500") return RES_800X500;
        if (str == "res_800x600") return RES_800X600;

        if (str == "res_1024x576") return RES_1024X576;
        if (str == "res_1024x640") return RES_1024X640;
        if (str == "res_1024x768") return RES_1024X768;

        if (str == "res_1280x720") return RES_1280X720;
        if (str == "res_1280x800") return RES_1280X800;
        if (str == "res_1280x960") return RES_1280X960;
        if (str == "res_1280x1024") return RES_1280X1024;

        if (str == "res_1400x787") return RES_1400X787;
        if (str == "res_1400x875") return RES_1400X875;
        if (str == "res_1400x1050") return RES_1400X1050;

        if (str == "res_1600x900") return RES_1600X900;
        if (str == "res_1600x1000") return RES_1600X1000;
        if (str == "res_1600x1200") return RES_1600X1200;

        if (str == "res_1920x1080") return RES_1920X1080;

        throw std::exception();
    }

    ScreenFormat ConfigLoader::stringToFormat(std::string str) {
        if (str == "crcgcb") return CRCGCB;
        if (str == "rgb24") return RGB24;
        if (str == "rgba32") return RGBA32;
        if (str == "argb32") return ARGB32;
        if (str == "cbcgcr") return CBCGCR;
        if (str == "bgr24") return BGR24;
        if (str == "bgra32") return BGRA32;
        if (str == "abgr32") return ABGR32;
        if (str == "gray8") return GRAY8;
        if (str == "doom_256_colors8") return DOOM_256_COLORS8;

        throw std::exception();
    }

    SamplingRate ConfigLoader::stringToSamplingRate(std::string str) {
        if (str == "sr_11025") return SR_11025;
        if (str == "sr_22050") return SR_22050;
        if (str == "sr_44100") return SR_44100;

        throw std::exception();
    }

    Button ConfigLoader::stringToButton(std::string str) {
        if (str == "attack") return ATTACK;
        if (str == "use") return USE;
        if (str == "jump") return JUMP;
        if (str == "crouch") return CROUCH;
        if (str == "turn180") return TURN180;
        if (str == "altattack") return ALTATTACK;
        if (str == "reload") return RELOAD;
        if (str == "zoom") return ZOOM;
        if (str == "speed") return SPEED;
        if (str == "strafe") return STRAFE;
        if (str == "move_right") return MOVE_RIGHT;
        if (str == "move_left") return MOVE_LEFT;
        if (str == "move_backward") return MOVE_BACKWARD;
        if (str == "move_forward") return MOVE_FORWARD;
        if (str == "turn_right") return TURN_RIGHT;
        if (str == "turn_left") return TURN_LEFT;
        if (str == "look_up") return LOOK_UP;
        if (str == "look_down") return LOOK_DOWN;
        if (str == "move_up") return MOVE_UP;
        if (str == "move_down") return MOVE_DOWN;
        if (str == "land") return LAND;

        if (str == "select_weapon1") return SELECT_WEAPON1;
        if (str == "select_weapon2") return SELECT_WEAPON2;
        if (str == "select_weapon3") return SELECT_WEAPON3;
        if (str == "select_weapon4") return SELECT_WEAPON4;
        if (str == "select_weapon5") return SELECT_WEAPON5;
        if (str == "select_weapon6") return SELECT_WEAPON6;
        if (str == "select_weapon7") return SELECT_WEAPON7;
        if (str == "select_weapon8") return SELECT_WEAPON8;
        if (str == "select_weapon9") return SELECT_WEAPON9;
        if (str == "select_weapon0") return SELECT_WEAPON0;

        if (str == "select_next_weapon") return SELECT_NEXT_WEAPON;
        if (str == "select_prev_weapon") return SELECT_PREV_WEAPON;
        if (str == "drop_selected_weapon") return DROP_SELECTED_WEAPON;
        if (str == "activate_selected_weapon") return ACTIVATE_SELECTED_ITEM;
        if (str == "select_next_item") return SELECT_NEXT_ITEM;
        if (str == "select_prev_item") return SELECT_PREV_ITEM;
        if (str == "drop_selected_item") return DROP_SELECTED_ITEM;

        if (str == "look_up_down_delta") return LOOK_UP_DOWN_DELTA;
        if (str == "turn_left_right_delta") return TURN_LEFT_RIGHT_DELTA;
        if (str == "move_forward_backward_delta")return MOVE_FORWARD_BACKWARD_DELTA;
        if (str == "move_left_right_delta") return MOVE_LEFT_RIGHT_DELTA;
        if (str == "move_up_down_delta") return MOVE_UP_DOWN_DELTA;

        throw std::exception();
    }

    GameVariable ConfigLoader::stringToGameVariable(std::string str) {
        if (str == "killcount") return KILLCOUNT;
        if (str == "itemcount") return ITEMCOUNT;
        if (str == "secretcount") return SECRETCOUNT;
        if (str == "fragcount") return FRAGCOUNT;
        if (str == "deathcount") return DEATHCOUNT;
        if (str == "hitcount") return HITCOUNT;
        if (str == "hits_taken") return HITS_TAKEN;
        if (str == "damagecount") return DAMAGECOUNT;
        if (str == "damage_taken") return DAMAGE_TAKEN;
        if (str == "health") return HEALTH;
        if (str == "armor") return ARMOR;
        if (str == "dead") return DEAD;
        if (str == "on_ground") return ON_GROUND;
        if (str == "attack_ready") return ATTACK_READY;
        if (str == "altattack_ready") return ALTATTACK_READY;
        if (str == "selected_weapon") return SELECTED_WEAPON;
        if (str == "selected_weapon_ammo") return SELECTED_WEAPON_AMMO;

        if (str == "ammo1") return AMMO1;
        if (str == "ammo2") return AMMO2;
        if (str == "ammo3") return AMMO3;
        if (str == "ammo4") return AMMO4;
        if (str == "ammo5") return AMMO5;
        if (str == "ammo6") return AMMO6;
        if (str == "ammo7") return AMMO7;
        if (str == "ammo8") return AMMO8;
        if (str == "ammo9") return AMMO9;
        if (str == "ammo0") return AMMO0;

        if (str == "weapon1") return WEAPON1;
        if (str == "weapon2") return WEAPON2;
        if (str == "weapon3") return WEAPON3;
        if (str == "weapon4") return WEAPON4;
        if (str == "weapon5") return WEAPON5;
        if (str == "weapon6") return WEAPON6;
        if (str == "weapon7") return WEAPON7;
        if (str == "weapon8") return WEAPON8;
        if (str == "weapon9") return WEAPON9;
        if (str == "weapon0") return WEAPON0;

        if (str == "user1") return USER1;
        if (str == "user2") return USER2;
        if (str == "user3") return USER3;
        if (str == "user4") return USER4;
        if (str == "user5") return USER5;
        if (str == "user6") return USER6;
        if (str == "user7") return USER7;
        if (str == "user8") return USER8;
        if (str == "user9") return USER9;
        if (str == "user10") return USER10;
        if (str == "user11") return USER11;
        if (str == "user12") return USER12;
        if (str == "user13") return USER13;
        if (str == "user14") return USER14;
        if (str == "user15") return USER15;
        if (str == "user16") return USER16;
        if (str == "user17") return USER17;
        if (str == "user18") return USER18;
        if (str == "user19") return USER19;
        if (str == "user20") return USER20;
        if (str == "user21") return USER21;
        if (str == "user22") return USER22;
        if (str == "user23") return USER23;
        if (str == "user24") return USER24;
        if (str == "user25") return USER25;
        if (str == "user26") return USER26;
        if (str == "user27") return USER27;
        if (str == "user28") return USER28;
        if (str == "user29") return USER29;
        if (str == "user30") return USER30;
        if (str == "user31") return USER31;
        if (str == "user32") return USER32;
        if (str == "user33") return USER33;
        if (str == "user34") return USER34;
        if (str == "user35") return USER35;
        if (str == "user36") return USER36;
        if (str == "user37") return USER37;
        if (str == "user38") return USER38;
        if (str == "user39") return USER39;
        if (str == "user40") return USER40;
        if (str == "user41") return USER41;
        if (str == "user42") return USER42;
        if (str == "user43") return USER43;
        if (str == "user44") return USER44;
        if (str == "user45") return USER45;
        if (str == "user46") return USER46;
        if (str == "user47") return USER47;
        if (str == "user48") return USER48;
        if (str == "user49") return USER49;
        if (str == "user50") return USER50;
        if (str == "user51") return USER51;
        if (str == "user52") return USER52;
        if (str == "user53") return USER53;
        if (str == "user54") return USER54;
        if (str == "user55") return USER55;
        if (str == "user56") return USER56;
        if (str == "user57") return USER57;
        if (str == "user58") return USER58;
        if (str == "user59") return USER59;
        if (str == "user60") return USER60;

        if (str == "position_x") return POSITION_X;
        if (str == "position_y") return POSITION_Y;
        if (str == "position_z") return POSITION_Z;
        if (str == "angle") return ANGLE;
        if (str == "pitch") return PITCH;
        if (str == "roll") return ROLL;
        if(str == "view_height") return VIEW_HEIGHT;
        if (str == "velocity_x") return VELOCITY_X;
        if (str == "velocity_y") return VELOCITY_Y;
        if (str == "velocity_z") return VELOCITY_Z;

        if (str == "camera_position_x") return CAMERA_POSITION_X;
        if (str == "camera_position_y") return CAMERA_POSITION_Y;
        if (str == "camera_position_z") return CAMERA_POSITION_Z;
        if (str == "camera_angle") return CAMERA_ANGLE;
        if (str == "camera_pitch") return CAMERA_PITCH;
        if (str == "camera_roll") return CAMERA_ROLL;
        if (str == "camera_fov") return CAMERA_FOV;

        if (str == "player_number") return PLAYER_NUMBER;
        if (str == "player_count") return PLAYER_COUNT;

        if (str == "player1_fragcount") return PLAYER1_FRAGCOUNT;
        if (str == "player2_fragcount") return PLAYER2_FRAGCOUNT;
        if (str == "player3_fragcount") return PLAYER3_FRAGCOUNT;
        if (str == "player4_fragcount") return PLAYER4_FRAGCOUNT;
        if (str == "player5_fragcount") return PLAYER5_FRAGCOUNT;
        if (str == "player6_fragcount") return PLAYER6_FRAGCOUNT;
        if (str == "player7_fragcount") return PLAYER7_FRAGCOUNT;
        if (str == "player8_fragcount") return PLAYER8_FRAGCOUNT;
        if (str == "player9_fragcount") return PLAYER9_FRAGCOUNT;
        if (str == "player10_fragcount") return PLAYER10_FRAGCOUNT;
        if (str == "player11_fragcount") return PLAYER11_FRAGCOUNT;
        if (str == "player12_fragcount") return PLAYER12_FRAGCOUNT;
        if (str == "player13_fragcount") return PLAYER13_FRAGCOUNT;
        if (str == "player14_fragcount") return PLAYER14_FRAGCOUNT;
        if (str == "player15_fragcount") return PLAYER15_FRAGCOUNT;
        if (str == "player16_fragcount") return PLAYER16_FRAGCOUNT;

        throw std::exception();
    }

    typedef b::tokenizer<b::char_separator<char> > tokenizer;

    bool ConfigLoader::parseListProperty(int &line_number, std::string &value, std::ifstream &input,
                                         std::vector<std::string> &output) {
        using namespace b::algorithm;
        int start_line = line_number;
        /* Find '{' */
        while (value.empty()) {
            if (!input.eof()) {
                ++line_number;
                std::getline(input, value);
                trim_all(value);
                if (!value.empty() && value[0] == '#')
                    value = "";
            } else
                break;
        }
        if (value.empty() || value[0] != '{') return false;

        value = value.substr(1);

        /* Find '}' */
        while ((value.empty() || value[value.size() - 1] != '}') && !input.eof()) {
            ++line_number;
            std::string newline;
            std::getline(input, newline);
            trim_all(newline);
            if (!newline.empty() && newline[0] != '#')
                value += std::string(" ") + newline;
        }
        if (value.empty() || value[value.size() - 1] != '}') return false;

        /* Fill the vector */
        value[value.size() - 1] = ' ';
        trim_all(value);
        to_lower(value);

        b::char_separator<char> separator(" ");
        tokenizer tok(value, separator);
        for (tokenizer::iterator it = tok.begin(); it != tok.end(); ++it) {
            output.push_back(*it);
        }
        return true;
    }

    bool ConfigLoader::load(std::string filePath) {

        std::string scenarioName = filePath;
        trim_all(scenarioName);
        to_lower(scenarioName);
        replace_all(scenarioName, " ", "_");
        if(!ends_with(scenarioName, ".cfg")) {
            scenarioName += ".cfg";
        }
        std::string workingConfigPath = "./scenarios/" + scenarioName;
        std::string sharedConfigPath = getThisSharedObjectPath() + "/scenarios/" + scenarioName;

        // Check if scenario exists in library's scenerios directory
        if (fileExists(filePath)) this->filePath = filePath;
        else if (fileExists(workingConfigPath)) this->filePath = workingConfigPath;
        else if (fileExists(sharedConfigPath)) this->filePath = sharedConfigPath;
        else throw FileDoesNotExistException(filePath + " | " + workingConfigPath + " | " + sharedConfigPath);

        bool success = true;
        std::ifstream file(filePath);


        std::string line;
        int lineNumber = 0;

        /* Process every line. */
        while (!file.eof()) {
            ++lineNumber;

            std::getline(file, line);

            /* Ignore empty and comment lines */
            trim_all(line);

            if (line.empty() || line[0] == '#') {
                continue;
            }

            bool append = false; //it looks for +=

            /* Check if '=' is there */
            size_t equals_sign_pos = line.find_first_of('=');
            size_t append_sign_pos = line.find("+=");

            std::string key;
            std::string val;
            std::string rawVal;
            if (append_sign_pos != std::string::npos) {
                key = line.substr(0, append_sign_pos);
                val = line.substr(append_sign_pos + 2);
                append = true;
            } else if (equals_sign_pos != std::string::npos) {
                key = line.substr(0, equals_sign_pos);
                val = line.substr(equals_sign_pos + 1);
            } else {
                std::cerr << "WARNING! Loading config from: \"" << filePath << "\". Syntax error in line #" <<
                          lineNumber << ". Line ignored.\n";

                success = false;
                continue;
            }


            rawVal = val;
            trim_all(key);
            trim_all(val);
            std::string originalVal = val;
            to_lower(val);
            to_lower(key);
            if (key.empty()) {
                std::cerr << "WARNING! Loading config from: \"" << filePath << "\". Empty key in line #" <<
                          lineNumber << ". Line ignored.\n";

                success = false;
                continue;
            }


            /* Parse enum list properties */

            if (key == "available_buttons" || key == "availablebuttons") {
                std::vector<std::string> strButtons;
                int start_line = lineNumber;
                bool parseSuccess = ConfigLoader::parseListProperty(lineNumber, val, file, strButtons);
                if (parseSuccess) {
                    unsigned int i = 0;
                    try {
                        std::vector<Button> buttons;
                        for (i = 0; i < strButtons.size(); ++i) {
                            buttons.push_back(ConfigLoader::stringToButton(strButtons[i]));

                        }
                        if (!append)
                            this->game->clearAvailableButtons();
                        for (i = 0; i < buttons.size(); ++i) {
                            this->game->addAvailableButton(buttons[i]);
                        }
                    }
                    catch (std::exception) {
                        std::cerr << "WARNING! Loading config from: \"" << filePath <<
                                  "\". Unsupported value in lines " << start_line << "-" << lineNumber << ": " <<
                                  strButtons[i] << ". Lines ignored.\n";

                        success = false;
                    }
                } else {
                    std::cerr << "WARNING! Loading config from: \"" << filePath << "\". Syntax error in lines " <<
                              start_line << "-" << lineNumber << ". Lines ignored.\n";

                    success = false;
                }

                continue;
            }

            if (key == "available_game_variables" || key == "availablegamevariables") {
                std::vector<std::string> str_variables;
                int start_line = lineNumber;
                bool parseSuccess = ConfigLoader::parseListProperty(lineNumber, val, file, str_variables);
                if (parseSuccess) {
                    unsigned int i = 0;
                    try {
                        std::vector<GameVariable> variables;
                        for (i = 0; i < str_variables.size(); ++i) {
                            variables.push_back(ConfigLoader::stringToGameVariable(str_variables[i]));

                        }
                        if (!append)
                            this->game->clearAvailableGameVariables();
                        for (i = 0; i < variables.size(); ++i) {
                            this->game->addAvailableGameVariable(variables[i]);
                        }
                    }
                    catch (std::exception) {
                        std::cerr << "WARNING! Loading config from: \"" << filePath <<
                                  "\". Unsupported value in lines " << start_line << "-" << lineNumber << ": " <<
                                  str_variables[i] << ". Lines ignored.\n";

                        success = false;
                    }
                } else {
                    std::cerr << "WARNING! Loading config from: \"" << filePath << "\". Syntax error in lines " <<
                              start_line << "-" << lineNumber << ". Lines ignored.\n";

                    success = false;
                }

                continue;
            }

            /* Parse game args which are string but enables "+=" */
            if (key == "game_args" || key == "gameargs") {
                if (!append) {
                    this->game->clearGameArgs();
                }
                this->game->addGameArgs(originalVal);
                continue;
            }

            /* Check if "+=" was not used for non-list property */
            if (append) {
                std::cerr << "WARNING! Loading config from: \"" << filePath <<
                          "\". \"+=\" is not supported for non-list properties. Line #" << lineNumber << " ignored.\n";

                success = false;
                continue;
            }


            /* Check if value is not empty */
            if (val.empty()) {
                std::cerr << "WARNING! Loading config from: \"" << filePath << "\". Empty value in line #" <<
                          lineNumber << ". Line ignored.\n";

                success = false;
                continue;
            }

            /* Parse int properties */
            try {
                if (key == "seed") {
                    this->game->setSeed(stringToUint(val));
                    continue;
                }
                if (key == "episode_timeout" || key == "episodetimeout") {
                    this->game->setEpisodeTimeout(stringToUint(val));
                    continue;
                }
                if (key == "episode_start_time" || key == "episodestarttime") {
                    this->game->setEpisodeStartTime(stringToUint(val));
                    continue;
                }
                if (key == "doom_skill" || key == "doomskill") {
                    this->game->setDoomSkill(stringToUint(val));
                    continue;
                }
                if (key == "ticrate") {
                    this->game->setTicrate(stringToUint(val));
                    continue;
                }
                if (key == "audio_buffer_size" || key == "audiobuffersize") {
                    this->game->setAudioBufferSize(stringToUint(val));
                    continue;
                }
            }
            catch (b::bad_lexical_cast &) {
                std::cerr << "WARNING! Loading config from: \"" << filePath <<
                          "\". Unsigned int value expected instead of: " << rawVal << " in line #" << lineNumber <<
                          ". Line ignored.\n";

                success = false;
                continue;
            }

            /* Parse float properties */
            try {
                if (key == "living_reward" || key == "livingreward") {
                    this->game->setLivingReward(b::lexical_cast<double>(val));
                    continue;
                }
                if (key == "death_penalty" || key == "deathpenalty") {
                    this->game->setDeathPenalty(b::lexical_cast<double>(val));
                    continue;
                }
            }
            catch (b::bad_lexical_cast &) {
                std::cerr << "WARNING! Loading config from: \"" << filePath << "\". Float value expected instead of: " <<
                          rawVal << " in line #" << lineNumber << ". Line ignored.\n";

                success = false;
                continue;
            }

            /* Parse string properties */
            if (key == "doom_map" || key == "doommap") {
                this->game->setDoomMap(val);
                continue;
            }
            if (key == "vizdoom_path" || key == "vizdoompath") {
                this->game->setViZDoomPath(relativePath(originalVal, this->filePath));
                continue;
            }
            if (key == "doom_game_path" || key == "doomgamepath") {
                this->game->setDoomGamePath(relativePath(originalVal, this->filePath));
                continue;
            }
            if (key == "doom_scenario_path" || key == "doomscenariopath") {
                this->game->setDoomScenarioPath(relativePath(originalVal, this->filePath));
                continue;
            }
            if (key == "doom_config_path" || key == "doomconfigpath") {
                this->game->setDoomConfigPath(relativePath(originalVal, this->filePath));
                continue;
            }

            /* Parse bool properties */
            try {
                if (key == "depth_buffer_enabled" || key == "depthbufferenabled") {
                    this->game->setDepthBufferEnabled(stringToBool(val));
                    continue;
                }
                if (key == "labels_buffer_enabled" || key == "labelsbufferenabled") {
                    this->game->setLabelsBufferEnabled(stringToBool(val));
                    continue;
                }
                if (key == "automap_buffer_enabled" || key == "automapbufferenabled") {
                    this->game->setAutomapBufferEnabled(stringToBool(val));
                    continue;
                }
                if (key == "automap_rotate" || key == "automaprotate") {
                    this->game->setAutomapRotate(stringToBool(val));
                    continue;
                }
                if (key == "automap_render_textures" || key == "automaprendertextures") {
                    this->game->setAutomapRenderTextures(stringToBool(val));
                    continue;
                }
                if (key == "objects_info_enabled" || key == "objectsinfoenabled") {
                    this->game->setObjectsInfoEnabled(stringToBool(val));
                    continue;
                }
                if (key == "sectors_info_enabled" || key == "sectorsinfoenabled") {
                    this->game->setSectorsInfoEnabled(stringToBool(val));
                    continue;
                }
                if (key == "render_hud" || key == "renderhud") {
                    this->game->setRenderHud(stringToBool(val));
                    continue;
                }
                if (key == "render_minimal_hud" || key == "renderminimalhud") {
                    this->game->setRenderMinimalHud(stringToBool(val));
                    continue;
                }
                if (key == "render_weapon" || key == "renderweapon") {
                    this->game->setRenderWeapon(stringToBool(val));
                    continue;
                }
                if (key == "render_crosshair" || key == "rendercrosshair") {
                    this->game->setRenderCrosshair(stringToBool(val));
                    continue;
                }
                if (key == "render_decals" || key == "renderdecals") {
                    this->game->setRenderDecals(stringToBool(val));
                    continue;
                }
                if (key == "render_particles" || key == "renderparticles") {
                    this->game->setRenderParticles(stringToBool(val));
                    continue;
                }
                if (key == "render_effects_sprites" || key == "rendereffectssprites") {
                    this->game->setRenderEffectsSprites(stringToBool(val));
                    continue;
                }
                if (key == "render_messages" || key == "rendermessages") {
                    this->game->setRenderMessages(stringToBool(val));
                    continue;
                }
                if (key == "render_corpses" || key == "rendercorpses") {
                    this->game->setRenderCorpses(stringToBool(val));
                    continue;
                }
                if (key == "render_screen_flashes" || key == "renderscreenflashes") {
                    this->game->setRenderScreenFlashes(stringToBool(val));
                    continue;
                }
                if (key == "render_all_frames" || key == "renderallframes") {
                    this->game->setRenderAllFrames(stringToBool(val));
                    continue;
                }
                if (key == "window_visible" || key == "windowvisible") {
                    this->game->setWindowVisible(stringToBool(val));
                    continue;
                }
                if (key == "console_enabled" || key == "consoleenabled") {
                    this->game->setConsoleEnabled(stringToBool(val));
                    continue;
                }
                if (key == "sound_enabled" || key == "soundenabled") {
                    this->game->setSoundEnabled(stringToBool(val));
                    continue;
                }
                if (key == "audio_buffer_enabled" || key == "audiobufferenabled") {
                    this->game->setAudioBufferEnabled(stringToBool(val));
                    continue;
                }
            }
            catch (std::exception) {
                std::cerr << "WARNING! Loading config from: \"" << filePath <<
                          "\". Boolean value expected insted of: " << rawVal << " in line #" << lineNumber <<
                          ". Line ignored.\n";

                success = false;
                continue;

            }

            /* Parse enum properties */

            if (key == "mode") {
                if (val == "player") {
                    this->game->setMode(PLAYER);
                    continue;
                }
                if (val == "spectator") {
                    this->game->setMode(SPECTATOR);
                    continue;
                }
                if (val == "async_player") {
                    this->game->setMode(ASYNC_PLAYER);
                    continue;
                }
                if (val == "async_spectator") {
                    this->game->setMode(ASYNC_SPECTATOR);
                    continue;
                }

                std::cerr << "WARNING! Loading config from: \"" << filePath <<
                          "\". (ASYNC_)SPECTATOR || PLAYER expected instead of: " << rawVal << " in line #"
                          << lineNumber <<
                          ". Line ignored.\n";

                success = false;
                continue;
            }

            if (key == "automap_mode" || key == "automapmode") {
                if (val == "normal") {
                    this->game->setAutomapMode(NORMAL);
                    continue;
                }
                if (val == "whole") {
                    this->game->setAutomapMode(WHOLE);
                    continue;
                }
                if (val == "objects") {
                    this->game->setAutomapMode(OBJECTS);
                    continue;
                }
                if (val == "objects_with_size") {
                    this->game->setAutomapMode(OBJECTS_WITH_SIZE);
                    continue;
                }

                std::cerr << "WARNING! Loading config from: \"" << filePath <<
                          "\". NORMAL || WHOLE || OBJECTS || OBJECTS_WITH_SIZE expected instead of: " << rawVal <<
                          " in line #" << lineNumber << ". Line ignored.\n";

                success = false;
                continue;
            }

            try {
                if (key == "screen_resolution" || key == "screenresolution") {
                    this->game->setScreenResolution(stringToResolution(val));
                    continue;
                }
                if (key == "screen_format" || key == "screenformat") {
                    this->game->setScreenFormat(stringToFormat(val));
                    continue;
                }
                if (key == "audio_sampling_rate" || key == "audiosamplingrate") {
                    this->game->setAudioSamplingRate(stringToSamplingRate(val));
                    continue;
                }
                if (key == "button_max_value" || key == "buttonmaxvalue") {
                    size_t space = val.find_first_of(' ');

                    if (space == std::string::npos) throw std::exception();

                    Button button = ConfigLoader::stringToButton(val.substr(0, space));
                    val = val.substr(space + 1);
                    unsigned int maxValue = b::lexical_cast < unsigned
                    int > (val);

                    if (val[0] == '-') throw b::bad_lexical_cast();

                    this->game->setButtonMaxValue(button, maxValue);
                    continue;
                }
            }
            catch (std::exception &) {
                std::cerr << "WARNING! Loading config from: \"" << filePath << "\". Unsupported value: " << rawVal <<
                          " in line #" << lineNumber << ". Line ignored.\n";

                success = false;
                continue;
            }

            std::cerr << "WARNING! Loading config from: \"" << filePath << "\". Unsupported key: " << key <<
                      " in line #" << lineNumber << ". Line ignored.\n";

            success = false;
        }

        file.close();

        return success;
    }
}