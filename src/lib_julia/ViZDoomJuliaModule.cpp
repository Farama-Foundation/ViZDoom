#include "ViZDoom.h"
#include <jlcxx/jlcxx.hpp>

using namespace vizdoom;

namespace jlcxx
{
template <>
struct IsBits<Mode> : std::true_type
{
};
template <>
struct IsBits<ScreenFormat> : std::true_type
{
};
template <>
struct IsBits<ScreenResolution> : std::true_type
{
};
template <>
struct IsBits<AutomapMode> : std::true_type
{
};
template <>
struct IsBits<Button> : std::true_type
{
};
template <>
struct IsBits<GameVariable> : std::true_type
{
};
} // namespace jlcxx

JULIA_CPP_MODULE_BEGIN(registry)
jlcxx::Module &mod = registry.create_module("ViZDoomWrapper");

/* Consts */
/*----------------------------------------------------------------------------------------------------------------*/
mod.set_const("SLOT_COUNT", SLOT_COUNT);
mod.set_const("MAX_PLAYERS", MAX_PLAYERS);
mod.set_const("MAX_PLAYER_NAME_LENGTH", MAX_PLAYER_NAME_LENGTH);
mod.set_const("USER_VARIABLE_COUNT", USER_VARIABLE_COUNT);
mod.set_const("DEFAULT_TICRATE", DEFAULT_TICRATE);

mod.set_const("BINARY_BUTTON_COUNT", BINARY_BUTTON_COUNT);
mod.set_const("DELTA_BUTTON_COUNT", DELTA_BUTTON_COUNT);
mod.set_const("BUTTON_COUNT", BUTTON_COUNT);

/* Enums */
/*----------------------------------------------------------------------------------------------------------------*/
mod.add_bits<Mode>("Mode");
mod.set_const("PLAYER", PLAYER);
mod.set_const("SPECTATOR", SPECTATOR);
mod.set_const("ASYNC_PLAYER", ASYNC_PLAYER);
mod.set_const("ASYNC_SPECTATOR", ASYNC_SPECTATOR);

mod.add_bits<ScreenFormat>("ScreenFormat");
mod.set_const("CRCGCB", CRCGCB);
mod.set_const("RGB24", RGB24);
mod.set_const("RGBA32", RGBA32);
mod.set_const("ARGB32", ARGB32);
mod.set_const("CBCGCR", CBCGCR);
mod.set_const("BGR24", BGR24);
mod.set_const("BGRA32", BGRA32);
mod.set_const("ABGR32", ABGR32);
mod.set_const("GRAY8", GRAY8);
mod.set_const("DOOM_256_COLORS8", DOOM_256_COLORS8);

mod.add_bits<ScreenResolution>("ScreenResolution");
mod.set_const("RES_160X120", RES_160X120);

mod.set_const("RES_200X125", RES_200X125);
mod.set_const("RES_200X150", RES_200X150);

mod.set_const("RES_256X144", RES_256X144);
mod.set_const("RES_256X160", RES_256X160);
mod.set_const("RES_256X192", RES_256X192);

mod.set_const("RES_320X180", RES_320X180);
mod.set_const("RES_320X200", RES_320X200);
mod.set_const("RES_320X240", RES_320X240);
mod.set_const("RES_320X256", RES_320X256);

mod.set_const("RES_400X225", RES_400X225);
mod.set_const("RES_400X250", RES_400X250);
mod.set_const("RES_400X300", RES_400X300);

mod.set_const("RES_512X288", RES_512X288);
mod.set_const("RES_512X320", RES_512X320);
mod.set_const("RES_512X384", RES_512X384);

mod.set_const("RES_640X360", RES_640X360);
mod.set_const("RES_640X400", RES_640X400);
mod.set_const("RES_640X480", RES_640X480);

mod.set_const("RES_800X450", RES_800X450);
mod.set_const("RES_800X500", RES_800X500);
mod.set_const("RES_800X600", RES_800X600);

mod.set_const("RES_1024X576", RES_1024X576);
mod.set_const("RES_1024X640", RES_1024X640);
mod.set_const("RES_1024X768", RES_1024X768);

mod.set_const("RES_1280X720", RES_1280X720);
mod.set_const("RES_1280X800", RES_1280X800);
mod.set_const("RES_1280X960", RES_1280X960);
mod.set_const("RES_1280X1024", RES_1280X1024);

mod.set_const("RES_1400X787", RES_1400X787);
mod.set_const("RES_1400X875", RES_1400X875);
mod.set_const("RES_1400X1050", RES_1400X1050);

mod.set_const("RES_1600X900", RES_1600X900);
mod.set_const("RES_1600X1000", RES_1600X1000);
mod.set_const("RES_1600X1200", RES_1600X1200);

mod.set_const("RES_1920X1080", RES_1920X1080);

mod.add_bits<AutomapMode>("AutomapMode");
mod.set_const("NORMAL", NORMAL);
mod.set_const("WHOLE", WHOLE);
mod.set_const("OBJECTS", OBJECTS);
mod.set_const("OBJECTS_WITH_SIZE", OBJECTS_WITH_SIZE);

mod.add_bits<Button>("Button");
mod.set_const("ATTACK", ATTACK);
mod.set_const("USE", USE);
mod.set_const("JUMP", JUMP);
mod.set_const("CROUCH", CROUCH);
mod.set_const("TURN180", TURN180);
mod.set_const("ALTATTACK", ALTATTACK);
mod.set_const("RELOAD", RELOAD);
mod.set_const("ZOOM", ZOOM);
mod.set_const("SPEED", SPEED);
mod.set_const("STRAFE", STRAFE);
mod.set_const("MOVE_RIGHT", MOVE_RIGHT);
mod.set_const("MOVE_LEFT", MOVE_LEFT);
mod.set_const("MOVE_BACKWARD", MOVE_BACKWARD);
mod.set_const("MOVE_FORWARD", MOVE_FORWARD);
mod.set_const("TURN_RIGHT", TURN_RIGHT);
mod.set_const("TURN_LEFT", TURN_LEFT);
mod.set_const("LOOK_UP", LOOK_UP);
mod.set_const("LOOK_DOWN", LOOK_DOWN);
mod.set_const("MOVE_UP", MOVE_UP);
mod.set_const("MOVE_DOWN", MOVE_DOWN);
mod.set_const("LAND", LAND);
mod.set_const("SELECT_WEAPON1", SELECT_WEAPON1);
mod.set_const("SELECT_WEAPON2", SELECT_WEAPON2);
mod.set_const("SELECT_WEAPON3", SELECT_WEAPON3);
mod.set_const("SELECT_WEAPON4", SELECT_WEAPON4);
mod.set_const("SELECT_WEAPON5", SELECT_WEAPON5);
mod.set_const("SELECT_WEAPON6", SELECT_WEAPON6);
mod.set_const("SELECT_WEAPON7", SELECT_WEAPON7);
mod.set_const("SELECT_WEAPON8", SELECT_WEAPON8);
mod.set_const("SELECT_WEAPON9", SELECT_WEAPON9);
mod.set_const("SELECT_WEAPON0", SELECT_WEAPON0);
mod.set_const("SELECT_NEXT_WEAPON", SELECT_NEXT_WEAPON);
mod.set_const("SELECT_PREV_WEAPON", SELECT_PREV_WEAPON);
mod.set_const("DROP_SELECTED_WEAPON", DROP_SELECTED_WEAPON);
mod.set_const("ACTIVATE_SELECTED_ITEM", ACTIVATE_SELECTED_ITEM);
mod.set_const("SELECT_NEXT_ITEM", SELECT_NEXT_ITEM);
mod.set_const("SELECT_PREV_ITEM", SELECT_PREV_ITEM);
mod.set_const("DROP_SELECTED_ITEM", DROP_SELECTED_ITEM);
mod.set_const("LOOK_UP_DOWN_DELTA", LOOK_UP_DOWN_DELTA);
mod.set_const("TURN_LEFT_RIGHT_DELTA", TURN_LEFT_RIGHT_DELTA);
mod.set_const("MOVE_FORWARD_BACKWARD_DELTA", MOVE_FORWARD_BACKWARD_DELTA);
mod.set_const("MOVE_LEFT_RIGHT_DELTA", MOVE_LEFT_RIGHT_DELTA);
mod.set_const("MOVE_UP_DOWN_DELTA", MOVE_UP_DOWN_DELTA);

mod.add_bits<GameVariable>("GameVariable");
mod.set_const("KILLCOUNT", KILLCOUNT);
mod.set_const("ITEMCOUNT", ITEMCOUNT);
mod.set_const("SECRETCOUNT", SECRETCOUNT);
mod.set_const("FRAGCOUNT", FRAGCOUNT);
mod.set_const("DEATHCOUNT", DEATHCOUNT);
mod.set_const("HITCOUNT", HITCOUNT);
mod.set_const("HITS_TAKEN", HITS_TAKEN);
mod.set_const("DAMAGECOUNT", DAMAGECOUNT);
mod.set_const("DAMAGE_TAKEN", DAMAGE_TAKEN);
mod.set_const("HEALTH", HEALTH);
mod.set_const("ARMOR", ARMOR);
mod.set_const("DEAD", DEAD);
mod.set_const("ON_GROUND", ON_GROUND);
mod.set_const("ATTACK_READY", ATTACK_READY);
mod.set_const("ALTATTACK_READY", ALTATTACK_READY);
mod.set_const("SELECTED_WEAPON", SELECTED_WEAPON);
mod.set_const("SELECTED_WEAPON_AMMO", SELECTED_WEAPON_AMMO);
mod.set_const("AMMO1", AMMO1);
mod.set_const("AMMO2", AMMO2);
mod.set_const("AMMO3", AMMO3);
mod.set_const("AMMO4", AMMO4);
mod.set_const("AMMO5", AMMO5);
mod.set_const("AMMO6", AMMO6);
mod.set_const("AMMO7", AMMO7);
mod.set_const("AMMO8", AMMO8);
mod.set_const("AMMO9", AMMO9);
mod.set_const("AMMO0", AMMO0);
mod.set_const("WEAPON1", WEAPON1);
mod.set_const("WEAPON2", WEAPON2);
mod.set_const("WEAPON3", WEAPON3);
mod.set_const("WEAPON4", WEAPON4);
mod.set_const("WEAPON5", WEAPON5);
mod.set_const("WEAPON6", WEAPON6);
mod.set_const("WEAPON7", WEAPON7);
mod.set_const("WEAPON8", WEAPON8);
mod.set_const("WEAPON9", WEAPON9);
mod.set_const("WEAPON0", WEAPON0);
mod.set_const("POSITION_X", POSITION_X);
mod.set_const("POSITION_Y", POSITION_Y);
mod.set_const("POSITION_Z", POSITION_Z);
mod.set_const("ANGLE", ANGLE);
mod.set_const("PITCH", PITCH);
mod.set_const("ROLL", ROLL);
mod.set_const("VIEW_HEIGHT", VIEW_HEIGHT);
mod.set_const("VELOCITY_X", VELOCITY_X);
mod.set_const("VELOCITY_Y", VELOCITY_Y);
mod.set_const("VELOCITY_Z", VELOCITY_Z);
mod.set_const("CAMERA_POSITION_X", CAMERA_POSITION_X);
mod.set_const("CAMERA_POSITION_Y", CAMERA_POSITION_Y);
mod.set_const("CAMERA_POSITION_Z", CAMERA_POSITION_Z);
mod.set_const("CAMERA_ANGLE", CAMERA_ANGLE);
mod.set_const("CAMERA_PITCH", CAMERA_PITCH);
mod.set_const("CAMERA_ROLL", CAMERA_ROLL);
mod.set_const("CAMERA_FOV", CAMERA_FOV);
mod.set_const("USER1", USER1);
mod.set_const("USER2", USER2);
mod.set_const("USER3", USER3);
mod.set_const("USER4", USER4);
mod.set_const("USER5", USER5);
mod.set_const("USER6", USER6);
mod.set_const("USER7", USER7);
mod.set_const("USER8", USER8);
mod.set_const("USER9", USER9);
mod.set_const("USER10", USER10);
mod.set_const("USER11", USER11);
mod.set_const("USER12", USER12);
mod.set_const("USER13", USER13);
mod.set_const("USER14", USER14);
mod.set_const("USER15", USER15);
mod.set_const("USER16", USER16);
mod.set_const("USER17", USER17);
mod.set_const("USER18", USER18);
mod.set_const("USER19", USER19);
mod.set_const("USER20", USER20);
mod.set_const("USER21", USER21);
mod.set_const("USER22", USER22);
mod.set_const("USER23", USER23);
mod.set_const("USER24", USER24);
mod.set_const("USER25", USER25);
mod.set_const("USER26", USER26);
mod.set_const("USER27", USER27);
mod.set_const("USER28", USER28);
mod.set_const("USER29", USER29);
mod.set_const("USER30", USER30);
mod.set_const("USER31", USER31);
mod.set_const("USER32", USER32);
mod.set_const("USER33", USER33);
mod.set_const("USER34", USER34);
mod.set_const("USER35", USER35);
mod.set_const("USER36", USER36);
mod.set_const("USER37", USER37);
mod.set_const("USER38", USER38);
mod.set_const("USER39", USER39);
mod.set_const("USER40", USER40);
mod.set_const("USER41", USER41);
mod.set_const("USER42", USER42);
mod.set_const("USER43", USER43);
mod.set_const("USER44", USER44);
mod.set_const("USER45", USER45);
mod.set_const("USER46", USER46);
mod.set_const("USER47", USER47);
mod.set_const("USER48", USER48);
mod.set_const("USER49", USER49);
mod.set_const("USER50", USER50);
mod.set_const("USER51", USER51);
mod.set_const("USER52", USER52);
mod.set_const("USER53", USER53);
mod.set_const("USER54", USER54);
mod.set_const("USER55", USER55);
mod.set_const("USER56", USER56);
mod.set_const("USER57", USER57);
mod.set_const("USER58", USER58);
mod.set_const("USER59", USER59);
mod.set_const("USER60", USER60);
mod.set_const("PLAYER_NUMBER", PLAYER_NUMBER);
mod.set_const("PLAYER_COUNT", PLAYER_COUNT);
mod.set_const("PLAYER1_FRAGCOUNT", PLAYER1_FRAGCOUNT);
mod.set_const("PLAYER2_FRAGCOUNT", PLAYER2_FRAGCOUNT);
mod.set_const("PLAYER3_FRAGCOUNT", PLAYER3_FRAGCOUNT);
mod.set_const("PLAYER4_FRAGCOUNT", PLAYER4_FRAGCOUNT);
mod.set_const("PLAYER5_FRAGCOUNT", PLAYER5_FRAGCOUNT);
mod.set_const("PLAYER6_FRAGCOUNT", PLAYER6_FRAGCOUNT);
mod.set_const("PLAYER7_FRAGCOUNT", PLAYER7_FRAGCOUNT);
mod.set_const("PLAYER8_FRAGCOUNT", PLAYER8_FRAGCOUNT);
mod.set_const("PLAYER9_FRAGCOUNT", PLAYER9_FRAGCOUNT);
mod.set_const("PLAYER10_FRAGCOUNT", PLAYER10_FRAGCOUNT);
mod.set_const("PLAYER11_FRAGCOUNT", PLAYER11_FRAGCOUNT);
mod.set_const("PLAYER12_FRAGCOUNT", PLAYER12_FRAGCOUNT);
mod.set_const("PLAYER13_FRAGCOUNT", PLAYER13_FRAGCOUNT);
mod.set_const("PLAYER14_FRAGCOUNT", PLAYER14_FRAGCOUNT);
mod.set_const("PLAYER15_FRAGCOUNT", PLAYER15_FRAGCOUNT);
mod.set_const("PLAYER16_FRAGCOUNT", PLAYER16_FRAGCOUNT);

/* Structs */
/*----------------------------------------------------------------------------------------------------------------*/
mod.add_type<Label>("Label")
    .method("object_id", [](Label &l) { return l.objectId; })
    .method("object_name", [](Label &l) { return l.objectName; })
    .method("value", [](Label &l) { return l.value; })
    .method("x", [](Label &l) { return l.x; })
    .method("y", [](Label &l) { return l.y; })
    .method("width", [](Label &l) { return l.width; })
    .method("height", [](Label &l) { return l.height; })
    .method("object_position_x", [](Label &l) { return l.objectPositionX; })
    .method("object_position_y", [](Label &l) { return l.objectPositionY; })
    .method("object_position_z", [](Label &l) { return l.objectPositionZ; })
    .method("object_angle", [](Label &l) { return l.objectAngle; })
    .method("object_pitch", [](Label &l) { return l.objectPitch; })
    .method("object_roll", [](Label &l) { return l.objectRoll; })
    .method("object_velocity_x", [](Label &l) { return l.objectVelocityX; })
    .method("object_velocity_y", [](Label &l) { return l.objectVelocityY; })
    .method("object_velocity_z", [](Label &l) { return l.objectVelocityZ; });

mod.add_type<Buffer>("Buffer");
mod.add_type<GameState>("GameState")
    .method("number", [](GameState &gs) { return gs.number; })
    .method("tic", [](GameState &gs) { return gs.tic; })
    .method("game_variables", [](GameState &gs) { return jlcxx::ArrayRef<double, 1>(&(gs.gameVariables[0]), gs.gameVariables.size()); })
    .method("screen_buffer", [](GameState &gs) { return gs.screenBuffer; })
    .method("depth_buffer", [](GameState &gs) { return gs.depthBuffer; })
    .method("labels_buffer", [](GameState &gs) { return gs.labelsBuffer; })
    .method("automap_buffer", [](GameState &gs) { return gs.automapBuffer; })
    .method("labels", [](GameState &gs) { return jlcxx::ArrayRef<Label, 1>(&(gs.labels[0]), gs.labels.size()); });

mod.add_type<ServerState>("ServerState")
    .method("player_count", [](ServerState &ss) { return ss.playerCount; })
    .method("players_in_game", [](ServerState &ss) { return ss.playersInGame; })
    .method("players_names", [](ServerState &ss) { return ss.playersNames; })
    .method("players_frags", [](ServerState &ss) { return ss.playersFrags; });

/* DoomGame */
/*----------------------------------------------------------------------------------------------------------------*/

mod.add_type<DoomGame>("DoomGame")
    .method("init", &DoomGame::init)
    .method("close", &DoomGame::close)
    .method("new_episode", &DoomGame::newEpisode)
    .method("new_episode", [](DoomGame& dg) {dg.newEpisode();})
    .method("load_config", &DoomGame::loadConfig)
    .method("is_running", &DoomGame::isRunning)
    .method("is_multiplayer_game", &DoomGame::isMultiplayerGame)
    .method("is_recording_episode", &DoomGame::isRecordingEpisode)
    .method("is_replaying_episode", &DoomGame::isReplayingEpisode)
    .method("replay_episode", &DoomGame::replayEpisode)
    .method("is_episode_finished", &DoomGame::isEpisodeFinished)
    .method("is_new_episode", &DoomGame::isNewEpisode)
    .method("is_player_dead", &DoomGame::isPlayerDead)
    .method("respawn_player", &DoomGame::respawnPlayer)
    .method("set_action", [](DoomGame &dg, jlcxx::ArrayRef<double, 1> actions) {
        std::vector<double> data(actions.begin(), actions.end());
        dg.setAction(data); })
    .method("make_action", [](DoomGame &dg, jlcxx::ArrayRef<double, 1> actions) {
        std::vector<double> data(actions.begin(), actions.end());
        return dg.makeAction(data);
    })
    .method("make_action", [](DoomGame &dg, jlcxx::ArrayRef<double, 1> actions, unsigned int tics) {
        std::vector<double> data(actions.begin(), actions.end());
        return dg.makeAction(data, tics);
    })
    .method("advance_action", &DoomGame::advanceAction)
    .method("advance_action", [](DoomGame& dg){dg.advanceAction();})
    .method("get_state", &DoomGame::getState)
    .method("get_server_state", &DoomGame::getServerState)
    .method("get_game_variable", &DoomGame::getGameVariable)
    .method("get_button", &DoomGame::getButton)
    .method("get_living_reward", &DoomGame::getLivingReward)
    .method("set_living_reward", &DoomGame::setLivingReward)
    .method("get_death_penalty", &DoomGame::getDeathPenalty)
    .method("set_death_penalty", &DoomGame::setDeathPenalty)
    .method("get_last_reward", &DoomGame::getLastReward)
    .method("get_total_reward", &DoomGame::getTotalReward)
    .method("get_last_action", &DoomGame::getLastAction)
    .method("get_available_game_variables", &DoomGame::getAvailableGameVariables)
    .method("set_available_game_variables", [](DoomGame &dg, jlcxx::ArrayRef<GameVariable, 1> gv) {
        std::vector<GameVariable> data(gv.begin(), gv.end());
        dg.setAvailableGameVariables(data);
    })
    .method("add_available_game_variable", &DoomGame::addAvailableGameVariable)
    .method("clear_available_game_variables", &DoomGame::clearAvailableGameVariables)
    .method("get_available_game_variables_size", &DoomGame::getAvailableGameVariablesSize)
    .method("get_available_buttons", &DoomGame::getAvailableButtons)
    .method("set_available_buttons", [](DoomGame &dg, jlcxx::ArrayRef<Button, 1> buttons) {
        std::vector<Button> data(buttons.begin(), buttons.end());
        dg.setAvailableButtons(data);
    })
    .method("add_available_button", &DoomGame::addAvailableButton)
    .method("add_available_button", [](DoomGame& dg, Button bt){dg.addAvailableButton(bt);})
    .method("clear_available_buttons", &DoomGame::clearAvailableButtons)
    .method("get_available_buttons_size", &DoomGame::getAvailableButtonsSize)
    .method("set_button_max_value", &DoomGame::setButtonMaxValue)
    .method("get_button_max_value", &DoomGame::getButtonMaxValue)
    .method("add_game_args", &DoomGame::addGameArgs)
    .method("clear_game_args", &DoomGame::clearGameArgs)
    .method("send_game_command", &DoomGame::sendGameCommand)
    .method("get_mode", &DoomGame::getMode)
    .method("set_mode", &DoomGame::setMode)
    .method("get_ticrate", &DoomGame::getTicrate)
    .method("set_ticrate", &DoomGame::setTicrate)
    .method("set_vizdoom_path", &DoomGame::setViZDoomPath)
    .method("set_doom_game_path", &DoomGame::setDoomGamePath)
    .method("set_doom_scenario_path", &DoomGame::setDoomScenarioPath)
    .method("set_doom_map", &DoomGame::setDoomMap)
    .method("set_doom_skill", &DoomGame::setDoomSkill)
    .method("set_doom_config_path", &DoomGame::setDoomConfigPath)
    .method("get_seed", &DoomGame::getSeed)
    .method("set_seed", &DoomGame::setSeed)
    .method("get_episode_start_time", &DoomGame::getEpisodeStartTime)
    .method("set_episode_start_time", &DoomGame::setEpisodeStartTime)
    .method("get_episode_timeout", &DoomGame::getEpisodeTimeout)
    .method("set_episode_timeout", &DoomGame::setEpisodeTimeout)
    .method("get_episode_time", &DoomGame::getEpisodeTime)
    .method("set_console_enabled", &DoomGame::setConsoleEnabled)
    .method("set_sound_enabled", &DoomGame::setSoundEnabled)
    .method("set_screen_resolution", &DoomGame::setScreenResolution)
    .method("set_screen_format", &DoomGame::setScreenFormat)
    .method("is_depth_buffer_enabled", &DoomGame::isDepthBufferEnabled)
    .method("set_depth_buffer_enabled", &DoomGame::setDepthBufferEnabled)
    .method("is_labels_buffer_enabled", &DoomGame::isLabelsBufferEnabled)
    .method("set_labels_buffer_enabled", &DoomGame::setLabelsBufferEnabled)
    .method("is_automap_buffer_enabled", &DoomGame::isAutomapBufferEnabled)
    .method("set_automap_buffer_enabled", &DoomGame::setAutomapBufferEnabled)
    .method("set_automap_mode", &DoomGame::setAutomapMode)
    .method("set_automap_rotate", &DoomGame::setAutomapRotate)
    .method("set_automap_render_textures", &DoomGame::setAutomapRenderTextures)
    .method("set_render_hud", &DoomGame::setRenderHud)
    .method("set_render_minimal_hud", &DoomGame::setRenderMinimalHud)
    .method("set_render_weapon", &DoomGame::setRenderWeapon)
    .method("set_render_crosshair", &DoomGame::setRenderCrosshair)
    .method("set_render_decals", &DoomGame::setRenderDecals)
    .method("set_render_particles", &DoomGame::setRenderParticles)
    .method("set_render_effects_sprites", &DoomGame::setRenderEffectsSprites)
    .method("set_render_messages", &DoomGame::setRenderMessages)
    .method("set_render_corpses", &DoomGame::setRenderCorpses)
    .method("set_render_screen_flashes", &DoomGame::setRenderScreenFlashes)
    .method("set_render_all_frames", &DoomGame::setRenderAllFrames)
    .method("set_window_visible", &DoomGame::setWindowVisible)
    .method("get_screen_width", &DoomGame::getScreenWidth)
    .method("get_screen_height", &DoomGame::getScreenHeight)
    .method("get_screen_channels", &DoomGame::getScreenChannels)
    .method("get_screen_size", &DoomGame::getScreenSize)
    .method("get_screen_pitch", &DoomGame::getScreenPitch)
    .method("get_screen_format", &DoomGame::getScreenFormat);

mod.method("doom_tics_to_ms", doomTicsToMs);
mod.method("ms_to_doom_tics", msToDoomTics);
mod.method("doom_tics_to_sec", doomTicsToSec);
mod.method("sec_to_doom_tics", secToDoomTics);
mod.method("doom_fixed_to_double", [](double x) { return doomFixedToDouble(x); });
mod.method("is_binary_button", isBinaryButton);
mod.method("is_delta_button", isDeltaButton);
JULIA_CPP_MODULE_END
