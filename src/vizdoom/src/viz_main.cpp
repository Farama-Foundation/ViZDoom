/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch
 Copyright (C) 2017 - 2022 by Marek Wydmuch, Michał Kempka, Wojciech Jaśkowski, and the respective contributors

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

#include "viz_main.h"
#include "viz_system.h"
#include "viz_input.h"
#include "viz_game.h"
#include "viz_buffers.h"
#include "viz_message_queue.h"

#include "d_main.h"
#include "g_game.h"
#include "sbar.h"
#include "c_dispatch.h"
#include "i_sound.h"
#include "i_system.h"


/* CVARs and CCMDs */
/*--------------------------------------------------------------------------------------------------------------------*/

//CVAR_ARCHIVE		= 1,	// set to cause it to be saved to config
//CVAR_USERINFO		= 2,	// added to userinfo  when changed
//CVAR_SERVERINFO	= 4,	// added to serverinfo when changed
//CVAR_NOSET		= 8,	// don't allow change from console at all, but can be set from the command line
//CVAR_LATCH		= 16,	// save changes until server restart
//CVAR_UNSETTABLE	= 32,	// can unset this var from console
//CVAR_DEMOSAVE		= 64,	// save the value of this cvar in a demo
//CVAR_ISDEFAULT	= 128,	// is cvar unchanged since creation?
//CVAR_AUTO			= 256,	// allocated; needs to be freed when destroyed
//CVAR_NOINITCALL	= 512,	// don't call callback at game start
//CVAR_GLOBALCONFIG	= 1024,	// cvar is saved to global config section
//CVAR_VIDEOCONFIG	= 2048, // cvar is saved to video config section (not implemented)
//CVAR_NOSAVE		= 4096, // when used with CVAR_SERVERINFO, do not save var to savegame
//CVAR_MOD			= 8192,	// cvar was defined by a mod
//CVAR_IGNORE		= 16384,// do not send cvar across the network/inaccesible from ACS (dummy mod cvar)

// debug
CVAR (Int, viz_debug, 0, CVAR_NOSET)
// 0 - no debug msg
// 1 - init debug msg
// 2 - tic basic debug msg
// 3 - tic detailed debug msg
// 4 - all
CVAR (Bool, viz_debug_instances, 0, CVAR_NOSET) // prints instance id with every message
CVAR (Int, viz_log, 0, CVAR_NOSET)

// control
CVAR (Bool, viz_controlled, false, CVAR_NOSET)
CVAR (String, viz_instance_id, "0", CVAR_NOSET)
CVAR (Int, viz_seed, 0, CVAR_NOSET)
CVAR (Bool, viz_cmd_filter, true, CVAR_NOSET)

// modes
CVAR (Bool, viz_async, false, CVAR_NOSET)
CVAR (Bool, viz_allow_input, false, CVAR_NOSET)
CVAR (Int, viz_sync_timeout, 1000, CVAR_NOSET | CVAR_SERVERINFO) // In milliseconds

// buffers
CVAR (Int, viz_screen_format, 0, 0)
CVAR (Bool, viz_depth, false, 0)
CVAR (Bool, viz_labels, false, 0)
CVAR (Bool, viz_automap, false, 0)
CVAR (Bool, viz_objects, false, 0)
CVAR (Bool, viz_sectors, false, 0)

// rendering options (bitset)
CVAR (Int, viz_render_mode, 0, 0)
CVAR (Int, viz_automap_mode, 0, 0)
CVAR (Bool, viz_render_corpses, true, 0)
CVAR (Bool, viz_render_flashes, true, 0)
CVAR (Bool, viz_ignore_render_mode, false, 0)

CVAR (Float, viz_am_scale, -1, 0)
CVAR (Bool, viz_am_center, false, 0)

// window/sound/console/rendering all frames
CVAR (Bool, viz_render_all, false, CVAR_NOSET)
CVAR (Bool, viz_window_hidden, false, CVAR_NOSET)
CVAR (Bool, viz_noxserver, false, CVAR_NOSET)
CVAR (Bool, viz_noconsole, false, CVAR_NOSET)
CVAR (Bool, viz_nosound, false, CVAR_NOSET)

// multiplayer/recordings
CVAR (Int, viz_override_player, 0, 0)
CVAR (Bool, viz_loop_map, false, CVAR_NOSET | CVAR_SERVERINFO)
CVAR (Bool, viz_nocheat, false, CVAR_NOSET | CVAR_SERVERINFO)
CVAR (Int, viz_respawn_delay, 1, CVAR_DEMOSAVE | CVAR_SERVERINFO)
CVAR (Bool, viz_spectator, false, CVAR_DEMOSAVE | CVAR_USERINFO) // players[playernum].userinfo.GetSpectator()
CVAR (Int, viz_afk_timeout, 60, CVAR_DEMOSAVE | CVAR_SERVERINFO) // In seconds
CVAR (Int, viz_connect_timeout, 60, CVAR_NOSET) // In seconds
CVAR (String, viz_bots_path, "", CVAR_NOSET)

// audio buffer related
CVAR (Bool, viz_soft_audio, false, 0)
CVAR (Int, viz_samp_freq, 44100, 0)
CVAR (Int, viz_audio_tics, 4, 0)

CCMD(viz_set_seed){
    viz_seed.CmdSet(argv[1]);
    rngseed = atoi(argv[1]);
    staticrngseed = rngseed;
    use_staticrng = true;
    VIZ_DebugMsg(2, VIZ_FUNC, "viz_seed changed to: %d.", rngseed);
}


/* Flow */
/*--------------------------------------------------------------------------------------------------------------------*/

int vizTime = 0;
bool vizNextTic = false;
bool vizUpdate = false;
unsigned int vizLastUpdate = 0;
int vizNodesRecv[VIZ_MAX_PLAYERS];

int vizSavedTime = 0;
bool vizFreeze = false;
int (*_I_GetTime)(bool);
int (*_I_WaitForTic)(int);
void (*_I_FreezeTime)(bool);

int VIZ_GetTime(bool saveMS){
    //if (*viz_allow_input) _I_GetTim(saveMS);
    //if(saveMS) vizSavedTime = vizTime;
    return vizTime;
}

int VIZ_WaitForTic(int tic){
    if(*viz_allow_input) _I_WaitForTic(tic);
    if(tic > vizTime) vizTime = tic;
    return VIZ_GetTime(false);
}

void VIZ_FreezeTime (bool frozen){
    //if (*viz_allow_input) _I_FreezeTime(frozen);
    vizFreeze = frozen;
}

void VIZ_Init(){
    if(*viz_controlled) {
        Printf("VIZ_Init: instance id: %s, async: %d, input: %d\n", *viz_instance_id, *viz_async, *viz_allow_input);

        VIZ_CVARsUpdate();

        VIZ_MQInit(*viz_instance_id);
        VIZ_SMInit(*viz_instance_id);

        VIZ_GameStateInit();
        VIZ_InputInit();
        VIZ_BuffersInit();

        VIZ_GameStateSMUpdate();

        vizNextTic = true;
        vizUpdate = true;

        if(!*viz_async) {
            vizTime = gametic + 1;
            _I_GetTime = I_GetTime;
            _I_WaitForTic = I_WaitForTic;
            _I_FreezeTime = I_FreezeTime;
            I_GetTime = &VIZ_GetTime;
            I_WaitForTic = &VIZ_WaitForTic;
            I_FreezeTime = &VIZ_FreezeTime;
        }
    }

    VIZ_CVARsInit();
}

void VIZ_Close(){
    if(*viz_controlled) {
        Printf("VIZ_Close: instance id: %s\n", *viz_instance_id);

        VIZ_InputClose();
        VIZ_GameStateClose();
        VIZ_ScreenClose();

        VIZ_SMClose();
        VIZ_MQClose();
    }
}

void VIZ_AsyncStartTic(){
    VIZ_InterruptionPoint();

    if (*viz_controlled){
        VIZ_MQTic();
        if(vizNextTic) VIZ_InputTic();
        ++vizTime;
    }
}

void VIZ_Tic(){

    VIZ_DebugMsg(2, VIZ_FUNC, "tic: %d, vizTime: %d", gametic, vizTime);
    VIZ_DebugMsg(4, VIZ_FUNC, "rngseed: %d, use_staticrng: %d, staticrngseed: %d", rngseed, use_staticrng, staticrngseed);

    if(*viz_debug >= 5){
        std::string vizCvarsStateMsg = std::string("viz_cvars: ")
            + "viz_controlled: %d, viz_instance_id: %d, viz_seed: %d, viz_async: %d, viz_allow_input: %d, viz_sync_timeout: %d"
            + ", viz_screen_format: %d, viz_depth: %d, viz_labels: %d, viz_automap: %d, viz_render_mode: %d, viz_automap_mode: %d"
            + ", viz_render_corpses: %d, viz_render_all: %d, viz_window_hidden: %d, viz_noxserver: %d, viz_noconsole: %d, viz_nosound: %d"
            + ", viz_override_player: %d, viz_loop_map: %d, viz_nocheat: %d, viz_respawn_delay: %d";

        VIZ_DebugMsg(5, VIZ_FUNC, vizCvarsStateMsg.c_str(),
                     *viz_controlled, *viz_instance_id, *viz_seed, *viz_async, *viz_allow_input, *viz_sync_timeout,
                     *viz_screen_format, *viz_depth, *viz_labels, *viz_automap, *viz_render_mode, *viz_automap_mode,
                     *viz_render_corpses, *viz_render_all, *viz_window_hidden, *viz_noxserver, *viz_noconsole, *viz_nosound,
                     *viz_override_player, *viz_loop_map, *viz_nocheat, *viz_respawn_delay);
    }

    VIZ_InterruptionPoint();

    if (*viz_controlled){
        if(vizNextTic) {
            VIZ_GameStateTic();
            if(vizUpdate) {
                VIZ_Update();
            }

            // Sound buffer will always be updated irrespective of update signal
            if (*viz_soft_audio) {
                VIZ_AudioUpdate();
            }

            VIZ_MQSend(VIZ_MSG_CODE_DOOM_DONE);
            vizNextTic = false;
        }

        if(!*viz_async){
            VIZ_MQTic();
            VIZ_InputTic();
            ++vizTime;
        }
    }

    if(*viz_log) {
        VIZ_PrintPlayers();
        VIZ_PrintInput();
    }
}

void VIZ_Update(){
    VIZ_DebugMsg(3, VIZ_FUNC, "tic: %d, vizTime: %d, lastupdate: %d", gametic, VIZ_TIME, vizLastUpdate);

    if(!*viz_nocheat && *viz_automap){
        VIZ_D_MapDisplay();
        VIZ_ScreenLevelMapUpdate();
    }
    VIZ_D_ScreenDisplay();
    VIZ_ScreenUpdate();
    VIZ_GameStateUpdate();

    vizLastUpdate = VIZ_TIME;
    vizUpdate = false;
}

bool VIZ_IsPaused(){
    return menuactive != MENU_Off;

    //&& (gamestate == GS_LEVEL || gamestate == GS_TITLELEVEL || gamestate == GS_INTERMISSION || gamestate == GS_FINALE)
    //&& !paused && !pauseext && menuactive == MENU_Off && ConsoleState != c_down && ConsoleState != c_falling
}


/* CVARs settings */
/*--------------------------------------------------------------------------------------------------------------------*/

// other
EXTERN_CVAR(Bool, vid_fps)
EXTERN_CVAR(Bool, cl_capfps)
EXTERN_CVAR(Bool, vid_vsync)
EXTERN_CVAR(Int, wipetype)

// hud
EXTERN_CVAR(Int, screenblocks)
EXTERN_CVAR(Bool, st_scale)
EXTERN_CVAR(Bool, hud_scale)
EXTERN_CVAR(Bool, hud_althud)

// player sprite
EXTERN_CVAR(Bool, r_drawplayersprites)

// crosshair
EXTERN_CVAR (Int, crosshair);
//EXTERN_CVAR (Bool, crosshairforce);
//EXTERN_CVAR (Color, crosshaircolor);
//EXTERN_CVAR (Bool, crosshairhealth);
//EXTERN_CVAR (Bool, crosshairscale);
//EXTERN_CVAR (Bool, crosshairgrow);

// decals
EXTERN_CVAR(Bool, cl_bloodsplats)
EXTERN_CVAR(Int, cl_maxdecals)
EXTERN_CVAR(Bool, cl_missiledecals)
EXTERN_CVAR(Bool, cl_spreaddecals)

// particles && effects sprites
EXTERN_CVAR(Bool, r_particles)
EXTERN_CVAR(Int, r_maxparticles)

EXTERN_CVAR(Int, cl_bloodtype)
EXTERN_CVAR(Int, cl_pufftype)
EXTERN_CVAR(Int, cl_rockettrails)

// flashes -> removed or GZDoom only?
//EXTERN_CVAR(Float, blood_fade_scalar)
//EXTERN_CVAR(Float, pickup_fade_scalar)

// messages
EXTERN_CVAR(Float, con_midtime)
EXTERN_CVAR(Float, con_notifytime)
EXTERN_CVAR(Bool, cl_showmultikills)
EXTERN_CVAR(Bool, cl_showsprees)

// automap
EXTERN_CVAR(Int, am_cheat)
EXTERN_CVAR(Int, am_rotate)
EXTERN_CVAR(Bool, am_textured)
EXTERN_CVAR(Bool, am_followplayer)
EXTERN_CVAR(Int, am_drawmapback)
EXTERN_CVAR(Bool, am_showtriggerlines)

EXTERN_CVAR(Bool, am_showitems)
EXTERN_CVAR(Bool, am_showmonsters)
EXTERN_CVAR(Bool, am_showsecrets)
EXTERN_CVAR(Bool, am_showtime)
EXTERN_CVAR(Bool, am_showtotaltime)

#ifdef VIZ_OS_WIN
    EXTERN_CVAR(Bool, vid_forceddraw);
#endif

void VIZ_CVARsInit(){
    if(*viz_spectator && !*viz_override_player){
        screenblocks.CmdSet("12");
        crosshair.CmdSet("0");
        r_drawplayersprites.CmdSet("0");
    }
}

void VIZ_CVARsUpdate(){

    VIZ_DebugMsg(3, VIZ_FUNC, "mode: %d", *viz_render_mode);

    cl_capfps.CmdSet("1");
    vid_vsync.CmdSet("0");
    wipetype.CmdSet("0");

    #ifdef VIZ_OS_WIN
        vid_forceddraw.CmdSet("1");
    #endif

    if(!*viz_ignore_render_mode) {
        // hud
        bool hud = (*viz_render_mode & 1) != 0;
        bool minHud = (*viz_render_mode & 2) != 0;

        if (minHud && hud) screenblocks.CmdSet("11");
        else if (hud) screenblocks.CmdSet("10");
        else screenblocks.CmdSet("12");

        st_scale.CmdSet("1");
        hud_scale.CmdSet("1");
        hud_althud.CmdSet("0");

        // players sprite (weapon)
        r_drawplayersprites.CmdSet((*viz_render_mode & 4) != 0 ? "1" : "0");

        // crosshair
        crosshair.CmdSet((*viz_render_mode & 8) != 0 ? "1" : "0");

        // decals
        bool decals = (*viz_render_mode & 16) != 0;
        cl_bloodsplats.CmdSet(decals ? "1" : "0");
        cl_maxdecals.CmdSet(decals ? "1024" : "0");
        cl_missiledecals.CmdSet(decals ? "1" : "0");
        cl_spreaddecals.CmdSet(decals ? "1" : "0");

        // particles && effects sprites
        bool particles = (*viz_render_mode & 32) != 0;
        bool sprites = (*viz_render_mode & 64) != 0;

        r_particles.CmdSet(particles ? "1" : "0");
        r_maxparticles.CmdSet(particles ? "4092" : "0");

        cl_bloodtype.CmdSet(sprites ? "1" : "2");
        cl_pufftype.CmdSet(sprites ? "0" : "1");
        cl_rockettrails.CmdSet(sprites ? "3" : "1");

        // messages
        bool messages = (*viz_render_mode & 128) != 0;
        con_midtime.CmdSet(messages ? "3" : "0");
        con_notifytime.CmdSet(messages ? "3" : "0");
        cl_showmultikills.CmdSet(messages ? "1" : "0");
        cl_showsprees.CmdSet(messages ? "1" : "0");

        // automap
        am_rotate.CmdSet((*viz_render_mode & 256) != 0 ? "1" : "0");
        am_textured.CmdSet((*viz_render_mode & 512) != 0 ? "1" : "0");
        am_showtriggerlines.CmdSet("2");
        am_drawmapback.CmdSet("0");
        if(!*viz_am_center)
            am_followplayer.CmdSet("1");

        am_showitems.CmdSet("0");
        am_showmonsters.CmdSet("0");
        am_showsecrets.CmdSet("0");
        am_showtime.CmdSet("0");
        am_showtotaltime.CmdSet("0");

        // bodies
        viz_render_corpses.CmdSet((*viz_render_mode & 1024) != 0 ? "1" : "0");

        // flashes
        viz_render_flashes.CmdSet((*viz_render_mode & 2048) != 0 ? "1" : "0");
        //blood_fade_scalar.CmdSet((*viz_render_mode & 2048) != 0 ? "1" : "0");
        //pickup_fade_scalar.CmdSet((*viz_render_mode & 2048) != 0 ? "1" : "0");
    }

    am_cheat = *viz_nocheat ? 0 : *viz_automap_mode;

    if(demoplayback && multiplayer && *viz_override_player){
        if(*viz_override_player >= 1 && *viz_override_player <= VIZ_MAX_PLAYERS && playeringame[*viz_override_player-1]) {
            consoleplayer = *viz_override_player - 1;
            S_UpdateSounds(players[consoleplayer].camera);
            StatusBar->AttachToPlayer(&players[consoleplayer]);
        }
        else VIZ_Error(VIZ_FUNC, "Player %d does not exist.", *viz_override_player);
    }

    VIZ_CVARsInit();
}


/* Error and debug handling */
/*--------------------------------------------------------------------------------------------------------------------*/

bool vizIgnoreNextError = false;

void VIZ_IgnoreNextDoomError(){
    vizIgnoreNextError = true;
}

void VIZ_DoomError(const char *error){
    if(vizIgnoreNextError){
        vizIgnoreNextError = false;
        return;
    }

    if(*viz_controlled){
        VIZ_MQSend(VIZ_MSG_CODE_DOOM_ERROR, error);
        exit(1);
    }
}

void VIZ_PrintFuncMsg(const char *func, const char *msg){
    int s = 0;
    while (func[s] != NULL && func[s] != ' ') ++s;
    int e = s;
    while (func[e] != NULL && func[e] != '(') ++e;

    if(*viz_debug_instances) Printf("%s: ", *viz_instance_id);
    if(e > s) Printf("%.*s: %s\n", e - s - 1, &func[s + 1], msg);
    else Printf("%s: %s\n", func, msg);
}

void VIZ_Error(const char *func, const char *error, ...){

    va_list arg_ptr;
    char error_msg[VIZ_MAX_ERROR_TEXT_LEN];

    va_start(arg_ptr, error);
    myvsnprintf(error_msg, VIZ_MAX_ERROR_TEXT_LEN, error, arg_ptr);
    va_end(arg_ptr);

    VIZ_PrintFuncMsg(func, error_msg);
    VIZ_MQSend(VIZ_MSG_CODE_DOOM_ERROR, error_msg);
    exit(1);
}

void VIZ_DebugMsg(int level, const char *func, const char *msg, ...){
    if(*viz_debug < level) return;

    va_list arg_ptr;
    char debug_msg[VIZ_MAX_DEBUG_TEXT_LEN];

    va_start(arg_ptr, msg);
    myvsnprintf(debug_msg, VIZ_MAX_DEBUG_TEXT_LEN, msg, arg_ptr);
    va_end(arg_ptr);

    VIZ_PrintFuncMsg(func, debug_msg);
}
