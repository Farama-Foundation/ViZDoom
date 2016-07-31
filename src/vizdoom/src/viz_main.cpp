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

#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/thread.hpp>

#include "viz_main.h"
#include "viz_defines.h"
#include "viz_input.h"
#include "viz_game.h"
#include "viz_screen.h"
#include "viz_shared_memory.h"
#include "viz_message_queue.h"

#include "d_main.h"
#include "d_net.h"

namespace b = boost;
namespace bt = boost::this_thread;

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

CVAR (Bool, viz_debug, false, CVAR_NOSET)

CVAR (Bool, viz_controlled, false, CVAR_NOSET)
CVAR (String, viz_instance_id, "0", CVAR_NOSET)

//modes
CVAR (Bool, viz_async, false, CVAR_NOSET)
CVAR (Bool, viz_allow_input, false, CVAR_NOSET)

//buffers
CVAR (Int, viz_screen_format, 0, 0)
CVAR (Bool, viz_depth, false, 0)
CVAR (Bool, viz_labels, false, 0)
CVAR (Bool, viz_automap, false, 0)
//CVAR (Int, viz_buf_mode, 0, 0)

//rendering options (bitset)
CVAR (Int, viz_render_mode, 0, 0)
CVAR (Int, viz_automap_mode, 0, 0)

//window/sound/console/rendering all frames
CVAR (Bool, viz_render_all, false, CVAR_NOSET)
CVAR (Bool, viz_window_hidden, false, CVAR_NOSET)
CVAR (Bool, viz_noxserver, false, CVAR_NOSET)
CVAR (Bool, viz_noconsole, false, CVAR_NOSET)
CVAR (Bool, viz_nosound, false, CVAR_NOSET)

CVAR (Bool, viz_loop_map, false, CVAR_NOSET)
CVAR (Bool, viz_nocheat, false, CVAR_NOSET)

int vizTime = 0;
bool vizNextTic = false;
bool vizUpdate = false;
unsigned int vizLastUpdate = 0;

void VIZ_Init(){
    Printf("VIZ_Init: Instance id: %s\n", *viz_instance_id);

    if(*viz_controlled) {

        VIZ_UpdateCVARs();

        VIZ_MQInit(*viz_instance_id);
        VIZ_SMInit(*viz_instance_id);

        VIZ_GameStateInit();
        VIZ_InputInit();
        VIZ_ScreenInit();

        vizNextTic = true;
        vizUpdate = true;
    }
}

void VIZ_Close(){
    if(*viz_controlled) {
        VIZ_InputClose();
        VIZ_GameStateClose();
        VIZ_ScreenClose();

        VIZ_SMClose();
        VIZ_MQClose();
    }
}

void VIZ_AsyncStartTic(){
    try{
        bt::interruption_point();
    }
    catch(b::thread_interrupted &ex ){
        exit(0);
    }

    if (*viz_controlled){
        VIZ_MQTic();
        if(vizNextTic) VIZ_InputTic();
    }
}

void VIZ_Tic(){

    try{
        bt::interruption_point();
    }
    catch(b::thread_interrupted &ex ){
        exit(0);
    }

    if (*viz_controlled){

        NetUpdate();

        if(vizUpdate) {
            VIZ_Update();
        }
        if(vizNextTic) {
            VIZ_GameStateTic();
            VIZ_MQSend(VIZ_MSG_CODE_DOOM_DONE);
            vizNextTic = false;
        }

        if(!*viz_async){
            VIZ_MQTic();
            VIZ_InputTic();
        }
    }
}

void VIZ_Update(){
    VIZ_DEBUG_PRINT("VIZ_Update: tic: %d, viztic: %d, lastupdate: %d\n", gametic, VIZ_TIME, vizLastUpdate);

    VIZ_D_MapDisplay();
    VIZ_ScreenLevelMapUpdate();
    VIZ_D_ScreenDisplay();
    VIZ_ScreenUpdate();
    VIZ_GameStateUpdateLabels();

    vizLastUpdate = VIZ_TIME;
    vizUpdate = false;
}

bool VIZ_IsPaused(){
    return menuactive != MENU_Off;

    //&& (gamestate == GS_LEVEL || gamestate == GS_TITLELEVEL || gamestate == GS_INTERMISSION || gamestate == GS_FINALE)
    //&& !paused && !pauseext && menuactive == MENU_Off && ConsoleState != c_down && ConsoleState != c_falling
}

// other
EXTERN_CVAR(Bool, vid_fps)

// hud
EXTERN_CVAR(Int, screenblocks)
EXTERN_CVAR (Bool, st_scale)
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

//particles && effects sprites
EXTERN_CVAR(Bool, r_particles)
EXTERN_CVAR(Int, r_maxparticles)

EXTERN_CVAR(Int, cl_bloodtype)
EXTERN_CVAR(Int, cl_pufftype)
EXTERN_CVAR(Int, cl_rockettrails)

//automap
EXTERN_CVAR(Int, am_cheat)
EXTERN_CVAR(Bool, am_rotate)
EXTERN_CVAR(Bool, am_textured)

EXTERN_CVAR(Bool, am_showitems)
EXTERN_CVAR(Bool, am_showmonsters)
EXTERN_CVAR(Bool, am_showsecrets)
EXTERN_CVAR(Bool, am_showtime)
EXTERN_CVAR(Bool, am_showtotaltime)



void VIZ_UpdateCVARs(){

    // hud
    bool hud = (*viz_render_mode & 1) != 0;
    bool minHud = (*viz_render_mode & 2) != 0;

    if (minHud && hud) screenblocks.CmdSet("11");
    else if (hud) screenblocks.CmdSet("10");
    else screenblocks.CmdSet("12");

    st_scale.CmdSet("1");
    hud_scale.CmdSet("1");
    hud_althud.CmdSet("0");

    //players sprite (weapon)
    r_drawplayersprites.CmdSet((*viz_render_mode & 4) != 0 ? "1" : "0");

    //crosshair
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

    // automap
    am_cheat = *viz_nocheat ? 0 : *viz_automap_mode;

    am_rotate.CmdSet((*viz_render_mode & 128) != 0 ? "1" : "0");
    am_textured.CmdSet((*viz_render_mode & 256) != 0 ? "1" : "0");

    am_showitems.CmdSet("0");
    am_showmonsters.CmdSet("0");
    am_showsecrets.CmdSet("0");
    am_showtime.CmdSet("0");
    am_showtotaltime.CmdSet("0");

};

