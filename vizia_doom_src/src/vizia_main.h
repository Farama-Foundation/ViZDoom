#ifndef VIZIA_MAIN_H
#define VIZIA_MAIN_H

#include "templates.h"
#include "version.h"
#include "doomdef.h"
#include "doomstat.h"

#include "d_main.h"
#include "d_net.h"
#include "d_event.h"
#include "d_player.h"
#include "d_netinf.h"

#include "g_game.h"
#include "g_level.h"

#include "c_console.h"
#include "c_cvars.h"
#include "c_bind.h"
#include "c_dispatch.h"

#include "intermission/intermission.h"
#include "m_argv.h"
#include "m_misc.h"

#include "p_tick.h"
#include "p_local.h"
#include "p_acs.h"

#include "v_video.h"
#include "r_renderer.h"

#include "p_acs.h"
#include "r_data/r_translate.h"
#include "resourcefiles/resourcefile.h"

#include "g_shared/a_pickups.h";
#include "g_shared/a_keys.h";

#define VIZIA_PLAYER players[consoleplayer]

struct player_t *viziaPlayer;
int viziaScreenHeight;
int viziaScreenWidth;



void Vizia_Init();

void Vizia_Tic();

void Vizia_Close();

#endif //VIZIA_MAIN_H
