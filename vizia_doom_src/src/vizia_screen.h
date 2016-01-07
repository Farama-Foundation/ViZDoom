#ifndef __VIZIA_SCREEN_H__
#define __VIZIA_SCREEN_H__

#include <stddef.h>

extern unsigned int viziaScreenWidth;
extern unsigned int viziaScreenHeight;
extern size_t viziaScreenPitch;
extern size_t viziaScreenSize;

void Vizia_ScreenInit();

void Vizia_ScreenUpdate();

void Vizia_ScreenClose();

#endif
