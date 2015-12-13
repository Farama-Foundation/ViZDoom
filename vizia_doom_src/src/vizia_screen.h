#ifndef __VIZIA_SCREEN_H__
#define __VIZIA_SCREEN_H__

#include <stddef.h>

#define VIZIA_SCREEN_CRCGCB 0
#define VIZIA_SCREEN_CRCGCBCA 1
#define VIZIA_SCREEN_RGB24 2
#define VIZIA_SCREEN_RGBA32 3
#define VIZIA_SCREEN_ARGB32 4
#define VIZIA_SCREEN_CBCGCR 5
#define VIZIA_SCREEN_CBCGCRCA 6
#define VIZIA_SCREEN_BGR24 7
#define VIZIA_SCREEN_BGRA32 8
#define VIZIA_SCREEN_ABGR32 9
#define VIZIA_SCREEN_GRAY8 10
#define VIZIA_SCREEN_DOOM_256_COLORS 11

extern unsigned int viziaScreenWidth;
extern unsigned int viziaScreenHeight;
extern size_t viziaScreenPitch;
extern size_t viziaScreenSize;

void Vizia_ScreenInit();

void Vizia_ScreenUpdate();

void Vizia_ScreenClose();

#endif
