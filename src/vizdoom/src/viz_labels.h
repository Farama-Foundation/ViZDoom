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

#ifndef __VIZ_LABELS_H__
#define __VIZ_LABELS_H__

//UNCOMMENT TO ENABLE LABELS BUFFER DEBUG WINDOW
//#define VIZ_LABELS_TEST 1

//UNCOMMENT TO ENABLE COLOR-BASED LABELS TEST
//#define VIZ_LABELS_COLORS 1

#include <vector>
#include <unordered_map>

#ifdef VIZ_LABELS_TEST
#include <SDL_video.h>
#endif

#include "basictypes.h"
#include "doomtype.h"
#include "actor.h"

#include "r_main.h"
#include "r_plane.h"
#include "r_draw.h"
#include "r_things.h"
#include "r_3dfloors.h"
#include "a_sharedglobal.h"
#include "g_level.h"

struct VIZSprite{
    AActor* actor;
    bool psprite;
    vissprite_t* vissprite;
    bool labeled;
    BYTE label;
    unsigned int minX;
    unsigned int maxX;
    unsigned int minY;
    unsigned int maxY;
    unsigned int pointCount;

    VIZSprite(){
        this->actor = NULL;
        this->vissprite = NULL;
        this->psprite = false;
        this->labeled = false;
        this->label = 0;
        this->minX = (unsigned int)-1;
        this->minY = (unsigned int)-1;
        this->maxX = 0;
        this->maxY = 0;
        this->pointCount = 0;
    };
};

class VIZLabelsBuffer{
public:

    VIZLabelsBuffer(unsigned int width, unsigned int height);
    ~VIZLabelsBuffer();

    BYTE *getBuffer();
    BYTE *getBufferPoint(unsigned int x, unsigned int y);
    void setPoint(unsigned int x, unsigned int y, BYTE label);
    void setPoint(unsigned int x, unsigned int y);
    unsigned int getBufferSize();
    unsigned int getBufferWidth();
    unsigned int getBufferHeight();
    void clearBuffer();
    void clearBuffer(BYTE color);
    void lock();
    void unlock();
    bool isLocked();
    void sizeUpdate();

    void updateBoundingBox(unsigned int x, unsigned int y);

    void addSprite(AActor *thing, vissprite_t* vis);
    void addPSprite(AActor *thing, vissprite_t* vis);
    BYTE getLabel(VIZSprite* sprite);
    BYTE getLabel(vissprite_t* vis);
    void setLabel(BYTE label);
    void setSprite(vissprite_t* vis);
    void unsetSprite();

    std::vector<VIZSprite> getSprites();

    #ifdef VIZ_LABELS_TEST
        void testUpdate();
    #endif

    std::vector<VIZSprite> sprites;

private:

    BYTE *buffer;
    unsigned int bufferSize;
    unsigned int bufferWidth;
    unsigned int bufferHeight;
    bool locked;

    BYTE labeled;
    BYTE currentLabel;
    VIZSprite *currentSprite;
    VIZSprite *pSprite;

    #ifdef VIZ_LABELS_TEST
        SDL_Window* window;
        SDL_Surface* surface;
        SDL_Color colors[256];
    #endif
};

extern VIZLabelsBuffer* vizLabels;
#endif
