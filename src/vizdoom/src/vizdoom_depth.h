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

#ifndef __VIZDOOM_DEPTH_H__
#define __VIZDOOM_DEPTH_H__

//UNCOMMENT TO ENABLE DEPTH BUFFER DEBUG WINDOW
//#define VIZDOOM_DEPTH_TEST 1

//UNCOMMENT TO ENABLE COLOR-BASED DEPTH TEST
//#define VIZDOOM_DEPTH_COLORS 1

#include "basictypes.h"

#ifdef VIZDOOM_DEPTH_TEST
#include <SDL_video.h>
#endif

class ViZDoomDepthBuffer{
public:
    BYTE *getBuffer();
    BYTE *getBufferPoint(unsigned int x, unsigned int y);
    void setPoint(unsigned int x, unsigned int y, BYTE depth);
    void setPoint(unsigned int x, unsigned int y);
    void setActualDepth(BYTE depth);
    void setActualDepthConv(int depth);
    void setDepthBoundries(int maxDepth, int minDepth);
    void updateActualDepth(int adsb);
    void storeX(int x);
    void storeY(int y);
    int getX(void);
    int getY(void);
    ViZDoomDepthBuffer(unsigned int width, unsigned int height);
    ~ViZDoomDepthBuffer();
    unsigned int getBufferSize();
    unsigned int getBufferWidth();
    unsigned int getBufferHeight();
    void clearBuffer();
    void clearBuffer(BYTE color);
    void lock();
    void unlock();
    bool isLocked();
    void sizeUpdate();
    unsigned int helperBuffer[4];
#ifdef VIZDOOM_DEPTH_TEST
    void Update();
#endif
private:
    BYTE *buffer;
    unsigned int bufferSize;
    unsigned int bufferWidth;
    unsigned int bufferHeight;
    BYTE actualDepth;
    int maxDepth;
    int minDepth;
    int convSteps;
    int tX, tY;
    bool locked;
#ifdef VIZDOOM_DEPTH_TEST
    SDL_Window* window;
    SDL_Surface* surface;
    SDL_Color colors[256];
#endif
};

extern ViZDoomDepthBuffer* depthMap;
#endif
