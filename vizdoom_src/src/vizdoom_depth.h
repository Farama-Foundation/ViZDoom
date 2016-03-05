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

class depthBuffer{
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
    depthBuffer(unsigned int width, unsigned int height);
    ~depthBuffer();
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

extern depthBuffer* depthMap;
#endif
