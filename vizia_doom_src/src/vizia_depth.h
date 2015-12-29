//
// Created by gregory on 22.12.15.
//

#ifndef VIZIAAPI_VIZIA_DEPTH_H
#define VIZIAAPI_VIZIA_DEPTH_H

#include <sys/types.h>
#include <stddef.h>



class depthBuffer{
public:
    u_int8_t *getBuffer();
    u_int8_t *getBufferPoint(unsigned int x, unsigned int y);
    void setPoint(unsigned int x, unsigned int y, u_int8_t depth);
    void setPoint(unsigned int x, unsigned int y);
    void setActualDepth(u_int8_t depth);
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
    void clearBuffer(u_int8_t color);
private:
    u_int8_t *buffer;
    unsigned int bufferSize;
    unsigned int bufferWidth;
    unsigned int bufferHeight;
    u_int8_t actualDepth;
    int maxDepth;
    int minDepth;
    int convSteps;
    int tX, tY;
};

extern depthBuffer* depthMap;
#endif //VIZIAAPI_VIZIA_DEPTH_H
