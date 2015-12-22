//
// Created by gregory on 22.12.15.
//

#ifndef VIZIAAPI_VIZIA_DEPTH_H
#define VIZIAAPI_VIZIA_DEPTH_H

#include <sys/types.h>

class depthBuffer{
public:
    u_int8_t *getBuffer();
    u_int8_t *getBufferPoint(unsigned int x, unsigned int y);
    void setPoint(unsigned int x, unsigned int y, u_int8_t depth);
    depthBuffer(unsigned int width, unsigned int height);
    ~depthBuffer();
    unsigned int getBufferSize();
    unsigned int getBufferWidth();
    unsigned int getBufferHeight();
private:
    u_int8_t *buffer;
    unsigned int bufferSize;
    unsigned int bufferWidth;
    unsigned int bufferHeight;
};

#endif //VIZIAAPI_VIZIA_DEPTH_H
