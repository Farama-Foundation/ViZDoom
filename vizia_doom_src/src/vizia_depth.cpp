//
// Created by gregory on 22.12.15.
//

#include "vizia_depth.h"

depthBuffer::depthBuffer(unsigned int width, unsigned int height) : bufferHeight(height), bufferWidth(width), bufferSize(height*width) {
    buffer = new u_int8_t[bufferSize];
    for(int i=0;i<bufferSize;i++)
        buffer[i]=0;
}

depthBuffer::~depthBuffer() {
    delete buffer;
}

u_int8_t* depthBuffer::getBuffer() { return buffer; }

u_int8_t* depthBuffer::getBufferPoint(unsigned int x, unsigned int y) {
    if( x < bufferWidth && y < bufferHeight )
        return buffer + x + y*bufferWidth;
    else
    return NULL;}

void depthBuffer::setPoint(unsigned int x, unsigned int y, u_int8_t depth) {
    u_int8_t *dpth = getBufferPoint(x, y);
    if(dpth!=NULL)
        *dpth = depth;
}

unsigned int depthBuffer::getBufferSize(){
    return bufferSize;
}

unsigned int depthBuffer::getBufferWidth(){
    return bufferWidth;
}

unsigned int depthBuffer::getBufferHeight(){
    return bufferHeight;
}