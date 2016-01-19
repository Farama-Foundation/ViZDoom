//
// Created by gregory on 22.12.15.
//
#include <SDL_events.h>
#include "vizia_depth.h"

depthBuffer* depthMap = NULL;

depthBuffer::depthBuffer(unsigned int width, unsigned int height) : bufferSize(height*width), bufferWidth(width), bufferHeight(height) {
    buffer = new u_int8_t[bufferSize];
    for(unsigned int i=0;i<bufferSize;i++) {
        buffer[i] = 0;
    }
#ifdef VIZIA_DEPTH_TEST
    for(int j = 0; j < 256; j++)
    {
        colors[j].r = colors[j].g = colors[j].b = j;
    }
    this->window = SDL_CreateWindow("Vizia Depth Buffer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width,
                                    height, SDL_WINDOW_SHOWN);
    this->surface =  SDL_GetWindowSurface( window );
#endif //VIZIA_DEPTH_TEST
}

depthBuffer::~depthBuffer() {
    delete[] buffer;
#ifdef VIZIA_DEPTH_TEST
    SDL_DestroyWindow( window );
    window = NULL;
    SDL_FreeSurface( surface);
    surface = NULL;
#endif //VIZIA_DEPTH_TEST
}
//get depth buffer pointer
u_int8_t* depthBuffer::getBuffer() { return buffer; }

//get pointer for requested pixel (x, y coords)
u_int8_t* depthBuffer::getBufferPoint(unsigned int x, unsigned int y) {
    if( x < bufferWidth && y < bufferHeight )
        return buffer + x + y*bufferWidth;
    else
    return NULL;}

//set point(x,y) value with depth stored in actualDepth
void depthBuffer::setPoint(unsigned int x, unsigned int y) {
    u_int8_t *dpth = getBufferPoint(x, y);
    if(dpth!=NULL)
        *dpth = actualDepth;
}

//set point(x,y) value with requested depth
void depthBuffer::setPoint(unsigned int x, unsigned int y, u_int8_t depth) {
    u_int8_t *dpth = getBufferPoint(x, y);
    if(dpth!=NULL)
        *dpth = depth;
}

//store depth value for later usage
void depthBuffer::setActualDepth(u_int8_t depth) {
    if(this->isLocked())
        return;
    this->actualDepth=depth;
}

//store depth value for later usage with automated conversion based on stored boundries
void depthBuffer::setActualDepthConv(int depth) {
    if(this->isLocked())
        return;
    if(depth>maxDepth)
        this->actualDepth=255;
    else if(depth<minDepth)
        this->actualDepth=0;
    else {
        depth-=minDepth;
        this->actualDepth = (unsigned int)  (depth-minDepth) / this->convSteps;
    }
}

//increase or decrease stored depth value by adsb
void depthBuffer::updateActualDepth(int adsb) {
    int act = this->actualDepth;
    if(this->isLocked())
        return;
    if(act+adsb>255)
        this->actualDepth=255;
    else if(act+adsb<0)
        this->actualDepth=0;
    else
        this->actualDepth+=adsb;
}
//store x value for later usage
void depthBuffer::storeX(int x) {this->tX=x; }

//store y value for later usage
void depthBuffer::storeY(int y) {this->tY=y; }

//get stored x value
int depthBuffer::getX(void) {return this->tX; }

//get stored y value
int depthBuffer::getY(void) {return this->tY; }

//get buffer size
unsigned int depthBuffer::getBufferSize(){
    return this->bufferSize;
}

//set boundries for storing depth value with conversion
void depthBuffer::setDepthBoundries(int maxDepth, int minDepth) {
    this->maxDepth=maxDepth;
    this->minDepth=minDepth;
    this->convSteps = (maxDepth-minDepth)/255;
}

//get buffer width
unsigned int depthBuffer::getBufferWidth(){
    return bufferWidth;
}

//get buffer height
unsigned int depthBuffer::getBufferHeight(){
    return bufferHeight;
}

//clear buffer - set every point to 0
void depthBuffer::clearBuffer() {
    for(unsigned int i=0;i<bufferSize;i++)
        buffer[i]=0;
}

//set every point to color value
void depthBuffer::clearBuffer(u_int8_t color) {
    for(unsigned int i=0;i<bufferSize;i++)
        buffer[i]=color;
}

void depthBuffer::lock() {this->locked=true; }

void depthBuffer::unlock() {this->locked=false; }

bool depthBuffer::isLocked() { return this->locked; }

#ifdef VIZIA_DEPTH_TEST
//update depth debug window
void depthBuffer::Update() {
    SDL_Surface* surf = SDL_CreateRGBSurfaceFrom(this->buffer, this->bufferWidth, this->bufferHeight, 8,
                                                 this->bufferWidth, 0, 0, 0, 0);
    SDL_SetPaletteColors(surf->format->palette, colors, 0, 256);
    SDL_BlitSurface(surf, NULL, this->surface, NULL);
    SDL_UpdateWindowSurface(this->window);
}
#endif