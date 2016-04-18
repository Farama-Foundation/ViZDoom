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

#include "vizdoom_depth.h"
#include "v_video.h"

#ifdef VIZDOOM_DEPTH_TEST
#include <SDL_events.h>
#endif
    
ViZDoomDepthBuffer* depthMap = NULL;

ViZDoomDepthBuffer::ViZDoomDepthBuffer(unsigned int width, unsigned int height) : bufferSize(height*width), bufferWidth(width), bufferHeight(height) {
    buffer = new BYTE[bufferSize];
    for(unsigned int i=0;i<bufferSize;i++) {
        buffer[i] = 0;
    }
    this->setDepthBoundries(120000000, 358000);
#ifdef VIZDOOM_DEPTH_TEST
    for(int j = 0; j < 256; j++)
    {
#ifndef VIZDOOM_DEPTH_COLORS
        colors[j].r = colors[j].g = colors[j].b = j;
#else
        colors[j].r= j%3==0 ? 255 : 0;
        colors[j].g= j%3==1 ? 255 : 0;
        colors[j].b= j%3==2 ? 255 : 0;
        if(j%2==0)
            colors[j].r=colors[j].g=colors[j].b = j;
        if(j%7==0) {
            colors[j].r = 100;
            colors[j].g = 200;
            colors[j].b = 255;

        }
#endif
    }
    this->window = SDL_CreateWindow("Vizia Depth Buffer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width,
                                    height, SDL_WINDOW_SHOWN);
    this->surface =  SDL_GetWindowSurface( window );
#endif //VIZDOOM_DEPTH_TEST
}

ViZDoomDepthBuffer::~ViZDoomDepthBuffer() {
    delete[] buffer;
#ifdef VIZDOOM_DEPTH_TEST
    SDL_DestroyWindow( window );
    window = NULL;
    SDL_FreeSurface( surface);
    surface = NULL;
#endif //VIZDOOM_DEPTH_TEST
}
//get depth buffer pointer
BYTE* ViZDoomDepthBuffer::getBuffer() { return buffer; }

//get pointer for requested pixel (x, y coords)
BYTE* ViZDoomDepthBuffer::getBufferPoint(unsigned int x, unsigned int y) {
    if( x < bufferWidth && y < bufferHeight )
        return buffer + x + y*bufferWidth;
    else
    return NULL;}

//set point(x,y) value with depth stored in actualDepth
void ViZDoomDepthBuffer::setPoint(unsigned int x, unsigned int y) {
    BYTE *dpth = getBufferPoint(x, y);
    if(dpth!=NULL)
        *dpth = actualDepth;
}

//set point(x,y) value with requested depth
void ViZDoomDepthBuffer::setPoint(unsigned int x, unsigned int y, BYTE depth) {
    BYTE *dpth = getBufferPoint(x, y);
    if(dpth!=NULL)
        *dpth = depth;
}

//store depth value for later usage
void ViZDoomDepthBuffer::setActualDepth(BYTE depth) {
    if(this->isLocked())
        return;
    if(this->bufferHeight==480)
        this->actualDepth=depth;
    else {
        int dpth=depth;
        dpth *= (double) this->bufferHeight / 480;
        if(dpth>255)
            dpth=255;
        this->actualDepth=(BYTE) dpth;
    }
}

//store depth value for later usage with automated conversion based on stored boundries
void ViZDoomDepthBuffer::setActualDepthConv(int depth) {
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
void ViZDoomDepthBuffer::updateActualDepth(int adsb) {
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
void ViZDoomDepthBuffer::storeX(int x) {this->tX=x; }

//store y value for later usage
void ViZDoomDepthBuffer::storeY(int y) {this->tY=y; }

//get stored x value
int ViZDoomDepthBuffer::getX(void) {return this->tX; }

//get stored y value
int ViZDoomDepthBuffer::getY(void) {return this->tY; }

//get buffer size
unsigned int ViZDoomDepthBuffer::getBufferSize(){
    return this->bufferSize;
}

//set boundries for storing depth value with conversion
void ViZDoomDepthBuffer::setDepthBoundries(int maxDepth, int minDepth) {
    this->maxDepth=maxDepth;
    this->minDepth=minDepth;
    this->convSteps = (maxDepth-minDepth)/255;
}

//get buffer width
unsigned int ViZDoomDepthBuffer::getBufferWidth(){
    return bufferWidth;
}

//get buffer height
unsigned int ViZDoomDepthBuffer::getBufferHeight(){
    return bufferHeight;
}

//clear buffer - set every point to 0
void ViZDoomDepthBuffer::clearBuffer() {
    for(unsigned int i=0;i<bufferSize;i++)
        buffer[i]=0;
}

//set every point to color value
void ViZDoomDepthBuffer::clearBuffer(BYTE color) {
    for(unsigned int i=0;i<bufferSize;i++)
        buffer[i]=color;
}

void ViZDoomDepthBuffer::lock() {this->locked=true; }

void ViZDoomDepthBuffer::unlock() {this->locked=false; }

bool ViZDoomDepthBuffer::isLocked() { return this->locked; }

void ViZDoomDepthBuffer::sizeUpdate() {
    if(this->bufferWidth!= (unsigned int)screen->GetWidth() || this->bufferHeight!=(unsigned int)screen->GetHeight())
    {
        delete[] this->buffer;
        this->bufferHeight=(unsigned)screen->GetHeight();
        this->bufferWidth= (unsigned)screen->GetWidth();
        this->bufferSize=this->bufferHeight*this->bufferWidth;
        this->buffer = new BYTE[this->bufferSize];
#ifdef VIZDOOM_DEPTH_TEST
        SDL_SetWindowSize(this->window, this->bufferWidth,this->bufferHeight);
        this->surface=SDL_GetWindowSurface(this->window);
#endif
    }
}

#ifdef VIZDOOM_DEPTH_TEST
//update depth debug window
void ViZDoomDepthBuffer::Update() {
    SDL_Surface* surf = SDL_CreateRGBSurfaceFrom(this->buffer, this->bufferWidth, this->bufferHeight, 8,
                                                 this->bufferWidth, 0, 0, 0, 0);
    SDL_SetPaletteColors(surf->format->palette, colors, 0, 256);
    SDL_BlitSurface(surf, NULL, this->surface, NULL);
    SDL_UpdateWindowSurface(this->window);
}
#endif