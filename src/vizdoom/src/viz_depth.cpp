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

#include "viz_depth.h"
#include "v_video.h"

#ifdef VIZ_DEPTH_TEST
#include <SDL_events.h>
#endif
    
VIZDepthBuffer* vizDepthMap = NULL;

VIZDepthBuffer::VIZDepthBuffer(unsigned int width, unsigned int height):
        bufferSize(height*width), bufferWidth(width), bufferHeight(height) {

    buffer = new BYTE[bufferSize];
    memset(buffer, 0, bufferSize);

    this->setDepthBoundries(120000000, 358000);

    #ifdef VIZ_DEPTH_TEST
        for(int j = 0; j < 256; j++){

            #ifndef VIZ_DEPTH_COLORS
                colors[j].r = colors[j].g = colors[j].b = j;
            #else
                colors[j].r= j%3==0 ? 255 : 0;
                colors[j].g= j%3==1 ? 255 : 0;
                colors[j].b= j%3==2 ? 255 : 0;
                if(j%2==0)
                    colors[j].r=colors[j].g=colors[j].b = j;
                if(j%7==0){
                    colors[j].r = 100;
                    colors[j].g = 200;
                    colors[j].b = 255;

                }
            #endif
        }
        this->window = SDL_CreateWindow("ViZDoom Depth Buffer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);
        this->surface =  SDL_GetWindowSurface( window );
    #endif
}

VIZDepthBuffer::~VIZDepthBuffer() {
    delete[] buffer;

    #ifdef VIZ_DEPTH_TEST
        SDL_DestroyWindow(this->window);
        window = NULL;
        SDL_FreeSurface(this->surface);
        surface = NULL;
    #endif
}

// Get depth buffer pointer
BYTE* VIZDepthBuffer::getBuffer() { return buffer; }

// Get pointer for requested pixel (x, y coords)
BYTE* VIZDepthBuffer::getBufferPoint(unsigned int x, unsigned int y) {
    if( x < bufferWidth && y < bufferHeight )
        return buffer + x + y * bufferWidth;
    else return NULL;
}

// Set point(x,y) value with depth stored in actualDepth
void VIZDepthBuffer::setPoint(unsigned int x, unsigned int y) {
    this->setPoint(x, y, this->actualDepth);
}

// Set point(x,y) value with requested depth
void VIZDepthBuffer::setPoint(unsigned int x, unsigned int y, BYTE depth) {
    BYTE *dpth = getBufferPoint(x, y);
    if(dpth!=NULL) *dpth = depth;
}

// Store depth value for later usage
void VIZDepthBuffer::setActualDepth(BYTE depth) {
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

// Store depth value for later usage with automated conversion based on stored boundries
void VIZDepthBuffer::setActualDepthConv(int depth) {
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

// Increase or decrease stored depth value by adsb
void VIZDepthBuffer::updateActualDepth(int adsb) {
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
// Store x value for later usage
void VIZDepthBuffer::storeX(int x) {this->tX=x; }

// Store y value for later usage
void VIZDepthBuffer::storeY(int y) {this->tY=y; }

// Get stored x value
int VIZDepthBuffer::getX(void) {return this->tX; }

// Get stored y value
int VIZDepthBuffer::getY(void) {return this->tY; }

// Get buffer size
unsigned int VIZDepthBuffer::getBufferSize(){
    return this->bufferSize;
}

// Set boundries for storing depth value with conversion
void VIZDepthBuffer::setDepthBoundries(int maxDepth, int minDepth) {
    this->maxDepth=maxDepth;
    this->minDepth=minDepth;
    this->convSteps = (maxDepth-minDepth)/255;
}

// Get buffer width
unsigned int VIZDepthBuffer::getBufferWidth(){
    return bufferWidth;
}

// Get buffer height
unsigned int VIZDepthBuffer::getBufferHeight(){
    return bufferHeight;
}

// Clear buffer - set every point to 0
void VIZDepthBuffer::clearBuffer() {
    for(unsigned int i=0;i<bufferSize;i++)
        buffer[i]=0;
}

// Set every point to color value
void VIZDepthBuffer::clearBuffer(BYTE color) {
    for(unsigned int i=0;i<bufferSize;i++)
        buffer[i]=color;
}

void VIZDepthBuffer::lock() {this->locked=true; }

void VIZDepthBuffer::unlock() {this->locked=false; }

bool VIZDepthBuffer::isLocked() { return this->locked; }

void VIZDepthBuffer::sizeUpdate() {
    if(this->bufferWidth!= (unsigned int)screen->GetWidth() || this->bufferHeight!=(unsigned int)screen->GetHeight()) {
        delete[] this->buffer;
        this->bufferHeight=(unsigned)screen->GetHeight();
        this->bufferWidth= (unsigned)screen->GetWidth();
        this->bufferSize=this->bufferHeight*this->bufferWidth;
        this->buffer = new BYTE[this->bufferSize];

        #ifdef VIZ_DEPTH_TEST
            SDL_SetWindowSize(this->window, this->bufferWidth,this->bufferHeight);
            this->surface=SDL_GetWindowSurface(this->window);
        #endif
    }
}

#ifdef VIZ_DEPTH_TEST
// Update depth debug window
void VIZDepthBuffer::testUpdate() {
    SDL_Surface* surf = SDL_CreateRGBSurfaceFrom(this->buffer, this->bufferWidth, this->bufferHeight, 8,
                                                 this->bufferWidth, 0, 0, 0, 0);
    SDL_SetPaletteColors(surf->format->palette, colors, 0, 256);
    SDL_BlitSurface(surf, NULL, this->surface, NULL);
    SDL_UpdateWindowSurface(this->window);
}
#endif