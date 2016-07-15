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

#include "viz_labels.h"
#include "v_video.h"

#ifdef VIZ_LABELS_TEST
#include <SDL_events.h>
#endif
    
VIZLabelsBuffer* vizLabels = NULL;

VIZLabelsBuffer::VIZLabelsBuffer(unsigned int width, unsigned int height):
        bufferSize(height * width), bufferWidth(width), bufferHeight(height) {

    buffer = new BYTE[bufferSize];
    memset(buffer, 0, bufferSize);
//    for(unsigned int i=0; i<bufferSize; ++i) {
//        buffer[i] = 0;
//    }

    this->nextLabel = 255;

    #ifdef VIZ_LABELS_TEST
        for(int j = 0; j < 256; j++)
        {
            #ifndef VIZ_LABELS_COLORS
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
        this->window = SDL_CreateWindow("ViZDoom Labels Buffer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);
        this->surface =  SDL_GetWindowSurface( window );
    #endif
}

VIZLabelsBuffer::~VIZLabelsBuffer() {
    delete[] buffer;

    #ifdef VIZ_LABELS_TEST
        SDL_DestroyWindow( window );
        window = NULL;
        SDL_FreeSurface( surface);
        surface = NULL;
    #endif
}

// Get labels buffer pointer
BYTE* VIZLabelsBuffer::getBuffer() { return buffer; }

// Get pointer for requested pixel (x, y coords)
BYTE* VIZLabelsBuffer::getBufferPoint(unsigned int x, unsigned int y) {
    if( x < bufferWidth && y < bufferHeight )
        return buffer + x + y*bufferWidth;
    else
    return NULL;
}

// Set point(x,y) value with next label
void VIZLabelsBuffer::setPoint(unsigned int x, unsigned int y) {
    this->setPoint(x, y, this->currentLabel);
}

// Set point(x,y) value with requested label
void VIZLabelsBuffer::setPoint(unsigned int x, unsigned int y, BYTE label) {
    BYTE *map = getBufferPoint(x, y);
    if(map != NULL) *map = label;
}

// Get buffer size
unsigned int VIZLabelsBuffer::getBufferSize(){
    return this->bufferSize;
}

// Get buffer width
unsigned int VIZLabelsBuffer::getBufferWidth(){
    return bufferWidth;
}

// Get buffer height
unsigned int VIZLabelsBuffer::getBufferHeight(){
    return bufferHeight;
}

// Clear buffer - set every point to 0
void VIZLabelsBuffer::clearBuffer() {
    this->clearBuffer(0);
}

// Set every point to color value
void VIZLabelsBuffer::clearBuffer(BYTE color) {
    memset(buffer, color, bufferSize);

    this->toLabel.clear();
    this->labeled = 0;
}

void VIZLabelsBuffer::lock() {this->locked=true; }

void VIZLabelsBuffer::unlock() {this->locked=false; }

bool VIZLabelsBuffer::isLocked() { return this->locked; }

void VIZLabelsBuffer::sizeUpdate() {
    if(this->bufferWidth != (unsigned int)screen->GetWidth() || this->bufferHeight!=(unsigned int)screen->GetHeight()) {
        delete[] this->buffer;
        this->bufferHeight=(unsigned)screen->GetHeight();
        this->bufferWidth= (unsigned)screen->GetWidth();
        this->bufferSize=this->bufferHeight*this->bufferWidth;
        this->buffer = new BYTE[this->bufferSize];

        #ifdef VIZ_LABELS_TEST
            SDL_SetWindowSize(this->window, this->bufferWidth,this->bufferHeight);
            this->surface=SDL_GetWindowSurface(this->window);
        #endif
    }
}

void VIZLabelsBuffer::addSprite(AActor *thing, vissprite_t* vis){
    VIZToLabel newSprite;
    newSprite.actor = thing;
    newSprite.psrpite = false;
    newSprite.vissprite = vis;

    this->toLabel.push_back(newSprite);
}

void VIZLabelsBuffer::addPSprite(AActor *thing, vissprite_t* vis){
    VIZToLabel newSprite;
    newSprite.actor = thing;
    newSprite.psrpite = true;
    newSprite.vissprite = vis;

    this->toLabel.push_back(newSprite);
}

BYTE VIZLabelsBuffer::getLabel(vissprite_t* vis){
    for(auto i = this->toLabel.begin(); i != this->toLabel.end(); ++i){
        if(i->vissprite == vis){
            if(i->psrpite) i->label = VIZ_MAX_LABELS - 1;
            else{
                ++labeled;
                i->label = labeled * (VIZ_MAX_LABELS - 1) / (this->toLabel.size() + 1);
            }
            return i->label;
        }
    }
    return 0;
}

void VIZLabelsBuffer::setLabel(BYTE label){
    this->currentLabel = label;
}

#ifdef VIZ_LABELS_TEST
// Update labels debug window
void VIZLabelsBuffer::testUpdate() {
    SDL_Surface* surf = SDL_CreateRGBSurfaceFrom(this->buffer, this->bufferWidth, this->bufferHeight, 8, this->bufferWidth, 0, 0, 0, 0);
    SDL_SetPaletteColors(surf->format->palette, colors, 0, 256);
    SDL_BlitSurface(surf, NULL, this->surface, NULL);
    SDL_UpdateWindowSurface(this->window);
}
#endif