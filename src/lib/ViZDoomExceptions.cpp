/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch
 Copyright (C) 2017 - 2022 by Marek Wydmuch, Michał Kempka, Wojciech Jaśkowski, and the respective contributors
 Copyright (C) 2023 - 2024 by Marek Wydmuch, Farama Foundation, and the respective contributors

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

#include "ViZDoomExceptions.h"
#include <cstring>

namespace vizdoom {

    /* FileDoesNotExistException */
    const char *FileDoesNotExistException::what() const throw() {
        std::string what = std::string("File \"") + this->path + "\" does not exist.";
        return strdup(what.c_str());
    }

    /* MessageQueueException */
    const char *MessageQueueException::what() const throw() {
        if (this->error.length()) return this->error.c_str();
        else return "Unknown message queue error.";
    }

    /* SharedMemoryException */
    const char *SharedMemoryException::what() const throw() {
        if (this->error.length()) return this->error.c_str();
        else return "Unknown shared memory error.";
    }

    /* ViZDoomErrorException */
    const char *ViZDoomErrorException::what() const throw() {
        if (this->error.length()) return this->error.c_str();
        else return "Controlled ViZDoom instance unknown error.";
    }

    /* ViZDoomIsNotRunningException */
    const char *ViZDoomIsNotRunningException::what() const throw() {
        return "Controlled ViZDoom instance is not running or not ready.";
    }

    /* ViZDoomUnexpectedExitException */
    const char *ViZDoomUnexpectedExitException::what() const throw() {
        return "Controlled ViZDoom instance exited unexpectedly.";
    }

    /* ViZDoomNoOpenALSoundException */
    const char *ViZDoomNoOpenALSoundException::what() const throw() {
        return "Controlled ViZDoom instance requires OpenAL sound, however it was not correctly initialized.";
    }

}