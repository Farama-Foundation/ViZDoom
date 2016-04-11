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

#ifndef __VIZDOOM_EXCEPTIONS_H__
#define __VIZDOOM_EXCEPTIONS_H__

#include <exception>
#include <string>
#include <cstring>

namespace vizdoom{

    class Exception : public std::exception {
    public:
        virtual const char* what() const throw(){ return "Unknown exception."; }
    };

    class SharedMemoryException : public Exception {
    public:
        const char* what() const throw(){ return "Shared memory error."; }
    };

    class MessageQueueException : public Exception {
    public:
        const char* what() const throw(){ return "Message queue error."; }
    };

    class ViZDoomErrorException : public Exception {
    public:
        const char* what() const throw(){ return "Controlled ViZDoom instance reported error."; }
    };

    class ViZDoomMismatchedVersionException : public Exception {
    public:
        const char* what() const throw(){ return "Controlled ViZDoom version does not match API version."; }
    };

    class ViZDoomUnexpectedExitException : public Exception {
    public:
        const char* what() const throw(){ return "Controlled ViZDoom instance exited unexpectedly."; }
    };

    class ViZDoomIsNotRunningException : public Exception {
    public:
        const char* what() const throw(){ return "Controlled ViZDoom instance is not running or not ready."; }
    };

    class FileDoesNotExistException : public Exception {
    public:
        FileDoesNotExistException(std::string path){
            this->path = path;
        }
        ~FileDoesNotExistException() throw(){}
        const char* what() const throw(){
            std::string what = std::string("File \"") + this->path + "\" does not exist.";
            return strdup(what.c_str());
        }

    private:
        std::string path;
    };
}

#endif