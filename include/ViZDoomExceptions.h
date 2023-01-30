/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch
 Copyright (C) 2017 - 2022 by Marek Wydmuch, Michał Kempka, Wojciech Jaśkowski, and the respective contributors

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

namespace vizdoom{

    class FileDoesNotExistException : public std::exception {
    public:
        FileDoesNotExistException(std::string path): path(path){}
        ~FileDoesNotExistException() throw(){}
        const char* what() const throw();
    private:
        std::string path;
    };

    class MessageQueueException : public std::exception {
    public:
        MessageQueueException(){}
        MessageQueueException(std::string error): error(error){}
        ~MessageQueueException() throw(){}
        const char* what() const throw();
    private:
        std::string error;
    };

    class SharedMemoryException : public std::exception {
    public:
        SharedMemoryException(){}
        SharedMemoryException(std::string error): error(error){}
        ~SharedMemoryException() throw(){}
        const char* what() const throw();
    private:
        std::string error;
    };

    class SignalException : public std::exception {
    public:
        SignalException(std::string signal): signal(signal){}
        ~SignalException() throw(){}
        const char* what() const throw();
    private:
        std::string signal;
    };

    class ViZDoomErrorException : public std::exception {
    public:
        ViZDoomErrorException(){}
        ViZDoomErrorException(std::string error): error(error){}
        ~ViZDoomErrorException() throw(){}
        const char* what() const throw();
    private:
        std::string error;
    };

    class ViZDoomIsNotRunningException : public std::exception {
    public:
        const char* what() const throw();
    };

    class ViZDoomUnexpectedExitException : public std::exception {
    public:
        const char* what() const throw();
    };
}

#endif
