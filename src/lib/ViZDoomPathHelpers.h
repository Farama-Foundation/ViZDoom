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

#ifndef __VIZDOOM_PATHHELPERS_H__
#define __VIZDOOM_PATHHELPERS_H__

#include <string>

/* OSes */
#ifdef __linux__
    #define OS_LINUX
#elif _WIN32
    #define OS_WIN
#elif __APPLE__
    #define OS_OSX
#endif

namespace vizdoom {

    std::string fileExtension(std::string filePath);

    bool hasExtension(std::string filePath);

    bool fileExists(std::string filePath);

    std::string relativePath(std::string relativePath, std::string basePath = "");

    std::string checkFile(std::string filePath, std::string expectedExt = "");

    std::string prepareFilePathArg(std::string filePath);

    std::string prepareFilePathCmd(std::string filePath);

    std::string prepareExeFilePath(std::string filePath);

    std::string prepareWadFilePath(std::string filePath);

    std::string prepareLmpFilePath(std::string filePath);

    std::string getThisSharedObjectPath();
}

#endif