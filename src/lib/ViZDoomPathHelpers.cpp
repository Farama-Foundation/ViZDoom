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

#include "ViZDoomPathHelpers.h"
#include "ViZDoomExceptions.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

namespace vizdoom {

    namespace b     = boost;
    namespace bfs   = boost::filesystem;

    std::string fileExtension(std::string filePath){
        bfs::path path(filePath);
        return path.extension().string();
    }

    bool hasExtension(std::string filePath){
        bfs::path path(filePath);
        return path.has_extension();
    }

    bool fileExists(std::string filePath){
        bfs::path path(filePath);
        return bfs::exists(path) && bfs::is_regular_file(path);
    }

    std::string pathFromBase(std::string basePath, std::string relativePath){
        bfs::path base(basePath);
        base.remove_filename();
        bfs::path relative(relativePath);
        bfs::path out = bfs::canonical(relative, base);
        return out.string();
    }

    std::string prepareFilePath(std::string filePath){
        b::trim_left_if(filePath, b::is_any_of(" \n\r\""));
        b::trim_right_if(filePath, b::is_any_of(" \n\r\""));
        if(b::find_first(filePath, " ")) filePath = '\"' + filePath  + '\"';
        return filePath;
    }

    std::string prepareExeFilePath(std::string filePath){
        if(!fileExists(filePath)){
        #ifdef OS_WIN
            if(hasExtension(filePath)) throw FileDoesNotExistException(filePath);
            if(!fileExists(filePath + ".exe")) throw FileDoesNotExistException(filePath + "(.exe)");
            filePath += ".exe";
        #else
            throw FileDoesNotExistException(filePath);
        #endif
        }

        return prepareFilePath(filePath);
    }

    std::string prepareWadFilePath(std::string filePath){
        return prepareFilePath(filePath);
    }

    std::string prepareLmpFilePath(std::string filePath){
        if(!fileExists(filePath)){
            if(hasExtension(filePath)) throw FileDoesNotExistException(filePath);
            if(!fileExists(filePath + ".lmp")) throw FileDoesNotExistException(filePath + "(.lmp)");
            filePath += ".lmp";
        }

        return prepareFilePath(filePath);
    }

}