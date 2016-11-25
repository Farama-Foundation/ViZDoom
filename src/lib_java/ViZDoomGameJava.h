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

#ifndef __VIZDOOM_GAME_JAVA_H__
#define __VIZDOOM_GAME_JAVA_H__

#include "ViZDoomJava.h"

namespace vizdoom {

    class DoomGameJava : public DoomGame {
    public:

        // Java doesn't support C++ 11 default arguments

        void newEpisode_() { this->newEpisode(); };
        void newEpisode_str(std::string _str) { this->newEpisode(_str); };

        double makeAction_vec(std::vector<int>& _vec){ return this->makeAction(_vec); }
        double makeAction_vec_int(std::vector<int>& _vec, unsigned int _int){ return this->makeAction(_vec, _int); }

        void advanceAction_() { this->advanceAction(); }
        void advanceAction_int(unsigned int _int) { this->advanceAction(_int); }
        void advanceAction_int_bool(unsigned int _int, bool _bool) { this->advanceAction(_int, _bool); }

        void addAvailableButton_btn(Button _btn) { this->addAvailableButton(_btn); }
        void addAvailableButton_btn_int(Button _btn, unsigned int _int) { this->addAvailableButton(_btn, _int); }

        void replayEpisode_str(std::string _str) { this->replayEpisode(_str); }
        void replayEpisode_str_int(std::string _str, unsigned int _int) { this->replayEpisode(_str, _int); }

    };
}

#define JAVA_PACKAGE vizdoom
#define JAVA_CLASS DoomGame
#define CPP_CLASS DoomGameJava

#ifdef __cplusplus
extern "C" {
#endif

JNI_EXPORT(void, DoomGameNative);
JNI_EXPORT(jboolean, loadConfig, jstring);
JNI_EXPORT(jboolean, init);
JNI_EXPORT(void, close);
JNI_EXPORT(void, newEpisode__);
JNI_EXPORT(void, newEpisode__Ljava_lang_String, jstring);
JNI_EXPORT(void, replayEpisode__Ljava_lang_String_2, jstring);
JNI_EXPORT(void, replayEpisode__Ljava_lang_String_2I, jstring, jint);
JNI_EXPORT(jboolean, isRunning);
JNI_EXPORT(void, setAction, jintArray);
JNI_EXPORT(void, advanceAction__);
JNI_EXPORT(void, advanceAction__I, jint);
JNI_EXPORT(void, advanceAction__IZ, jint, jboolean);
JNI_EXPORT(jdouble, makeAction___3I, jintArray);
JNI_EXPORT(jdouble, makeAction___3II, jintArray, jint);
JNI_EXPORT(jobject, getState);
JNI_EXPORT(jintArray, getLastAction);
JNI_EXPORT(jboolean, isNewEpisode);
JNI_EXPORT(jboolean, isEpisodeFinished);
JNI_EXPORT(jboolean, isPlayerDead);
JNI_EXPORT(void, respawnPlayer);

JNI_EXPORT(jobjectArray, getAvailableButtons);
JNI_EXPORT(void, setAvailableButtons, jobjectArray);
JNI_EXPORT(void, addAvailableButton__Lvizdoom_Button_2, jobject);
JNI_EXPORT(void, addAvailableButton__Lvizdoom_Button_2I, jobject, jint);
JNI_EXPORT(void, clearAvailableButtons);
JNI_EXPORT(jint, getAvailableButtonsSize);
JNI_EXPORT(void, setButtonMaxValue, jobject, jint);
JNI_EXPORT(jint, getButtonMaxValue, jobject);

JNI_EXPORT(jobjectArray, getAvailableGameVariables);
JNI_EXPORT(void, setAvailableGameVariables, jobjectArray);
JNI_EXPORT(void, addAvailableGameVariable, jobject);
JNI_EXPORT(void, clearAvailableGameVariables);
JNI_EXPORT(jint, getAvailableGameVariablesSize);

JNI_EXPORT(void, addGameArgs, jstring);
JNI_EXPORT(void, clearGameArgs);
JNI_EXPORT(void, sendGameCommand, jstring);
JNI_EXPORT(jint, getModeNative);
JNI_EXPORT(void, setMode, jobject);
JNI_EXPORT(jint, getTicrate);
JNI_EXPORT(void, setTicrate, jint);
JNI_EXPORT(jdouble, getGameVariable, jobject);
JNI_EXPORT(jdouble, getLivingReward);
JNI_EXPORT(void, setLivingReward, jdouble);
JNI_EXPORT(jdouble, getDeathPenalty);
JNI_EXPORT(void, setDeathPenalty, jdouble);
JNI_EXPORT(jdouble, getLastReward);
JNI_EXPORT(jdouble, getTotalReward);

JNI_EXPORT(void, setViZDoomPath, jstring);
JNI_EXPORT(void, setDoomGamePath, jstring);
JNI_EXPORT(void, setDoomScenarioPath, jstring);
JNI_EXPORT(void, setDoomMap, jstring);
JNI_EXPORT(void, setDoomSkill, jint);
JNI_EXPORT(void, setDoomConfigPath, jstring);
JNI_EXPORT(jint, getSeed);
JNI_EXPORT(void, setSeed, jint);
JNI_EXPORT(jint, getEpisodeStartTime);
JNI_EXPORT(void, setEpisodeStartTime, jint);
JNI_EXPORT(jint, getEpisodeTimeout);
JNI_EXPORT(void, setEpisodeTimeout, jint);
JNI_EXPORT(jint, getEpisodeTime);

JNI_EXPORT(jboolean, isDepthBufferEnabled);
JNI_EXPORT(void, setDepthBufferEnabled, jboolean);

JNI_EXPORT(jboolean, isLabelsBufferEnabled);
JNI_EXPORT(void, setLabelsBufferEnabled, jboolean);

JNI_EXPORT(jboolean, isAutomapBufferEnabled);
JNI_EXPORT(void, setAutomapBufferEnabled, jboolean);
JNI_EXPORT(void, setAutomapMode, jobject);
JNI_EXPORT(void, setAutomapRotate, jboolean);
JNI_EXPORT(void, setAutomapRenderTextures, jboolean);

JNI_EXPORT(void, setScreenResolution, jobject);
JNI_EXPORT(void, setScreenFormat, jobject);
JNI_EXPORT(void, setRenderHud, jboolean);
JNI_EXPORT(void, setRenderMinimalHud, jboolean);
JNI_EXPORT(void, setRenderWeapon, jboolean);
JNI_EXPORT(void, setRenderCrosshair, jboolean);
JNI_EXPORT(void, setRenderDecals, jboolean);
JNI_EXPORT(void, setRenderParticles, jboolean);
JNI_EXPORT(void, setRenderEffectsSprites, jboolean);
JNI_EXPORT(void, setRenderMessages, jboolean);
JNI_EXPORT(void, setRenderCorpses, jboolean);
JNI_EXPORT(void, setWindowVisible, jboolean);
JNI_EXPORT(void, setConsoleEnabled, jboolean);
JNI_EXPORT(void, setSoundEnabled, jboolean);

JNI_EXPORT(jint, getScreenWidth);
JNI_EXPORT(jint, getScreenHeight);
JNI_EXPORT(jint, getScreenChannels);
JNI_EXPORT(jint, getScreenPitch);
JNI_EXPORT(jint, getScreenSize);
JNI_EXPORT(jint, getScreenFormatNative);

JNI_EXPORT(jdouble, doomTics2Ms, jdouble, jint);
JNI_EXPORT(jdouble, ms2DoomTics, jdouble, jint);
JNI_EXPORT(jdouble, doomTics2Sec, jdouble, jint);
JNI_EXPORT(jdouble, sec2DoomTics, jdouble, jint);
JNI_EXPORT(jdouble, doomFixedToDouble, jdouble);
JNI_EXPORT(jboolean, isBinaryButton, jobject);
JNI_EXPORT(jboolean, isDeltaButton, jobject);

#ifdef __cplusplus
}
#endif
#endif
