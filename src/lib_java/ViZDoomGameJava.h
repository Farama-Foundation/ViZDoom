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

#include "ViZDoomJava.h"

#ifndef __VIZDOOM_GAME_JAVA_H__
#define __VIZDOOM_GAME_JAVA_H__

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
        void advanceAction_int_bool_bool(unsigned int _int, bool _bool1, bool _bool2) { this->advanceAction(_int, _bool1, _bool2); }

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

JNI_EXPORT_0_ARG(void, DoomGameNative);
JNI_EXPORT_1_ARG(jboolean, loadConfig, jstring);
JNI_EXPORT_0_ARG(jboolean, init);
JNI_EXPORT_0_ARG(void, close);
JNI_EXPORT_0_ARG(void, newEpisode__);
JNI_EXPORT_1_ARG(void, newEpisode__Ljava_lang_String, jstring);
JNI_EXPORT_1_ARG(void, replayEpisode__Ljava_lang_String_2, jstring);
JNI_EXPORT_2_ARG(void, replayEpisode__Ljava_lang_String_2I, jstring, jint);
JNI_EXPORT_0_ARG(jboolean, isRunning);
JNI_EXPORT_1_ARG(void, setAction, jintArray);
JNI_EXPORT_0_ARG(void, advanceAction__);
JNI_EXPORT_1_ARG(void, advanceAction__I, jint);
JNI_EXPORT_2_ARG(void, advanceAction__IZ, jint, jboolean);
JNI_EXPORT_3_ARG(void, advanceAction__IZZ, jint, jboolean, jboolean);
JNI_EXPORT_1_ARG(jdouble, makeAction___3I, jintArray);
JNI_EXPORT_2_ARG(jdouble, makeAction___3II, jintArray, jint);
JNI_EXPORT_0_ARG(jobject, getState);
JNI_EXPORT_0_ARG(jintArray, getLastAction);
JNI_EXPORT_0_ARG(jboolean, isNewEpisode);
JNI_EXPORT_0_ARG(jboolean, isEpisodeFinished);
JNI_EXPORT_0_ARG(jboolean, isPlayerDead);
JNI_EXPORT_0_ARG(void, respawnPlayer);
JNI_EXPORT_1_ARG(void, addAvailableButton__Lvizdoom_Button_2, jobject);
JNI_EXPORT_2_ARG(void, addAvailableButton__Lvizdoom_Button_2I, jobject, jint);
JNI_EXPORT_0_ARG(void, clearAvailableButtons);
JNI_EXPORT_0_ARG(jint, getAvailableButtonsSize);
JNI_EXPORT_2_ARG(void, setButtonMaxValue, jobject, jint);
JNI_EXPORT_1_ARG(jint, getButtonMaxValue, jobject);
JNI_EXPORT_1_ARG(void, addAvailableGameVariable, jobject);
JNI_EXPORT_0_ARG(void, clearAvailableGameVariables);
JNI_EXPORT_0_ARG(jint, getAvailableGameVariablesSize);
JNI_EXPORT_1_ARG(void, addGameArgs, jstring);
JNI_EXPORT_0_ARG(void, clearGameArgs);
JNI_EXPORT_1_ARG(void, sendGameCommand, jstring);
JNI_EXPORT_0_ARG(jintArray, getGameScreen);
JNI_EXPORT_0_ARG(jint, getModeNative);
JNI_EXPORT_1_ARG(void, setMode, jobject);
JNI_EXPORT_0_ARG(jint, getTicrate);
JNI_EXPORT_1_ARG(void, setTicrate, jint);
JNI_EXPORT_1_ARG(jint, getGameVariable, jobject);
JNI_EXPORT_0_ARG(jdouble, getLivingReward);
JNI_EXPORT_1_ARG(void, setLivingReward, jdouble);
JNI_EXPORT_0_ARG(jdouble, getDeathPenalty);
JNI_EXPORT_1_ARG(void, setDeathPenalty, jdouble);
JNI_EXPORT_0_ARG(jdouble, getLastReward);
JNI_EXPORT_0_ARG(jdouble, getTotalReward);
JNI_EXPORT_1_ARG(void, setViZDoomPath, jstring);
JNI_EXPORT_1_ARG(void, setDoomGamePath, jstring);
JNI_EXPORT_1_ARG(void, setDoomScenarioPath, jstring);
JNI_EXPORT_1_ARG(void, setDoomMap, jstring);
JNI_EXPORT_1_ARG(void, setDoomSkill, jint);
JNI_EXPORT_1_ARG(void, setDoomConfigPath, jstring);
JNI_EXPORT_0_ARG(jint, getSeed);
JNI_EXPORT_1_ARG(void, setSeed, jint);
JNI_EXPORT_0_ARG(jint, getEpisodeStartTime);
JNI_EXPORT_1_ARG(void, setEpisodeStartTime, jint);
JNI_EXPORT_0_ARG(jint, getEpisodeTimeout);
JNI_EXPORT_1_ARG(void, setEpisodeTimeout, jint);
JNI_EXPORT_0_ARG(jint, getEpisodeTime);

JNI_EXPORT_0_ARG(jboolean, isDepthBufferEnabled);
JNI_EXPORT_1_ARG(void, setDepthBufferEnabled, jboolean);

JNI_EXPORT_0_ARG(jboolean, isLabelsBufferEnabled);
JNI_EXPORT_1_ARG(void, setLabelsBufferEnabled, jboolean);

JNI_EXPORT_0_ARG(jboolean, isAutomapBufferEnabled);
JNI_EXPORT_1_ARG(void, setAutomapBufferEnabled, jboolean);
JNI_EXPORT_1_ARG(void, setAutomapMode, jobject);
JNI_EXPORT_1_ARG(void, setAutomapRotate, jboolean);
JNI_EXPORT_1_ARG(void, setAutomapRenderTextures, jboolean);

JNI_EXPORT_1_ARG(void, setScreenResolution, jobject);
JNI_EXPORT_1_ARG(void, setScreenFormat, jobject);
JNI_EXPORT_1_ARG(void, setRenderHud, jboolean);
JNI_EXPORT_1_ARG(void, setRenderMinimalHud, jboolean);
JNI_EXPORT_1_ARG(void, setRenderWeapon, jboolean);
JNI_EXPORT_1_ARG(void, setRenderCrosshair, jboolean);
JNI_EXPORT_1_ARG(void, setRenderDecals, jboolean);
JNI_EXPORT_1_ARG(void, setRenderParticles, jboolean);
JNI_EXPORT_1_ARG(void, setRenderEffectsSprites, jboolean);
JNI_EXPORT_1_ARG(void, setRenderMessages, jboolean);
JNI_EXPORT_1_ARG(void, setWindowVisible, jboolean);
JNI_EXPORT_1_ARG(void, setConsoleEnabled, jboolean);
JNI_EXPORT_1_ARG(void, setSoundEnabled, jboolean);

JNI_EXPORT_0_ARG(jint, getScreenWidth);
JNI_EXPORT_0_ARG(jint, getScreenHeight);
JNI_EXPORT_0_ARG(jint, getScreenChannels);
JNI_EXPORT_0_ARG(jint, getScreenPitch);
JNI_EXPORT_0_ARG(jint, getScreenSize);
JNI_EXPORT_0_ARG(jint, getScreenFormatNative);

JNI_EXPORT_2_ARG(jdouble, doomTics2Ms, jdouble, jint);
JNI_EXPORT_2_ARG(jdouble, ms2DoomTics, jdouble, jint);
JNI_EXPORT_2_ARG(jdouble, doomTics2Sec, jdouble, jint);
JNI_EXPORT_2_ARG(jdouble, sec2DoomTics, jdouble, jint);
JNI_EXPORT_1_ARG(jdouble, doomFixedToDouble, jint);
JNI_EXPORT_1_ARG(jboolean, isBinaryButton, jobject);
JNI_EXPORT_1_ARG(jboolean, isDeltaButton, jobject);

#ifdef __cplusplus
}
#endif
#endif
