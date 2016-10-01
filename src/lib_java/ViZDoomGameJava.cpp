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

#include "ViZDoomGameJava.h"

JNI_EXPORT_0_ARG(void, DoomGameNative) {
    constructJavaObject<DoomGameJava>(jEnv, jObj);
}

JNI_METHOD_RETT_1_ARG(jboolean, loadConfig, loadConfig, jstring)
JNI_METHOD_RETT_0_ARG(jboolean, init, init)
JNI_METHOD_VOID_0_ARG(void, close, close)
JNI_METHOD_VOID_0_ARG(void, newEpisode__, newEpisode_)
JNI_METHOD_VOID_1_ARG(void, newEpisode__Ljava_lang_String, newEpisode_str, jstring)
JNI_METHOD_VOID_1_ARG(void, replayEpisode__Ljava_lang_String_2, replayEpisode_str, jstring)
JNI_METHOD_VOID_2_ARG(void, replayEpisode__Ljava_lang_String_2I, replayEpisode_str_int, jstring, jint)
JNI_METHOD_RETT_0_ARG(jboolean, isRunning, isRunning)
JNI_METHOD_VOID_1_ARG(void, setAction, setAction, jintArray)
JNI_METHOD_VOID_0_ARG(void, advanceAction__, advanceAction_)
JNI_METHOD_VOID_1_ARG(void, advanceAction__I, advanceAction_int, jint)
JNI_METHOD_VOID_2_ARG(void, advanceAction__IZ, advanceAction_int_bool, jint, jboolean)
JNI_METHOD_VOID_3_ARG(void, advanceAction__IZZ, advanceAction_int_bool_bool, jint, jboolean, jboolean)
JNI_METHOD_RETT_1_ARG(jdouble, makeAction___3I, makeAction_vec, jintArray)
JNI_METHOD_RETT_2_ARG(jdouble, makeAction___3II, makeAction_vec_int, jintArray, jint)

JNI_EXPORT_0_ARG(jobject, getState){
    auto state = callObjMethod(jEnv, jObj, &DoomGameJava::getState);
    if (state == nullptr) return NULL;

    jclass jStateClass = jEnv->FindClass("vizdoom/GameState");
    if (jStateClass == 0) return NULL;

    jintArray jGameVariables = castTojintArray(jEnv, state->gameVariables);
    jintArray jScreenBuffer = state->screenBuffer != nullptr ? castTojintArray(jEnv, *state->screenBuffer) : 0;
    jintArray jDepthBuffer = state->depthBuffer != nullptr ? castTojintArray(jEnv, *state->depthBuffer) : 0;
    jintArray jLabelsBuffer = state->labelsBuffer != nullptr ? castTojintArray(jEnv, *state->labelsBuffer) : 0;
    jintArray jAutomapBuffer = state->automapBuffer != nullptr ? castTojintArray(jEnv, *state->automapBuffer) : 0;

    jclass jLabelClass = jEnv->FindClass("vizdoom/Label");
    if (jLabelClass == 0) return NULL;
    jobjectArray jLabels = jEnv->NewObjectArray(state->labels.size(), jLabelClass, NULL);
    jmethodID jLabelConstructor = jEnv->GetMethodID(jLabelClass, "<init>", "(ILjava/lang/StringI)V");
    if (jLabelConstructor == 0) return NULL;

    for(size_t i = 0; i < state->labels.size(); ++i){
        jobject jLabel = jEnv->NewObject(jLabelClass, jLabelConstructor, (jint)state->labels[i].objectId,
                                         castTojstring(jEnv, state->labels[i].objectName), (jint)state->labels[i].value);
        jEnv->SetObjectArrayElement(jLabels, i, jLabel);
    }

    jmethodID jStateConstructor = jEnv->GetMethodID(jStateClass, "<init>", "(I[I[I[I[I[I[Lvizdoom/Label)V");
    if (jStateConstructor == 0) return NULL;
    jobject jState = jEnv->NewObject(jStateClass, jStateConstructor, (jint)state->number,
        jGameVariables, jScreenBuffer, jDepthBuffer, jLabelsBuffer, jAutomapBuffer, jLabels);

    return jState;
}

JNI_EXPORT_0_ARG(jintArray, getLastAction){
    auto lastAction = callObjMethod(jEnv, jObj, &DoomGameJava::getLastAction);
    return castTojintArray(jEnv, lastAction);
}

JNI_METHOD_RETT_0_ARG(jboolean, isNewEpisode, isNewEpisode)
JNI_METHOD_RETT_0_ARG(jboolean, isEpisodeFinished, isEpisodeFinished)
JNI_METHOD_RETT_0_ARG(jboolean, isPlayerDead, isPlayerDead)
JNI_METHOD_VOID_0_ARG(void, respawnPlayer, respawnPlayer)

JNI_EXPORT_1_ARG(void, addAvailableButton__Lvizdoom_Button_2, jobject){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::addAvailableButton_btn, arg1);
}

JNI_EXPORT_2_ARG(void, addAvailableButton__Lvizdoom_Button_2I, jobject, jint){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    auto arg2 = jintCast(jarg2);
    callObjMethod(jEnv, jObj, &DoomGameJava::addAvailableButton_btn_int, arg1, arg2);
}

JNI_METHOD_VOID_0_ARG(void, clearAvailableButtons, clearAvailableButtons)
JNI_METHOD_RETT_0_ARG(jint, getAvailableButtonsSize, getAvailableButtonsSize)

JNI_EXPORT_2_ARG(void, setButtonMaxValue, jobject, jint){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    auto arg2 = jintCast(jarg2);
    callObjMethod(jEnv, jObj, &DoomGameJava::setButtonMaxValue, arg1, arg2);
}

JNI_EXPORT_1_ARG(jint, getButtonMaxValue, jobject){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    return castTojint(callObjMethod(jEnv, jObj, &DoomGameJava::getButtonMaxValue, arg1));
}

JNI_EXPORT_1_ARG(void, addAvailableGameVariable, jobject){
    auto arg1 = jobjectCastToEnum<GameVariable>(jEnv, "vizdoom/GameVariable", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::addAvailableGameVariable, arg1);
}

JNI_EXPORT_1_ARG(jint, getGameVariable, jobject){
    auto arg1 = jobjectCastToEnum<GameVariable>(jEnv, "vizdoom/GameVariable", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::getGameVariable, arg1);
}

JNI_METHOD_VOID_0_ARG(void, clearAvailableGameVariables, clearAvailableGameVariables)
JNI_METHOD_RETT_0_ARG(jint, getAvailableGameVariablesSize, getAvailableGameVariablesSize)
JNI_METHOD_VOID_1_ARG(void, addGameArgs, addGameArgs, jstring)
JNI_METHOD_VOID_0_ARG(void, clearGameArgs, clearGameArgs)
JNI_METHOD_VOID_1_ARG(void, sendGameCommand, sendGameCommand, jstring)
JNI_METHOD_RETT_0_ARG(jint, getModeNative, getMode)

JNI_EXPORT_1_ARG(void, setMode, jobject){
    auto arg1 = jobjectCastToEnum<Mode>(jEnv, "vizdoom/Mode", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::setMode, arg1);
}

JNI_METHOD_RETT_0_ARG(jint, getTicrate, getTicrate)
JNI_METHOD_VOID_1_ARG(void, setTicrate, setTicrate, jint)
JNI_METHOD_RETT_0_ARG(jdouble, getLivingReward, getLivingReward)
JNI_METHOD_VOID_1_ARG(void, setLivingReward, setLivingReward, jdouble)
JNI_METHOD_RETT_0_ARG(jdouble, getDeathPenalty, getDeathPenalty)
JNI_METHOD_VOID_1_ARG(void, setDeathPenalty, setDeathPenalty, jdouble)
JNI_METHOD_RETT_0_ARG(jdouble, getLastReward, getLastReward)
JNI_METHOD_RETT_0_ARG(jdouble, getTotalReward, getTotalReward)
JNI_METHOD_VOID_1_ARG(void, setViZDoomPath, setViZDoomPath, jstring)
JNI_METHOD_VOID_1_ARG(void, setDoomGamePath, setDoomGamePath, jstring)
JNI_METHOD_VOID_1_ARG(void, setDoomScenarioPath, setDoomScenarioPath, jstring)
JNI_METHOD_VOID_1_ARG(void, setDoomMap, setDoomMap, jstring)
JNI_METHOD_VOID_1_ARG(void, setDoomSkill, setDoomSkill, jint)
JNI_METHOD_VOID_1_ARG(void, setDoomConfigPath, setDoomConfigPath, jstring)
JNI_METHOD_RETT_0_ARG(jint, getSeed, getSeed)
JNI_METHOD_VOID_1_ARG(void, setSeed, setSeed, jint)
JNI_METHOD_RETT_0_ARG(jint, getEpisodeStartTime, getEpisodeStartTime)
JNI_METHOD_VOID_1_ARG(void, setEpisodeStartTime, setEpisodeTimeout, jint)
JNI_METHOD_RETT_0_ARG(jint, getEpisodeTimeout, getEpisodeTimeout)
JNI_METHOD_VOID_1_ARG(void, setEpisodeTimeout, setEpisodeTimeout, jint)
JNI_METHOD_RETT_0_ARG(jint, getEpisodeTime, getEpisodeTime)

JNI_EXPORT_1_ARG(void, setScreenResolution, jobject){
    auto arg1 = jobjectCastToEnum<ScreenResolution>(jEnv, "vizdoom/ScreenResolution", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::setScreenResolution, arg1);
}

JNI_EXPORT_1_ARG(void, setScreenFormat, jobject){
    auto arg1 = jobjectCastToEnum<ScreenFormat>(jEnv, "vizdoom/ScreenFormat", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::setScreenFormat, arg1);
}

JNI_METHOD_RETT_0_ARG(jboolean, isDepthBufferEnabled, isDepthBufferEnabled);
JNI_METHOD_VOID_1_ARG(void, setDepthBufferEnabled, setDepthBufferEnabled, jboolean);

JNI_METHOD_RETT_0_ARG(jboolean, isLabelsBufferEnabled, isLabelsBufferEnabled);
JNI_METHOD_VOID_1_ARG(void, setLabelsBufferEnabled, setLabelsBufferEnabled, jboolean);

JNI_METHOD_RETT_0_ARG(jboolean, isAutomapBufferEnabled, isAutomapBufferEnabled);
JNI_METHOD_VOID_1_ARG(void, setAutomapBufferEnabled, setAutomapBufferEnabled, jboolean);

JNI_EXPORT_1_ARG(void, setAutomapMode, jobject){
    auto arg1 = jobjectCastToEnum<AutomapMode>(jEnv, "vizdoom/AutomapMode", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::setAutomapMode, arg1);
}

JNI_METHOD_VOID_1_ARG(void, setAutomapRotate, setAutomapRotate, jboolean);
JNI_METHOD_VOID_1_ARG(void, setAutomapRenderTextures, setAutomapRenderTextures, jboolean);

JNI_METHOD_VOID_1_ARG(void, setRenderHud, setRenderHud, jboolean)
JNI_METHOD_VOID_1_ARG(void, setRenderMinimalHud, setRenderMinimalHud, jboolean)
JNI_METHOD_VOID_1_ARG(void, setRenderWeapon, setRenderWeapon, jboolean)
JNI_METHOD_VOID_1_ARG(void, setRenderCrosshair, setRenderCrosshair, jboolean)
JNI_METHOD_VOID_1_ARG(void, setRenderDecals, setRenderDecals, jboolean)
JNI_METHOD_VOID_1_ARG(void, setRenderParticles, setRenderParticles, jboolean)
JNI_METHOD_VOID_1_ARG(void, setRenderEffectsSprites, setRenderEffectsSprites, jboolean)
JNI_METHOD_VOID_1_ARG(void, setRenderMessages, setRenderMessages, jboolean)
JNI_METHOD_VOID_1_ARG(void, setWindowVisible, setWindowVisible, jboolean)
JNI_METHOD_VOID_1_ARG(void, setConsoleEnabled, setConsoleEnabled, jboolean)
JNI_METHOD_VOID_1_ARG(void, setSoundEnabled, setSoundEnabled, jboolean)
JNI_METHOD_RETT_0_ARG(jint, getScreenWidth, getScreenWidth)
JNI_METHOD_RETT_0_ARG(jint, getScreenHeight, getScreenHeight)
JNI_METHOD_RETT_0_ARG(jint, getScreenChannels, getScreenChannels)
JNI_METHOD_RETT_0_ARG(jint, getScreenPitch, getScreenPitch)
JNI_METHOD_RETT_0_ARG(jint, getScreenSize, getScreenSize)
JNI_METHOD_RETT_0_ARG(jint, getScreenFormatNative, getScreenFormat)

JNI_EXPORT_2_ARG(jdouble, doomTicsToMs, jdouble, jint){
    return (jdouble) doomTicsToMs(jarg1, jarg2);
}

JNI_EXPORT_2_ARG(jdouble, msToDoomTics, jdouble, jint){
    return (jdouble) msToDoomTics(jarg1, jarg2);
}

JNI_EXPORT_2_ARG(jdouble, doomTicsToSec, jdouble, jint){
    return (jdouble) doomTicsToSec(jarg1, jarg2);
}

JNI_EXPORT_2_ARG(jdouble, secToDoomTics, jdouble, jint){
    return (jdouble) secToDoomTics(jarg1, jarg2);
}

JNI_EXPORT_1_ARG(jdouble, doomFixedToDouble, jint){
    return (jdouble) doomFixedToDouble(jarg1);
}

JNI_EXPORT_1_ARG(jboolean, isBinaryButton, jobject){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    return (jboolean) isBinaryButton(arg1);
}

JNI_EXPORT_1_ARG(jboolean, isDeltaButton, jobject){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    return (jboolean) isDeltaButton(arg1);
}
