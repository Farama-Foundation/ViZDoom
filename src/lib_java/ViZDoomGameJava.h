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

#include <jni.h>

/* Header for class DoomGame */

#ifndef _Included_DoomGame
#define _Included_DoomGame

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     DoomGame
 * Method:    DoomTics2Ms
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_DoomTics2Ms
  (JNIEnv *, jobject, jint);

/*
 * Class:     DoomGame
 * Method:    Ms2DoomTics
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_Ms2DoomTics
  (JNIEnv *, jobject, jint);

/*
 * Class:     DoomGame
 * Method:    DoomFixedToDouble
 * Signature: (I)D
 */
JNIEXPORT jdouble JNICALL Java_vizdoom_DoomGame_DoomFixedToDouble
  (JNIEnv *, jobject, jint);


/*
 * Class:     DoomGame
 * Method:    isBinaryButton
 * Signature: (Lvizdoom/Button)Z
 */
JNIEXPORT jboolean JNICALL Java_vizdoom_DoomGame_isBinaryButton
  (JNIEnv *, jobject, jobject);

/*
 * Class:     DoomGame
 * Method:    isDeltaButton
 * Signature: (Lvizdoom/Button)Z
 */
JNIEXPORT jboolean JNICALL Java_vizdoom_DoomGame_isDeltaButton
  (JNIEnv *, jobject, jobject);

/*
 * Class:     DoomGame
 * Method:    DoomGame
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_DoomGame
  (JNIEnv *, jobject);


/*
 * Class:     DoomGame
 * Method:    loadConfig
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_vizdoom_DoomGame_loadConfig
  (JNIEnv *, jobject, jstring);

/*
 * Class:     DoomGame
 * Method:    init
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_vizdoom_DoomGame_init
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    close
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_close
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    newEpisode
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_newEpisode
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    isRunning
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_vizdoom_DoomGame_isRunning
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    setAction
 * Signature: ([I)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setAction
  (JNIEnv *, jobject, jintArray);

/*
 * Class:     DoomGame
 * Method:    advanceAction
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_advanceAction__
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    advanceAction
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_advanceAction__I
  (JNIEnv *, jobject, jint);

/*
 * Class:     DoomGame
 * Method:    advanceAction
 * Signature: (IZZ)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_advanceAction__IZZ
  (JNIEnv *, jobject, jint, jboolean, jboolean);

/*
 * Class:     DoomGame
 * Method:    makeAction
 * Signature: ([I)F
 */
JNIEXPORT jdouble JNICALL Java_vizdoom_DoomGame_makeAction___3I
  (JNIEnv *, jobject, jintArray);

/*
 * Class:     DoomGame
 * Method:    makeAction
 * Signature: ([II)F
 */
JNIEXPORT jdouble JNICALL Java_vizdoom_DoomGame_makeAction___3II
  (JNIEnv *, jobject, jintArray, jint);

/*
 * Class:     DoomGame
 * Method:    getState
 * Signature: ()LGameState;
 */
JNIEXPORT jobject JNICALL Java_vizdoom_DoomGame_getState
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    getLastAction
 * Signature: ()[I
 */
JNIEXPORT jintArray JNICALL Java_vizdoom_DoomGame_getLastAction
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    isNewEpisode
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_vizdoom_DoomGame_isNewEpisode
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    isEpisodeFinished
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_vizdoom_DoomGame_isEpisodeFinished
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    isPlayerDead
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_vizdoom_DoomGame_isPlayerDead
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    respawnPlayer
 * Signature: ()Z
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_respawnPlayer
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    addAvailableButton
 * Signature: (Lvizdoom/Button;)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_addAvailableButton__Lvizdoom_Button_2
  (JNIEnv *, jobject, jobject);

/*
 * Class:     DoomGame
 * Method:    addAvailableButton
 * Signature: (Lvizdoom/Button;I)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_addAvailableButton__Lvizdoom_Button_2I
  (JNIEnv *, jobject, jobject, jint);

/*
 * Class:     DoomGame
 * Method:    clearAvailableButtons
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_clearAvailableButtons
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    getAvailableButtonsSize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_getAvailableButtonsSize
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    setButtonMaxValue
 * Signature: (Lvizdoom/Button;I)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setButtonMaxValue
  (JNIEnv *, jobject, jobject, jint);

/*
 * Class:     DoomGame
 * Method:    getButtonMaxValue
 * Signature: (Lvizdoom/Button)V
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_getButtonMaxValue
  (JNIEnv *, jobject, jobject);

/*
 * Class:     DoomGame
 * Method:    addAvailableGameVariable
 * Signature: (Lvizdoom/GameVariable;)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_addAvailableGameVariable
  (JNIEnv *, jobject, jobject);

/*
 * Class:     DoomGame
 * Method:    clearAvailableGameVariables
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_clearAvailableGameVariables
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    getAvailableGameVariablesSize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_getAvailableGameVariablesSize
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    addGameArgs
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_addGameArgs
  (JNIEnv *, jobject, jstring);

/*
 * Class:     DoomGame
 * Method:    clearGameArgs
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_clearGameArgs
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    sendGameCommand
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_sendGameCommand
  (JNIEnv *, jobject, jstring);

/*
 * Class:     DoomGame
 * Method:    getGameScreen
 * Signature: ()[I
 */
JNIEXPORT jintArray JNICALL Java_vizdoom_DoomGame_getGameScreen
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    getMod
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_getMod
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    setMode
 * Signature: (Lvizdoom/Mode;)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setMode
  (JNIEnv *, jobject, jobject);

/*
 * Class:     DoomGame
 * Method:    getGameVariable
 * Signature: (Lvizdoom/GameVariable;)I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_getGameVariable
  (JNIEnv *, jobject, jobject);

/*
 * Class:     DoomGame
 * Method:    getLivingReward
 * Signature: ()D
 */
JNIEXPORT jdouble JNICALL Java_vizdoom_DoomGame_getLivingReward
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    setLivingReward
 * Signature: (F)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setLivingReward
  (JNIEnv *, jobject, jdouble);

/*
 * Class:     DoomGame
 * Method:    getDeathPenalty
 * Signature: ()F
 */
JNIEXPORT jdouble JNICALL Java_vizdoom_DoomGame_getDeathPenalty
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    setDeathPenalty
 * Signature: (F)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setDeathPenalty
  (JNIEnv *, jobject, jdouble);

/*
 * Class:     DoomGame
 * Method:    getLastReward
 * Signature: ()F
 */
JNIEXPORT jdouble JNICALL Java_vizdoom_DoomGame_getLastReward
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    getTotalReward
 * Signature: ()F
 */
JNIEXPORT jdouble JNICALL Java_vizdoom_DoomGame_getTotalReward
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    setViZDoomPath
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setViZDoomPath
  (JNIEnv *, jobject, jstring);

/*
 * Class:     DoomGame
 * Method:    setDoomGamePath
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setDoomGamePath
  (JNIEnv *, jobject, jstring);

/*
 * Class:     DoomGame
 * Method:    setDoomScenarioPath
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setDoomScenarioPath
  (JNIEnv *, jobject, jstring);

/*
 * Class:     DoomGame
 * Method:    setDoomMap
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setDoomMap
  (JNIEnv *, jobject, jstring);

/*
 * Class:     DoomGame
 * Method:    setDoomSkill
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setDoomSkill
  (JNIEnv *, jobject, jint);

/*
 * Class:     DoomGame
 * Method:    setDoomConfigPath
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setDoomConfigPath
  (JNIEnv *, jobject, jstring);

/*
 * Class:     DoomGame
 * Method:    getSeed
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_getSeed
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    setSeed
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setSeed
  (JNIEnv *, jobject, jint);

/*
 * Class:     DoomGame
 * Method:    getEpisodeStartTime
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_getEpisodeStartTime
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    setEpisodeStartTime
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setEpisodeStartTime
  (JNIEnv *, jobject, jint);

/*
 * Class:     DoomGame
 * Method:    getEpisodeTimeout
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_getEpisodeTimeout
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    setEpisodeTimeout
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setEpisodeTimeout
  (JNIEnv *, jobject, jint);

/*
 * Class:     DoomGame
 * Method:    getEpisodeTime
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_getEpisodeTime
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    setScreenResolution
 * Signature: (Lvizdoom/ScreenResolution;)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setScreenResolution
  (JNIEnv *, jobject, jobject);

/*
 * Class:     DoomGame
 * Method:    setScreenFormat
 * Signature: (Lvizdoom/ScreenFormat;)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setScreenFormat
  (JNIEnv *, jobject, jobject);

/*
 * Class:     DoomGame
 * Method:    setRenderHud
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setRenderHud
  (JNIEnv *, jobject, jboolean);

/*
 * Class:     DoomGame
 * Method:    setRenderWeapon
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setRenderWeapon
  (JNIEnv *, jobject, jboolean);

/*
 * Class:     DoomGame
 * Method:    setRenderCrosshair
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setRenderCrosshair
  (JNIEnv *, jobject, jboolean);

/*
 * Class:     DoomGame
 * Method:    setRenderDecals
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setRenderDecals
  (JNIEnv *, jobject, jboolean);

/*
 * Class:     DoomGame
 * Method:    setRenderParticles
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setRenderParticles
  (JNIEnv *, jobject, jboolean);

/*
 * Class:     DoomGame
 * Method:    setWindowVisible
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setWindowVisible
  (JNIEnv *, jobject, jboolean);

/*
 * Class:     DoomGame
 * Method:    setConsoleEnabled
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setConsoleEnabled
  (JNIEnv *, jobject, jboolean);

/*
 * Class:     DoomGame
 * Method:    setSoundEnabled
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_vizdoom_DoomGame_setSoundEnabled
  (JNIEnv *, jobject, jboolean);

/*
 * Class:     DoomGame
 * Method:    getScreenWidth
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_getScreenWidth
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    getScreenHeight
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_getScreenHeight
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    getScreenChannels
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_getScreenChannels
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    getScreenPitch
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_getScreenPitch
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    getScreenSize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_getScreenSize
  (JNIEnv *, jobject);

/*
 * Class:     DoomGame
 * Method:    getScreenForma
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_vizdoom_DoomGame_getScreenForma
  (JNIEnv *, jobject);

#ifdef __cplusplus
}
#endif
#endif
