/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
///////////////////////////////////////////////////////////////////
#include <opencv2/core/eigen.hpp>
///////////////////////////////////////////////////////////////////

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

///////////////////////////////////////////////////////////////////
/*
    int distance = 15;
    nFeatures = 960 * 540 / distance/ distance;
*/
// 這邊就是為啥每次特徵點都取2304點的原因
///////////////////////////////////////////////////////////////////

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


// 函式流程
// 1. 將影像轉為灰度影像
// 2. 構造Frame類
// 3. 跟踪
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
// 1. 將影像轉為灰度影像
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

// 2. 構造Frame類

/*
    // 初始化时直接提取，平常就跟踪就好
    // 沒有成功初始化的前一個狀態就是 NO_IMAGES_YET
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    // 跟踪
    Track();
*/

///////////////////////////////////////////////////////////////////
    // 初始化时直接提取，平常就跟踪就好
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth, true);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth, false);
     
    Track(); // 跟踪
///////////////////////////////////////////////////////////////////

    return mCurrentFrame.mTcw.clone();
}

// 函式流程
// 1. 初始化
// 2. 跟踪進入正常SLAM模式，有地圖更新
// 3. 純定位模式，只進行跟踪tracking，局部地圖不工作
// 4. 在跟踪得到當前幀初始姿態後，對局部地圖進行跟踪得到更多的匹配，並優化當前姿態
// 5. 跟踪成功，更新恆速運動模型
// 6. 清除觀測不到的地圖點
// 7. 清除恆速模型跟踪中 UpdateLastFrame中為當前幀臨時添加的MapPoints（僅雙目和rgbd）
// 8. 檢測並插入關鍵幀，對於雙目或RGB-D會產生新的地圖點
// 9. 刪除那些在bundle adjustment中檢測為outlier的地圖點
// 10. 如果初始化後不久就跟踪失敗，並且relocation也沒有搞定，只能重新Reset
// 11. 記錄位姿信息，用於最後保存所有的軌跡
void Tracking::Track()
{
// 1. 初始化    
    // 確認是否初始化
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }
    
    // 儲存了Tracking最新的狀態，用於FrameDrawer中的繪製
    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    // 鎖地圖點更新
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    // 沒初始化的時候
    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();  // 單目地圖初始化

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
 
 // 2. 跟踪進入正常SLAM模式，有地圖更新  
    else
    {
        // System is initialized. Track Frame.
        // bOK為臨時變數，用於表示每個函數是否執行成功
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        // mbOnlyTracking 等於false表示正常SLAM模式（定位+地圖更新），mbOnlyTracking等於true表示僅定位模式
        // tracking 類構造時默認為false。在viewer中有個開關ActivateLocalizationMode，可以控制是否開啟mbOnlyTracking
        if(!mbOnlyTracking)  //非僅追踪模式
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            // 是否初始化成功
            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                // 2.1. 檢查並更新上一幀被替換的MapPoints
                // 局部建圖線程則可能會對原有的地圖點進行替換.在這裡進行檢查
                CheckReplacedInLastFrame();

                // 2.2. 運動模型是空的或剛完成重定位，跟踪參考關鍵幀；否則恆速模型跟踪
                // 第一個條件,如果運動模型為空,說明是剛初始化開始，或者已經跟丟了
                // 第二個條件,如果當前幀緊緊地跟著在重定位的幀的後面，我們將重定位幀來恢復位姿
                // mnLastRelocFrameId 上一次重定位的那一幀
                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    // 用最近的關鍵幀來跟踪當前的普通幀
                    // 通過BoW的方式在參考幀中找當前幀特徵點的匹配點
                    // 優化每個特徵點都對應3D點重投影誤差即可得到位姿
                    bOK = TrackReferenceKeyFrame(); //使用關鍵幀追踪
                }
                else    //先用當前幀追踪(恆速模式)，如果追踪失敗，則使用關鍵點幀追踪
                {
                    // 用最近的普通幀來跟踪當前的普通幀
                    // 根據恆速模型設定當前幀的初始位姿
                    // 通過投影的方式在參考幀中找當前幀特徵點的匹配點
                    // 優化每個特徵點所對應3D點的投影誤差即可得到位姿                
                    bOK = TrackWithMotionModel();
                    if(!bOK)                    
                        //根據恆速模型失敗了，只能根據參考關鍵幀來跟踪
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else    // 如果跟踪失敗，則進行重定位
            {
                // 如果跟踪狀態不成功,那麼就只能重定位了
                // BOW搜索，EPnP求解位姿
                bOK = Relocalization();
            }
        }

// 3. 純定位模式，只進行跟踪tracking，局部地圖不工作
        else
        {
            // Localization Mode: Local Mapping is deactivated
            if(mState==LOST)
            {
                // 3.1 如果跟丟了，只能重定位
                bOK = Relocalization();
            }
            else
            {
                // mbVO是mbOnlyTracking為true時的才有的一個變量
                // mbVO為false表示此幀匹配了很多的MapPoints，跟踪很正常 (注意有點反直覺)
                // mbVO為true表明此幀匹配了很少的MapPoints，少於10個，要跪的節奏
                if(!mbVO)
                {
                    // 3.2 如果跟踪正常，使用恆速模型 或 參考關鍵幀跟踪
                    // In last frame we tracked enough MapPoints in the map
                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                        // ? 為了和前面模式統一，這個地方是不是應該加上
                        // if(!bOK)
                        //    bOK = TrackReferenceKeyFrame();

                    }
                    else
                    {
                        // 如果恆速模型不被滿足,那麼就只能夠通過參考關鍵幀來定位
                        bOK = TrackReferenceKeyFrame();
                    }
                }

                //mbVO為true，表明此幀匹配了很少（小於10）的地圖點，既做跟踪又做重定位
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    // MM=Motion Model,通過運動模型進行跟踪的結果
                    bool bOKMM = false;

                    // 通過重定位方法來跟踪的結果
                    bool bOKReloc = false;
                
                    // 運動模型中構造的地圖點
                    vector<MapPoint*> vpMPsMM;

                    // 在追踪運動模型後發現的外點
                    vector<bool> vbOutMM;

                    // 運動模型得到的位姿
                    cv::Mat TcwMM;

// 3.3. 當運動模型有效的時候,根據運動模型計算位姿
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        
                        // 將恆速模型跟踪結果暫存到這幾個變量中，因為後面重定位會改變這些變量
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }

                    // 3.4. 使用重定位的方法來得到當前幀的位姿
                    bOKReloc = Relocalization();

                    // 3.5. 根據前面的恆速模型、重定位結果來更新狀態
                    if(bOKMM && !bOKReloc)
                    {

                        // 恆速模型成功、重定位失敗，重新使用之前暫存的恆速模型結果
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;
                  
                  
                        //? 疑似bug！這段代碼是不是重複增加了觀測次數？後面 TrackLocalMap 函數中會有這些操作
                        // 如果當前幀匹配的3D點很少，增加當前可視地圖點的被觀測次數
                        if(mbVO)
                        {
                        
                            // 更新當前幀的地圖點被觀測次數
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                // 如果這個特徵點形成了地圖點,並且也不是外點的時候
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    // 增加能觀測到該地圖點的幀數
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        // 只要重定位成功整個跟踪過程正常進行（重定位與跟踪，更相信重定位）
                        mbVO = false;
                    }
                    // 有一個成功我們就認為執行成功了
                    bOK = bOKReloc || bOKMM;
                }
            }
        }

// 4. 在跟踪得到當前幀初始姿態後，對局部地圖進行跟踪得到更多的匹配，並優化當前姿態
        // 將最新的關鍵幀作為當前幀的參考關鍵幀
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        // 前面只是跟踪一幀得到初始位姿，這裡搜索局部關鍵幀、局部地圖點，和當前幀進行投影匹配，得到更多匹配的MapPoints後進行Pose優化
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            
            // 重定位成功
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }
        
        // 根據上面的操作來判斷是否追踪成功
        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        // Step 4：更新顯示線程中的圖像、特徵點、地圖點等信息
        mpFrameDrawer->Update(this);

// 5. 跟踪成功，更新恆速運動模型
        // If tracking were good, check if we insert a keyframe
        // 只有在成功追踪時才考慮生成關鍵幀的問題
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                // 更新恆速運動模型 TrackWithMotionModel 中的mVelocity
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                
                // mVelocity = Tcl = Tcw * Twl,表示上一幀到當前幀的變換， 其中 Twl = LastTwc
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else            
                // 否則速度為空
                mVelocity = cv::Mat();

            // 更新顯示中的位姿
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

// 6. 清除觀測不到的地圖點
            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }


// 7. 清除恆速模型跟踪中 UpdateLastFrame中為當前幀臨時添加的MapPoints（僅雙目和rgbd）
            // Delete temporal MapPoints
            // 步驟6中只是在當前幀中將這些MapPoints剔除，這裡從MapPoints數據庫中刪除
            // 臨時地圖點僅僅是為了提高雙目或rgbd攝像頭的幀間跟踪效果，用完以後就扔了，沒有添加到地圖中
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }

            // 這裡不僅僅是清除mlpTemporalPoints，通過delete pMP還刪除了指針指向的MapPoint
            // 不能夠直接執行這個是因為其中存儲的都是指針,之前的操作都是為了避免內存洩露
            mlpTemporalPoints.clear();


// 8. 檢測並插入關鍵幀，對於雙目或RGB-D會產生新的地圖點

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();


/////////////////////////////////////////////////////////////////////////
/*
            if(NeedNewKeyFrame())
            {
                mCurrentFrame.add();
                CreateNewKeyFrame();
            }
*/            
/////////////////////////////////////////////////////////////////////////


            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            // 作者這裡說允許在BA中被Huber核函數判斷為外點的傳入新的關鍵幀中，讓後續的BA來審判他們是不是真正的外點
            // 但是估計下一幀位姿的時候我們不想用這些外點，所以刪掉

// 9. 刪除那些在bundle adjustment中檢測為outlier的地圖點
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                // 這裡第一個條件還要執行判斷是因為, 前面的操作中可能刪除了其中的地圖點
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

// 10. 如果初始化後不久就跟踪失敗，並且relocation也沒有搞定，只能重新Reset
        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            // 如果地圖中的關鍵幀信息過少的話,直接重新進行初始化了
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        // 確保已經設置了參考關鍵幀
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // 保存上一幀的數據,當前幀變上一幀
        mLastFrame = Frame(mCurrentFrame);
    }

// 11. 記錄位姿信息，用於最後保存所有的軌跡
    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        // 計算相對姿態Tcr = Tcw * Twr, Twr = Trw^-1
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        
        // 保存各種狀態
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        // 如果跟踪失敗，則相對位姿使用上一次值
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }
}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

// 函式流程
// 1. 構建單目初始器
// 2. 如果當前幀特徵點太少，重新構造初始器
// 3. 尋找初始幀與當前幀的匹配特徵點對
// 4. 驗證匹配結果，如果匹配點太少就從新初始化
// 5. 用匹配結果計算H與F矩陣
// 6. 初始化成功後，刪除無法三角化的匹配點
// 7. 將初始化的第一幀作為世界座標中心
// 8. 創建初始化地圖點
void Tracking::MonocularInitialization()
{
// 1. 構建單目初始器
    if(!mpInitializer)
    {
        // Set Reference Frame
        // 單目初始幀提取的特徵點數必須大於100，否則放棄該幀圖像
        if(mCurrentFrame.mvKeys.size()>100)
        {
            
            // 步驟1：得到用於初始化的第一幀，初始化需要兩幀
            mInitialFrame = Frame(mCurrentFrame);

            // 記錄最近的一幀
            mLastFrame = Frame(mCurrentFrame);

            // mvbPrevMatched最大的情況就是所有特徵點都被跟踪上
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            // 這兩句是多餘的
            if(mpInitializer)
                delete mpInitializer;

            // 由當前幀構造初始器 sigma:1.0 iterations:200
            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            //初始化為-1。表示沒有任何匹配，儲存匹配的ID
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }

// 2. 如果當前幀特徵點太少，重新構造初始器，只有連續兩幀的特徵點個數都大於100時，才能繼續進行初始化過程
    else
    {
        // Try to initialize
        // 當當前幀特徵點小於100
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

// 3. 尋找初始幀與當前幀的匹配特徵點對，尋找mInitialFrame與mCurrentFrame的匹配特徵點對
        // Find correspondences
        // 建構特徵匹配器
        ORBmatcher matcher(0.9,true);

//        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);
////////////////////////////////////////////////////////////////////////
        int nmatches;
        nmatches = matcher.SearchForInitializationWithGMS(mInitialFrame,mCurrentFrame,mvIniMatches);
////////////////////////////////////////////////////////////////////////

// 4. 驗證匹配結果，如果匹配點太少就從新初始化
        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

// 5. 用匹配結果計算H與F矩陣
        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        // 通過H模型或F模型進行單目初始化，得到兩幀間相對運動、初始MapPoints
        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {

// 6. 初始化成功後，刪除無法三角化的匹配點
            // 刪除那些無法進行三角化的匹配點
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

// 7. 將初始化的第一幀作為世界座標中心
            // Set Frame Poses
            // 將初始化的第一幀作為世界坐標系，因此第一幀變換矩陣為單位矩陣
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

            // 由Rcw和tcw構造Tcw,並賦值給mTcw，mTcw為世界坐標係到該幀的變換矩陣
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

// 8. 創建初始化地圖點
            // 步驟6：將三角化得到的3D點包裝成MapPoints
            // Initialize函數會得到mvIniP3D，
            // mvIniP3D是cv::Point3f類型的一個容器，是個存放3D點的臨時變量，
            // CreateInitialMapMonocular將3D點包裝成MapPoint類型存入KeyFrame和Map中
            CreateInitialMapMonocular();
        }
    }
}


// 函式流程
// 1. 創立關鍵幀
// 2. 將初始關鍵幀轉為詞袋
// 3. 將關鍵幀插入到地圖中
// 4. 用初始化的3D點生成地圖點
// 5. 為該地圖點添加屬性
// 6. 在地圖中添加這個地圖點
// 7. 更新關鍵幀之間的連接關係
// 8. 全域的BA優化，優化所有姿態及三維點
// 9. 將MapPoints的中值深度歸一化
// 10. 將兩幀之間的變換歸一化
// 11. 將三維點歸一化
// 12. 將關鍵幀插入局部地圖，更新姿態與局部地圖點
void Tracking::CreateInitialMapMonocular()
{
// 1. 創立關鍵幀
    // Create KeyFrames
    // 將初始幀與當前幀都視為關鍵幀
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

// 2. 將初始關鍵幀轉為詞袋
    // 步驟1：將初始關鍵幀的描述子轉為BoW
    pKFini->ComputeBoW();

    // 步驟2：將當前關鍵幀的描述子轉為BoW
    pKFcur->ComputeBoW();

// 3. 將關鍵幀插入到地圖中
    // Insert KFs in the map
    // 凡是關鍵幀，都要插入地圖
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

// 4. 用初始化的3D點生成地圖點
    // Create MapPoints and asscoiate to keyframes
    // 遍歷所有匹配點
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        // 沒有匹配關係
        if(mvIniMatches[i]<0)
            continue;
            
        // Create MapPoint.
        // 將3D點放到世界座標中
        cv::Mat worldPos(mvIniP3D[i]);

        // 用3D點構造地圖點
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

// 5. 為該地圖點添加屬性
        // a.觀測到該MapPoint的關鍵幀
        // b.該MapPoint的描述子
        // c.該MapPoint的平均觀測方向和深度範圍

        // 該KeyFrame的哪個特徵點可以觀測到哪個3D點
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        // a.表示該MapPoint可以被哪個KeyFrame的哪個特徵點觀測到(添加觀測關係)
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        // b.從眾多觀測到該MapPoint的特徵點中挑選區分讀最高的描述子
        pMP->ComputeDistinctiveDescriptors();

        // c.更新該MapPoint平均觀測方向以及觀測距離的範圍
        pMP->UpdateNormalAndDepth();

///////////////////////////////////////////////////////////////////

        // add by lyc
        pMP->feature = mCurrentFrame.track_feature_pts_[mvIniMatches[i]];
        pMP->feature->first_kf = pKFcur;
        pMP->feature->is_3d = true;
        pMP->feature->mp = pMP;

///////////////////////////////////////////////////////////////////

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

// 6. 在地圖中添加這個地圖點
        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

// 7. 更新關鍵幀之間的連接關係
    // Update Connections
    // 更新關鍵幀間的連接關係，對於一個新創建的關鍵幀都會執行一次關鍵連接關係更新
    // 在3D點和關鍵幀之間建立邊，每個邊有一個權重，邊的權重是該關鍵幀與當前幀公共3D點的個數
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

// 8. 全域的BA優化，優化所有姿態及三維點
    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    // 步驟5：BA優化
    Optimizer::GlobalBundleAdjustemnt(mpMap,20);


// 9. 將MapPoints的中值深度歸一化
    // Set median depth to 1
    // 單目傳感器無法恢復真實的深度，這裡將點雲中值深度（歐式距離，不是指z）歸一化到1
    // 評估關鍵幀場景深度，q=2表示中值
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    // 當平均深度小於0，或是在當前幀中被觀測到的地圖點小於100個，代表初始化失敗
    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

// 10. 將兩幀之間的變換歸一化
    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    
    // 根據點雲歸一化比例縮放平移量，x/z,y/z
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

// 11. 將三維點歸一化
    // Scale points
    // 把3D點的尺度也歸一化到1
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

// 12. 將關鍵幀插入局部地圖，更新姿態與局部地圖點
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    // 更新局部地圖點
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK; // 初始化成功，至此，初始化過程完成
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
/////////////////////////////////////////////////////////////////////
    // 提取orb特征点
    mCurrentFrame.extract();
/////////////////////////////////////////////////////////////////////

    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;
/////////////////////////////////////////////////////////////////////
    int nmatches;
    
    //{
    nmatches = matcher.SearchWithGMS(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
    //}

/////////////////////////////////////////////////////////////////////
// A-ORB-SLAM2 刪除 int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
    // int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
///////////////////////////////////////////////////////////////////////
    // add
    mLastFrame.update_mps();
///////////////////////////////////////////////////////////////////////

    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    

 
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }


    if(nmatches<20)
        return false;


    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();

///////////////////////////////////////////////////////////////////

//    SearchLocalPoints();
    return true;

    //SearchLocalPoints();
///////////////////////////////////////////////////////////////////
    
    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

/////////////////////////////////////////////////////////////////////////
bool Tracking::NeedNewKeyFrameNew()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;
    if (last_kf == nullptr)
        return true;
    // Compute median parallax
    double med_rot_parallax = 0.;
    // unrot : false / median : true / only_2d : false
    {
        int cnt = 0 ;
        for (int i = 0;i < mCurrentFrame.mvpMapPoints.size(); i++){
            if (mCurrentFrame.mvpMapPoints.at(i) != nullptr){
                auto pt1 = mCurrentFrame.mvKeysUn.at(i).pt;
                int id2 = mCurrentFrame.mvpMapPoints.at(i)->GetIndexInKeyFrame(last_kf);
                if (id2 != -1){
                    auto pt2 = last_kf->mvKeysUn.at(id2).pt;
                    med_rot_parallax += cv::norm(pt2 - pt1);
                    cnt ++;
                }
            }
        }
        med_rot_parallax /= cnt;
    }

    // Id diff with last KF
    int nbimfromkf = (int)(mCurrentFrame.mnId - last_kf->mnFrameId);
    std::cout<<" c1 ";
    /// 1
    if( mCurrentFrame.mvKeys.size() < 400
        && nbimfromkf >= 5)
    {
        return true;
    }

    /// 2
    int cnt = 0;
    for (const auto& mp:mCurrentFrame.mvpMapPoints) {
        if (mp != nullptr && !mp->isBad()){
            cnt ++;
        }
    }
    if( cnt < 20 &&
        nbimfromkf >= 2 )
    {
        return true;
    }
    std::cout<<" c2 ";

    /// 3
    if( cnt > 500
        && nbimfromkf < 2)
    {
        return false;
    }
    std::cout<<" c3 ";
    
    /// 4
    bool cx = med_rot_parallax >= 15;
    bool c0 = med_rot_parallax >= 30;
    int cnt2 = 0;
    for (const auto& mp:last_kf->GetMapPointMatches()) {
        if (mp != nullptr && !mp->isBad()){
            cnt2 ++;
        }
    }
    bool c1 = cnt < 0.75 * cnt2;
    bool c2 = mCurrentFrame.mvKeys.size() < 500
              && cnt < 0.85 * cnt;

    bool bkfreq = (c0 || c1 || c2) && cx;
    std::cout<<" c4 "<<bkfreq;
    return bkfreq;
}
/////////////////////////////////////////////////////////////////////////

bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

//////////////////////////////////////////////////////////////////////
    //mMaxFrames = 5;
//////////////////////////////////////////////////////////////////////

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

//////////////////////////////////////////////////////////////////////
    int cnt = 0;
    for (const auto& mp:mCurrentFrame.mvpMapPoints) {
        if (mp != nullptr && !mp->isBad()){
            cnt ++;
        }
    }
//////////////////////////////////////////////////////////////////////
//    if((c1a||c1b||c1c)&&c2)
    if(((c1a||c1b||c1c)&&c2) || cnt < 200 || mCurrentFrame.mvKeys.size() < 400)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

//////////////////////////////////////////////////////////////////////
/*
    last_kf = pKF;
    local_queue.emplace(pKF);
    while (local_queue.size() > 5)
        local_queue.pop();
    for (int i = 0; i < pKF->GetMapPointMatches().size();i ++)
    {
        auto mp = pKF->GetMapPoint(i);
        if (mp != nullptr)
        {
            mp->AddObservation(pKF, i);
        }
    }
*/    
//////////////////////////////////////////////////////////////////////


    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
////////////////////////////////////////////////////////////////////////
    //int track_num = 0;
////////////////////////////////////////////////////////////////////////

    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
////////////////////////////////////////////////////////////////////////
                //track_num ++;
////////////////////////////////////////////////////////////////////////
            }
        }
    }

////////////////////////////////////////////////////////////////////////
/*
    ORBmatcher matcher(0.8);
    int cnt = 0;
    int match_num = 0;
    std::vector<KeyFrame*> kfs;
    kfs.resize(local_queue.size(), nullptr);
    for (int i = 0;i < local_queue.size(); i++)
    {
        kfs.at(local_queue.size() - i - 1) = local_queue.front();
        local_queue.emplace(local_queue.front());
        //std::cout<<"kf id "<<local_queue.front()->mnFrameId<<" match local "<<num<<std::endl;
        local_queue.pop();
    }
    for (auto &kf:kfs)
    {
        match_num += num;
        if (num < 5)
            break;
    }
*/
////////////////////////////////////////////////////////////////////////

// 原本的tracking.cc有這段code
    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
// 原本的tracking.cc有這段code
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
