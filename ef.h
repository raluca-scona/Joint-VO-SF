#ifndef EF_Container_H
#define EF_Container_H

#include "ElasticFusion.h"
#include "Shaders/Resize.h"
#include "GUI.h"
#include <Eigen/Core>


class EF_Container {

public:

    ElasticFusion * eFusion;
    Resize * resizeStream;
    GUI * gui;

    Eigen::Matrix4f fromNomToVis;
    Eigen::Matrix4f fromVisToNom;

    int icpCountThresh;
    float icpErrThresh;
    float covThresh;
    bool openLoop;
    bool iclnuim;
    bool reloc;
    float photoThresh;
    float confidence;
    float depth;
    float icp;
    bool fastOdom;
    float fernThresh;
    bool so3;
    bool frameToFrameRGB;
    const std::string fileName;
    int timeDelta;

    EF_Container() {

        //various params and thresholds
        icpCountThresh = 40000;
        icpErrThresh = 5e-05;
        covThresh = 1e-05;
        openLoop = false;
        iclnuim = false;
        reloc = true;
        photoThresh = 115;
        confidence = 2.0f;
        depth = 5.0f; //usually 3.5m
        icp = 10.0f;
        fastOdom = false;
        fernThresh = 0.3095f;
        so3 = true; //remember to update this
        frameToFrameRGB = false;
        std::string fileName = "cf-mesh";
        timeDelta = 200;

        fromNomToVis = Eigen::Matrix4f::Zero();
        fromVisToNom = Eigen::Matrix4f::Zero();

        Eigen::Quaternionf fromNomToVisQuat = Eigen::AngleAxisf(0.5*M_PI, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(0.5*M_PI, Eigen::Vector3f::UnitX());
        Eigen::Matrix3f fromNomToVisRot = fromNomToVisQuat.normalized().toRotationMatrix();

        fromNomToVis.topLeftCorner(3, 3) = fromNomToVisRot;
        fromNomToVis(3, 3) = 1;

        fromVisToNom = fromNomToVis.inverse();

        Resolution::getInstance(640, 480);
        Intrinsics::getInstance(528, 528, 320, 240);

        gui = new GUI(fileName.length() == 0, false);

        gui->flipColors->Ref().Set(true);
        gui->rgbOnly->Ref().Set(false);
        gui->pyramid->Ref().Set(true);
        gui->fastOdom->Ref().Set(fastOdom);
        gui->confidenceThreshold->Ref().Set(confidence);
        gui->depthCutoff->Ref().Set(depth);
        gui->icpWeight->Ref().Set(icp);
        gui->so3->Ref().Set(so3);
        gui->frameToFrameRGB->Ref().Set(frameToFrameRGB);

        resizeStream = new Resize(Resolution::getInstance().width(),
                                     Resolution::getInstance().height(),
                                     Resolution::getInstance().width() / 2,
                                     Resolution::getInstance().height() / 2);


        eFusion = new ElasticFusion(openLoop ? std::numeric_limits<int>::max() / 2 : timeDelta,
                                      icpCountThresh,
                                      icpErrThresh,
                                      covThresh,
                                      !openLoop,
                                      iclnuim,
                                      reloc,
                                      photoThresh,
                                      confidence,
                                      depth,
                                      icp,
                                      fastOdom,
                                      fernThresh,
                                      so3,
                                      frameToFrameRGB,
                                      fileName);

    }

    void updateGUI() {

        if(gui->followPose->Get())
        {
            pangolin::OpenGlMatrix mv;

            Eigen::Matrix4f currPose = eFusion->getCurrPose();
            Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);

            Eigen::Quaternionf currQuat(currRot);
            Eigen::Vector3f forwardVector(0, 0, 1);
            Eigen::Vector3f upVector(0, iclnuim ? 1 : -1, 0);

            Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
            Eigen::Vector3f up = (currQuat * upVector).normalized();

            Eigen::Vector3f eye(currPose(0, 3), currPose(1, 3), currPose(2, 3));

            eye -= forward;

            Eigen::Vector3f at = eye + forward;

            Eigen::Vector3f z = (eye - at).normalized();  // Forward
            Eigen::Vector3f x = up.cross(z).normalized(); // Right
            Eigen::Vector3f y = z.cross(x);

            Eigen::Matrix4d m;
            m << x(0),  x(1),  x(2),  -(x.dot(eye)),
                 y(0),  y(1),  y(2),  -(y.dot(eye)),
                 z(0),  z(1),  z(2),  -(z.dot(eye)),
                    0,     0,     0,              1;

            memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));

            gui->s_cam.SetModelViewMatrix(mv);
        }


        gui->preCall();

        Eigen::Matrix4f pose = eFusion->getCurrPose();

        if(gui->drawRawCloud->Get() || gui->drawFilteredCloud->Get() || gui->drawStaticCloud->Get())
        {
            eFusion->computeFeedbackBuffers();
        }


        if(gui->drawRawCloud->Get())
        {
            eFusion->getFeedbackBuffers().at(FeedbackBuffer::RAW)->render(gui->s_cam.GetProjectionModelViewMatrix(), pose, gui->drawNormals->Get(), gui->drawColors->Get());
        }


        if(gui->drawFilteredCloud->Get())
        {
            eFusion->getFeedbackBuffers().at(FeedbackBuffer::FILTERED)->render(gui->s_cam.GetProjectionModelViewMatrix(), pose, gui->drawNormals->Get(), gui->drawColors->Get());
        }

        if(gui->drawStaticCloud->Get())
        {
            eFusion->getFeedbackBuffers().at(FeedbackBuffer::STATIC)->render(gui->s_cam.GetProjectionModelViewMatrix(), pose, gui->drawNormals->Get(), gui->drawColors->Get());
        }





        if(gui->drawGlobalModel->Get())
        {
            glFinish();


            eFusion->getGlobalModel().renderPointCloud(gui->s_cam.GetProjectionModelViewMatrix(),
                                                       eFusion->getConfidenceThreshold(),
                                                       gui->drawUnstable->Get(),
                                                       gui->drawNormals->Get(),
                                                       gui->drawColors->Get(),
                                                       gui->drawPoints->Get(),
                                                       gui->drawWindow->Get(),
                                                       gui->drawTimes->Get(),
                                                       eFusion->getTick(),
                                                       eFusion->getTimeDelta());

            glFinish();
        }



        if(eFusion->getLost())
        {
            glColor3f(1, 1, 0);
        }
        else
        {
            glColor3f(1, 0, 1);
        }
        gui->drawFrustum(pose);


        glColor3f(1, 1, 1);

        eFusion->normaliseDepth(0.3f, gui->depthCutoff->Get());

        for(std::map<std::string, GPUTexture*>::const_iterator it = eFusion->getTextures().begin(); it != eFusion->getTextures().end(); ++it)
        {
            if(it->second->draw)
            {
                gui->displayImg(it->first, it->second);
            }
        }

        eFusion->getIndexMap().renderDepth(gui->depthCutoff->Get());


        gui->displayImg("ModelImg", eFusion->getIndexMap().imageTex());
        gui->displayImg("Model", eFusion->getIndexMap().drawTex());

        gui->postCall();


        eFusion->setRgbOnly(gui->rgbOnly->Get());
        eFusion->setPyramid(gui->pyramid->Get());
        eFusion->setFastOdom(gui->fastOdom->Get());
        eFusion->setConfidenceThreshold(gui->confidenceThreshold->Get());
        eFusion->setDepthCutoff(gui->depthCutoff->Get());
        eFusion->setIcpWeight(gui->icpWeight->Get());
        eFusion->setSo3(gui->so3->Get());
        eFusion->setFrameToFrameRGB(gui->frameToFrameRGB->Get());

        if(pangolin::Pushed(*gui->save))
        {
            eFusion->savePly();
        }
    }

};

#endif // EF

