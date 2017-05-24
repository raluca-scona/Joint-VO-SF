/*********************************************************************************
**Fast Odometry and Scene Flow from RGB-D Cameras based on Geometric Clustering	**
**------------------------------------------------------------------------------**
**																				**
**	Copyright(c) 2017, Mariano Jaimez Tarifa, University of Malaga & TU Munich	**
**	Copyright(c) 2017, Christian Kerl, TU Munich								**
**	Copyright(c) 2017, MAPIR group, University of Malaga						**
**	Copyright(c) 2017, Computer Vision group, TU Munich							**
**																				**
**  This program is free software: you can redistribute it and/or modify		**
**  it under the terms of the GNU General Public License (version 3) as			**
**	published by the Free Software Foundation.									**
**																				**
**  This program is distributed in the hope that it will be useful, but			**
**	WITHOUT ANY WARRANTY; without even the implied warranty of					**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the				**
**  GNU General Public License for more details.								**
**																				**
**  You should have received a copy of the GNU General Public License			**
**  along with this program. If not, see <http://www.gnu.org/licenses/>.		**
**																				**
*********************************************************************************/

#include <string.h>
#include <joint_vo_sf.h>

#include <chrono>
#include <iostream>
#include <fstream>


// -------------------------------------------------------------------------------
//								Instructions:
// You need to click on the window of the 3D Scene to be able to interact with it.
// 'n' - Load new frame and solve
// 's' - Turn on/off continuous estimation
// 'e' - Finish/exit
// -------------------------------------------------------------------------------

int main()
{	
    const unsigned int res_factor = 2;
	VO_SF cf(res_factor);


	//Set first image to load, decimation factor and the sequence dir
	unsigned int im_count = 1;
    const unsigned int decimation = 5;
    std::string dir = "/usr/prakt/p025/datasets/Giraff-loop-m-format/"; im_count = 220;

	//Load image and create pyramid
	cf.loadImageFromSequence(dir, im_count, res_factor);
	cf.createImagePyramid();

	//Create the 3D Scene
	cf.initializeSceneImageSeq();

	//Auxiliary variables
	int pushed_key = 0;
	bool continuous_exec = false, stop = false;

    //                                  EF PARAMS
    // -------------------------------------------------------------------------------

    int icpCountThresh = 40000;
    float icpErrThresh = 5e-05;
    float covThresh = 1e-05;
    bool openLoop = true;
    bool iclnuim = false;
    bool reloc = false;
    float photoThresh = 115;
    float confidence = 5.0f;
    float depth = 5.0f; //usually 3.5m -> pose drifts a lot with this setting
    float icp = 10.0f;
    bool fastOdom = false;
    float fernThresh = 0.3095f;
    bool so3 = true; //remember to update this
    bool frameToFrameRGB = false;
    const std::string fileName = "cf-mesh";
    int timeDelta = 200;

    Resolution::getInstance(640, 480);
    Intrinsics::getInstance(528, 528, 320, 240);

    cf.gui = new GUI(fileName.length() == 0, false);

    cf.gui->flipColors->Ref().Set(true);
    cf.gui->rgbOnly->Ref().Set(false);
    cf.gui->pyramid->Ref().Set(true);
    cf.gui->fastOdom->Ref().Set(fastOdom);
    cf.gui->confidenceThreshold->Ref().Set(confidence);
    cf.gui->depthCutoff->Ref().Set(depth);
    cf.gui->icpWeight->Ref().Set(icp);
    cf.gui->so3->Ref().Set(so3);
    cf.gui->frameToFrameRGB->Ref().Set(frameToFrameRGB);

    cf.resizeStream = new Resize(Resolution::getInstance().width(),
                                 Resolution::getInstance().height(),
                                 Resolution::getInstance().width() / 2,
                                 Resolution::getInstance().height() / 2);


    cf.eFusion = new ElasticFusion(openLoop ? std::numeric_limits<int>::max() / 2 : timeDelta,
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



    Eigen::Matrix4f * currentPose = 0;
    float weightMultiplier = 1;

    std::chrono::duration<double, std::milli> processFrameDuration;
    std::chrono::high_resolution_clock::time_point efProcessFrame0;
    std::chrono::high_resolution_clock::time_point efProcessFrame1;

    cv::Mat weightedImageColumnMajor;
    cv::Mat fullSizeWeightedImage;
    cv::Mat smallestWeightedImage;
    cv::Mat weightedImage;

    std::vector<float> level0WeightedImage (Resolution::getInstance().width() * Resolution::getInstance().height(), 0.0);
    std::vector<float> level1WeightedImage (Resolution::getInstance().width() / 2 * Resolution::getInstance().height() / 2, 0.0 );
    std::vector<float> level2WeightedImage (Resolution::getInstance().width() / 4 * Resolution::getInstance().height() / 4, 0.0 );

    std::vector<std::vector<float> > weightedImagePyramid(3);
    unsigned char * rgbImage;

    //GUI hacks
    std::stringstream stri;
    std::stringstream stre;

    Eigen::Matrix4f pose;
    bool resetButton = false;

    int totalRepeatFrames = 50;

    int counter = 0;

    //                                  DONE WITH EF PARAMS
    // -------------------------------------------------------------------------------
	
	while (!stop)
	{	
        if (cf.window.keyHit())
            pushed_key = cf.window.getPushedKey();
        else
            pushed_key = 0;

		switch (pushed_key) {

        //Load new image and solve
        case 'n':

            im_count += decimation;

            if (im_count > 960) {
                cf.eFusion->savePly();
            }

            stop = cf.loadImageFromSequence(dir, im_count, res_factor);
            cf.run_VO_SF(true);

            //RUNNING EF SOON
            efProcessFrame0 = std::chrono::high_resolution_clock::now();

            weightedImageColumnMajor = cv::Mat(320, 240, CV_32F, cf.b_segm_image_warped.data());

            cv::flip(weightedImageColumnMajor, weightedImage, 1);

            cv::transpose(weightedImage, weightedImage);  //stored in row major order

            cv::resize(weightedImage, fullSizeWeightedImage,  cv::Size(640, 480) , 0,0);  //resize to full image
            cv::resize(weightedImage, smallestWeightedImage,  cv::Size(640/4, 480/4) , 0,0);  //resize to small image

//            weightedImagePyramid[0] = level0WeightedImage;
//            weightedImagePyramid[1] = level1WeightedImage;
//            weightedImagePyramid[2] = level2WeightedImage;

            weightedImagePyramid[0].assign((float *) fullSizeWeightedImage.data, (float *) fullSizeWeightedImage.data + 640 / 1 * 480 / 1);
            weightedImagePyramid[1].assign((float *) weightedImage.data, (float *) weightedImage.data + 640 / 2 * 480 / 2);
            weightedImagePyramid[2].assign((float *) smallestWeightedImage.data, (float *) smallestWeightedImage.data + 640 / 4 * 480 / 4);

            rgbImage = cf.color_full.data;

            for(int i = 0; i < 640 * 480 * 3; i += 3)
            {
                std::swap(rgbImage[i + 0], rgbImage[i + 2]);   //flipping the colours
            }

            cf.eFusion->processFrame(rgbImage, (unsigned short *) cf.depth_full.data, weightedImagePyramid , im_count, currentPose, weightMultiplier);

            efProcessFrame1 = std::chrono::high_resolution_clock::now();

            processFrameDuration = efProcessFrame1 - efProcessFrame0;

            std::cout<<"\nef process frame time: "<<processFrameDuration.count()<<"\n";

            //ALL THE GUI

            if (true) {

                if(cf.gui->followPose->Get())
                {
                    pangolin::OpenGlMatrix mv;

                    Eigen::Matrix4f currPose = cf.eFusion->getCurrPose();
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

                    cf.gui->s_cam.SetModelViewMatrix(mv);
                }


                cf.gui->preCall();

                pose = cf.eFusion->getCurrPose();

                if(cf.gui->drawRawCloud->Get() || cf.gui->drawFilteredCloud->Get() || cf.gui->drawStaticCloud->Get())
                {
                    cf.eFusion->computeFeedbackBuffers();
                }


                if(cf.gui->drawRawCloud->Get())
                {
                    cf.eFusion->getFeedbackBuffers().at(FeedbackBuffer::RAW)->render(cf.gui->s_cam.GetProjectionModelViewMatrix(), pose, cf.gui->drawNormals->Get(), cf.gui->drawColors->Get());
                }


                if(cf.gui->drawFilteredCloud->Get())
                {
                    cf.eFusion->getFeedbackBuffers().at(FeedbackBuffer::FILTERED)->render(cf.gui->s_cam.GetProjectionModelViewMatrix(), pose, cf.gui->drawNormals->Get(), cf.gui->drawColors->Get());
                }

                if(cf.gui->drawStaticCloud->Get())
                {
                    cf.eFusion->getFeedbackBuffers().at(FeedbackBuffer::STATIC)->render(cf.gui->s_cam.GetProjectionModelViewMatrix(), pose, cf.gui->drawNormals->Get(), cf.gui->drawColors->Get());
                }





                if(cf.gui->drawGlobalModel->Get())
                {
                    glFinish();


                    cf.eFusion->getGlobalModel().renderPointCloud(cf.gui->s_cam.GetProjectionModelViewMatrix(),
                                                               cf.eFusion->getConfidenceThreshold(),
                                                               cf.gui->drawUnstable->Get(),
                                                               cf.gui->drawNormals->Get(),
                                                               cf.gui->drawColors->Get(),
                                                               cf.gui->drawPoints->Get(),
                                                               cf.gui->drawWindow->Get(),
                                                               cf.gui->drawTimes->Get(),
                                                               cf.eFusion->getTick(),
                                                               cf.eFusion->getTimeDelta());

                    glFinish();
                }



                if(cf.eFusion->getLost())
                {
                    glColor3f(1, 1, 0);
                }
                else
                {
                    glColor3f(1, 0, 1);
                }
                cf.gui->drawFrustum(pose);


                glColor3f(1, 1, 1);

                cf.eFusion->normaliseDepth(0.3f, cf.gui->depthCutoff->Get());

                for(std::map<std::string, GPUTexture*>::const_iterator it = cf.eFusion->getTextures().begin(); it != cf.eFusion->getTextures().end(); ++it)
                {
                    if(it->second->draw)
                    {
                        cf.gui->displayImg(it->first, it->second);
                    }
                }

                cf.eFusion->getIndexMap().renderDepth(cf.gui->depthCutoff->Get());


                cf.gui->displayImg("ModelImg", cf.eFusion->getIndexMap().imageTex());
                cf.gui->displayImg("Model", cf.eFusion->getIndexMap().drawTex());

                cf.gui->postCall();


                cf.eFusion->setRgbOnly(cf.gui->rgbOnly->Get());
                cf.eFusion->setPyramid(cf.gui->pyramid->Get());
                cf.eFusion->setFastOdom(cf.gui->fastOdom->Get());
                cf.eFusion->setConfidenceThreshold(cf.gui->confidenceThreshold->Get());
                cf.eFusion->setDepthCutoff(cf.gui->depthCutoff->Get());
                cf.eFusion->setIcpWeight(cf.gui->icpWeight->Get());
                cf.eFusion->setSo3(cf.gui->so3->Get());
                cf.eFusion->setFrameToFrameRGB(cf.gui->frameToFrameRGB->Get());

                resetButton = pangolin::Pushed(*cf.gui->reset);


                if(resetButton)
                {
                    break;
                }

                if(pangolin::Pushed(*cf.gui->save))
                {
                    cf.eFusion->savePly();
                }

            }

            //NO MORE EF GUI


            //DONE WITH EF


            cf.createImagesOfSegmentations();
            cf.updateSceneImageSeq();
            break;

		//Start/Stop continuous estimation
		case 's':
			continuous_exec = !continuous_exec;
			break;
			
		//Close the program
		case 'e':
			stop = true;
			break;
		}
	
		if ((continuous_exec)&&(!stop))
		{
			im_count += decimation;
			stop = cf.loadImageFromSequence(dir, im_count, res_factor);
            cf.run_VO_SF(true);
            cf.createImagesOfSegmentations();
			cf.updateSceneImageSeq();
		}
	}

	return 0;
}

