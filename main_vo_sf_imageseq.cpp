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
#include <ef.h>

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
   //
   // std::string dir = "/usr/prakt/p025/datasets/two-people-moving-1/"; im_count = 1;
   // std::string dir = "/usr/prakt/p025/datasets/Me-standing-moving-cam-1/"; im_count = 1;


    //                                  EF PARAMS
    // -------------------------------------------------------------------------------

    EF_Container ef;

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

    cv::Mat colorPrediction;
    cv::Mat depthPrediction;
    cv::Mat weightPrediction;


    Eigen::Matrix4f poseEFCoords;

    //                                  DONE WITH EF PARAMS
    // -------------------------------------------------------------------------------

    //Load image
    cf.loadImageFromSequence(dir, im_count, res_factor);

    rgbImage = cf.color_full.data;

    for(int i = 0; i < 640 * 480 * 3; i += 3)
    {
        std::swap(rgbImage[i + 0], rgbImage[i + 2]);   //flipping the colours for EF
    }

    // Initialise a weighted pyramid with zeros
    weightedImagePyramid[0].assign((float *) level0WeightedImage.data(), (float *) level0WeightedImage.data() + 640 / 1 * 480 / 1);
    weightedImagePyramid[1].assign((float *) level1WeightedImage.data(), (float *) level1WeightedImage.data() + 640 / 2 * 480 / 2);
    weightedImagePyramid[2].assign((float *) level2WeightedImage.data(), (float *) level2WeightedImage.data() + 640 / 4 * 480 / 4);

    //Initialise model in EF to the first frame we get
    ef.eFusion->processFrame(rgbImage, (unsigned short *) cf.depth_full.data, weightedImagePyramid , im_count, 0 , 1);


	//Create the 3D Scene
	cf.initializeSceneImageSeq();

	//Auxiliary variables
	int pushed_key = 0;
	bool continuous_exec = false, stop = false;

    const float norm_factor = 1.f/255.f;

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

            stop = cf.loadImageFromSequence(dir, im_count, res_factor);

            efProcessFrame0 = std::chrono::high_resolution_clock::now();

            //Got images from the model, should overwrite the old images and re-do the pyramid
            ef.eFusion->getPredictedImages(colorPrediction, depthPrediction, weightPrediction);

            //overwriting the previous colour and depth images
            for (unsigned int v=0; v<cf.height; v++) {
                for (unsigned int u=0; u<cf.width; u++)
                {
                    cv::Vec3b color_here = colorPrediction.at<cv::Vec3b>(res_factor*v, res_factor*u);
                    cf.im_r_old(v,u) = norm_factor*color_here[0];
                    cf.im_g_old(v,u) = norm_factor*color_here[1];
                    cf.im_b_old(v,u) = norm_factor*color_here[2];
                    cf.intensity_wf_old(v,u) = 0.299f* cf.im_r_old(v,u) + 0.587f*cf.im_g_old(v,u) + 0.114f*cf.im_b_old(v,u);

                    cf.depth_wf_old(v,u) = depthPrediction.at<float>(res_factor*v, res_factor*u);
                }
            }

            cf.createImagePyramid(true);   //pyramid for the old model

            cf.run_VO_SF(true); //run algorithm using pyramid for new images
            poseEFCoords = ef.fromVisToNom * cf.T_odometry * ef.fromNomToVis; //EF uses the coordinate frame with Z forwards, Y upwards

            weightedImageColumnMajor = cv::Mat(320, 240, CV_32F, cf.b_segm_image_warped.data()); //Eigen returns data in column-major
            //When running in this mode, images are flipped upside down. If I want to pass them in their original orientation, I need to do flip them.
            //Now I am just passing them as they have been preprocessed in CF, so upside down.
            //cv::flip(weightedImageColumnMajor, weightedImage, 1);
            weightedImage = weightedImageColumnMajor;
            cv::transpose(weightedImage, weightedImage);  //stored in row major order
            cv::resize(weightedImage, fullSizeWeightedImage,  cv::Size(640, 480) , 0,0);  //resize to full image
            cv::resize(weightedImage, smallestWeightedImage,  cv::Size(640/4, 480/4) , 0,0);  //resize to small image

            weightedImagePyramid[0].assign((float *) fullSizeWeightedImage.data, (float *) fullSizeWeightedImage.data + 640 / 1 * 480 / 1);
            weightedImagePyramid[1].assign((float *) weightedImage.data, (float *) weightedImage.data + 640 / 2 * 480 / 2);
            weightedImagePyramid[2].assign((float *) smallestWeightedImage.data, (float *) smallestWeightedImage.data + 640 / 4 * 480 / 4);

            rgbImage = cf.color_full.data;

            for(int i = 0; i < 640 * 480 * 3; i += 3)
            {
                std::swap(rgbImage[i + 0], rgbImage[i + 2]);   //get the colour image of the latest frame
            }

            ef.eFusion->processFrame(rgbImage, (unsigned short *) cf.depth_full.data, weightedImagePyramid, im_count, &(poseEFCoords), 1);

            efProcessFrame1 = std::chrono::high_resolution_clock::now();

            processFrameDuration = efProcessFrame1 - efProcessFrame0;

            std::cout<<"\n Run-time of whole thing "<<processFrameDuration.count()<<"\n";

            ef.updateGUI();

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

