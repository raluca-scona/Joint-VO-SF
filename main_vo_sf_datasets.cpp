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

#include <stdio.h>
#include <joint_vo_sf.h>
#include <ef.h>
#include <datasets.h>


// -------------------------------------------------------------------------------
//								Instructions:
// You need to click on the window of the 3D Scene to be able to interact with it.
// 'n' - Load new frame and solve
// 's' - Turn on/off continuous estimation
// 'e' - Finish/exit
//
// Set the flag "save_results" to true if you want to save the estimated trajectory
// -------------------------------------------------------------------------------

int main()
{	
    //                                  EF PARAMS
    // -------------------------------------------------------------------------------

    EF_Container ef;

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

    int im_count = 1;

    Eigen::Matrix4f poseEFCoords;

    const float norm_factor = 1.f/255.f;


    //                                  DONE WITH EF PARAMS
    // -------------------------------------------------------------------------------



	const bool save_results = true;
    unsigned int res_factor = 2;
	VO_SF cf(res_factor);
	Datasets dataset(res_factor);

    cv::Mat depth_full = cv::Mat(cf.height * res_factor, cf.width * res_factor,  CV_16U, 0.0);
    cv::Mat color_full = cv::Mat(cf.height * res_factor, cf.width * res_factor,  CV_8UC3,  cv::Scalar(0,0,0));


	//Set dir of the Rawlog file
    dataset.filename = "/usr/prakt/p025/datasets/tum-benchmark-mrpt/rawlog_rgbd_dataset_freiburg3_walking_static/rgbd_dataset_freiburg3_walking_static.rawlog";
   // dataset.filename = "/usr/prakt/p025/datasets/tum-benchmark-mrpt/rawlog_rgbd_dataset_freiburg2_desk_with_person/rgbd_dataset_freiburg2_desk_with_person.rawlog";

	//Create the 3D Scene
	cf.initializeSceneDatasets();

	//Initialize
	if (save_results)
		dataset.CreateResultsFile();
    dataset.openRawlog();

    dataset.loadFrameAndPoseFromDataset(cf.depth_wf_old, cf.intensity_wf_old, cf.im_r_old, cf.im_g_old, cf.im_b_old, depth_full, color_full);
    cf.cam_pose = dataset.gt_pose; cf.cam_oldpose = dataset.gt_pose;

    rgbImage = color_full.data;
    for(int i = 0; i < 640 * 480 * 3; i += 3)
    {
        std::swap(rgbImage[i + 0], rgbImage[i + 2]);   //flipping the colours for EF
    }
    // Initialise a weighted pyramid with zeros
    weightedImagePyramid[0].assign((float *) level0WeightedImage.data(), (float *) level0WeightedImage.data() + 640 / 1 * 480 / 1);
    weightedImagePyramid[1].assign((float *) level1WeightedImage.data(), (float *) level1WeightedImage.data() + 640 / 2 * 480 / 2);
    weightedImagePyramid[2].assign((float *) level2WeightedImage.data(), (float *) level2WeightedImage.data() + 640 / 4 * 480 / 4);
    //Initialise model in EF to the first frame we get
    ef.eFusion->processFrame(rgbImage, (unsigned short *) depth_full.data, weightedImagePyramid , im_count, 0 , 1);


	//Auxiliary variables
	int pushed_key = 0, stop = 0;
	bool anything_new = false, continuous_exec = false;
	
	while (!stop)
	{	

        if (cf.window.keyHit())
            pushed_key = cf.window.getPushedKey();
        else
            pushed_key = 0;

		switch (pushed_key) {
			
        //Load new frame and solve
		case  'n':
            dataset.loadFrameAndPoseFromDataset(cf.depth_wf, cf.intensity_wf, cf.im_r, cf.im_g, cf.im_b, depth_full, color_full);

            im_count++;

            //Got images from the model, should overwrite the old images and re-do the pyramid
            ef.eFusion->getPredictedImages(colorPrediction, depthPrediction);

            std::cout<<cf.height<<" "<<cf.width<<"\n";

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

            cf.run_VO_SF(true);

            std::cout<<"also  here\n";


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

            rgbImage = color_full.data;

            for(int i = 0; i < 640 * 480 * 3; i += 3)
            {
                std::swap(rgbImage[i + 0], rgbImage[i + 2]);   //get the colour image of the latest frame
            }

            ef.eFusion->processFrame(rgbImage, (unsigned short *) depth_full.data, weightedImagePyramid, im_count, &(poseEFCoords), 1);

            ef.updateGUI();


            cf.createImagesOfSegmentations();
			if (save_results)
				dataset.writeTrajectoryFile(cf.cam_pose, cf.ddt);
            anything_new = 1;




			break;

        //Turn on/off continuous estimation
        case 's':
            continuous_exec = !continuous_exec;
            break;
		
		//Close the program
		case 'e':
			stop = 1;
			break;
		}

        if (continuous_exec)
        {
            dataset.loadFrameAndPoseFromDataset(cf.depth_wf, cf.intensity_wf, cf.im_r, cf.im_g, cf.im_b, depth_full, color_full);
            cf.run_VO_SF(true);
            cf.createImagesOfSegmentations();
			if (save_results)
				dataset.writeTrajectoryFile(cf.cam_pose, cf.ddt);
            anything_new = 1;

			if (dataset.dataset_finished)
				continuous_exec = false;
        }
	
		if (anything_new)
		{
			bool aux = false;
			cf.updateSceneDatasets(dataset.gt_pose, dataset.gt_oldpose);
			anything_new = 0;
		}
	}

	if (save_results)
		dataset.f_res.close();

	return 0;
}

