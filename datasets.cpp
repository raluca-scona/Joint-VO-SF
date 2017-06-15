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

#include <datasets.h>

using namespace mrpt;
using namespace mrpt::obs;
using namespace std;


Datasets::Datasets(unsigned int res_factor)
{
    downsample = res_factor; // (1 - 640 x 480, 2 - 320 x 240)
	max_distance = 6.f;
	dataset_finished = false;
	rawlog_count = 0;
}

void Datasets::openRawlog()
{

	//						Open Rawlog File
	//==================================================================
	if (!dataset.loadFromRawLogFile(filename))
		throw std::runtime_error("\nCouldn't open rawlog dataset file for input...");

	// Set external images directory:
	const string imgsPath = CRawlog::detectImagesDirectory(filename);
	utils::CImage::IMAGES_PATH_BASE = imgsPath;


	//					Load ground-truth
	//=========================================================
	filename = system::extractFileDirectory(filename);
	filename.append("/groundtruth.txt");
	f_gt.open(filename.c_str());
	if (f_gt.fail())
		throw std::runtime_error("\nError finding the groundtruth file: it should be contained in the same folder than the rawlog file");

	//Count number of lines (of the file)
	unsigned int number_of_lines = 0;
    std::string line;
    while (std::getline(f_gt, line))
        ++number_of_lines;

	gt_matrix.resize(number_of_lines-3, 8);
    f_gt.clear();
	f_gt.seekg(0, ios::beg);

	//Store the gt data in a matrix
	char aux[100];
	f_gt.getline(aux, 100);
	f_gt.getline(aux, 100);
	f_gt.getline(aux, 100);
	for (unsigned int k=0; k<number_of_lines-3; k++)
	{
		f_gt >> gt_matrix(k,0); f_gt >> gt_matrix(k,1); f_gt >> gt_matrix(k,2); f_gt >> gt_matrix(k,3);
		f_gt >> gt_matrix(k,4); f_gt >> gt_matrix(k,5); f_gt >> gt_matrix(k,6); f_gt >> gt_matrix(k,7);
		f_gt.ignore(10,'\n');	
	}

	f_gt.close();
	last_gt_row = 0;
}

void Datasets::loadFrameAndPoseFromDataset(Eigen::MatrixXf &depth_wf, Eigen::MatrixXf &intensity_wf, Eigen::MatrixXf &im_r, Eigen::MatrixXf &im_g, Eigen::MatrixXf &im_b, cv::Mat depth_full, cv::Mat color_full)
{
	if (dataset_finished)
	{
		printf("\n End of the dataset reached. Stop estimating motion!");
		return;
	}
	
	//Read images
	//-------------------------------------------------------
	CObservationPtr alfa = dataset.getAsObservation(rawlog_count);

	while (!IS_CLASS(alfa, CObservation3DRangeScan))
	{
		rawlog_count++;
		if (dataset.size() <= rawlog_count)
		{
			dataset_finished = true;
			return;
		}
		alfa = dataset.getAsObservation(rawlog_count);
	}

	CObservation3DRangeScanPtr obs3D = CObservation3DRangeScanPtr(alfa);
	obs3D->load();
	const Eigen::MatrixXf range = obs3D->rangeImage;
	const utils::CImage int_image =  obs3D->intensityImage;
	const unsigned int height = range.getRowCount();
	const unsigned int width = range.getColCount();
	const unsigned int cols = width/downsample, rows = height/downsample;

    math::CMatrixFloat intensity, r, g, b;
    intensity.resize(height, width);
	r.resize(height, width); g.resize(height, width); b.resize(height, width);
	int_image.getAsMatrix(intensity);
	int_image.getAsRGBMatrices(r, g, b);

	for (unsigned int j = 0; j<cols; j++)
		for (unsigned int i = 0; i<rows; i++)
		{
			intensity_wf(i,j) = intensity(height-downsample*i-1, width-downsample*j-1);
			const float z = range(height-downsample*i-1, width-downsample*j-1);
			if (z < max_distance)	depth_wf(i,j) = z;
			else					depth_wf(i,j) = 0.f;

			//Color image, just for the visualization
			im_r(i,j) = b(height-downsample*i-1, width-downsample*j-1);
			im_g(i,j) = g(height-downsample*i-1, width-downsample*j-1);
			im_b(i,j) = r(height-downsample*i-1, width-downsample*j-1);
        }


    for (unsigned int j = 0; j<width; j++)
        for (unsigned int i = 0; i<height; i++)
        {
            const float z = range(height-i-1, width-j-1);
            if (z < max_distance)	depth_full.at<unsigned short>(i,j) = z * 1000.0;
            else					depth_full.at<unsigned short>(i,j) = 0.f;

            color_full.at<cv::Vec3b>(i,j) = cv::Vec3b( r(height-i-1, width-j-1) * 255, g(height-i-1, width-j-1)* 255, b(height-i-1, width-j-1)* 255);
        }


	timestamp_obs = mrpt::system::timestampTotime_t(obs3D->timestamp);

	obs3D->unload();
	rawlog_count++;

	if (dataset.size() <= rawlog_count)
		dataset_finished = true;

	//Groundtruth
	//--------------------------------------------------
	//Check whether the current gt is the closest one or we should read new gt
	const float current_dif_tim = abs(gt_matrix(last_gt_row,0) - timestamp_obs);
	const float next_dif_tim = abs(gt_matrix(last_gt_row+1,0) - timestamp_obs);

	while (abs(gt_matrix(last_gt_row,0) - timestamp_obs) > abs(gt_matrix(last_gt_row+1,0) - timestamp_obs))
	{
		last_gt_row++;
		if (last_gt_row >= gt_matrix.rows())
		{
			dataset_finished = true;
			return;		
		}
	}

	//Get the pose of the closest ground truth
	double x,y,z,qx,qy,qz,w;
	x = gt_matrix(last_gt_row,1); y = gt_matrix(last_gt_row,2); z = gt_matrix(last_gt_row,3);
	qx = gt_matrix(last_gt_row,4); qy = gt_matrix(last_gt_row,5); qz = gt_matrix(last_gt_row,6);
	w = gt_matrix(last_gt_row,7);

	math::CMatrixDouble33 mat;
	mat(0,0) = 1 - 2*qy*qy - 2*qz*qz;
	mat(0,1) = 2*(qx*qy - w*qz);
	mat(0,2) = 2*(qx*qz + w*qy);
	mat(1,0) = 2*(qx*qy + w*qz);
	mat(1,1) = 1 - 2*qx*qx - 2*qz*qz;
	mat(1,2) = 2*(qy*qz - w*qx);
	mat(2,0) = 2*(qx*qz - w*qy);
	mat(2,1) = 2*(qy*qz + w*qx);
	mat(2,2) = 1 - 2*qx*qx - 2*qy*qy;

	poses::CPose3D gt, transf;
	gt.setFromValues(x,y,z,0,0,0);
	gt.setRotationMatrix(mat);
	transf.setFromValues(0,0,0,0.5*M_PI, -0.5*M_PI, 0); //Needed because we use different coordinates

	gt_oldpose = gt_pose;
	gt_pose = gt + transf;
}


void Datasets::CreateResultsFile()
{
	//Create file with the first free file-name.
	char	aux[100];
	int     nFile = 0;
	bool    free_name = false;

	system::createDirectory("./odometry_results");

	while (!free_name)
	{
		nFile++;
		sprintf(aux, "./odometry_results/experiment_%03u.txt", nFile );
		free_name = !system::fileExists(aux);
	}

	// Open log file:
	f_res.open(aux);
	printf(" Saving results to file: %s \n", aux);
}

void Datasets::writeTrajectoryFile(poses::CPose3D &cam_pose, Eigen::MatrixXf &ddt)
{	
	//Don't take into account those iterations with consecutive equal depth images
	if (abs(ddt.sumAll()) > 0)
	{		
		mrpt::math::CQuaternionDouble quat;
		poses::CPose3D auxpose, transf;
		transf.setFromValues(0,0,0,0.5*M_PI, -0.5*M_PI, 0);

		auxpose = cam_pose - transf;
		auxpose.getAsQuaternion(quat);
	
		char aux[24];
		sprintf(aux,"%.04f", timestamp_obs);
		f_res << aux << " " << cam_pose[0] << " " << cam_pose[1] << " " << cam_pose[2] << " ";
		f_res << quat(2) << " " << quat(3) << " " << -quat(1) << " " << -quat(0) << endl;
	}
}




