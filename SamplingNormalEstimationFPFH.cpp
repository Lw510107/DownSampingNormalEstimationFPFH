#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/correspondence.h>                 //对应表示两个实体之间的匹配（例如，点，描述符等）。
#include <pcl/features/normal_3d.h>             //法线
#include <pcl/features/fpfh.h>
#include <pcl/filters/uniform_sampling.h>    //均匀采样
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>   //可视化
#include <pcl/common/transforms.h>              //转换矩阵
#include <pcl/console/parse.h>
#include <pcl/point_types.h>

#include <string>
#include <iostream>
#include <stdio.h>
using namespace std;

typedef pcl::PointXYZRGB PointT;             //Point with color
typedef pcl::PointCloud<PointT> PointCloud;   //PointCloud with color

int main()
{
	string model_filename = "../dataset/boat.ply";
	bool display = true;
	bool downSampling = false;

	//load pcd/ply point cloud
	pcl::PointCloud<PointT>::Ptr model(new pcl::PointCloud<PointT>()); //模型点云
	if (pcl::io::loadPLYFile(model_filename, *model) < 0)
	{
		std::cerr << "Error loading model cloud." << std::endl;
		return (-1);
	}

	if (downSampling)
	{
		//create the filtering object
		std::cout << "Number of points before downSampling: " << model->points.size() << std::endl;
		pcl::VoxelGrid<PointT> sor;
		sor.setInputCloud(model);
		sor.setLeafSize(0.01, 0.01, 0.01);
		sor.filter(*model);
		std::cout << "Number of points after downSampling: " << model->points.size() << std::endl;
	}

	// Normal estimation
	pcl::NormalEstimation<PointT, pcl::PointNormal> ne;
	pcl::PointCloud<pcl::PointNormal>::Ptr normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
	ne.setInputCloud(model);
	ne.setSearchMethod(tree);
	ne.setKSearch(10);
	//boost::shared_ptr<pcl::PointIndices> indices;
	//indices->indices.push_back(200);
	//ne.setIndices(indices);
	ne.compute(*normals);

	for(int i = 100; i<101; ++i)
		printf("points[%d]=[%f, %f, %f], normal=[%f, %f, %f], curvature = %f\n",i, model->points[i].x, model->points[i].y, model->points[i].z, normals->points[i].normal_x, normals->points[i].normal_y, normals->points[i].normal_z, normals->points[i].curvature);
	//for (int i = 0; i < normals->points.size(); ++i)
	//	printf("points[%f, %f, %f], normal[%f, %f, %f]", normals->points[i].x, normals->points[i].y, normals->points[i].z, normals->points[i].normal_x, normals->points[i].normal_x, normals->points[i].y, normals->points[i].normal_z);
	
	//create the FPFH estimation class, and pass the input dataset+normals to it
	pcl::FPFHEstimation<PointT, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());
	fpfh.setInputCloud(model);
	fpfh.setInputNormals(normals);
	fpfh.setSearchMethod(tree);
	fpfh.setRadiusSearch(0.02); // Use all neighbors in a sphere of radius 5cm, the radius used here has to be larger than the radius used to estimate the surface normals!!!
	fpfh.compute(*fpfhs);// Compute the features


	FILE* fid = fopen("source.bin", "wb");
	int nV = normals->size(), nDim = 33;
	fwrite(&nV, sizeof(int), 1, fid);            // size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream)
	fwrite(&nDim, sizeof(int), 1, fid);
	for (int v = 0; v < nV; v++)
	{
		const pcl::PointNormal &pt = normals->points[v];	
		float xyz[3] = { pt.normal_x, pt.normal_y, pt.normal_z };
		fwrite(xyz, sizeof(float), 3, fid);
		const pcl::FPFHSignature33 &feature = fpfhs->points[v];
		fwrite(feature.histogram, sizeof(float), 33, fid);
	}
	fclose(fid);

	if (display)
	{
		// Concatenate the XYZ and normal fields*
		//pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
		//pcl::concatenateFields(*model, *normals, *cloud_with_normals);
		//pcl::io::savePLYFile("result.ply", *cloud_with_normals);
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Viewer"));
		viewer->setBackgroundColor(0, 0, 0);
		viewer->addPointCloud<PointT>(model, "model");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "model");
		viewer->addPointCloudNormals<PointT, pcl::PointNormal>(model, normals, 10, 0.05, "normals");  //display every 1 points, and the scale of the arrow is 10
		viewer->addCoordinateSystem(1.0);
		viewer->initCameraParameters();
		while (!viewer->wasStopped())
		{
			viewer->spinOnce(100);
			boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}
	}
	return 0;
}
