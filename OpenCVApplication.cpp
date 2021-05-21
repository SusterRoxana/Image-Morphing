// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cvstd.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/core//types_c.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


using namespace std;

// function that reads an image and its txt with points 
void read_image(char *path_image, char *path_points, Mat &image, vector<Point2f> &key_points)
{
	image = imread(path_image);
	ifstream f(path_points);
	vector<int> points;
	int x, y;
	while (f >> x >> y)
	{
		// we will use the function emplace_back for the vectors -> it creates and inserts a new element at the end
		key_points.emplace_back(x, y);
	}
}

// Read points stored in the text files
vector<Point2f> readPoints(string pointsFileName)
{
	vector<Point2f> points;
	ifstream ifs(pointsFileName);
	float x, y;
	while (ifs >> x >> y)
	{
		points.push_back(Point2f(x, y));
	}

	return points;
}

void draw_point(Mat& image, const Point2f& fp, const Scalar& color)
{
	circle(image, fp, 2, color, FILLED, LINE_AA, 0);
}

void draw_delaunayTriungulation(Mat& image, Subdiv2D& subdiv, const Scalar& delaunay_color)
{
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> point(3);
	Size img_size = image.size();
	int width = img_size.width;
	int height = img_size.height;
	Rect rect(0, 0, width, height);
	for (Vec6f t : triangleList) {
		point[0] = Point(cvRound(t[0]), cvRound(t[1]));
		point[1] = Point(cvRound(t[2]), cvRound(t[3]));
		point[2] = Point(cvRound(t[4]), cvRound(t[5]));

		//Draw circles just inside the image 

		if (rect.contains(point[0]) && rect.contains(point[1]) && rect.contains(point[2]))
		{
			line(image, point[0], point[1], delaunay_color, 1, LINE_AA, 0);
			line(image, point[1], point[2], delaunay_color, 1, LINE_AA, 0);
			line(image, point[2], point[0], delaunay_color, 1, LINE_AA, 0);
		}
	}
}

Subdiv2D get_delaunayTriangulation(Mat& image, vector<Point2f>& points)
{
	Size img_size = image.size();
	int width = img_size.width;
	int height = img_size.height;
	Rect rect(0, 0, width, height);
	Subdiv2D subdiv(rect);
	for (Point2f& point : points)
	{
		subdiv.insert(point);
	}
	return subdiv;
}

//affine calculated using source triangle and destination triangle tothe src
void affine_transform(Mat& warpImage, Mat& src, vector<Point2f>& srcT, vector<Point2f>& dstT)
{
	//find the affine transform, having a pair of triangles
	Mat warpMat = getAffineTransform(srcT, dstT);
	//apply the affine transform on the source image 
	// we will use: inter_linear -> biliniar interpolation 
	//              border_reflect -> border will be mirror reflection of the border elements 
	warpAffine(src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

// Warps and alpha blends triangular regions from img1 and img2 to img
void morphTriagle(Mat &image1, Mat &image2, Mat &image, vector<Point2f>  &t1, vector<Point2f> &t2, vector<Point2f> &t, float alpha)
{
	//find the bouncing rectangle for each triangle 
	Rect r = boundingRect(t);
	Rect r1 = boundingRect(t1);
	Rect r2 = boundingRect(t2);

	//offset points by left top corner of the respective rectangles 
	vector<Point2f> t1Rect, t2Rect, tRect;
	vector<Point> tRectInt;
	for (int i = 0; i < 3; i++) {
		tRect.emplace_back(t[i].x - r.x, t[i].y - r.y);
		tRectInt.emplace_back(t[i].x - r.x, t[i].y - r.y); // for fillConvexPoly
		t1Rect.emplace_back(t1[i].x - r1.x, t1[i].y - r1.y);
		t2Rect.emplace_back(t2[i].x - r2.x, t2[i].y - r2.y);
	}
	// Get mask by filling triangle
	Mat mask = Mat::zeros(r.height, r.width, CV_32FC3);
	fillConvexPoly(mask, tRectInt, Scalar(1.0, 1.0, 1.0), 16, 0);
	// Apply warpImage to small rectangular patches
	Mat img1Rect, img2Rect;
	image1(r1).copyTo(img1Rect);
	image2(r2).copyTo(img2Rect);

	Mat warpImage1 = Mat::zeros(r.height, r.width, img1Rect.type());
	Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());

	affine_transform(warpImage1, img1Rect, t1Rect, tRect);
	affine_transform(warpImage2, img2Rect, t2Rect, tRect);
	// Alpha blend rectangular patches
	Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;

	// Copy triangular region of the rectangular patch to the output image
	multiply(imgRect, mask, imgRect);
	multiply(image(r), Scalar(1.0, 1.0, 1.0) - mask, image(r));
	image(r) = image(r) + imgRect;
}


int main(int argc, char* argv[]) {

	printf("ciao\n");
  
	string filename1("Hilary.jpg");
	string filename2("ted.jpg");

	printf("hello\n");

	//alpha controls the degree of morph
	double alpha = 0.7;

	//Read input images
	Mat img1 = imread(filename1);
	Mat img2 = imread(filename2);

	printf("aici");

	//convert Mat to float data type
	img1.convertTo(img1, CV_32F);
	img2.convertTo(img2, CV_32F);

	printf("converted\n");


	//empty average image
	Mat imgMorph = Mat::zeros(img1.size(), CV_32FC3);


	//Read points
	vector<Point2f> points1 = readPoints("Hilary.txt");
	vector<Point2f> points2 = readPoints("ted.txt");
	vector<Point2f> points;


	auto subdiv1 = get_delaunayTriangulation(img1, points1);
	auto subdiv2 = get_delaunayTriangulation(img2, points2);

	Mat img1_copy, img2_copy;
	img1.copyTo(img1_copy);
	img2.copyTo(img2_copy);
	Scalar color(255, 255, 255);
	draw_delaunayTriungulation(img1_copy, subdiv1, color);
	draw_delaunayTriungulation(img2_copy, subdiv2, color);
	imwrite("HilaryT.jpg", img1_copy);
	imwrite("tedT.jpg", img2_copy);


	//compute weighted average point coordinates
	for (int i = 0; i < points1.size(); i++)
	{
		float x, y;
		x = (1 - alpha) * points1[i].x + alpha * points2[i].x;
		y = (1 - alpha) * points1[i].y + alpha * points2[i].y;

		points.push_back(Point2f(x, y));

	}

	printf("step\n");

	vector<Vec6f> triangles1, triangles2;
	subdiv1.getTriangleList(triangles1);
	auto len = triangles1.size();

	vector<vector<int>> index;
	Size size = img1.size();
	for (int i = 0; i < len; ++i) {
		vector<pair<Point2f, int>> cur;
		for (int j = 0; j <= 4; j += 2) {
			cur.emplace_back(Point2f(triangles1[i][j], triangles1[i][j + 1]), 0);
			for (auto k = 0; k < points1.size(); ++k) {
				if (points1[k] == cur.back().first) {
					cur[cur.size() - 1].second = k;
					break;
				}
			}
		}
		Rect rect(0, 0, size.width, size.height);
		if (rect.contains(cur[0].first) && rect.contains(cur[1].first) && rect.contains(cur[2].first)) {
			index.emplace_back(vector<int>({ cur[0].second, cur[1].second, cur[2].second }));
		}
	}



	//Read triangle indices
	//ifstream ifs(index);
	//int x, y, z;

	/*while (ifs >> x >> y >> z)*/
	for (const auto& i : index) 
	{
		// Triangles
		vector<Point2f> t1, t2, t;

		// Triangle corners for image 1.
		t1.push_back(points1[i[0]]);
		t1.push_back(points1[i[1]]);
		t1.push_back(points1[i[2]]);

		// Triangle corners for image 2.
		t2.push_back(points2[i[0]]);
		t2.push_back(points2[i[1]]);
		t2.push_back(points2[i[2]]);
			
		// Triangle corners for morphed image.
		t.push_back(points[i[0]]);
		t.push_back(points[i[1]]);
		t.push_back(points[i[2]]);

		printf("while\n");
		morphTriagle(img1, img2, imgMorph, t1, t2, t, alpha);

	}

	printf("end\n");

	// Display Result
	imshow("Morphed Face", imgMorph / 255.0);
	waitKey(0);

	return 0;
}

//struct Conf {
//    cv::String model_path;
//    double scaleFactor;
//    Conf(cv::String s, double d) {
//        model_path = s;
//        scaleFactor = d;
//        face_detector.load(model_path);
//    };
//    CascadeClassifier face_detector;
//};
//bool myDetector(InputArray image, OutputArray faces, Conf* conf) {
//    Mat gray;
//    if (image.channels() > 1)
//        cvtColor(image, gray, COLOR_BGR2GRAY);
//    else
//        gray = image.getMat().clone();
//    equalizeHist(gray, gray);
//    std::vector<Rect> faces_;
//    conf->face_detector.detectMultiScale(gray, faces_, conf->scaleFactor, 2, CASCADE_SCALE_IMAGE, Size(30, 30));
//    Mat(faces_).copyTo(faces);
//    return true;
//}
//
//int main()
//{
//    // CREATE FACEMARK INSTANCE
//    //FacemarkLBF::Params params;
//    //params.model_filename = "helen.model"; // the trained model will be saved using this filename
//    //Ptr<Facemark> facemark = FacemarkLBF::create(params);
//    //Conf config("lbpcascade_frontalface.xml", 1.4);
//    //facemark->setFaceDetector(myDetector, &config); // we must guarantee proper lifetime of "config" object
//}