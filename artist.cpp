/*
Artist is a program to create computer-generated abstract paintings from photographs
Author: Tania Papandrea taniapapandrea@gmail.com
Final Project for BU CS585:Computer Vision
*/


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

//Reduces an input image into triangles on a set grid. 
//Input: an input image, output image, and user-specified number of triangles
//Resulting dst: an image of the same size filled with triangles with colors mimicking the original image
void blindTriangulation(Mat& src, Mat& dst, int threshold);

//Helper function for blindTriangulation: breaks a rectangle into many triangles
//Input: height and width of image, and user-specified number of triangles
//Output: vector of vectors containing vertices for all newly-created triangles
vector < vector < Point > > findTriangles(int rows, int columns, int n);

//Draws triangles to mimic an input image based on density of edges
//Input: an input image, output image, and edge map produced by an edge detector. Threshold is unused.
//Output: an image of the same size filled with triangles with colors mimicking the original image
void edgeDetection(Mat& src, Mat& edges, Mat& dst, bool sort, int smallest);

//Helper function for edgeDetection: given a category map that classifies every pixel to a value between 0 and n, this
//method partitions the image into clusters of neighboring pixels. A neighboring pixel is a pixel to the North, South, East
//or West of the starting pixel, which shares the same value as the starting pixel on the category map.
//Input: a 1-channel Mat that contains a value between 0 and n for every pixel, and int n.
//Output: a vector of vectors containing binary 1-channel Mats - the outer vector determines the category (0...n) and the inner vector
//	stores separate binary images for each cluster
vector < vector < Mat > > findClusters(Mat& d, int categories);

//Helper function for edgeDetection: forms a bounding triangle around a cluster of pixels
//Input: a binary 1-channel Mat containing one cluster
//Output: a vector of three points for the vertices of the bounding triangle
vector < Point > boundingTriangle(Mat& img);

//Helper function for edgeDetection and colorDetection: creates an image of triangles from a category map by clustering, bounding each cluster,
//and drawing triangles in dst with appropriate colors
//Input: an input image, output image, category map (1-channel Mat that contains a value between 0 and n for every pixel), and int n
//Resulting dst: an image filled with triangles
void clusterBoundAndColor(Mat& src, Mat& dst, Mat& categories, int numCategories, bool sort, int smallest);

//Helper function for clusterBoundAndColor: counts the number of white pixels in a binary image
//Input: a binary image
//Output: an integer
int numPixels(Mat& img);

//Draws triangles to mimic an input image based on color
//Input: an input image and output image. Threshold is unused.
//Output: an image of the same size filled with triangles with colors mimicking the original image
void colorDetection(Mat& src, Mat& dst, int threshold, bool sort, int smallest);

//Helper function for colorDetections: finds the absolute difference between two (r,g,b) values
//Input: two Vec3b colors
//Output: an integer value for the channel-wise difference between the two colors
int vecDiff(Vec3b one, Vec3b two);

//stores remainder information for blindTriangulation
int row_remainder=0;
int col_remainder=0;

int main()
{
	//allow user to choose the image
	Mat image;
	int imgNum;
	cout << "Select an image:\n1.Low Contrast\n2.High Contrast\n3.Low Saturation\n4.High Saturation\n5.Simple\n6.Complex\n";
	cin >> imgNum;
	if (imgNum==1){
		image = imread("boston.jpg", CV_LOAD_IMAGE_COLOR);
	} else if (imgNum==2){
		image = imread("high_contrast.jpg", CV_LOAD_IMAGE_COLOR);
	} else if (imgNum==3){
		image = imread("low_saturation.jpg", CV_LOAD_IMAGE_COLOR);
	} else if (imgNum==4){
		image = imread("high_saturation.jpg", CV_LOAD_IMAGE_COLOR);
	} else if (imgNum==5){
		image = imread("simple.jpg", CV_LOAD_IMAGE_COLOR);
	} else if (imgNum==6){
		image = imread("complex.jpg", CV_LOAD_IMAGE_COLOR);
	}
    namedWindow( "Original Image", 1 );
    imshow("Original Image", image);

    //allow user to choose the method
	int method;
	cout << "Methods:\n1. Blind Triangulation\n2. Edge Density\n3. Greedy Color\n";
	cout << "Choose a method by entering its number: ";
	cin >> method;

    if (method==1) {
		/*numTriangles is not the exact number of triangles, but the number of vertical squares used to make the triangles. 
		The number of horizontal squares will be determined by the image width, and each square will become 4 triangles. */
		int numTriangles;
		cout << "\n\nHow many rows of triangles? Enter a whole number greater than 0:\n";
		cin >> numTriangles;
	
		//Create an abstract image from the original using Blind Triangulation
		Mat bt_image = Mat::zeros(image.rows, image.cols, CV_LOAD_IMAGE_COLOR);
		bt_image = image.clone();
		blindTriangulation(image, bt_image, numTriangles);
		namedWindow("Blind Triangulation", 1);
		imshow("Blind Triangulation", bt_image);
		//Write new image to file
		imwrite("bt_image.jpg", bt_image);

	} else if (method==2) {
		bool sort=false;
		int sort_;
		int smallest;
		cout << "\n\nHow many pixels can the smallest triangle be? Enter a whole number greater than 0:\n";
		cin >> smallest;
		cout << "\n\nChoose an option:\n1.Mixed order\n2.Smaller triangles in front\n";
		cin >> sort_;
		if (sort_==2){
			sort = true;
		}
		//Create an abstract image from the original using Edge Detections
		Mat ed_image = Mat::zeros(image.rows, image.cols, CV_LOAD_IMAGE_COLOR);
		ed_image = image.clone();
		Mat edges = Mat::zeros(image.rows, image.cols, CV_8UC1);
		edgeDetection(image, edges, ed_image, sort, smallest);
		imshow("Edge Density", ed_image);
		//write the new image to file
		imwrite("ed_image.jpg", ed_image);

	} else if (method==3) {
		/*threshold determines pixel color uniqueness
		Higher values mean fewer colors, lower values mean more colors*/
		int threshold;
		cout << "\n\nHow many colors? Smaller numbers will produce more colors. Enter a whole number between 20 and 200:";
		cin >> threshold;
		bool sort=false;
		int sort_;
		int smallest;
		cout << "\n\nHow many pixels can the smallest triangle be? Enter a whole number greater than 0:\n";
		cin >> smallest;
		cout << "\n\nChoose an option:\n1.Mixed order\n2.Smaller triangles in front\n";
		cin >> sort_;
		if (sort_==2){
			sort = true;
		}
		//Create an abstract image from the original using Edge Detections
		Mat cd_image = Mat::zeros(image.rows, image.cols, CV_LOAD_IMAGE_COLOR);
		cd_image = image.clone();
		colorDetection(image, cd_image, threshold, sort, smallest);
		imshow("Greedy Color Algorithm", cd_image);
		//write the new image to file
		imwrite("cd_image.jpg", cd_image);
	}

	char key = waitKey(0);
    cout<<"You pressed a key."<<endl;
    return 0;
}

void blindTriangulation(Mat& src, Mat& dst, int n) {
	int rows = src.rows;
	int columns = src.cols;
	//random colors for testing purposes only
	vector < vector <int> > rand_colors;
    for (int i=0; i<70; i++){
        vector <int> color;
        color.push_back(rand()%255);
        color.push_back(rand()%255);
        color.push_back(rand()%255);
        rand_colors.push_back(color);
    }
	//create the triangles, denoted by their vertices
	std::vector < vector < Point > > triangles = findTriangles(src.rows, src.cols, n);
	int size = triangles.size();
	//iterate through all triangles
	for (int i=0; i<size; i++) {
		vector < Point > vertices = triangles[i];
		//estimate average color of triangle using its 3 vertices
		int r_value=0;
		int g_value=0;
		int b_value=0;
		for (int j=0; j<3; j++) {
			Vec3b vertexColor = src.at<Vec3b>(vertices[j].y, vertices[j].x);
			r_value += vertexColor[0];
			g_value += vertexColor[1];
			b_value += vertexColor[2];
		}
		r_value /= 3;
		g_value /= 3;
		b_value /= 3;
		//fill triangle in the output image
		//fillConvexPoly(dst, vertices, Scalar(rand_colors[i%70][0], rand_colors[i%70][1], rand_colors[i%70][2]), 8, 0);
		fillConvexPoly(dst, vertices, Scalar(r_value, g_value, b_value), 8, 0);
	}
	//fill any remaining space with white rectangles
	//horizontal rectangle at bottom
	if (row_remainder>0){
		Point br = Point(columns, rows);
		Point tr = Point(columns, rows-row_remainder);
		Point bl = Point(0, rows);
		Point tl = Point(0, rows-row_remainder);
		vector < Point > t1;
		t1.push_back(tl);
		t1.push_back(tr);
		t1.push_back(bl);
		vector < Point > t2;
		t2.push_back(tr);
		t2.push_back(br);
		t2.push_back(bl);
		fillConvexPoly(dst, t1, Scalar(255, 255, 255, 255), 8, 0);
		fillConvexPoly(dst, t2, Scalar(255, 255, 255, 255), 8, 0);
	}
	//vertical rectangle at right
	if (col_remainder>0){
		Point br = Point(columns, rows);
		Point tr = Point(columns, 0);
		Point bl = Point(columns-col_remainder, rows);
		Point tl = Point(columns-col_remainder, 0);
		vector < Point > t1;
		t1.push_back(tl);
		t1.push_back(tr);
		t1.push_back(bl);
		vector < Point > t2;
		t2.push_back(tr);
		t2.push_back(br);
		t2.push_back(bl);
		fillConvexPoly(dst, t1, Scalar(255, 255, 255, 255), 8, 0);
		fillConvexPoly(dst, t2, Scalar(255, 255, 255, 255), 8, 0);
	}
}

vector < vector < Point > > findTriangles(int rows, int columns, int n) {
	vector < vector < Point > > triangles;
	int s = rows/n;
	int r = rows - s*(n+1);
	int num_row_boxes = n;
	if (r > 0){
		int extra_boxes = (r/s);
		num_row_boxes += extra_boxes;
	}
	int num_col_boxes = columns/s;
	for (int r=0; r<num_row_boxes; r++){
		for (int c=0; c<num_col_boxes; c++){
			//find top, bottom, right and left coordinates of square
			int top = s*r;
			int bottom = s*(r+1);
			int left = s*c;
			int right = s*(c+1);
			//find center point
			Point center = Point(left+(s/2),top+(s/2));
			//form left triangle
			vector < Point > left_tri;
			left_tri.push_back(Point(left, top));
			left_tri.push_back(center);
			left_tri.push_back(Point(left, bottom));
			triangles.push_back(left_tri);
			//form right triangle
			vector < Point > right_tri;
			right_tri.push_back(Point(right, top));
			right_tri.push_back(center);
			right_tri.push_back(Point(right, bottom));
			triangles.push_back(right_tri);
			//form top triangle
			vector < Point > top_tri;
			top_tri.push_back(Point(left, top));
			top_tri.push_back(center);
			top_tri.push_back(Point(right, top));
			triangles.push_back(top_tri);
			//form bottom triangle
			vector < Point > bottom_tri;
			bottom_tri.push_back(Point(left, bottom));
			bottom_tri.push_back(center);
			bottom_tri.push_back(Point(right, bottom));
			triangles.push_back(bottom_tri);
		}
	}
	//store remainder data
	row_remainder = rows - num_row_boxes*s;
	col_remainder = columns - num_col_boxes*s;
	return triangles;
}

void edgeDetection(Mat& src, Mat& edges, Mat& dst, bool sort, int smallest){
		Mat neighbors_(src.rows, src.cols, CV_8UC1);
		Mat density_greyscale(src.rows, src.cols, CV_8UC1);
		Mat density_categories(src.rows, src.cols, CV_8UC1);

		//threshold for how many pixels is considered "closeby"
		int c_thresh = 6;
		/*Documentation for OpenCV Canny function:
		http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny */
		Canny(src, edges, 50, 75, 3);
		imshow("Edge Detection", edges);

		//determine density map of the source image
		//maxDensity will help us keep track of the image's highest density
		int maxDensity=0;
		//iterate through all pixels in image
		for (int r=0; r<src.rows; r++){
			for (int c=0; c<src.cols; c++){
				//define a bounding box around this pixel (determined by c_thresh) - 
				//where we will search for neighboring pixels
				int top_bound = r-c_thresh;
				if (top_bound<0){
					top_bound=0;
				}
				int bottom_bound = r+c_thresh;
				if (bottom_bound>src.rows-1){
					bottom_bound=src.rows-1;
				}
				int left_bound = c-c_thresh;
				if (left_bound<0){
					left_bound=0;
				}
				int right_bound = c+c_thresh;
				if (right_bound>src.cols-1){
					right_bound=src.cols-1;
				}
				//count the number of neighboring pixels
				int neighbors=0;
				for (int i=left_bound; i<right_bound+1; i++){
					for (int j=top_bound; j<bottom_bound+1; j++){
						if(edges.at<uchar>(j,i)==255) {
							neighbors +=1;
						}
					}
				}
				//update maximum density
				neighbors_.at<uchar>(r,c)=neighbors;
				if (neighbors>maxDensity){
					maxDensity=neighbors;
				}
			}
		}

		//create four density-threshold groups based on the maximum density in the image
		int thresh=maxDensity/4;
		//iterate through each pixel
		for (int r=0; r<src.rows; r++){
			for (int c=0; c<src.cols; c++){
				//assign each pixel to the appropriate group based on its density
				if (neighbors_.at<uchar>(r,c)>(thresh*3)){
					density_greyscale.at<uchar>(r,c)=255;
					density_categories.at<uchar>(r,c)=3;
				} else if (neighbors_.at<uchar>(r,c)>(thresh*2)){
					density_greyscale.at<uchar>(r,c)=150;
					density_categories.at<uchar>(r,c)=2;
				} else if (neighbors_.at<uchar>(r,c)>(thresh)){
					density_greyscale.at<uchar>(r,c)=75;
					density_categories.at<uchar>(r,c)=1;
				}
			}
		}
		imshow("Edge-Based Density", density_greyscale);

		//draw the final result based on the edge-based densities
		int numCategories=4;
		clusterBoundAndColor(src, dst, density_categories, numCategories, sort, smallest);
}

vector < vector < Mat > > findClusters(Mat& d, int categories){
	Mat d_ = d.clone();
	vector < vector < Mat > > all_clusters;
	//for each category
	for (int i=0; i<categories; i++){
		vector < Mat > i_clusters;
		//iterate through all points
		for (int r=0; r<d.rows; r++){
			for (int c=0; c<d.cols; c++){
				//if the category is right
				if (d_.at<uchar>(r,c)==i){
					//depth-first search to find all its touching pixels
					Mat new_cluster = Mat::zeros(d_.rows+2,d_.cols+2, CV_8UC1);
					/*OpenCV Documentation for floodFill function:
					http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html*/
					int filling = floodFill(d_, new_cluster, Point(c,r), Scalar(255), 0, Scalar(0), Scalar(0), 4 | (255 << 8));
					i_clusters.push_back(new_cluster);
				}
			}
		}
		all_clusters.push_back(i_clusters);
	}
	return all_clusters;
}

vector < Point > boundingTriangle(Mat& img){
	//collect all pixels into a vector
	vector < Point2f > triangle;
	vector < Point > pts;
	for (int r=0; r<img.rows; r++){
		for (int c=0; c<img.cols; c++){
			if (img.at<uchar>(r,c)==255){
				pts.push_back(Point(c,r));
			}
		}
	}
	/*Documentation for OpenCV minEnclosingTriangle function:
	http://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#minenclosingtriangle*/
	minEnclosingTriangle(pts, triangle);
	//convert triangle vertices into integers
	vector < Point > vertices;
	vertices.push_back(Point((int)triangle[0].x, (int)triangle[0].y));
	vertices.push_back(Point((int)triangle[1].x, (int)triangle[1].y));
	vertices.push_back(Point((int)triangle[2].x, (int)triangle[2].y));
	return vertices;
}

void clusterBoundAndColor(Mat& src, Mat& dst, Mat& categories, int numCategories, bool sort, int smallest){
	//random colors for testing purposes only
	vector < vector <int> > rand_colors;
	for (int i=0; i<70; i++){
    	vector <int> color;
        color.push_back(rand()%255);
        color.push_back(rand()%255);
        color.push_back(rand()%255);
        rand_colors.push_back(color);
    }
    //sample random points to obtain estimates for each category's average color,
    //as well as the image's average color
    int image_r=0; 
 	int image_g=0;
  	int image_b =0;
    for (int i=0; i<51; i++){
     	int x = rand()%(src.cols);
     	int y = rand()%(src.rows);
	   	image_r += src.at<Vec3b>(y,x)[0];
     	image_g += src.at<Vec3b>(y,x)[1];
     	image_b += src.at<Vec3b>(y,x)[2];
     }
    image_r/= 50; 
 	image_g/= 50;
   	image_b/= 50;
    //fill background
    vector < Point > lt;
    vector < Point > rt;
    lt.push_back(Point(0,0));
    lt.push_back(Point(src.cols-1, 0));
    lt.push_back(Point(0, src.rows-1));
    rt.push_back(Point(src.cols-1, src.rows-1));
    rt.push_back(Point(src.cols-1, 0));
    rt.push_back(Point(0, src.rows-1));
    fillConvexPoly(dst, lt, Scalar(image_r, image_b, image_g), 8, 0);
    fillConvexPoly(dst, rt, Scalar(image_r, image_b, image_g), 8, 0);
    //divide the pixels of each group into clusters of touching pixels
    vector < vector < Mat > > clusters = findClusters(categories, numCategories);
    //put all clusters into one vector
    vector < Mat > all_clusters;
    vector < int > cluster_sizes;
    for (int i=0; i<numCategories; i++){
    	for (int j=0; j<clusters[i].size(); j++){
    		Mat current = clusters[i][j];
     		int size = numPixels(current);
     		if (size>smallest){
     			all_clusters.push_back(current);
     			cluster_sizes.push_back(size);
     		}
     	}
    }
    //sort clusters by size
    if(sort){
    	//fill a new array with clusters in order of decreasing size
    	vector < Mat > ordered_clusters;
    	for (int i=0; i<all_clusters.size(); i++){
 	   		int max=0;
 	   		int max_index=0;
 	   		//locate the largest remaining cluster
 	   		for (int j=0; j<cluster_sizes.size(); j++){
 	   			if (cluster_sizes[j]>max){
 	   				max = cluster_sizes[j];
 	   				max_index=j;
 	   			}
 	   		}
 	   		//add it to our ordered array
 	   		ordered_clusters.push_back(all_clusters[max_index]);
    		//remove it from the list of remaining clusters
    		cluster_sizes[max_index]=0;
    	}
    	all_clusters=ordered_clusters;
    }
    //for each group
    for (int i=0; i<all_clusters.size(); i++){
    	//iterate through each cluster
    	Mat current = all_clusters[i];
    	//obtain vertices for bounding triangle
    	vector < Point > vertices = boundingTriangle(current);
    	//keep all triangle vertices in bounds
		if (vertices[0].x<0) {
			vertices[0].x=0;
		} else if (vertices[0].x>src.cols-1){
			vertices[0].x=src.cols-1;
		}
		if (vertices[0].y<0) {
			vertices[0].y=0;
		} else if (vertices[0].y>src.rows-1){
			vertices[0].y=src.rows-1;
		}
		if (vertices[1].x<0) {
			vertices[1].x=0;
		} else if (vertices[1].x>src.cols-1){
			vertices[1].x=src.cols-1;
		}
		if (vertices[1].y<0) {
			vertices[1].y=0;
		} else if (vertices[1].y>src.rows-1){
			vertices[1].y=src.rows-1;
		}
		if (vertices[2].x<0) {
			vertices[2].x=0;
		} else if (vertices[2].x>src.cols-1){
			vertices[2].x=src.cols-1;
		}
		if (vertices[2].y<0) {
			vertices[2].y=0;
		} else if (vertices[2].y>src.rows-1){
			vertices[2].y=src.rows-1;
		}
		//estimate average color of triangle using its 3 vertices
		int r_value=0;
		int g_value=0;
		int b_value=0;
		for (int j=0; j<3; j++) {
			Vec3b vertexColor = src.at<Vec3b>(vertices[j].y, vertices[j].x);
			r_value += vertexColor[0];
			g_value += vertexColor[1];
			b_value += vertexColor[2];
		}
		r_value /= 3;
		g_value /= 3;
		b_value /= 3;
    	//fill triangle
    	fillConvexPoly(dst, vertices, Scalar(r_value, g_value, b_value), 8, 0);
    	//fillConvexPoly(dst, vertices, Scalar(rand_colors[i%70][0], rand_colors[i%70][1], rand_colors[i%70][2]), 8, 0);
    }
}

int numPixels(Mat& img){
	//count the number of pixels whose value is 255
	int count=0;
	for (int r=0; r<img.rows; r++){
		for (int c=0; c<img.cols; c++){
			if ((int)img.at<uchar>(r,c)==255){
				count++;
			}
		}
	}
	return count;
}

void colorDetection(Mat& src, Mat& dst, int threshold, bool sort, int smallest){
	//create an array of unique colors that appear in the image
	vector < vector < int > > colors;
	for (int r=0; r<src.rows; r++){
		for (int c=0; c<src.cols; c++){
			//the color that we will compare with all existing colors
			Vec3b current = src.at<Vec3b>(r,c);
			int current_r = current[0];
			int current_g = current[1];
			int current_b = current[2];
			bool unique = true;
			//check if the current color is unique enough from the rest
			for (int i=0; i<colors.size(); i++){
				int tmp_r = colors[i][0];
				int tmp_g = colors[i][1];
				int tmp_b = colors[i][2];
				Vec3b tmp = Vec3b(tmp_r, tmp_g, tmp_b);
				if (vecDiff(tmp, current) < threshold){
					unique=false;
					break;
			 	}
			}
			//add the new color to our list
			if (unique){
				vector < int > new_color;
				new_color.push_back(current_r);
				new_color.push_back(current_g);
				new_color.push_back(current_b);
				colors.push_back(new_color);
			}
		}
	}
	//create a category map that assigns every pixel to a color group
	Mat color_categories(src.rows, src.cols, CV_8UC1);
	Mat color_grayscale(src.rows, src.cols, CV_8UC1);
	int numColors = colors.size();
	int grayscale_step = 255/numColors;
	//iterate through each pixel
	for (int r=0; r<src.rows; r++){
		for (int c=0; c<src.cols; c++){
			//the color that we are matching
			Vec3b current = src.at<Vec3b>(r,c);
			int current_r = current[0];
			int current_g = current[1];
			int current_b = current[2];
			//find the color it matches with
			for (int i=0; i<colors.size(); i++){
				int tmp_r = colors[i][0];
				int tmp_g = colors[i][1];
				int tmp_b = colors[i][2];
				Vec3b tmp = Vec3b(tmp_r, tmp_g, tmp_b);
				if (vecDiff(tmp, current) < threshold){
					color_grayscale.at<uchar>(r,c)=(grayscale_step*i);
					color_categories.at<uchar>(r,c)=i;
					break;
			 	}
			}
		}
	}
	//imshow("Color Groups", color_grayscale);

	//draw the final result based on the edge-based densities
	clusterBoundAndColor(src, dst, color_categories, numColors, sort, smallest);
}

int vecDiff(Vec3b one, Vec3b two){
	int diff=0;
	//for each color channel, add the absolute difference
	for (int i=0; i<3; i++){
		diff+=abs(one[i]-two[i]);
	}
	return diff;
}

