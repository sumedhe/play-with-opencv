#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <stdexcept>
#include <iostream>
#include <string>


using namespace cv;
using namespace std;

#define IMG_1_PATH "./data/Fish/img/0001.jpg"
#define IMG_2_PATH "./data/Fish/img/0199.jpg"

#define CROP_LEFT   134
#define CROP_TOP    55
#define CROP_WIDTH  60
#define CROP_HEIGHT 88

#define MIN_HESSIAN 400


// Open Image
cv::Mat open_image(String image_path) {
    cv::Mat img;
    img = cv::imread(image_path);
    if (img.empty()){
        cout << "Could not open or find the image!" << endl;
        throw;
    }
    return img;
}

// Show Image
void show_image(cv::Mat img, String img_name) {
    cv::namedWindow(img_name);
    cv::imshow(img_name, img);
}

// Save Image
void save_image(cv::Mat img, String filepath) {
    cv::imwrite(filepath, img);
}

// Draw Rectangle
void draw_rectangle(cv::Mat img, int left, int top, int width, int height) {
    cv::rectangle(img, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0));
}

// Crop Image
cv::Mat crop_image(cv::Mat img){
    cv::Rect rect(CROP_LEFT, CROP_TOP, CROP_WIDTH, CROP_HEIGHT);
    return img(rect);
}

// Surf Detector
std::vector<KeyPoint> surf_detection(cv::Mat img) {
	Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(MIN_HESSIAN);
	std::vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);
    return keypoints;
}

// Export keypoints to CSV
void export_keypoints(vector<KeyPoint> keypoints, String filepath) {
    std::ofstream outfile(filepath);
	for (int i = 0; i < keypoints.size(); i++) {
		outfile << keypoints[i].pt.x << "," << keypoints[i].pt.y << std::endl;
	}
}

// Calculate descriptors
cv::Mat calc_descriptors(cv::Mat img, vector<KeyPoint> keypoints) {
    cv::Mat descriptors;
    cv::Ptr<cv::xfeatures2d::SURF> extractor = cv::xfeatures2d::SURF::create();
    extractor->compute(img, keypoints, descriptors);
    return descriptors;
}

// Match Images Bruteforce
vector<cv::DMatch> match_images_bf(cv::Mat descriptors_1, cv::Mat descriptors_2) {
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_L2);
    matcher.match(descriptors_1, descriptors_2, matches);
    return matches;
}

// Match Images FLANN
vector<cv::DMatch> match_images_flann(cv::Mat descriptors_1, cv::Mat descriptors_2) {
	std::vector<cv::DMatch > matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	matcher->match(descriptors_1, descriptors_2, matches);
    return matches;
}


int main(int argc, char** argv) {
    // Load images
    cv::Mat img_1 = open_image(IMG_1_PATH);
    cv::Mat img_2 = open_image(IMG_2_PATH);

    // Draw rectangle
    draw_rectangle(img_1, CROP_LEFT, CROP_TOP, CROP_WIDTH, CROP_HEIGHT);

    // Crop image
    cv::Mat img_1_crop = crop_image(img_1);

	// Surf Feature Detection
    std::vector<KeyPoint> keypoints_1 = surf_detection(img_1);
    std::vector<KeyPoint> keypoints_2 = surf_detection(img_2);

	// Export keypoints to csv
    export_keypoints(keypoints_1, "./output/features0001.csv");
    export_keypoints(keypoints_2, "./output/features0199.csv");

    // Calc descriptors
    cv::Mat descriptors_1 = calc_descriptors(img_1_crop, keypoints_1);
    cv::Mat descriptors_2 = calc_descriptors(img_2, keypoints_2);
    
    // Match images BF
    cv::Mat img_matches_bf;
    vector<cv::DMatch> matches_bf = match_images_bf(descriptors_1, descriptors_2);
    drawMatches(img_1_crop, keypoints_1, img_2, keypoints_2, matches_bf, img_matches_bf);
    save_image(img_matches_bf, "./output/output_matching.jpg");

    // Match images FLANN
    cv::Mat img_matches_flann;
    vector<cv::DMatch> matches_flann = match_images_flann(descriptors_1, descriptors_2);
    drawMatches(img_1_crop, keypoints_1, img_2, keypoints_2, matches_flann, img_matches_flann);
    save_image(img_matches_flann, "./output/output_matching_flann.jpg");


    // Show Images
    String img_1_name = "img_1";
    show_image(img_1, img_1_name);

    String img_2_name = "img_2";
    show_image(img_2, img_2_name);

    String img_1_crop_name = "img_1_crop";
    show_image(img_1_crop, img_1_crop_name);

    String img_matches_name = "img_matches_name";
    show_image(img_matches_bf, img_matches_name);

    String img_matches_flann_name = "img_matches_flann_name";
    show_image(img_matches_flann, img_matches_flann_name);

    waitKey(0);

    destroyWindow(img_1_name);
    destroyWindow(img_2_name);
    destroyWindow(img_1_crop_name);
    destroyWindow(img_matches_name);
    destroyWindow(img_matches_flann_name);

    return 0;


}