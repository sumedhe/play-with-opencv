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

#define KEY_PATH "./data/img/chroma_key.jpg"
#define BG_PATH  "./data/img/background.jpg"

using namespace cv;
using namespace std;

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


int main(int argc, char** argv) {

    // Open images
    cv::Mat img_bg  = open_image(BG_PATH);
    cv::Mat img_key = open_image(KEY_PATH);


    // Resize
    cv::resize(img_bg, img_bg, img_key.size());

    // Convert to HSV
    cv::Mat img_key_hsv;
    cv::cvtColor(img_key, img_key_hsv, COLOR_BGR2HSV);

    int sensitivity = 20;
    cv::Vec3b low_bound(60 - sensitivity, 100, 50); 
    cv::Vec3b upper_bound(60 + sensitivity, 255, 255);

    // Get green color area mask
    cv::Mat mask, mask_inverse;
    cv::inRange(img_key_hsv, low_bound, upper_bound, mask);
    cv::bitwise_not(mask, mask_inverse);

    // Combine
    cv::Mat combined;
    cv::bitwise_and(img_bg, img_bg, combined, mask);
    cv::bitwise_and(img_key, img_key, combined, mask_inverse);

    show_image(combined, "combined");
    save_image(combined, "./output/combined.jpg");

    cv::waitKey(0);

    cv::destroyWindow("combined");

}