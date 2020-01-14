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

#define IMG_1_PATH "./data/0001.jpg"
#define IMG_2_PATH "./data/0199.jpg"

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

// Detect Edges using Canny
cv::Mat detect_edges_canny(cv::Mat img, int low_threshold, int ratio, int kernel_size) {
    // Create empty image
    cv::Mat img_edges;
    img_edges.create(img.size(), img.type());

    // Grayscale
    cv::Mat img_gray;
    cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    // Blur
    cv::Mat detected_edges;
    blur(img_gray, detected_edges, cv::Size(3, 3));

    // Canny
    Canny(detected_edges, detected_edges, low_threshold, low_threshold * ratio, kernel_size);

    // Output
    img_edges = cv::Scalar::all(0);
    img.copyTo(img_edges, detected_edges);

    return detected_edges;
}

int main(int argc, char** argv) {
    cv::Mat img_1 = open_image(IMG_1_PATH);
    cv::Mat img_2 = open_image(IMG_2_PATH);

    cv::Mat img_edges_1 = detect_edges_canny(img_1, 40, 3, 3);
    cv::Mat img_edges_2 = detect_edges_canny(img_2, 40, 3, 3);

    show_image(img_edges_1, "img_edges_1");
    show_image(img_edges_2, "img_edges_2");

    save_image(img_edges_1, "output2/output3.jpg");
    save_image(img_edges_2, "output2/output4.jpg");

    waitKey(0);

    destroyWindow("img_edges_1");
    destroyWindow("img_edges_2");

    return 1;
}