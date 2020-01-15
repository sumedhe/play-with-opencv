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

#define KEY_PATH "./data/vid/myvideo.avi"
#define BG_PATH  "./data/vid/background.avi"

using namespace cv;
using namespace std;

// Open Video
cv::VideoCapture open_video(String vid_path) {
    cv::VideoCapture vid(vid_path);
    if (!vid.isOpened()){
        cout << "Could not open or find the video!" << endl;
        throw;
    }
    return vid;
}

int main(int argc, char** argv) {

    // Open vidoes
    cv::VideoCapture vid_key = open_video(KEY_PATH);
    cv::VideoCapture vid_bg  = open_video(BG_PATH);
    
    int frame_height = vid_key.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frame_width  = vid_key.get(cv::CAP_PROP_FRAME_WIDTH);


    cv::VideoWriter video_writer("./output/combined_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frame_width, frame_height));


    while (1) {
        cv::Mat img_key;
        vid_key >> img_key;

        cv::Mat img_bg;
        vid_bg >> img_bg;

        if (img_bg.empty() || img_key.empty()) {
            break;
        }

        // Resize bg frame up to key frame size
        cv::Mat img_bg_resized;
        cv::resize(img_bg, img_bg_resized, img_key.size());

        // Convert key frame to HSV
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

        imshow("combined", combined);
        video_writer.write(combined);

        char c = (char) cv::waitKey(25);
        if (c == 27)
            break;
    }

    vid_key.release();
    vid_bg.release();
    
    destroyAllWindows();

    return 0;
}