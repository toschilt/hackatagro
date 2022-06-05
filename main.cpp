#include <iostream>
#include <vector>
#include <string>
#include <fstream> 

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int debug = 0;

/*
    Obtém o espectro de cor.
    type = 0 -> BGR
    type = 1 -> HSV
*/
std::vector<cv::Mat> ColorHistogram(cv::Mat input_image, int type, std::string window_name)
{
    std::vector<cv::Mat> color_planes;

    switch(type)
    {
        case 0:
            cv::split(input_image, color_planes);
            break;

        case 1:
            cv::Mat hsv_image;
            cv::cvtColor(input_image, hsv_image, cv::COLOR_BGR2HSV);
            cv::split(hsv_image, color_planes);
            break;
    }

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;

    cv::Mat blue_plane, green_plane, red_plane;
    cv::calcHist(&color_planes[0], 1, 0, cv::Mat(), blue_plane, 1, &histSize, histRange, uniform, accumulate);
    cv::calcHist(&color_planes[1], 1, 0, cv::Mat(), green_plane, 1, &histSize, histRange, uniform, accumulate);
    cv::calcHist(&color_planes[2], 1, 0, cv::Mat(), red_plane, 1, &histSize, histRange, uniform, accumulate);

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double) hist_w/histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0,0,0));
    cv::normalize(blue_plane, blue_plane, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(green_plane, green_plane, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(red_plane, red_plane, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    for(int i = 1; i < histSize; i++)
    {
        line(histImage, cv::Point(bin_w*(i-1), hist_h - cvRound(blue_plane.at<float>(i-1))),
              cv::Point(bin_w*(i), hist_h - cvRound(blue_plane.at<float>(i))),
              cv::Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, cv::Point(bin_w*(i-1), hist_h - cvRound(green_plane.at<float>(i-1))),
              cv::Point(bin_w*(i), hist_h - cvRound(green_plane.at<float>(i))),
              cv::Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, cv::Point(bin_w*(i-1), hist_h - cvRound(red_plane.at<float>(i-1))),
              cv::Point(bin_w*(i), hist_h - cvRound(red_plane.at<float>(i))),
              cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    cv::imshow(window_name, histImage);

    std::vector<cv::Mat> histograms;
    histograms.push_back(blue_plane);
    histograms.push_back(green_plane);
    histograms.push_back(red_plane);

    return histograms;
}

/*
    Do BGR and HSV historgrams.
*/
std::vector<float> BGRAndHSVHistograms(cv::Mat input_image)
{
    /* BGR HISTOGRAM CONSTRUCTION */
    std::vector<cv::Mat> bgr_histograms;
    bgr_histograms = ColorHistogram(input_image, 0, "bgr_hist_image");
    cv::Mat b_hist = bgr_histograms[0];
    cv::Mat g_hist = bgr_histograms[1];
    cv::Mat r_hist = bgr_histograms[2];

    std::cout << "BGR HISTOGRAM" << std::endl;
    std::vector<float> bgr_hist_average;
    for(int i = 0; i < bgr_histograms.size(); i++)
    {
        float sum = 0;
        for(int j = 0; j < bgr_histograms[i].rows; j++)
        {
            sum += bgr_histograms[i].at<float>(j, 0);
        }
        sum = sum/bgr_histograms[i].rows;
        bgr_hist_average.push_back(sum);
        std::cout << sum << std::endl;
    }

     /* BGR HISTOGRAM CONSTRUCTION */
    std::vector<cv::Mat> hsv_histograms;
    hsv_histograms = ColorHistogram(input_image, 1, "hsv_hist_image");
    cv::Mat h_hist = hsv_histograms[0];
    cv::Mat s_hist = hsv_histograms[1];
    cv::Mat v_hist = hsv_histograms[2];

    std::cout << "HSV HISTOGRAM" << std::endl;
    std::vector<float> hsv_hist_average;
    for(int i = 0; i < hsv_histograms.size(); i++)
    {
        float sum = 0;
        for(int j = 0; j < hsv_histograms[i].rows; j++)
        {
            sum += hsv_histograms[i].at<float>(j, 0);
        }
        sum = sum/hsv_histograms[i].rows;
        hsv_hist_average.push_back(sum);
        std::cout << sum << std::endl;
    }

    std::vector<float> histograms;
    histograms.insert(histograms.end(), bgr_hist_average.begin(), bgr_hist_average.end());
    histograms.insert(histograms.end(), hsv_hist_average.begin(), hsv_hist_average.end());
    
    return histograms;
}

/*
    Do object tracking in the image.
*/
std::vector<int> tracking(cv::Mat &outputImage,
                          cv::Mat &binaryImage,
                          int &minAreaThreshold,
                          int &maxAreaThreshold)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(binaryImage, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    std::vector<cv::Moments> m(contours.size());
    for(int i = 0; i < contours.size(); i++)
    {
        m[i] = cv::moments(contours[i]);
    }

    std::vector<cv::Point2f> centroids(contours.size());
    for(int i = 0; i < contours.size(); i++)
    {
        float pos_x = m[i].m10/m[i].m00;
        float pos_y = m[i].m01/m[i].m00;
        centroids[i] = cv::Point2f(pos_x, pos_y);
    }

    for(int i = 0; i < contours.size(); i++)
    {
        if(cv::contourArea(contours[i]) > minAreaThreshold &&
           cv::contourArea(contours[i]) < maxAreaThreshold)
        {
            cv::drawContours(outputImage, contours, i, cv::Scalar(0, 0, 255), 1, 8, hierarchy, 0, cv::Point());
            cv::circle(outputImage, centroids[i], 1, cv::Scalar(0, 255, 0), -1);
        }
    }

    std::vector<int> output_info;
    output_info.push_back(contours.size());
    return output_info;
}

/*
    Creates all OpenCV trackbars for testing and visualization.
*/
void trackbarSetup(std::vector<int*> gaussian_parameters,
                   std::vector<int*> trackbar_parameters,
                   std::vector<int*> erode_parameters,
                   std::vector<int*> dilate_parameters,
                   std::vector<int*> contours_parameters)
{
    /* GAUSSIAN TRACKBAR */
    cv::createTrackbar("GAUSSIAN_SIZE", "gaussian_control_image", gaussian_parameters[0], 20);
    cv::setTrackbarMin("GAUSSIAN_SIZE", "gaussian_control_image", 0);

    /* HSV TRACKBAR */
    cv::createTrackbar("HSV_H_LOW", "hsv_control_image", trackbar_parameters[0], 255);
    cv::createTrackbar("HSV_S_LOW", "hsv_control_image", trackbar_parameters[1], 255);
    cv::createTrackbar("HSV_V_LOW", "hsv_control_image",  trackbar_parameters[2], 255);
    cv::createTrackbar("HSV_H_HIGH", "hsv_control_image", trackbar_parameters[3], 255);
    cv::createTrackbar("HSV_S_HIGH", "hsv_control_image", trackbar_parameters[4], 255);
    cv::createTrackbar("HSV_V_HIGH", "hsv_control_image", trackbar_parameters[5], 255);

    /* ERODE TRACKBAR */
    cv::createTrackbar("ERODE_SIZE", "erode_control_image", erode_parameters[0], 20);
    cv::setTrackbarMin("ERODE_SIZE", "erode_control_image", 1);

    /* DILATE TRACKBAR */
    cv::createTrackbar("DILATE_SIZE", "dilate_control_image", dilate_parameters[0], 20);
    cv::setTrackbarMin("DILATE_SIZE", "dilate_control_image", 1);

    /* CONTOURS TRACKBAR */
    cv::createTrackbar("MIN_AREA_THRESHOLD", "contours_image", contours_parameters[0], 25);
    cv::createTrackbar("MAX_AREA_THRESHOLD", "contours_image", contours_parameters[1], 1000);
}

/*
    Creates all OpenCV windows for visualization.
*/
void windowSetup(std::vector<std::string> WINDOW_NAMES, 
                 std::vector<int> WINDOW_DISTANCES)
{
    int lines = 0;
    int columns = 0;
    for(int i = 0; i < WINDOW_NAMES.size(); i++)
    {
        if(i != 0 && (i % 7) == 0)
        {
            lines++;
            columns = 0;
        }
        cv::namedWindow(WINDOW_NAMES[i], cv::WINDOW_KEEPRATIO);
        cv::moveWindow(WINDOW_NAMES[i], 
                       WINDOW_DISTANCES[0] + columns*WINDOW_DISTANCES[1],
                       WINDOW_DISTANCES[0] + lines*WINDOW_DISTANCES[2]);
        columns++;
    }
}

int main(int argc, char *argv[])
{
    std::vector<std::string> input_images_filenames_compilation {
        std::string("./lateriteSoil.txt")
    };

    for(int i = 0; i < input_images_filenames_compilation.size(); i++)
    {
        /* GETTING IMAGES FILENAMES */
        std::ifstream filenames_compilation;

        //Getting number of image files
        filenames_compilation.open(input_images_filenames_compilation[i]);
        int number_lines = 0;
        while(!filenames_compilation.eof())
        {
            std::string temp;
            std::getline(filenames_compilation, temp);
            number_lines++;
        }
        filenames_compilation.close();

        //Opening a file to write parameters
        std::size_t pos = input_images_filenames_compilation[i].find("Soil");
        std::ofstream parameters_file(input_images_filenames_compilation[i].substr(0, pos+4) + ".csv");
        parameters_file << input_images_filenames_compilation[i] << std::endl;
        parameters_file << "GAUSSIAN_FILTER, HSV_H_LOW, HSV_S_LOW, HSV_V_LOW, HSV_H_HIGH, HSV_S_HIGH, HSV_V_HIGH, ERODE_SIZE, DILATE_SIZE, MIN_AREA_THRESHOLD, MAX_AREA_THRESHOLD";
        parameters_file << ", BLUE_AVERAGE, GREEN_AVERAGE, RED_AVERAGE, H_AVERAGE, S_AVERAGE, V_AVERAGE" << std::endl;

        filenames_compilation.open(input_images_filenames_compilation[i]);

        int GAUSSIAN_FILTER = 1;
        int HSV_H_LOW = 0, HSV_S_LOW = 0, HSV_V_LOW = 0;
        int HSV_H_HIGH = 0, HSV_S_HIGH = 0, HSV_V_HIGH = 0;
        int ERODE_SIZE = 1;
        int DILATE_SIZE = 1;
        int MIN_AREA_THRESHOLD = 1;
        int MAX_AREA_THRESHOLD = 1;

        for(int j = 0; j < number_lines; j++)
        {
            std::string file_path;
            if(filenames_compilation.is_open())
            {
                std::getline(filenames_compilation, file_path);
            }
            //std::cout << file_path << std::endl;

            /* INPUT IMAGE CONSTRUCTION */
            cv::Mat input_image;
            input_image = cv::imread(file_path);
            int input_image_width = input_image.size().width;
            int input_image_height = input_image.size().height;

            /* GAUSSIAN BLUR CONSTRUCTION */
            cv::Mat gaussian_control_image;
            std::vector<int*> GAUSSIAN_PARAMETERS {&GAUSSIAN_FILTER};

            /* CSV IMAGE CONSTRUCTION */
            cv::Mat hsv_image;
            cv::cvtColor(input_image, hsv_image, cv::COLOR_BGR2HSV);

            /* THRESHOLD IMAGE CONSTRUCTION */
            cv::Mat hsv_control_image;
            std::vector<int*> HSV_PARAMETERS {&HSV_H_LOW, &HSV_S_LOW, &HSV_V_LOW,
                                              &HSV_H_HIGH, &HSV_S_HIGH, &HSV_V_HIGH};

            /* ERODED IMAGE CONSTRUCTION */
            cv::Mat erode_control_image;
            std::vector<int*> ERODE_PARAMETERS {&ERODE_SIZE};

            /* DILATED IMAGE CONSTRUCTION */
            cv::Mat dilate_control_image;
            std::vector<int*> DILATE_PARAMETERS {&DILATE_SIZE};

            /* CONTOURS IMAGE CONSTRUCTION */
            cv::Mat contours_image;
            std::vector<int*> CONTOURS_PARAMETERS {&MIN_AREA_THRESHOLD, &MAX_AREA_THRESHOLD};

            int last_num_contours = 0;

            /* IMAGE VISUALIZATION */
            int xyOffset = 0;
            std::vector<std::string> WINDOW_NAMES {"input_image",
                                                "gaussian_control_image", 
                                                "hsv_image",
                                                "hsv_control_image",
                                                "erode_control_image",
                                                "dilate_control_image",
                                                "contours_image",
                                                "bgr_hist_image",
                                                "hsv_hist_image"};
            std::vector<int> WINDOW_DISTANCES {xyOffset, 
                                            input_image_width,
                                            input_image_height*2};
            windowSetup(WINDOW_NAMES,
                        WINDOW_DISTANCES);
            trackbarSetup(GAUSSIAN_PARAMETERS,
                          HSV_PARAMETERS,
                          ERODE_PARAMETERS,
                          DILATE_PARAMETERS,
                          CONTOURS_PARAMETERS);
            cv::imshow("input_image", input_image);

            std::vector<cv::Mat> hsv_vec;
            cv::split(hsv_image, hsv_vec);
            cv::Mat &H = hsv_vec[0];
            cv::Mat &S = hsv_vec[1];
            cv::Mat &V = hsv_vec[2];
            cv::merge(hsv_vec, hsv_image);
            
            cv::inRange(hsv_image,
                        cv::Scalar(0, 0, 0),
                        cv::Scalar(255, 150, 255),
                        hsv_image);
            cv::imshow("hsv_image", hsv_image);

            /* BGR AND HSV HISTOGRAMS CONSTRUCTION */
            std::vector<float> histograms_average = BGRAndHSVHistograms(input_image);

            int num_contours = 0;
            while(true)
            {
                /* GAUSSIAN BLUR IMAGE UPDATE */
                int ODD_GAUSSIAN_FILTER = GAUSSIAN_FILTER*2 + 1;
                cv::GaussianBlur(input_image, 
                                gaussian_control_image,
                                cv::Size(ODD_GAUSSIAN_FILTER, ODD_GAUSSIAN_FILTER), 0, 0);
                cv::imshow("gaussian_control_image", gaussian_control_image);

                /* THRESHOLD IMAGE UPDATE */
                cv::inRange(gaussian_control_image,
                            cv::Scalar(HSV_H_LOW, HSV_S_LOW, HSV_V_LOW),
                            cv::Scalar(HSV_H_HIGH, HSV_S_HIGH, HSV_V_HIGH),
                            hsv_control_image);
                cv::imshow("hsv_control_image", hsv_control_image);

                /* ERODE IMAGE UPDATE */
                cv::erode(hsv_control_image,
                        erode_control_image,
                        cv::getStructuringElement(cv::MORPH_ELLIPSE, 
                                                    cv::Size(ERODE_SIZE, ERODE_SIZE)));
                cv::imshow("erode_control_image", erode_control_image);
                
                /* DILATE IMAGE UPDATE */
                cv::dilate(erode_control_image,
                        dilate_control_image,
                        cv::getStructuringElement(cv::MORPH_ELLIPSE, 
                                                    cv::Size(DILATE_SIZE, DILATE_SIZE)));
                cv::imshow("dilate_control_image", dilate_control_image);


                /* TRACKING IMAGE UPDATE */
                input_image.copyTo(contours_image);
                std::vector<int> tracking_info = tracking(contours_image, dilate_control_image, MIN_AREA_THRESHOLD, MAX_AREA_THRESHOLD);
                if(tracking_info[0] != last_num_contours)
                {
                    std::cout << "Quantidade de contornos = " << tracking_info[0] << std::endl;
                }
                last_num_contours = tracking_info[0];
                num_contours = tracking_info[0];
                cv::imshow("contours_image", contours_image);

                if(cv::waitKey(20) == 'q')
                    break;
            }

            cv::destroyAllWindows();

            std::size_t pos = file_path.find(".jpg");
            //std::cout<< file_path << std::endl;

            std::string input_image_id = file_path.substr(pos-2, 2);
            //std::cout<< input_image_id << std::endl;

            pos = input_images_filenames_compilation[i].find("Soil");
            std::string output_images_folder("./image_output/" + input_images_filenames_compilation[i].substr(2, pos+2) + "/");

            cv::imwrite(output_images_folder + input_image_id + "_gaussian_control_image" + ".png", gaussian_control_image);
            cv::imwrite(output_images_folder + input_image_id + "_hsv_control_image" + ".png", hsv_control_image);
            cv::imwrite(output_images_folder + input_image_id + "_erode_control_image" + ".png", erode_control_image);
            cv::imwrite(output_images_folder + input_image_id + "_dilate_control_image" + ".png", dilate_control_image);
            cv::imwrite(output_images_folder + input_image_id + "_contours_image" + ".png", contours_image);
            cv::imwrite(output_images_folder + input_image_id + "_hsv_image" + ".png", hsv_image);

            parameters_file << GAUSSIAN_FILTER << ", ";
            parameters_file << HSV_H_LOW << ", ";
            parameters_file << HSV_S_LOW << ", ";
            parameters_file << HSV_V_LOW << ", ";
            parameters_file << HSV_H_HIGH << ", ";
            parameters_file << HSV_S_HIGH << ", ";
            parameters_file << HSV_V_HIGH << ", ";
            parameters_file << HSV_V_HIGH << ", ";
            parameters_file << ERODE_SIZE << ", ";
            parameters_file << DILATE_SIZE << ", ";
            parameters_file << MIN_AREA_THRESHOLD << ", ";
            parameters_file << MAX_AREA_THRESHOLD << ", ";
            parameters_file << num_contours << ", ";
            parameters_file << histograms_average[0] << ", ";
            parameters_file << histograms_average[1] << ", ";
            parameters_file << histograms_average[2] << ", ";
            parameters_file << histograms_average[3] << ", ";
            parameters_file << histograms_average[4] << ", ";
            parameters_file << histograms_average[5];
            parameters_file << std::endl;
        }

        parameters_file.close();
    }
}