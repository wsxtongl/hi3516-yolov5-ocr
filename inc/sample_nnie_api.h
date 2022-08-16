
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif

int yolo_init(const char *yolo_model_path);
int yolo_run(const char *rgb, int input_w, int input_h);
int yolo_unit();
int yolov3_inference(cv::Mat& imgRGB);
#ifdef __cplusplus
}
#endif //__cplusplus
