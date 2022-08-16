#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <sys/time.h>
#include "sample_nnie_api.h"
#include <unistd.h>


#include <chrono>
#include "paddle_api.h" // NOLINT
#include "paddle_place.h"

#include "cls_process.h"
#include "crnn_process.h"
#include "db_post_process.h"
#include "lite_autolog.h"

using namespace paddle::lite_api; // NOLINT
//using namespace std;



#define INPUT_W 640
#define INPUT_H 640

using namespace std;
using namespace cv;


void NeonMeanScale(const float *din, float *dout, int size,
                   const std::vector<float> mean,
                   const std::vector<float> scale) {
  if (mean.size() != 3 || scale.size() != 3) {
    std::cerr << "[ERROR] mean or scale size must equal to 3" << std::endl;
    exit(1);
  }
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(scale[2]);

  float *dout_c0 = dout;
  float *dout_c1 = dout + size;
  float *dout_c2 = dout + size * 2;

  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(din);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dout_c0, vs0);
    vst1q_f32(dout_c1, vs1);
    vst1q_f32(dout_c2, vs2);

    din += 12;
    dout_c0 += 4;
    dout_c1 += 4;
    dout_c2 += 4;
  }
  for (; i < size; i++) {
    *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
    *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
    *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
  }
}

// resize image to a size multiple of 32 which is required by the network
cv::Mat DetResizeImg(const cv::Mat img, int max_size_len,
                     std::vector<float> &ratio_hw) {
  int w = img.cols;
  int h = img.rows;

  float ratio = 1.f;
  int max_wh = w >= h ? w : h;
  if (max_wh > max_size_len) {
    if (h > w) {
      ratio = static_cast<float>(max_size_len) / static_cast<float>(h);
    } else {
      ratio = static_cast<float>(max_size_len) / static_cast<float>(w);
    }
  }

  int resize_h = static_cast<int>(float(h) * ratio);
  int resize_w = static_cast<int>(float(w) * ratio);
  if (resize_h % 32 == 0)
    resize_h = resize_h;
  else if (resize_h / 32 < 1 + 1e-5)
    resize_h = 32;
  else
    resize_h = (resize_h / 32 - 1) * 32;

  if (resize_w % 32 == 0)
    resize_w = resize_w;
  else if (resize_w / 32 < 1 + 1e-5)
    resize_w = 32;
  else
    resize_w = (resize_w / 32 - 1) * 32;

  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));

  ratio_hw.push_back(static_cast<float>(resize_h) / static_cast<float>(h));
  ratio_hw.push_back(static_cast<float>(resize_w) / static_cast<float>(w));
  return resize_img;
}

cv::Mat RunClsModel(cv::Mat img, std::shared_ptr<PaddlePredictor> predictor_cls,
                    const float thresh = 0.9) {
  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

  cv::Mat srcimg;
  img.copyTo(srcimg);
  cv::Mat crop_img;
  img.copyTo(crop_img);
  cv::Mat resize_img;

  int index = 0;
  float wh_ratio =
      static_cast<float>(crop_img.cols) / static_cast<float>(crop_img.rows);

  resize_img = ClsResizeImg(crop_img);
  resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);

  const float *dimg = reinterpret_cast<const float *>(resize_img.data);

  std::unique_ptr<Tensor> input_tensor0(std::move(predictor_cls->GetInput(0)));
  input_tensor0->Resize({1, 3, resize_img.rows, resize_img.cols});
  auto *data0 = input_tensor0->mutable_data<float>();

  NeonMeanScale(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);
  // Run CLS predictor
  predictor_cls->Run();

  // Get output and run postprocess
  std::unique_ptr<const Tensor> softmax_out(
      std::move(predictor_cls->GetOutput(0)));
  auto *softmax_scores = softmax_out->mutable_data<float>();
  auto softmax_out_shape = softmax_out->shape();
  float score = 0;
  int label = 0;
  for (int i = 0; i < softmax_out_shape[1]; i++) {
    if (softmax_scores[i] > score) {
      score = softmax_scores[i];
      label = i;
    }
  }
  if (label % 2 == 1 && score > thresh) {
    cv::rotate(srcimg, srcimg, 1);
  }
  return srcimg;
}

void RunRecModel(std::vector<std::vector<std::vector<int>>> boxes, cv::Mat img,
                 std::shared_ptr<PaddlePredictor> predictor_crnn,
                 std::vector<std::string> &rec_text,
                 std::vector<float> &rec_text_score,
                 std::vector<std::string> charactor_dict,
                 std::shared_ptr<PaddlePredictor> predictor_cls,
                 int use_direction_classify) {
  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

  cv::Mat srcimg;
  img.copyTo(srcimg);
  cv::Mat crop_img;
  cv::Mat resize_img;

  int index = 0;

  std::vector<double> time_info = {0, 0, 0};
  for (int i = boxes.size() - 1; i >= 0; i--) {
    auto preprocess_start = std::chrono::steady_clock::now();
    crop_img = GetRotateCropImage(srcimg, boxes[i]);
    if (use_direction_classify >= 1) {
      crop_img = RunClsModel(crop_img, predictor_cls);
    }
    float wh_ratio =
        static_cast<float>(crop_img.cols) / static_cast<float>(crop_img.rows);

    resize_img = CrnnResizeImg(crop_img, wh_ratio);
    resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);

    const float *dimg = reinterpret_cast<const float *>(resize_img.data);

    std::unique_ptr<Tensor> input_tensor0(
        std::move(predictor_crnn->GetInput(0)));
    input_tensor0->Resize({1, 3, resize_img.rows, resize_img.cols});
    auto *data0 = input_tensor0->mutable_data<float>();

    NeonMeanScale(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);
    auto preprocess_end = std::chrono::steady_clock::now();
    //// Run CRNN predictor
    auto inference_start = std::chrono::steady_clock::now();
    predictor_crnn->Run();

    // Get output and run postprocess
    std::unique_ptr<const Tensor> output_tensor0(
        std::move(predictor_crnn->GetOutput(0)));
    auto *predict_batch = output_tensor0->data<float>();
    auto predict_shape = output_tensor0->shape();
    auto inference_end = std::chrono::steady_clock::now();

    // ctc decode

    //auto postprocess_start = std::chrono::steady_clock::now();

    std::string str_res;
    int argmax_idx;
    int last_index = 0;
    float score = 0.f;
    int count = 0;
    float max_value = 0.0f;

    for (int n = 0; n < predict_shape[1]; n++) {
      argmax_idx = int(Argmax(&predict_batch[n * predict_shape[2]],
                              &predict_batch[(n + 1) * predict_shape[2]]));
      max_value =
          float(*std::max_element(&predict_batch[n * predict_shape[2]],
                                  &predict_batch[(n + 1) * predict_shape[2]]));
      if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
        score += max_value;
        count += 1;
        str_res += charactor_dict[argmax_idx];
        //std::cout << "index:" << argmax_idx << charactor_dict[argmax_idx] << std::endl;
	
      }
      last_index = argmax_idx;
    }
    score /= count;
    rec_text.push_back(str_res);
    rec_text_score.push_back(score);
    
    //auto postprocess_end = std::chrono::steady_clock::now();
  }
}

std::vector<std::vector<std::vector<int>>>
RunDetModel(std::shared_ptr<PaddlePredictor> predictor, cv::Mat img,
            std::map<std::string, double> Config, std::vector<double> *times) {
  // Read img
  int max_side_len = int(Config["max_side_len"]);
  int det_db_use_dilate = int(Config["det_db_use_dilate"]);

  cv::Mat srcimg;
  img.copyTo(srcimg);
  
  auto preprocess_start = std::chrono::steady_clock::now();
  std::vector<float> ratio_hw;
  img = DetResizeImg(img, max_side_len, ratio_hw);
  cv::Mat img_fp;
  img.convertTo(img_fp, CV_32FC3, 1.0 / 255.f);

  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor0(std::move(predictor->GetInput(0)));
  input_tensor0->Resize({1, 3, img_fp.rows, img_fp.cols});
  auto *data0 = input_tensor0->mutable_data<float>();

  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  const float *dimg = reinterpret_cast<const float *>(img_fp.data);
  NeonMeanScale(dimg, data0, img_fp.rows * img_fp.cols, mean, scale);
  auto preprocess_end = std::chrono::steady_clock::now();

  // Run predictor
  auto inference_start = std::chrono::steady_clock::now();
  predictor->Run();

  // Get output and post process
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  auto *outptr = output_tensor->data<float>();
  auto shape_out = output_tensor->shape();
  auto inference_end = std::chrono::steady_clock::now();

  // Save output
  auto postprocess_start = std::chrono::steady_clock::now();
  float pred[shape_out[2] * shape_out[3]];
  unsigned char cbuf[shape_out[2] * shape_out[3]];

  for (int i = 0; i < int(shape_out[2] * shape_out[3]); i++) {
    pred[i] = static_cast<float>(outptr[i]);
    cbuf[i] = static_cast<unsigned char>((outptr[i]) * 255);
  }

  cv::Mat cbuf_map(shape_out[2], shape_out[3], CV_8UC1,
                   reinterpret_cast<unsigned char *>(cbuf));
  cv::Mat pred_map(shape_out[2], shape_out[3], CV_32F,
                   reinterpret_cast<float *>(pred));

  const double threshold = double(Config["det_db_thresh"]) * 255;
  const double max_value = 255;
  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, max_value, cv::THRESH_BINARY);
  if (det_db_use_dilate == 1) {
    cv::Mat dilation_map;
    cv::Mat dila_ele =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(bit_map, dilation_map, dila_ele);
    bit_map = dilation_map;
  }
  auto boxes = BoxesFromBitmap(pred_map, bit_map, Config);

  std::vector<std::vector<std::vector<int>>> filter_boxes =
      FilterTagDetRes(boxes, ratio_hw[0], ratio_hw[1], srcimg);
  auto postprocess_end = std::chrono::steady_clock::now();

  std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
  times->push_back(double(preprocess_diff.count() * 1000));
  std::chrono::duration<float> inference_diff = inference_end - inference_start;
  times->push_back(double(inference_diff.count() * 1000));
  std::chrono::duration<float> postprocess_diff = postprocess_end - postprocess_start;
  times->push_back(double(postprocess_diff.count() * 1000));

  return filter_boxes;
}

std::shared_ptr<PaddlePredictor> loadModel(std::string model_file, int num_threads) {
  MobileConfig config;
  config.set_model_from_file(model_file);

  config.set_threads(num_threads);
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);
  return predictor;
}

cv::Mat Visualization(cv::Mat srcimg,
                      std::vector<std::vector<std::vector<int>>> boxes) {
  cv::Point rook_points[boxes.size()][4];
  for (int n = 0; n < boxes.size(); n++) {
    for (int m = 0; m < boxes[0].size(); m++) {
      rook_points[n][m] = cv::Point(static_cast<int>(boxes[n][m][0]),
                                    static_cast<int>(boxes[n][m][1]));
    }
  }
  cv::Mat img_vis;
  srcimg.copyTo(img_vis);
  for (int n = 0; n < boxes.size(); n++) {
    const cv::Point *ppt[1] = {rook_points[n]};
    int npt[] = {4};
    cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
  }

  cv::imwrite("./vis.jpg", img_vis);
  std::cout << "The detection visualized image saved in ./vis.jpg" << std::endl;
  return img_vis;
}

std::vector<std::string> split(const std::string &str,
                               const std::string &delim) {
  std::vector<std::string> res;
  if ("" == str)
    return res;
  char *strs = new char[str.length() + 1];
  std::strcpy(strs, str.c_str());

  char *d = new char[delim.length() + 1];
  std::strcpy(d, delim.c_str());

  char *p = std::strtok(strs, d);
  while (p) {
    string s = p;
    res.push_back(s);
    p = std::strtok(NULL, d);
  }

  return res;
}

std::map<std::string, double> LoadConfigTxt(std::string config_path) {
  auto config = ReadDict(config_path);

  std::map<std::string, double> dict;
  for (int i = 0; i < config.size(); i++) {
    std::vector<std::string> res = split(config[i], " ");
    dict[res[0]] = stod(res[1]);
  }
  return dict;
}

void rec() {
  std::string rec_model_file = "debug/ch_ppocr_mobile_v2.0_rec_slim_opt.nb";
  std::string runtime_device = "cpu";
  std::string precision = "FP32";
  std::string num_threads = "2";
  std::string batchsize = "1";
  std::string img_path = "debug/cut.jpg";
  std::string dict_path = "debug/ppocr_keys_v1.txt";

  /*if (strcmp(argv[4], "FP32") != 0 && strcmp(argv[4], "INT8") != 0) {
      std::cerr << "Only support FP32 or INT8." << std::endl;
      exit(1);
  }*/

  //std::vector<cv::String> cv_all_img_names;
  //cv::glob(img_dir, cv_all_img_names);

  auto charactor_dict = ReadDict(dict_path);
  charactor_dict.insert(charactor_dict.begin(), "#"); // blank char for ctc
  charactor_dict.push_back(" ");

  auto rec_predictor = loadModel(rec_model_file, std::stoi(num_threads));

  std::shared_ptr<PaddlePredictor> cls_predictor;

  std::vector<double> time_info = {0, 0, 0};
  std::cout << "The predict img: " << img_path << std::endl;
  cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);

  if (!srcimg.data) {
      std::cerr << "[ERROR] image read failed! image path: " << img_path << std::endl;
      exit(1);
  }

  int width = srcimg.cols;
  int height = srcimg.rows;
  std::vector<int> upper_left = {0, 0};
  std::vector<int> upper_right = {width, 0};
  std::vector<int> lower_right = {width, height};
  std::vector<int> lower_left  = {0, height};
  std::vector<std::vector<int>> box = {upper_left, upper_right, lower_right, lower_left};
  std::vector<std::vector<std::vector<int>>> boxes = {box};
  std::vector<std::string> rec_text;
  std::vector<float> rec_text_score;
  
  struct timeval tv5,tv6;
  long t5,t6, time_ocr;
  gettimeofday(&tv5, NULL);
  RunRecModel(boxes, srcimg, rec_predictor, rec_text, rec_text_score,
	charactor_dict, cls_predictor, 0);

  // print recognized text
  //std::cout << "size:" << rec_text.size() << std::endl;
  for (int i = 0; i < rec_text.size(); i++) {
  std::cout << i << "\t" << rec_text[i] << "\t" << rec_text_score[i]<< std::endl; 
  }
  gettimeofday(&tv6, NULL);
  t5 = tv6.tv_sec - tv5.tv_sec;
  t6 = tv6.tv_usec - tv5.tv_usec;
  time_ocr = (long)(t5 * 1000 + t6 / 1000);
  printf("ocr inference time : %dms\n", time_ocr);
}











cv::Mat preprocess_img(cv::Mat& img) {
    int w, h, x, y;
    float r_w = INPUT_W / (img.cols*1.0);
    float r_h = INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    } else {
        w = r_h * img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}


int test_image_yuv(const char *rgb, int width, int height)
{
    struct timeval tv1,tv2,tv3,tv4;
    long t1, t2,t3,t4, time;
    
    int size=width*height;
  
    int model_load_flag=yolo_init("yolov5_temp_inst.wk");
    printf("model_load_flag : %d\n", model_load_flag);
    //gettimeofday(&tv1, NULL);

    Mat imgrgb = imread(rgb);

    gettimeofday(&tv3, NULL);
    cv::Mat imgRGB = preprocess_img(imgrgb);
    gettimeofday(&tv4, NULL);
    t3 = tv4.tv_sec - tv3.tv_sec;
    t4 = tv4.tv_usec - tv3.tv_usec;
    time = (long)(t3 * 1000 + t4 / 1000);

    printf("preprocess time : %dms\n", time);
    gettimeofday(&tv1, NULL);
    int yolo_run_flag = yolov3_inference(imgRGB);
    printf("yolo_run return flag : %d\n", yolo_run_flag);
    gettimeofday(&tv2, NULL);

    t1 = tv2.tv_sec - tv1.tv_sec;
    t2 = tv2.tv_usec - tv1.tv_usec;
    time = (long)(t1 * 1000 + t2 / 1000);
    printf("yolo_run inference time : %dms\n", time);
    //usleep(300000);
    return 0;
}

int main()
{
    //yuv图像格式为420
	const char *image_path = "car.jpg";
    int flag=test_image_yuv(image_path,640,640);
    rec();
    return 0;
}
