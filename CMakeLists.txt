cmake_minimum_required(VERSION 3.6)
project(main)

set(TARGET yolov5_ocr)
SET(CMAKE_CXX_STANDARD 11)

# ADD_DEFINITIONS(-D __DEBUG__)  # is Debug?

# ------------------------------- FLAGS -----------------------------
SET(CMAKE_C_COMPILER arm-himix200-linux-gcc)
SET(CMAKE_CXX_COMPILER arm-himix200-linux-g++)
SET(CMAKE_C_STRIP arm-himix200-linux-strip)

#SET(CMAKE_C_FLAGS_PUBLIC "-mcpu=cortex-a7 -mfloat-abi=softfp -mfpu=neon-vfpv4 -ffunction-sections -mno-unaligned-access -fno-aggressive-loop-optimizations -mapcs-frame -rdynamic")
# SET(CMAKE_C_FLAGS_DEBUG "-Wall -ggdb3 -DNM_DEBUG ${CMAKE_C_FLAGS_PUBLIC}")
#SET(CMAKE_C_FLAGS_RELEASE  "-Wall -O3  ${CMAKE_C_FLAGS_PUBLIC}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math  -fpermissive -fpic")

add_definitions(-std=c++11 -O3 -pthread -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=softfp)
add_definitions(-ffunction-sections -mno-unaligned-access -fno-aggressive-loop-optimizations -mapcs-frame -rdynamic -fpermissive -fpic)
#SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)
#SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

# --------------------------------
# Hisi
SET(HI_ROOT libhisi)
SET(HI_INCLUDE ${HI_ROOT}/include)
SET(HI_LIB_DIR ${HI_ROOT}/lib)
# opencv
SET(OPENCV_ROOT libopencv)
SET(OPENCV_INCLUDE ${OPENCV_ROOT}/include)
SET(OPENCV_LIB_DIR ${OPENCV_ROOT}/lib)
FILE(GLOB OPENCV_LIBS "${OPENCV_LIB_DIR}/*")


# Paddle-Lite 
SET(Paddle_Lite_ROOT paddle-lite-inference)
set(Paddle_Lite_INCLUDE ${Paddle_Lite_ROOT}/include)
set(Paddle_Lite_LIB_DIR ${Paddle_Lite_ROOT}/lib)


# ------------ lib ----------------
LINK_LIBRARIES(mpi.a)
SET(HI_LIBS _hiawb.a _hiae.a _hidehaze.a upvqe.a dnvqe.a md.a
        _hildci.a _hidrc.a securec.a nnie.a ive.a tde.a VoiceEngine.a isp.a )

INCLUDE_DIRECTORIES(inc ${OPENCV_INCLUDE} ${HI_INCLUDE} ${Paddle_Lite_INCLUDE})
LINK_DIRECTORIES( ${OPENCV_LIB_DIR} ${HI_LIB_DIR} ${Paddle_Lite_LIB_DIR})

FILE(GLOB SRC "src/*.cpp" "src/*c")

#FILE(GLOB OCR "ocr/*.cpp" "ocr/*.cc")

# For YOLO ocr_db_crnn.cc
ADD_EXECUTABLE(${TARGET} ${SRC} main.cpp 
	crnn_process.cc
	clipper.cpp
	cls_process.cc
	db_post_process.cc)
TARGET_LINK_LIBRARIES(${TARGET}  ${HI_LIBS} ${OPENCV_LIBS} -lpaddle_light_api_shared -fopenmp -lstdc++ -lpthread -lm -ldl)
