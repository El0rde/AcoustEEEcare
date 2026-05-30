# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(CAREM7)

zephyr_include_directories(
    ${ZEPHYR_BASE}/modules/hal/nordic/nrfx
    ${ZEPHYR_BASE}/modules/hal/nordic/nrfx/drivers/include
    ${ZEPHYR_BASE}/modules/hal/nordic/nrfx/mdk
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# v7.2: generated MFCC tables (define heart/lung_mfcc_config + table arrays).
# v8.4: added src/tflm_inference.cc — TFLite Micro heart + lung inference.
#       Pulled in automatically: TFLM library + tensorflow/lite/... headers
#       via CONFIG_TENSORFLOW_LITE_MICRO=y; C++ via CONFIG_CPP/STD_CPP17.
target_sources(app PRIVATE
    src/main.c
    src/dsp_mfcc.c
    src/tflm_inference.cc          # ★ NEW
    src/generated/heart_mfcc_tables.c
    src/generated/lung_mfcc_tables.c
)

# Math (sinf/cosf/log10f/powf) — used by dsp_mfcc.c.
target_link_libraries(app PRIVATE m)