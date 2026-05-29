# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(CAREM7)

zephyr_include_directories(
    ${ZEPHYR_BASE}/modules/hal/nordic/nrfx
    ${ZEPHYR_BASE}/modules/hal/nordic/nrfx/drivers/include
    ${ZEPHYR_BASE}/modules/hal/nordic/nrfx/mdk
)

# CHANGE v7.2: added src/generated/heart_mfcc_tables.c and
#              src/generated/lung_mfcc_tables.c.
#
# heart_mfcc_config.h and lung_mfcc_config.h declare:
#   extern const dsp_mfcc_config_t heart_mfcc_config;
#   extern const dsp_mfcc_config_t lung_mfcc_config;
#   extern const float32_t heart_bp_coeffs[], heart_hamming[], ...;
#   extern const float32_t lung_bp_coeffs[],  lung_hamming[],  ...;
#
# The definitions live in these generated files (produced by
# tools/gen_mfcc_tables.py).  Without them the linker fails with
# undefined references to heart_mfcc_config / lung_mfcc_config
# and all their table arrays.
#
# To regenerate:
#   python tools/gen_mfcc_tables.py \
#       --heart heart_winning_config.json \
#       --lung  lung_winning_config.json  \
#       --out   src/generated/
target_sources(app PRIVATE
    src/main.c
    src/dsp_mfcc.c
    src/generated/heart_mfcc_tables.c
    src/generated/lung_mfcc_tables.c
)

# CMSIS-DSP is bundled with Zephyr's SDK — we just need to declare
# the dependency so Kconfig enables arm_math.h and the right sources.
# This is handled via prj.conf: CONFIG_CMSIS_DSP=y
# No extra include_directories needed — Zephyr exports arm_math.h
# automatically when CONFIG_CMSIS_DSP=y.

# Required for sinf(), cosf(), log10f(), powf() etc.
target_link_libraries(app PRIVATE m)