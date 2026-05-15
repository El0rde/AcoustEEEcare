# SPDX-License-Identifier: Apache-2.0
 
cmake_minimum_required(VERSION 3.20.0)
 
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(CAREM7)
 
zephyr_include_directories(
    ${ZEPHYR_BASE}/modules/hal/nordic/nrfx
    ${ZEPHYR_BASE}/modules/hal/nordic/nrfx/drivers/include
    ${ZEPHYR_BASE}/modules/hal/nordic/nrfx/mdk
)
 
target_sources(app PRIVATE
    src/main.c
    # src/dsp_mfcc.c
)
 
# CMSIS-DSP is bundled with Zephyr's SDK — we just need to declare
# the dependency so Kconfig enables arm_math.h and the right sources.
# This is handled via prj.conf: CONFIG_CMSIS_DSP=y
# No extra include_directories needed — Zephyr exports arm_math.h
# automatically when CONFIG_CMSIS_DSP=y.
 
# Required for sinf(), cosf(), log10f(), powf() etc.
target_link_libraries(app PRIVATE m)