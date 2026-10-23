#!/usr/bin/env bash
# CatchUp paper configuration: LLaVA-1.5-7B + Q-VER + LTC.
# Source this file before evaluation and profiling.
# Based on the final no-norm experiment script; verify parity with saved paper logs.
export REROUTE_CORRECTION_MODE=qver
export REROUTE_CORRECTION_ANCHOR_MODE=semantic_spatial
export REROUTE_CORRECTION_SNAPSHOT_POLICY=live
export REROUTE_CORRECTION_SNAPSHOT_DEBUG=0
export REROUTE_CORRECTION_MIN_LAYER_GAP=1
export REROUTE_CORRECTION_GAMMA_MAX=0.80
export REROUTE_CORRECTION_NORM_PRESERVE=0
export REROUTE_CORRECTION_NORM_TARGET=none
export REROUTE_CORRECTION_QUERY_ETA=0.0
export REROUTE_CORRECTION_QUERY_NORM=minmax
export REROUTE_CORRECTION_TOPM=8
export REROUTE_CORRECTION_SPATIAL_LAMBDA=0.15
export REROUTE_CORRECTION_TAU_L=0.07
