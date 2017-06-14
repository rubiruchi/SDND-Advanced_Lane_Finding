#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 23:51:43 2017

Line Class

@author: ucalegon
"""

# TODO: Clean this up, keep only what you need

# Define a class to receive the characteristics of each line detection
class Lane():
    def __init__(self):

        # When initialized, set first_frame property as True
        # Initiates a more expensive lane detection method requrired when there is no previous frames
        # from which to draw.
        self.first_frame = True

        # Was the line detected in the last iteration?
        # TODO: do you need this?
        self.detected = False

        # X positions of window centers for last n frames
        # Instead of looking at the histogram, use this for 2...n frames.
        self.window_base = None

        # Keep individual (non_averaged) lane line coeffs for the last n frames
        self.recent_fits = []

        # polynomial coefficients for the most recent fit (no avg)
        self.current_fit = None

        # The avg fitted line given current (if detected/normal) and recent fits
        self.best_fit = None

        # radius of curvature of the line in some units
        self.curverad = None

        # curve radius to display - updated every few frames for readability
        self.curverad_display = None
