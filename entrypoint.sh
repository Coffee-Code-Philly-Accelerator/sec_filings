#!/bin/bash
# Start Xvfb on display :99
Xvfb :99 -screen 0 1280x720x16 &

# Wait for Xvfb to start
sleep 5
