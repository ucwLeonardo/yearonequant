#!/bin/bash

# copy important files to share directory
cp ./*.py ./*.ipynb ./*.csv ../../share/

# print current date and time, will be redirected into log file when called by cron
now="$(date)"
printf "******** Backup done at %s ********\n" "$now"
