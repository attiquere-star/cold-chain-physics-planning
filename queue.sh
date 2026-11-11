#!/bin/bash
cd "$(dirname "$0")"
until [ "$(ls out/main_*.json 2>/dev/null | wc -l)" -ge 12 ]; do sleep 30; done
xargs -P 4 -I{} bash -c "{}" < jobs.txt
echo ALLDONE > out/queue_done.txt
