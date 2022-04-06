#!/bin/bash

file_id="18xpj33rMsj_VtAECOoIKRKUB4qKtXASH"
file_name="vgg16_thermal.weights"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}
