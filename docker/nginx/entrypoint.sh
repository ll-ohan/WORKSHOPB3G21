#!/bin/sh
set -e

: "${API_PROXY_PASS:=http://api:8000/}"

envsubst '${API_PROXY_PASS}' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf
