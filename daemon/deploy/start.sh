STATION_ID=$(echo "$(hostname)" | sed 's/[^0-9]//g') docker compose up -d --remove-orphans
