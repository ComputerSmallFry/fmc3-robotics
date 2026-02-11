sudo groupadd input
sudo usermod -aG input $USER
newgrp input

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker

