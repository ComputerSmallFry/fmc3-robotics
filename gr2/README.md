cd farts_deploy

cd /home/farts/farts_deploy && ./run_t5d.sh gr2t2d oak_97 t5dtest use_depth=true robot.visualize=true recording.pilot=-1 recording.operator=-1 hydra.job.chdir=false hand=fourier_dexpilot_dhx use_head=true

cd /home/farts/farts_deploy && ./run_t5d.sh gr2t2d_12dof oak_97 t5dtest use_depth=true robot.visualize=true recording.pilot=-1 recording.operator=-1 hydra.job.chdir=false hand=fourier_12dof_dexpilot use_head=true hand.use_tactile=false

cd /home/farts/farts_deploy && ./run_t5d_orbbec.sh gr2t2d orbbec t5dtestorbbec use_depth=false robot.visualize=true recording.pilot=-1 recording.operator=-1 hydra.job.chdir=false hand=fourier_dexpilot_dhx use_head=true camera.instance.display_config.mode=mono

cd /etc/udev/rules.d
sudo udevadm control --reload-rules
sudo udevadm trigger

chmod +x run_t5d.sh

sudo chown $USER /mnt/Data

http://127.0.0.1:7000/static/

docker tag docker.fftaicorp.com/farts/daemon:not-resize docker.fftaicorp.com/farts/daemon:latest