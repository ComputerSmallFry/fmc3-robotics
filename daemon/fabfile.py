from pathlib import Path

from fabric import Config
from fabric import ThreadingGroup as Group
from invoke import task
from loguru import logger


SUDO_PASS = "1"
USER = "farts"
HOSTS = "11,12,13,14,15,16,17,18,21,22,23,24"


def make_group(host_ids: list[int]) -> Group:
    hosts = [f"farts@{i:03d}.farts.com" for i in host_ids]
    return Group(
        *hosts,
        config=Config(overrides={"sudo": {"password": SUDO_PASS}}),
        forward_agent=True,
        connect_kwargs={
            "password": SUDO_PASS,
            "look_for_keys": False,
            "allow_agent": False,
        },
    )


def make_pool(hosts: str) -> Group:
    hosts = sorted([int(h) for h in hosts.split(",")])
    logger.info(f"Connecting to hosts: {hosts}")
    return make_group(hosts)


@task
def rput(c, hosts: str, local, remote):
    pool = make_pool(hosts)
    pool.run(f"mkdir -p {remote}")
    local = Path(local).resolve()

    if local.is_dir():
        for f in local.rglob("*"):
            if f.is_file():
                logger.info(f"Copying {f} to {remote + str(f.relative_to(local))}")
                pool.put(str(f), remote + str(f.relative_to(local)))
            if f.is_dir():
                logger.info(f"Creating directory {f} in {remote}")
                pool.run(f"mkdir -p {remote}/{f.relative_to(local)}")

    else:
        logger.info(f"Copying {local} to {remote}")
        pool.put(str(local), remote)


@task
def run(c, cmd, hosts):
    pool = make_pool(hosts)
    logger.info(f"Running command: {cmd} on hosts: {hosts}")
    res_dict = pool.run(cmd, pty=True)

    for conn, r in res_dict.items():
        logger.info(conn.host)
        print(r.stdout)


@task
def srun(c, cmd, hosts):
    pool = make_pool(hosts)
    logger.info(f"Running command: {cmd} on hosts: {hosts}")
    res_dict = pool.sudo(cmd, pty=True, hide=True)

    for conn, r in res_dict.items():
        logger.info(conn.host)
        print(r.stdout)


@task
def permissions(c, hosts):
    pool = make_pool(hosts)

    pool.sudo("chown -R $USER:$USER /mnt/Data/recordings", pty=True)
    pool.sudo("chmod -R 755 /mnt/Data/recordings", pty=True)

    pool.sudo("chown -R $USER:$USER ~/.farts/outputs", pty=True)
    pool.sudo("chmod -R 755 ~/.farts/outputs", pty=True)


@task
def du(c, hosts=HOSTS, path="/mnt/Data/recordings"):
    pool = make_pool(hosts)
    cmd = f"sudo du -d2 -h {path} | grep G  || echo 'No recordings'"
    logger.info(f"Running command: {cmd} on hosts: {hosts}")
    res_dict = pool.sudo(cmd, pty=True, hide=True)

    logger.info("Results:")
    for conn, r in res_dict.items():
        logger.info(conn.host)
        print(r.stdout)


@task
def rm_data(c, hosts=HOSTS, path="~/.farts/data/recordings"):
    pool = make_pool(hosts)
    cmd = f"rm -rf {path}/*"
    logger.info(f"Running command: {cmd} on hosts: {hosts}")
    res_dict = pool.sudo(cmd, pty=True, hide=True)

    logger.info("Results:")
    for conn, r in res_dict.items():
        logger.info(conn.host)
        print(r.stdout)


@task
def rm_processed_data(c, hosts="11,12,15,16"):
    pool = make_pool(hosts)
    cmd = "rm -rf ~/factory"
    logger.info(f"Running command: {cmd} on hosts: {hosts}")
    res_dict = pool.run(cmd, pty=True, hide=True)

    cmd = "rm -rf /media/Data"
    logger.info(f"Running command: {cmd} on hosts: {hosts}")
    res_dict = pool.run(cmd, pty=True, hide=True)

    logger.info("Results:")
    for conn, r in res_dict.items():
        logger.info(conn.host)
        print(r.stdout)


@task
def deploy(c):
    c.run("bash deploy.sh", pty=True)

