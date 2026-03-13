#!/usr/bin/env bash
set -euo pipefail

# 上传本地数据集目录到 Hugging Face Hub，支持 LeRobot 和 Dora 原始数据。
#
# 用法：
#   bash upload_lerobot_dataset_to_hf.sh --dataset /path/to/dataset_dir
#
# 约定：
# - 只需要传本地数据集目录。
# - 仓库名默认取目录名，例如：
#     /data/fmc3_gr2_grab_bottle_into_box_lerobot_ds
#   会上传为当前登录账号下的：
#     <your_hf_username>/fmc3_gr2_grab_bottle_into_box_lerobot_ds
# - 也就是说“数据集名字”和“repo_id 的 repo name 部分”保持一致。
#
# 依赖：
# - 已安装 `hf` CLI
# - 本地已保存可用 Hugging Face token（`hf auth login` 或 `HF_TOKEN`）
#
# 可选环境变量：
# - HF_UPLOAD_WORKERS：上传并发数，默认 16
# - HF_PRIVATE=1：创建私有 dataset repo
# - HF_REQUIRE_LEROBOT=1：只允许上传 LeRobot 结构；若检测到 Dora 或未知结构则直接报错
# - HF_FORCE_NO_PROXY=1：强制上传时忽略当前 shell 中的代理环境变量

usage() {
    cat <<'EOF'
用法:
  bash upload_lerobot_dataset_to_hf.sh --dataset /path/to/dataset_dir

示例:
  bash upload_lerobot_dataset_to_hf.sh \
    --dataset \
    /home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_into_box_lerobot_ds

  bash upload_lerobot_dataset_to_hf.sh \
    --dataset \
    /home/phl/workspace/dataset/fourier/gr2/muticams/dora/fmc3_gr2_grab_bottle_into_box_dora_ds
EOF
}

detect_dataset_type() {
    local dataset_path="$1"

    if [ -d "${dataset_path}/meta" ] && [ -d "${dataset_path}/data" ] && [ -d "${dataset_path}/videos" ]; then
        echo "lerobot"
        return 0
    fi

    if find "${dataset_path}" -type d -name 'episode_*' -print -quit 2>/dev/null | grep -q .; then
        echo "dora"
        return 0
    fi

    echo "unknown"
}

has_hf_token() {
    if [ -n "${HF_TOKEN:-}" ]; then
        return 0
    fi

    if hf auth list 2>/dev/null | grep -qE '^\*|HF_TOKEN'; then
        return 0
    fi

    return 1
}

proxy_env_present() {
    [ -n "${http_proxy:-}${https_proxy:-}${HTTP_PROXY:-}${HTTPS_PROXY:-}${ALL_PROXY:-}${all_proxy:-}" ]
}

run_hf_command() {
    if [ "${HF_FORCE_NO_PROXY:-0}" = "1" ]; then
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy "$@"
    else
        "$@"
    fi
}

run_hf_command_without_proxy() {
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy "$@"
}

DATASET_ARG=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataset)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --dataset 缺少目录参数" >&2
                usage
                exit 1
            fi
            DATASET_ARG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] 不支持的参数: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [ -z "${DATASET_ARG}" ]; then
    echo "[ERROR] 必须传入 --dataset <目录>" >&2
    usage
    exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
    echo "[ERROR] 未找到 hf 命令，请先安装 huggingface_hub CLI" >&2
    exit 1
fi

INPUT_PATH="${DATASET_ARG}"
if [ ! -d "${INPUT_PATH}" ]; then
    echo "[ERROR] 数据集目录不存在: ${INPUT_PATH}" >&2
    exit 1
fi

DATASET_PATH="$(cd "${INPUT_PATH}" && pwd)"
DATASET_NAME="$(basename "${DATASET_PATH}")"
UPLOAD_WORKERS="${HF_UPLOAD_WORKERS:-16}"
DATASET_TYPE="$(detect_dataset_type "${DATASET_PATH}")"

PRIVATE_FLAG=()
if [ "${HF_PRIVATE:-0}" = "1" ]; then
    PRIVATE_FLAG+=(--private)
fi

if [ "${HF_REQUIRE_LEROBOT:-0}" = "1" ] && [ "${DATASET_TYPE}" != "lerobot" ]; then
    echo "[ERROR] HF_REQUIRE_LEROBOT=1，但检测到的数据集结构不是 LeRobot: ${DATASET_TYPE}" >&2
    exit 1
fi

echo "检查本地 Hugging Face token..."
if ! has_hf_token; then
    echo "[ERROR] 未检测到本地 Hugging Face token，请先执行 hf auth login 或设置 HF_TOKEN" >&2
    exit 1
fi

echo "准备上传数据集"
echo "  local_path: ${DATASET_PATH}"
echo "  dataset_type: ${DATASET_TYPE}"
echo "  repo_type : dataset"
echo "  repo_name : ${DATASET_NAME}"
echo "  workers   : ${UPLOAD_WORKERS}"
if [ "${HF_PRIVATE:-0}" = "1" ]; then
    echo "  visibility: private"
else
    echo "  visibility: public"
fi
if [ "${HF_FORCE_NO_PROXY:-0}" = "1" ]; then
    echo "  network   : force_no_proxy"
elif proxy_env_present; then
    echo "  network   : use_current_proxy_env"
else
    echo "  network   : direct"
fi

if [ "${DATASET_TYPE}" = "unknown" ]; then
    echo "[WARN] 未识别为标准 LeRobot 或 Dora 目录，脚本仍会按普通 dataset 文件夹上传" >&2
fi

CREATE_ERR_FILE="$(mktemp)"
UPLOAD_ERR_FILE="$(mktemp)"
RETRY_WITHOUT_PROXY=0

upload_with_current_mode() {
    # 只传 repo_name，不显式传 username。
    # hf 会自动使用当前登录账号，最终远端 repo 实际上会是 <username>/<repo_name>。
    run_hf_command hf repo create "${DATASET_NAME}" --repo-type dataset --exist-ok "${PRIVATE_FLAG[@]}" 2>"${CREATE_ERR_FILE}"
    run_hf_command hf upload-large-folder "${DATASET_NAME}" "${DATASET_PATH}" \
        --repo-type dataset \
        --num-workers "${UPLOAD_WORKERS}" \
        "${PRIVATE_FLAG[@]}" 2>"${UPLOAD_ERR_FILE}"
}

upload_without_proxy() {
    run_hf_command_without_proxy hf repo create "${DATASET_NAME}" --repo-type dataset --exist-ok "${PRIVATE_FLAG[@]}" 2>"${CREATE_ERR_FILE}"
    run_hf_command_without_proxy hf upload-large-folder "${DATASET_NAME}" "${DATASET_PATH}" \
        --repo-type dataset \
        --num-workers "${UPLOAD_WORKERS}" \
        "${PRIVATE_FLAG[@]}" 2>"${UPLOAD_ERR_FILE}"
}

if upload_with_current_mode; then
    rm -f "${CREATE_ERR_FILE}" "${UPLOAD_ERR_FILE}"
    echo "上传完成: ${DATASET_NAME} (${DATASET_TYPE})"
    exit 0
fi

if proxy_env_present && [ "${HF_FORCE_NO_PROXY:-0}" != "1" ]; then
    RETRY_WITHOUT_PROXY=1
    echo "[WARN] 使用当前代理环境上传失败，准备自动重试一次（不使用代理）" >&2
    [ -n "${http_proxy:-}" ] && echo "  http_proxy=${http_proxy}" >&2
    [ -n "${https_proxy:-}" ] && echo "  https_proxy=${https_proxy}" >&2
    [ -n "${HTTP_PROXY:-}" ] && echo "  HTTP_PROXY=${HTTP_PROXY}" >&2
    [ -n "${HTTPS_PROXY:-}" ] && echo "  HTTPS_PROXY=${HTTPS_PROXY}" >&2
    [ -n "${ALL_PROXY:-}" ] && echo "  ALL_PROXY=${ALL_PROXY}" >&2
    [ -n "${all_proxy:-}" ] && echo "  all_proxy=${all_proxy}" >&2
    echo "[WARN] 首次失败输出：" >&2
    sed -n '1,120p' "${CREATE_ERR_FILE}" >&2
    sed -n '1,120p' "${UPLOAD_ERR_FILE}" >&2

    : > "${CREATE_ERR_FILE}"
    : > "${UPLOAD_ERR_FILE}"
    if upload_without_proxy; then
        rm -f "${CREATE_ERR_FILE}" "${UPLOAD_ERR_FILE}"
        echo "上传完成: ${DATASET_NAME} (${DATASET_TYPE})"
        exit 0
    fi
fi

echo "[ERROR] 上传失败" >&2
if [ "${RETRY_WITHOUT_PROXY}" = "1" ]; then
    echo "[ERROR] 已分别尝试：当前代理环境、无代理环境" >&2
else
    echo "[ERROR] 已尝试当前网络环境" >&2
fi
echo "[ERROR] repo create 输出：" >&2
sed -n '1,120p' "${CREATE_ERR_FILE}" >&2
echo "[ERROR] upload-large-folder 输出：" >&2
sed -n '1,120p' "${UPLOAD_ERR_FILE}" >&2
rm -f "${CREATE_ERR_FILE}" "${UPLOAD_ERR_FILE}"
exit 1
