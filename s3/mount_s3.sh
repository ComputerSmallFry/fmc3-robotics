#!/bin/bash

# S3 Mount Script with Command-Line Parameters
# Usage: ./mount_s3.sh [OPTIONS]
# Example: ./mount_s3.sh --bucket my-bucket --mount-path /mnt/custom

set -e

# Default configuration
S3_FAST_MINIO_ENDPOINT="s3.fftaicorp.com:443"
S3_FAST_MINIO_ACCESS_KEY="Rau5gTQzfadPfaqA3F77BaDgzDCWM0om"
S3_FAST_MINIO_SECRET_KEY="3mec9hne6azdevq7n4atqbjnejoluya3"
S3_FAST_MINIO_REGION="us-east-1"
S3_FAST_MINIO_BUCKET="farther-data"
S3_FAST_SECURE="true"
MOUNT_PATH="${HOME}/s3/${S3_FAST_MINIO_BUCKET}"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --endpoint|-e)
            S3_FAST_MINIO_ENDPOINT="$2"
            shift 2
            ;;
        --access-key|-a)
            S3_FAST_MINIO_ACCESS_KEY="$2"
            shift 2
            ;;
        --secret-key|-s)
            S3_FAST_MINIO_SECRET_KEY="$2"
            shift 2
            ;;
        --region|-r)
            S3_FAST_MINIO_REGION="$2"
            shift 2
            ;;
        --bucket|-b)
            S3_FAST_MINIO_BUCKET="$2"
            shift 2
            ;;
        --secure)
            S3_FAST_SECURE="$2"
            shift 2
            ;;
        --mount-path|-m)
            MOUNT_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "S3 Mount Script - Mount S3/MinIO bucket using s3fs"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --endpoint, -e <endpoint>      S3 endpoint (default: s3.fftaicorp.com:443)"
            echo "  --access-key, -a <key>         Access key (default: configured)"
            echo "  --secret-key, -s <key>         Secret key (default: configured)"
            echo "  --region, -r <region>          S3 region (default: us-east-1)"
            echo "  --bucket, -b <bucket>          S3 bucket name (default: farther-data)"
            echo "  --secure <true|false>          Use HTTPS (default: true)"
            echo "  --mount-path, -m <path>        Mount path (default: /mnt/s3)"
            echo "  --help, -h                     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use all defaults"
            echo "  $0 --bucket my-bucket                 # Custom bucket"
            echo "  $0 -b test -m /mnt/test               # Custom bucket and mount path"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build S3 endpoint URL
if [[ "${S3_FAST_SECURE}" == "true" ]]; then
    S3_URL="https://${S3_FAST_MINIO_ENDPOINT}"
else
    S3_URL="http://${S3_FAST_MINIO_ENDPOINT}"
fi

# Set AWS credentials environment variables (required by s3fs)
export AWS_ACCESS_KEY_ID="${S3_FAST_MINIO_ACCESS_KEY}"
export AWS_SECRET_ACCESS_KEY="${S3_FAST_MINIO_SECRET_KEY}"

# Ensure mount directory exists
mkdir -p "${MOUNT_PATH}"

# Display configuration
echo "================================"
echo "S3 Mount Configuration"
echo "================================"
echo "Endpoint:    ${S3_FAST_MINIO_ENDPOINT}"
echo "Region:      ${S3_FAST_MINIO_REGION}"
echo "Bucket:      ${S3_FAST_MINIO_BUCKET}"
echo "Mount Path:  ${MOUNT_PATH}"
echo "Secure:      ${S3_FAST_SECURE}"
echo "URL:         ${S3_URL}"
echo "================================"

# Check if s3fs is installed
if ! command -v s3fs &> /dev/null; then
    echo "Error: s3fs is not installed. Please install it first."
    echo "  Ubuntu/Debian: sudo apt-get install s3fs"
    echo "  RHEL/CentOS:   sudo yum install s3fs-fuse"
    echo "  macOS:         brew install s3fs"
    exit 1
fi

# Check and enable user_allow_other in fuse.conf if needed
FUSE_CONF="/etc/fuse.conf"
if [ -f "${FUSE_CONF}" ]; then
    if ! grep -q "^user_allow_other" "${FUSE_CONF}"; then
        echo "Enabling user_allow_other in ${FUSE_CONF}..."
        echo "user_allow_other" >> "${FUSE_CONF}"
    fi
fi

# Check if already mounted
if mountpoint -q "${MOUNT_PATH}"; then
    echo "Warning: ${MOUNT_PATH} is already mounted. Unmounting first..."
    fusermount -u "${MOUNT_PATH}" 2>/dev/null || umount "${MOUNT_PATH}" 2>/dev/null || true
fi

# Mount S3 bucket
echo "Mounting S3 bucket..."
s3fs "${S3_FAST_MINIO_BUCKET}" "${MOUNT_PATH}" \
    -o url="${S3_URL}" \
    -o use_path_request_style \
    -o allow_other \
    -o uid=$(id -u "${SUDO_USER:-$USER}") \
    -o gid=$(id -g "${SUDO_USER:-$USER}") \
    -o umask=022 \
    -o nonempty

if [ $? -eq 0 ]; then
    echo "✓ Successfully mounted ${S3_FAST_MINIO_BUCKET} to ${MOUNT_PATH}"
else
    echo "✗ Failed to mount S3 bucket"
    exit 1
fi