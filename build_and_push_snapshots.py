"""Build and push a unified Docker image with all Prometheus snapshots.

The image bundles the OpenTelemetry Demo source, ALL snapshot data
(Prometheus TSDB snapshots, OpenSearch data/snapshots), and the
entrypoint/health-check scripts.  At runtime the entrypoint selects
the appropriate Prometheus snapshot based on the SNAPSHOT_NAME env var.

Usage (from the otel-demo directory):
    # Build and push (skips if image already exists on Docker Hub):
    python build_and_push_snapshots.py --data-dir data-0216

    # Force rebuild even if already on Docker Hub:
    python build_and_push_snapshots.py --data-dir data-0216 --force
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from utils import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATES_DIR = Path("harbor-template")


def assemble_build_context(
    *,
    data_dir: Path,
    build_ctx: Path,
    templates_dir: Path,
) -> None:
    """Assemble a Docker build context directory with all snapshots."""
    # OpenTelemetry Demo source
    logger.info("Copying OpenTelemetry Demo source to build context ...")
    shutil.copytree(
        Path("opentelemetry-demo").resolve(),
        build_ctx / "opentelemetry-demo",
        symlinks=False,
        dirs_exist_ok=True,
        copy_function=os.link,
    )

    # Snapshot data
    data_dest = build_ctx / "data"
    data_dest.mkdir(parents=True, exist_ok=True)

    # ALL Prometheus TSDB snapshots
    prom_snapshots_src = data_dir / "prometheus" / "snapshots"
    if prom_snapshots_src.exists():
        logger.info(f"Copying all Prometheus snapshots from {prom_snapshots_src} ...")
        shutil.copytree(
            prom_snapshots_src,
            data_dest / "prometheus" / "snapshots",
            symlinks=False,
            dirs_exist_ok=True,
            copy_function=os.link,
        )
    else:
        logger.warning(
            f"Prometheus snapshots directory not found: {prom_snapshots_src}"
        )

    # Prometheus read-only config
    prom_config_src = Path("prometheus-config-readonly.yaml").resolve()
    if prom_config_src.exists():
        logger.info(
            f"Copying Prometheus config from {prom_config_src} to build context ..."
        )
        shutil.copy2(prom_config_src, data_dest / "prometheus-config.yaml")

    # OpenSearch snapshot repo (but NOT opensearch-data — OpenSearch starts
    # with an empty node and the snapshot restore recreates the indices).
    os_repo_src = data_dir / "opensearch-repo-snapshots"
    if os_repo_src.exists():
        logger.info(
            f"Copying OpenSearch repo snapshots from {os_repo_src} to build context ..."
        )
        shutil.copytree(
            os_repo_src,
            data_dest / "opensearch-repo-snapshots",
            symlinks=False,
            dirs_exist_ok=True,
            copy_function=os.link,
        )
    # Create empty opensearch-data dir (will be populated by snapshot restore at runtime)
    (data_dest / "opensearch-data").mkdir(parents=True, exist_ok=True)

    # docker-compose-base.yml (no bind-mount volumes)
    compose_src = Path("docker-compose-base.yml").resolve()
    if compose_src.exists():
        logger.info(
            f"Copying docker-compose-base.yml from {compose_src} to build context ..."
        )
        shutil.copy2(compose_src, build_ctx / "docker-compose-base.yml")
    else:
        logger.warning(f"docker-compose-base.yml not found: {compose_src}")

    # Snapshot-mode overrides (read-only Prometheus, OTel Collector, Jaeger)
    for snapshot_file in [
        "docker-compose.snapshot.yml",
        "otelcol-config-snapshot.yml",
        "jaeger-config-snapshot.yml",
    ]:
        src = Path(snapshot_file).resolve()
        if src.exists():
            logger.info(f"Copying {snapshot_file} to build context ...")
            shutil.copy2(src, build_ctx / snapshot_file)

    # Entrypoint and health check
    logger.info("Copying entrypoint and health check scripts to build context ...")
    shutil.copy2(
        (templates_dir / "environment" / "entrypoint.sh").resolve(),
        build_ctx / "entrypoint.sh",
    )
    shutil.copy2(
        Path("check_health.sh").resolve(),
        build_ctx / "check_health.sh",
    )

    # Dockerfile
    snapshot_dockerfile = templates_dir / "snapshot-base" / "Dockerfile"
    logger.info(f"Copying Dockerfile from {snapshot_dockerfile} to build context ...")
    shutil.copy2(snapshot_dockerfile.resolve(), build_ctx / "Dockerfile")


def build_image(tag: str, build_ctx: Path) -> bool:
    """Build a Docker image and return True on success."""
    cmd = ["docker", "build", "--progress=plain", "-t", tag, str(build_ctx)]
    logger.info(f"Building {tag} ...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"Build failed for {tag}")
        return False
    logger.info(f"Built {tag}")
    return True


def push_image(tag: str) -> bool:
    """Push a Docker image and return True on success."""
    cmd = ["docker", "push", tag]
    logger.info(f"Pushing {tag} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Push failed for {tag}:\n{result.stderr}")
        return False
    logger.info(f"Pushed {tag}")
    return True


def image_exists_on_hub(tag: str) -> bool:
    """Check if a Docker image already exists on Docker Hub."""
    cmd = ["docker", "manifest", "inspect", tag]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build and push a unified Docker image with all Prometheus snapshots."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to telemetry data directory. The directory name is used as the image tag.",
    )
    parser.add_argument(
        "--docker-repo",
        default="orcabench/sre-otel-snapshot",
        help="Docker Hub repository (e.g. user/repo).",
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=DEFAULT_TEMPLATES_DIR,
        help="Path to the Harbor task templates directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Build and push even if the image already exists on Docker Hub.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()
    setup_logging(getattr(logging, args.log_level))

    data_dir = args.data_dir.resolve()
    tag = f"{args.docker_repo}:{args.data_dir.name}-{args.templates_dir.name}"

    if not args.force and image_exists_on_hub(tag):
        logger.info(f"Skipping {tag} (already exists on Docker Hub)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        build_ctx = Path(tmp)
        assemble_build_context(
            data_dir=data_dir, build_ctx=build_ctx, templates_dir=args.templates_dir
        )
        if not build_image(tag, build_ctx):
            sys.exit(1)

    if not push_image(tag):
        sys.exit(1)

    logger.info(f"Built and pushed {tag} successfully.")


if __name__ == "__main__":
    main()
