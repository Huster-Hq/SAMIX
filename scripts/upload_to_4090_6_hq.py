import argparse
import posixpath
from pathlib import Path

import paramiko


def load_host_config(alias: str) -> dict[str, str | int]:
    config_path = Path.home() / ".ssh" / "config"
    text = config_path.read_text(encoding="utf-8")
    blocks: dict[str, list[str]] = {}
    current: str | None = None

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("host "):
            current = stripped[5:].strip()
            blocks[current] = []
            continue
        if current is not None:
            blocks[current].append(raw_line)

    matched_alias = next((name for name in blocks if name.lower() == alias.lower()), None)
    if matched_alias is None:
        raise ValueError(f"SSH alias not found: {alias}")

    host = None
    user = None
    port = 22
    password = None

    for line in blocks[matched_alias]:
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("hostname "):
            host = stripped.split(None, 1)[1].strip()
        elif lowered.startswith("user "):
            user = stripped.split(None, 1)[1].strip()
        elif lowered.startswith("port "):
            port = int(stripped.split(None, 1)[1].strip())
        elif stripped.startswith("#"):
            comment = stripped[1:].strip()
            if not comment:
                continue
            if " " not in comment:
                password = comment
            elif comment.lower().startswith("password:"):
                password = comment.split(":", 1)[1].strip()

    if not host or not user or not password:
        raise ValueError(f"Incomplete SSH config for alias: {matched_alias}")

    return {"host": host, "user": user, "port": port, "password": password}


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = remote_dir.strip("/").split("/")
    current = "/"
    for part in parts:
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def should_skip(path: Path) -> bool:
    skip_names = {".git", "__pycache__"}
    return any(part in skip_names for part in path.parts)


def iter_files(root: Path):
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(root)
        if should_skip(rel):
            continue
        yield path, rel


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload local workspace files to 4090-6-HQ.")
    parser.add_argument("--alias", default="4090-6-HQ")
    parser.add_argument("--local-root", default=".")
    parser.add_argument("--remote-root", required=True)
    args = parser.parse_args()

    local_root = Path(args.local_root).resolve()
    config = load_host_config(args.alias)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=str(config["host"]),
        port=int(config["port"]),
        username=str(config["user"]),
        password=str(config["password"]),
        timeout=10,
        banner_timeout=10,
        auth_timeout=10,
    )

    try:
        sftp = client.open_sftp()
        ensure_remote_dir(sftp, args.remote_root)
        count = 0
        for src, rel in iter_files(local_root):
            remote_path = posixpath.join(args.remote_root, rel.as_posix())
            ensure_remote_dir(sftp, posixpath.dirname(remote_path))
            sftp.put(str(src), remote_path)
            print(f"uploaded {rel.as_posix()}")
            count += 1
        print(f"uploaded_files={count}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
