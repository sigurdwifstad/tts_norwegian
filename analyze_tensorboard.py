#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util


EVENT_FILE_PREFIX = "events.out.tfevents."
DEFAULT_TENSORBOARD_HOST = "127.0.0.1"
DEFAULT_TENSORBOARD_PORT = 6006


@dataclass(frozen=True)
class ScalarPoint:
    step: int
    value: float
    wall_time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch TensorBoard or inspect scalar training metrics.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="models",
        help="TensorBoard log directory or event file (default: models).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch TensorBoard on localhost instead of printing summaries.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print scalar summaries instead of launching TensorBoard.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_TENSORBOARD_HOST,
        help="TensorBoard host when serving (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_TENSORBOARD_PORT,
        help="TensorBoard port when serving (default: 6006).",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the TensorBoard URL in the default browser after launch.",
    )
    parser.add_argument(
        "--tag",
        help="Optional regex filter for metric tags, e.g. 'loss|learning_rate'.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "csv"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--output",
        help="Write output to a file instead of stdout.",
    )
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="List available scalar tags per run instead of summaries.",
    )
    return parser.parse_args()


def build_tensorboard_command(logdir: Path, host: str, port: int) -> List[str]:
    return [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--host",
        host,
        "--port",
        str(port),
    ]


def launch_tensorboard(logdir: Path, host: str, port: int, open_browser: bool) -> int:
    command = build_tensorboard_command(logdir, host, port)
    subprocess.Popen(command, start_new_session=True)
    url = f"http://{host}:{port}/"
    print(f"TensorBoard started at {url}")
    print(f"Logdir: {logdir}")
    if open_browser:
        webbrowser.open(url)
    return 0


def discover_run_dirs(path: Path) -> List[Path]:
    if path.is_file():
        if path.name.startswith(EVENT_FILE_PREFIX):
            return [path.parent]
        raise FileNotFoundError(f"{path} is not a TensorBoard event file")

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    run_dirs = {
        event_file.parent
        for event_file in path.rglob(f"{EVENT_FILE_PREFIX}*")
        if event_file.is_file()
    }
    return sorted(run_dirs)


def _tensor_value(tensor_event) -> Optional[float]:
    array = tensor_util.make_ndarray(tensor_event.tensor_proto)
    if array.size != 1:
        return None
    if array.dtype.kind not in "biufc":
        return None
    try:
        return float(array.reshape(()))
    except (TypeError, ValueError):
        return None


def load_runs(run_dirs: Iterable[Path]) -> Dict[Path, Dict[str, List[ScalarPoint]]]:
    runs: Dict[Path, Dict[str, Dict[int, ScalarPoint]]] = {}

    for run_dir in run_dirs:
        accumulator = event_accumulator.EventAccumulator(
            str(run_dir),
            size_guidance={
                event_accumulator.SCALARS: 0,
                event_accumulator.TENSORS: 0,
            },
        )
        accumulator.Reload()

        tag_store: Dict[str, Dict[int, ScalarPoint]] = {}
        tags = accumulator.Tags()

        for tag in tags.get("scalars", []):
            points = tag_store.setdefault(tag, {})
            for scalar in accumulator.Scalars(tag):
                point = ScalarPoint(
                    step=int(scalar.step),
                    value=float(scalar.value),
                    wall_time=float(scalar.wall_time),
                )
                previous = points.get(point.step)
                if previous is None or point.wall_time >= previous.wall_time:
                    points[point.step] = point

        for tag in tags.get("tensors", []):
            points = tag_store.setdefault(tag, {})
            for tensor_event in accumulator.Tensors(tag):
                value = _tensor_value(tensor_event)
                if value is None:
                    continue
                point = ScalarPoint(
                    step=int(tensor_event.step),
                    value=value,
                    wall_time=float(tensor_event.wall_time),
                )
                previous = points.get(point.step)
                if previous is None or point.wall_time >= previous.wall_time:
                    points[point.step] = point

        runs[run_dir] = {
            tag: sorted(points.values(), key=lambda item: (item.step, item.wall_time))
            for tag, points in tag_store.items()
            if points
        }

    return runs


def filter_runs(
    runs: Dict[Path, Dict[str, List[ScalarPoint]]],
    tag_pattern: Optional[str],
) -> Dict[Path, Dict[str, List[ScalarPoint]]]:
    if not tag_pattern:
        return runs

    regex = re.compile(tag_pattern)
    filtered: Dict[Path, Dict[str, List[ScalarPoint]]] = {}

    for run_dir, tags in runs.items():
        matching = {tag: points for tag, points in tags.items() if regex.search(tag)}
        if matching:
            filtered[run_dir] = matching

    return filtered


def summarize(points: List[ScalarPoint]) -> dict:
    values = [point.value for point in points]
    min_point = min(points, key=lambda point: point.value)
    max_point = max(points, key=lambda point: point.value)

    return {
        "count": len(points),
        "first_step": points[0].step,
        "last_step": points[-1].step,
        "min": min_point.value,
        "min_step": min_point.step,
        "max": max_point.value,
        "max_step": max_point.step,
        "final": points[-1].value,
        "mean": sum(values) / len(values),
    }


def _run_label(run_dir: Path, base_path: Path) -> str:
    try:
        return str(run_dir.relative_to(base_path))
    except ValueError:
        return str(run_dir)


def render_text_summary(runs: Dict[Path, Dict[str, List[ScalarPoint]]], base_path: Path) -> str:
    blocks: List[str] = []

    for run_dir in sorted(runs):
        rows = []
        for tag in sorted(runs[run_dir]):
            stats = summarize(runs[run_dir][tag])
            rows.append(
                [
                    tag,
                    str(stats["count"]),
                    str(stats["first_step"]),
                    str(stats["last_step"]),
                    f'{stats["min"]:.6f}',
                    str(stats["min_step"]),
                    f'{stats["max"]:.6f}',
                    str(stats["max_step"]),
                    f'{stats["final"]:.6f}',
                    f'{stats["mean"]:.6f}',
                ]
            )

        headers = ["tag", "count", "first", "last", "min", "min@", "max", "max@", "final", "mean"]
        widths = [
            max([len(headers[i])] + [len(row[i]) for row in rows])
            for i in range(len(headers))
        ]

        header_line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
        separator_line = "  ".join("-" * widths[i] for i in range(len(headers)))
        row_lines = [
            "  ".join(row[i].ljust(widths[i]) for i in range(len(row)))
            for row in rows
        ]

        blocks.append(
            f"Run: {_run_label(run_dir, base_path)}\n{header_line}\n{separator_line}\n"
            + "\n".join(row_lines)
        )

    return "\n\n".join(blocks)


def build_json_payload(runs: Dict[Path, Dict[str, List[ScalarPoint]]], base_path: Path) -> list:
    payload = []

    for run_dir in sorted(runs):
        payload.append(
            {
                "run": _run_label(run_dir, base_path),
                "metrics": {
                    tag: {
                        "summary": summarize(points),
                        "points": [
                            {"step": point.step, "value": point.value, "wall_time": point.wall_time}
                            for point in points
                        ],
                    }
                    for tag, points in sorted(runs[run_dir].items())
                },
            }
        )

    return payload


def build_csv_rows(runs: Dict[Path, Dict[str, List[ScalarPoint]]], base_path: Path) -> List[dict]:
    rows = []
    for run_dir in sorted(runs):
        for tag, points in sorted(runs[run_dir].items()):
            for point in points:
                rows.append(
                    {
                        "run": _run_label(run_dir, base_path),
                        "tag": tag,
                        "step": point.step,
                        "value": point.value,
                        "wall_time": point.wall_time,
                    }
                )
    return rows


def render_tag_list(runs: Dict[Path, Dict[str, List[ScalarPoint]]], base_path: Path) -> str:
    blocks = []

    for run_dir in sorted(runs):
        tags = "\n".join(f"  - {tag}" for tag in sorted(runs[run_dir]))
        blocks.append(f"Run: {_run_label(run_dir, base_path)}\n{tags}")

    return "\n\n".join(blocks)


def write_text(text: str, output_path: Optional[str]) -> None:
    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
    else:
        print(text)


def write_csv(rows: List[dict], output_path: Optional[str]) -> None:
    fieldnames = ["run", "tag", "step", "value", "wall_time"]

    if output_path:
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def main() -> int:
    args = parse_args()
    target_path = Path(args.path).expanduser().resolve()

    if args.serve or not (args.summary or args.tag or args.list_tags or args.format != "text" or args.output):
        try:
            discover_run_dirs(target_path)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        return launch_tensorboard(target_path, args.host, args.port, args.open_browser)

    try:
        run_dirs = discover_run_dirs(target_path)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not run_dirs:
        print(f"No TensorBoard event files found under {target_path}", file=sys.stderr)
        return 1

    runs = filter_runs(load_runs(run_dirs), args.tag)

    if not runs:
        print("No matching scalar data found.", file=sys.stderr)
        return 1

    base_path = target_path if target_path.is_dir() else target_path.parent

    if args.list_tags:
        write_text(render_tag_list(runs, base_path), args.output)
        return 0

    if args.format == "text":
        write_text(render_text_summary(runs, base_path), args.output)
        return 0

    if args.format == "json":
        write_text(json.dumps(build_json_payload(runs, base_path), indent=2), args.output)
        return 0

    write_csv(build_csv_rows(runs, base_path), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
