#!/usr/bin/env python3
"""Build a raster plot from mnist-learn spikes_inference_raw.csv."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SpikeEvent:
    send_time: int
    sender_name: str
    sender_uid: str
    neuron_index: int


@dataclass
class SenderTrack:
    sender_name: str
    sender_uid: str
    label: str
    start_index: int
    end_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to spikes_inference_raw.csv")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PNG path. Default: <csv_path>.png",
    )
    parser.add_argument(
        "--sender",
        action="append",
        default=[],
        help="Filter by sender_name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--title",
        default="Inference Spike Raster",
        help="Figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG DPI.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=8.0,
        help="Scatter marker size.",
    )
    parser.add_argument(
        "--neuron-gap",
        type=int,
        default=4,
        help="Vertical gap between sender blocks.",
    )
    return parser.parse_args()


def load_events(csv_path: Path, sender_filters: set[str]) -> list[SpikeEvent]:
    events: list[SpikeEvent] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"send_time", "sender_name", "sender_uid", "neuron_index"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{csv_path} must contain columns: {', '.join(sorted(required_columns))}"
            )

        for row in reader:
            event = SpikeEvent(
                send_time=int(row["send_time"]),
                sender_name=row["sender_name"],
                sender_uid=row["sender_uid"],
                neuron_index=int(row["neuron_index"]),
            )
            if sender_filters and event.sender_name not in sender_filters:
                continue
            events.append(event)

    if not events:
        raise ValueError("No spikes matched the selected filters.")

    return events


def build_tracks(events: list[SpikeEvent], neuron_gap: int) -> tuple[list[SenderTrack], dict[str, int]]:
    sender_order: OrderedDict[str, tuple[str, int]] = OrderedDict()
    max_neuron_per_sender: dict[str, int] = {}
    uids_per_name: dict[str, set[str]] = defaultdict(set)

    for event in events:
        sender_key = event.sender_uid
        if sender_key not in sender_order:
            sender_order[sender_key] = (event.sender_name, event.neuron_index)
        max_neuron_per_sender[sender_key] = max(
            max_neuron_per_sender.get(sender_key, event.neuron_index), event.neuron_index
        )
        uids_per_name[event.sender_name].add(event.sender_uid)

    tracks: list[SenderTrack] = []
    sender_offsets: dict[str, int] = {}
    current_offset = 0

    for sender_uid, (sender_name, _) in sender_order.items():
        max_neuron = max_neuron_per_sender[sender_uid]
        label = sender_name
        if len(uids_per_name[sender_name]) > 1:
            label = f"{sender_name} ({sender_uid[:8]})"

        start_index = current_offset
        end_index = current_offset + max_neuron
        tracks.append(
            SenderTrack(
                sender_name=sender_name,
                sender_uid=sender_uid,
                label=label,
                start_index=start_index,
                end_index=end_index,
            )
        )
        sender_offsets[sender_uid] = current_offset
        current_offset = end_index + 1 + neuron_gap

    return tracks, sender_offsets


def build_output_path(csv_path: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    return csv_path.with_suffix(".png")


def plot_raster(
    events: list[SpikeEvent],
    tracks: list[SenderTrack],
    sender_offsets: dict[str, int],
    output_path: Path,
    title: str,
    dpi: int,
    marker_size: float,
) -> None:
    import matplotlib.pyplot as plt

    x_values = [event.send_time for event in events]
    y_values = [sender_offsets[event.sender_uid] + event.neuron_index for event in events]

    max_time = max(x_values)
    max_y = max(y_values)

    fig_width = max(10.0, min(18.0, max_time / 40.0 + 4.0))
    fig_height = max(5.0, min(14.0, max_y / 60.0 + 3.0))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    ax.scatter(x_values, y_values, s=marker_size, c="black", marker="|", linewidths=0.7)

    for track in tracks:
        ax.axhline(track.start_index - 0.5, color="0.85", linewidth=0.8)
        y_center = (track.start_index + track.end_index) / 2.0
        ax.text(
            max_time + 1,
            y_center,
            track.label,
            va="center",
            ha="left",
            fontsize=8,
            color="0.35",
        )

    ax.axhline(tracks[-1].end_index + 0.5, color="0.85", linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("stacked neuron index")
    ax.set_xlim(-1, max_time + max(2, max_time * 0.08))
    ax.set_ylim(-1, max_y + 1)
    ax.grid(axis="x", color="0.92", linewidth=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    sender_filters = set(args.sender)
    events = load_events(args.csv_path, sender_filters)
    tracks, sender_offsets = build_tracks(events, args.neuron_gap)
    output_path = build_output_path(args.csv_path, args.output)
    plot_raster(
        events=events,
        tracks=tracks,
        sender_offsets=sender_offsets,
        output_path=output_path,
        title=args.title,
        dpi=args.dpi,
        marker_size=args.marker_size,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
