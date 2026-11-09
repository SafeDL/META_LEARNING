"""Vehicle geometry and real-time GIF output for recorded HighwayEnv trajectories."""

import math

import numpy as np
from PIL import Image


def body_vertices(actor, origin):
    c, s = math.cos(actor["heading"]), math.sin(actor["heading"])
    vertices = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    vertices *= [actor["length"] / 2, actor["width"] / 2]
    return vertices @ np.array([[c, s], [-s, c]]) + [actor["x"] - origin, actor["y"]]


def capture_frame(figure):
    figure.canvas.draw()
    # Store indexed images, not an RGB copy of the entire animation in memory.
    return Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3]).quantize(
        colors=256, dither=Image.Dither.NONE
    )


def save_gif(images, times, destination):
    """Keep every measured frame and its timestamp; pause only at the final frame."""
    times = np.asarray(times, dtype=float)
    if not images or len(images) != len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("GIF frames require aligned, increasing trajectory timestamps")
    # GIF delays are in centiseconds. Quantize timestamps before taking differences
    # so rounding does not accumulate throughout the recording.
    ticks = np.rint((times - times[0]) * 100).astype(int)
    durations = (np.diff(ticks) * 10).tolist() + [500]
    if any(duration < 20 for duration in durations):
        raise ValueError("Trajectory frames are faster than the supported 50 FPS GIF playback")
    images[0].save(
        destination,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=durations,
        optimize=False,
        disposal=2,
    )
    return {
        "frames": len(images),
        "playback_speed": 1.0,
        "frame_interval_ms": durations[:-1],
        "final_hold_ms": durations[-1],
    }
