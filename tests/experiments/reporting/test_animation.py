from PIL import Image, ImageSequence

from experiments.plotting.animation import save_gif


def test_gif_follows_irregular_timestamps_instead_of_assuming_constant_frame_rate(tmp_path):
    images = [Image.new("RGB", (8, 8), color) for color in ("red", "green", "blue")]
    path = tmp_path / "timing.gif"
    save_gif(images, [2.0, 2.05, 2.2], path)
    with Image.open(path) as gif:
        assert [frame.info["duration"] for frame in ImageSequence.Iterator(gif)] == [50, 150, 500]
