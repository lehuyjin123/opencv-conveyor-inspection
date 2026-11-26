# OpenCV Conveyor Inspection

A Python and OpenCV pipeline that tracks parts in a fixed inspection region, classifies them by color, size, and center-dot placement, and counts each part as it crosses a line.

## Run

Use Python 3.10 or newer and provide an input video.

```bash
python -m pip install -r requirements.txt
python conveyor_inspection.py --input conveyor.mp4 --output result.mp4 --roi 400 110 575 300 --line 480
```

Adjust the region and line to your camera framing. The command writes an annotated video, CSV and JSON summaries, and frame timings. Use `--no-threads` to compare sequential video I/O.

The original 640×360 classroom clip produced 12 counted parts: 6 PASS and 6 FAIL. See [example inspection results](result.csv).
