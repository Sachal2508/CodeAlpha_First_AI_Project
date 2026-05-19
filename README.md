# AI Python Demo Collection

Lightweight collection of AI demo scripts for translation, FAQ chatbot, MIDI music generation, and object detection — suitable for experimentation and local demos.

## Features
- Translation demo: `translation_tool.py`
- FAQ chatbot: `FAQ_chatbot.py`
- MIDI music generation: `music_generation.py` (sample output: `generated_music.mid`)
- Object detection demo using a YOLO model: `object_detection.py` (model file: `yolov8n.pt`)
- Sample MIDI files in the `midi_songs/` folder

## Quick Start
1. Ensure you have Python 3.8 or newer installed.
2. (Optional) Create and activate a virtual environment.
3. Install any required packages (see `requirements.txt` if present).
4. Run a script:

```bash
python translation_tool.py
python FAQ_chatbot.py
python music_generation.py
python object_detection.py
```

## Notes
- Example MIDI files are in `midi_songs/`.
- `yolov8n.pt` is included for the object detection demo; ensure compatible dependencies (Ultralytics/PyTorch) are installed before running `object_detection.py`.

If you want, I can also generate a `requirements.txt` with likely dependencies or add short usage examples for each script.
