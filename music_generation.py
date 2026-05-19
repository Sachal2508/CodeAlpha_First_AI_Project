
import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from music21 import converter, instrument, note, chord, stream

# ============================================================
# STEP 1: LOAD AND PARSE MIDI FILES
# ============================================================
def load_notes(midi_folder="midi_songs"):
    """
    Reads all .mid files in a folder.
    Extracts each note/chord as a string.
    Returns a list of note strings.
    """
    all_notes = []

    files = glob.glob(os.path.join(midi_folder, "*.mid"))
    print(f"Found {len(files)} MIDI files.")

    for file in files:
        print(f"  Parsing: {file}")
        midi = converter.parse(file)

        parts = instrument.partitionByInstrument(midi)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                all_notes.append(str(element.pitch))           # e.g. "C4"
            elif isinstance(element, chord.Chord):
                all_notes.append('.'.join(str(n) for n in element.normalOrder))

    print(f"Total notes extracted: {len(all_notes)}")
    return all_notes

# ============================================================
# STEP 2: PREPARE DATA FOR THE LSTM
# ============================================================
def prepare_sequences(notes, sequence_length=50):
    """
    Creates input-output pairs.
    Input : a sequence of `sequence_length` notes
    Output: the next note after that sequence
    """
    unique_notes = sorted(set(notes))
    num_unique   = len(unique_notes)

    note_to_int = {n: i for i, n in enumerate(unique_notes)}
    int_to_note = {i: n for i, n in enumerate(unique_notes)}

    X, y = [], []
    for i in range(len(notes) - sequence_length):
        seq_in  = notes[i : i + sequence_length]
        seq_out = notes[i + sequence_length]
        X.append([note_to_int[n] for n in seq_in])
        y.append(note_to_int[seq_out])

    # Shape: (samples, timesteps, 1) — normalised to [0, 1]
    X = np.reshape(X, (len(X), sequence_length, 1)) / float(num_unique)
    y = np.array(y)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return X_tensor, y_tensor, note_to_int, int_to_note, num_unique

# ============================================================
# STEP 3: BUILD THE LSTM MODEL
# ============================================================
class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(MusicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc1     = nn.Linear(hidden_size, 128)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # Take the last timestep
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out)         # Raw logits (CrossEntropyLoss handles softmax)

def build_model(sequence_length, num_unique, device):
    model = MusicLSTM(
        input_size=1,
        hidden_size=256,
        num_layers=2,
        num_classes=num_unique,
        dropout=0.3
    ).to(device)
    print(model)
    return model

# ============================================================
# STEP 4: TRAIN THE MODEL
# ============================================================
def train_model(model, X, y, epochs=50, batch_size=64, device="cpu"):
    dataset    = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}]  Loss: {avg_loss:.4f}")

# ============================================================
# STEP 5: GENERATE NEW MUSIC
# ============================================================
def generate_music(model, notes, note_to_int, int_to_note,
                   num_unique, sequence_length=50, num_notes=100, device="cpu"):
    """
    Uses the trained model to generate `num_notes` new notes.
    Starts with a random seed from the training data.
    """
    model.eval()
    start   = random.randint(0, len(notes) - sequence_length - 1)
    pattern = [note_to_int[n] for n in notes[start: start + sequence_length]]

    generated_notes = []
    with torch.no_grad():
        for _ in range(num_notes):
            x = np.reshape(pattern, (1, sequence_length, 1)) / float(num_unique)
            x = torch.tensor(x, dtype=torch.float32).to(device)

            output     = model(x)
            next_index = torch.argmax(output, dim=1).item()
            next_note  = int_to_note[next_index]

            generated_notes.append(next_note)
            pattern.append(next_index)
            pattern = pattern[1:]   # Slide the window

    return generated_notes

# ============================================================
# STEP 6: SAVE GENERATED NOTES AS MIDI FILE
# ============================================================
def save_as_midi(generated_notes, output_file="generated_music.mid"):
    output_notes = []
    offset = 0

    for element in generated_notes:
        if '.' in element:          # It's a chord
            notes_in_chord = element.split('.')
            chord_notes    = [note.Note(int(n)) for n in notes_in_chord]
            new_chord          = chord.Chord(chord_notes)
            new_chord.offset   = offset
            output_notes.append(new_chord)
        else:                       # It's a single note
            new_note                  = note.Note(element)
            new_note.offset           = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5               # Each note is 0.5 beats apart

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)
    print(f"Music saved to: {output_file}")

# ============================================================
# MAIN — RUN EVERYTHING
# ============================================================
if __name__ == "__main__":
    MIDI_FOLDER     = "midi_songs"
    SEQUENCE_LENGTH = 50
    EPOCHS          = 50
    BATCH_SIZE      = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Make sure the folder exists ---
    if not os.path.exists(MIDI_FOLDER):
        os.makedirs(MIDI_FOLDER)
        print(f"Created folder '{MIDI_FOLDER}'.")
        print("Please put some .mid files in it and run again.")
        print("Free MIDI files: https://www.midiworld.com/")
        exit()

    # Step 1: Load notes
    notes = load_notes(MIDI_FOLDER)
    if len(notes) < SEQUENCE_LENGTH + 1:
        print("Not enough notes. Add more MIDI files.")
        exit()

    # Step 2: Prepare sequences
    X, y, note_to_int, int_to_note, num_unique = prepare_sequences(notes, SEQUENCE_LENGTH)
    print(f"Unique notes: {num_unique},  Training samples: {len(X)}")

    # Step 3: Build model
    model = build_model(SEQUENCE_LENGTH, num_unique, device)

    # Step 4: Train
    print("Training model... (this may take a while)")
    train_model(model, X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, device=device)

    # Step 5: Generate
    print("Generating music...")
    generated = generate_music(model, notes, note_to_int, int_to_note,
                                num_unique, SEQUENCE_LENGTH, num_notes=100, device=device)

    # Step 6: Save
    save_as_midi(generated, "generated_music.mid")
    print("Done! Open 'generated_music.mid' in any media player.")