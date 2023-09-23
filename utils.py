import math
import numpy as np
import miditoolkit
import copy
import mir_eval

# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch)

# read notes from midi and shift all notes
def get_note_items(midi_path, melody_annotation_path, only_melody=False):
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)

    melody_note_items = [Item(name='Note', start=note.start, end=note.end, velocity=note.velocity, pitch=note.pitch) for note in midi_obj.instruments[0].notes]
    other_note_items = []
    if len(midi_obj.instruments) > 1:
        for inst in midi_obj.instruments[1:]:
            for note in inst.notes:
                other_note_items.append(Item(name='Note', start=note.start, end=note.end, velocity=note.velocity, pitch=note.pitch))
    if only_melody:
        note_items = melody_note_items
    else:
        note_items = melody_note_items + other_note_items
    note_items.sort(key=lambda x: (x.start, x.pitch))

    if melody_annotation_path is not None:
        with open(melody_annotation_path) as f:
            melody_annotation = f.read().splitlines()
        note_number, duration = map(int, melody_annotation[0].split())
        melody_start = 1  # Shift for an anacrusis
        if note_number == 0:
            melody_start += duration / DEFAULT_FRACTION  # Shift for offset of the melody's first note

        ticks_per_bar = DEFAULT_RESOLUTION * 4
        shift = int(melody_start * ticks_per_bar) - melody_note_items[0].start
        for note_item in note_items:
            note_item.start += shift
            note_item.end += shift

    return note_items

# quantize items
def quantize_items(items, ticks=120):
    # grid
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items      

# read chords from annotation
def get_chord_items(chord_annotation_path):
    with open(chord_annotation_path) as f:
        chord_annotation = f.read().splitlines()
    ticks_per_beat, ticks_per_bar = DEFAULT_RESOLUTION, DEFAULT_RESOLUTION * 4
    root_integration_table = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    chord_items = [Item(name='Chord', start=0, end=ticks_per_bar, velocity=None, pitch='N:N')]
    for element in chord_annotation:
        chord, *_, beat_duration = element.split()
        if chord.startswith('N'):
            chord = 'N:N'
        else:
            root, symbol = chord.split(':')
            if 'min' in symbol: symbol = 'min'
            elif 'maj' in symbol: symbol = 'maj'
            elif 'dim' in symbol: symbol = 'dim'
            elif 'aug' in symbol: symbol = 'aug'
            elif 'sus4' in symbol: symbol = 'sus4'
            elif 'sus2' in symbol: symbol = 'sus2'
            else: symbol = 'maj'  # 7, 9, ...
            root = root_integration_table.get(root, root)
            chord = f'{root}:{symbol}'
        start = chord_items[-1].end
        end = start + int(beat_duration) * ticks_per_beat
        if chord == chord_items[-1].pitch:
            chord_items[-1].end = end
        else:
            chord_items.append(Item(name='Chord', start=start, end=end, velocity=None, pitch=chord))
    return chord_items

# group items
def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION*4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups

# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)

# item to event
def item2event(groups):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event(
            name='Bar',
            time=None, 
            value=None,
            text='{}'.format(n_downbeat)))
        for item in groups[i][1:-1]:
            # position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.start))
            events.append(Event(
                name='Position', 
                time=item.start,
                value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                text='{}'.format(item.start)))
            if item.name == 'Note':
                # pitch
                events.append(Event(
                    name='Note On',
                    time=item.start, 
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                # duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                events.append(Event(
                    name='Note Duration',
                    time=item.start,
                    value=DEFAULT_DURATION_BINS[index] / 120,
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
            elif item.name == 'Chord':
                events.append(Event(
                    name='Chord', 
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
    return events

#############################################################################################
# WRITE MIDI
#############################################################################################
def word_to_event(words, word2event):
    events = []
    for word in words:
        event_name, event_value = word2event.get(word).split('_')
        events.append(Event(event_name, None, event_value, None))
    return events

def write_midi(words, word2event, output_path, prompt_path=None, write_chord=True):
    events = word_to_event(words, word2event)
    # get downbeat and note (no time)
    temp_notes = []
    temp_chords = []
    temp_tempos = []
    for i in range(len(events)-3):
        if events[i].name == 'Bar' and i > 0:
            temp_notes.append('Bar')
            temp_chords.append('Bar')
            temp_tempos.append('Bar')
        elif events[i].name == 'Position' and \
            events[i+1].name == 'Note On' and \
            events[i+2].name == 'Note Duration':
            # start time and end time from position
            position = int(events[i].value.split('/')[0]) - 1
            # pitch
            pitch = int(events[i+1].value)
            # duration
            duration = int(float(events[i+2].value) * 120)
            # adding
            temp_notes.append([position, pitch, duration])
        elif events[i].name == 'Position' and events[i+1].name == 'Chord':
            position = int(events[i].value.split('/')[0]) - 1
            temp_chords.append([position, events[i+1].value])
    # get specific time for notes
    ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = DEFAULT_RESOLUTION * 4 # assume 4/4
    notes = []
    current_bar = 0
    for note in temp_notes:
        if note == 'Bar':
            current_bar += 1
        else:
            position, pitch, duration = note
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            # duration (end time)
            et = st + duration
            notes.append(miditoolkit.Note(60, pitch, st, et))
    # get specific time for chords
    if len(temp_chords) > 0:
        chords = []
        current_bar = 0
        for chord in temp_chords:
            if chord == 'Bar':
                current_bar += 1
            else:
                position, value = chord
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                st = flags[position]
                chords.append([st, value])
    # get specific time for tempos
    tempos = []
    current_bar = 0
    for tempo in temp_tempos:
        if tempo == 'Bar':
            current_bar += 1
        else:
            position, value = tempo
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            tempos.append([int(st), value])
    # write
    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        #
        last_time = DEFAULT_RESOLUTION * 4 * 4
        # note shift
        for note in notes:
            note.start += last_time
            note.end += last_time
        midi.instruments[0].notes.extend(notes)
        # tempo changes
        temp_tempos = []
        for tempo in midi.tempo_changes:
            if tempo.time < DEFAULT_RESOLUTION*4*4:
                temp_tempos.append(tempo)
            else:
                break
        for st, bpm in tempos:
            st += last_time
            temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = temp_tempos
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]+last_time))
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        # write instrument
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)
        # write tempo
        tempo_changes = []
        for st, bpm in tempos:
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = tempo_changes
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))

        if len(temp_chords) > 0 and write_chord:
            inst_chd = miditoolkit.midi.containers.Instrument(0, is_drum=False)
            for i, c in enumerate(chords):
                if c[1] == "N:N": continue
                st = c[0]
                et = chords[i + 1][0] if i < len(chords) - 1 else notes[-1].end
                root, bitmap, bass = mir_eval.chord.encode(c[1])
                root = 48 + root  # pitch, 48 is c3
                for tone, bit in enumerate(bitmap):
                    if bit == 1:
                        inst_chd.notes.append(miditoolkit.Note(60, root + tone, st, et))
            midi.instruments.append(inst_chd)

    # write
    midi.dump(output_path)
