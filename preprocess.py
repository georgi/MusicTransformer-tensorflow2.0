import utils
import os
import pickle
import magenta.music as mm
import note_seq
import pretty_midi
from magenta.models.score2perf.music_encoders import MidiPerformanceEncoder
from note_seq.sequences_lib import (
    quantize_note_sequence_absolute,
    stretch_note_sequence,
    transpose_note_sequence,
    apply_sustain_control_changes,
    concatenate_sequences
)

NUM_VELOCITY_BINS = 32
STEPS_PER_SECOND = 100
MIN_PITCH = 21
MAX_PITCH = 108


class MidiEncoder:
    
    def __init__(self):
        self.vocab_size = midi_encoder.vocab_size
        self.midi_encoder = (
            steps_per_second=STEPS_PER_SECOND,
            num_velocity_bins=NUM_VELOCITY_BINS,
            min_pitch=MIN_PITCH,
            max_pitch=MAX_PITCH,
        )


    def encode_note_sequence(self, ns):
        performance = note_seq.Performance(
            note_seq.quantize_note_sequence_absolute(ns, self.midi_encoder._steps_per_second),
            num_velocity_bins=self.midi_encoder._num_velocity_bins)

        event_ids = [self.midi_encoder._encoding.encode_event(event) + 
                     self.midi_encoder.num_reserved_ids
                     for event in performance]

        # Greedily encode performance event n-grams as new indices.
        ids = []
        j = 0

        while j < len(event_ids):
            ngram = ()
            best_ngram = None
            for i in range(j, len(event_ids)):
                ngram += (event_ids[i],)
                if self.midi_encoder._ngrams_trie.has_key(ngram):
                    best_ngram = ngram
                if not self.midi_encoder._ngrams_trie.has_subtrie(ngram):
                    break
            if best_ngram is not None:
                ids.append(self.midi_encoder._ngrams_trie[best_ngram])
                j += len(best_ngram)
            else:
                j += 1

        if self.midi_encoder._add_eos:
            ids.append(text_encoder.EOS_ID)

        return ids


    def decode_ids(self, ids):
        event_ids = []
        for i in ids:
            if i >= self.midi_encoder.unigram_vocab_size:
                event_ids += self.midi_encoder._ngrams[i - self.midi_encoder.unigram_vocab_size]
            else:
                event_ids.append(i)

        performance = note_seq.Performance(
            quantized_sequence=None,
            steps_per_second=self.midi_encoder._steps_per_second,
            num_velocity_bins=self.midi_encoder._num_velocity_bins
        )
        for i in event_ids:
            if i >= self.midi_encoder.num_reserved_ids:
                performance.append(self.midi_encoder._encoding.decode_event(i - self.midi_encoder.num_reserved_ids))

        return performance.to_sequence()


def convert_midi_to_proto(self, path):
    midi = pretty_midi.PrettyMIDI(path)
    for i, inst in enumerate(midi.instruments):
        num_distinct_pitches = sum([i > 5 for i in inst.get_pitch_class_histogram()])
        if inst.is_drum or num_distinct_pitches < 5 or len(inst.notes) < 30:
            midi.instruments.remove(inst)
    ns = mm.midi_to_note_sequence(midi)
    ns = apply_sustain_control_changes(ns)
    del ns.control_changes[:]
    out_file = os.path.join(data_dir, os.path.basename(path)) + '.pb'
    with open(out_file, 'wb') as f:
        f.write(ns.SerializeToString())