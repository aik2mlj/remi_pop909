from model import PopMusicTransformer
import pickle
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_split_file(split_fn):
    split_data = np.load(split_fn)
    train_inds = split_data['train_inds']
    valid_inds = split_data['valid_inds']
    return train_inds, valid_inds

def main():
    # declare model
    model = PopMusicTransformer(
        checkpoint='chord',
        is_training=True)
    # prepare data
    unused_pieces = [
        6, 18, 23, 34, 46, 56, 62, 63, 68, 79, 80, 88, 98, 102, 107, 123, 140, 152, 158, 171, 173, 176, 194, 196, 203, 208, 215, 224, 225, 229, 231, 236, 237, 251, 254, 255, 271, 278, 279, 280, 289,
        307, 310, 311, 316, 321, 322, 324, 328, 331, 333, 338, 341, 348, 350, 354, 355, 360, 369, 370, 379, 388, 389, 390, 391, 393, 394, 400, 412, 448, 449, 454, 455, 456, 457, 458, 464, 471, 474,
        487, 489, 506, 509, 511, 522, 531, 533, 549, 563, 584, 586, 587, 592, 609, 624, 629, 632, 633, 653, 654, 662, 665, 667, 675, 678, 689, 693, 714, 727, 733, 741, 744, 746, 748, 749, 756, 764,
        770, 771, 775, 779, 786, 787, 788, 791, 797, 799, 800, 801, 802, 803, 804, 806, 807, 818, 843, 869, 872, 883, 884, 887, 888, 897, 899, 900, 905
    ]
    train_inds, valid_inds = load_split_file("./split.npz")
    train_inds += 1
    valid_inds += 1
    print(len(train_inds))
    paths = [{
        'midi_path': f"POP909-Dataset/POP909/{i:03}/{i:03}.mid",
        'melody_annotation_path': f"hierarchical-structure-analysis/POP909/{i:03}/melody.txt",
        'chord_annotation_path': f"hierarchical-structure-analysis/POP909/{i:03}/finalized_chord.txt",
        'phrase_annotation_path': f"hierarchical-structure-analysis/POP909/{i:03}/human_label1.txt",
    } for i in train_inds if i not in unused_pieces]
    training_data, dictionary = model.prepare_data(paths)

    # check output checkpoint folder
    ####################################
    # if you use "REMI-tempo-chord-checkpoint" for the pre-trained checkpoint
    # please name your output folder as something with "chord"
    # for example: my-love-chord, cute-doggy-chord, ...
    # if use "REMI-tempo-checkpoint"
    # for example: my-love, cute-doggy, ...
    ####################################
    output_checkpoint_folder = 'REMI-chord' # your decision
    if not os.path.exists(output_checkpoint_folder):
        os.mkdir(output_checkpoint_folder)
    
    # save dictionary
    pickle.dump(dictionary, open(f'{output_checkpoint_folder}/dictionary.pkl', 'wb'))

    # finetune
    model.finetune(
        training_data=training_data,
        output_checkpoint_folder=output_checkpoint_folder)

    ####################################
    # after finetuning, please choose which checkpoint you want to try
    # and change the checkpoint names you choose into "model"
    # and copy the "dictionary.pkl" into the your output_checkpoint_folder
    # ***** the same as the content format in "REMI-tempo-checkpoint" *****
    # and then, you can use "main.py" to generate your own music!
    # (do not forget to revise the checkpoint path to your own in "main.py")
    ####################################

    # close
    model.close()

if __name__ == '__main__':
    main()
