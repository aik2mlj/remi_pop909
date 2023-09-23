from model import PopMusicTransformer
from datetime import datetime
import os
import argparse
import pickle
from finetune import load_split_file

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--len", help="the generation length")
    parser.add_argument("-n", default=1, help="how many sample to generate")
    parser.add_argument("--only-melody", action="store_true")
    parser.add_argument("--prompt", help="the prompt midi path")
    parser.add_argument("--prompt-chord", help="the chord of prompt midi path")
    args = parser.parse_args()

    chkpt_name = 'REMI-chord-melody' if args.only_melody else "REMI-chord"
    n_target_bar = int(args.len)

    # declare model
    model = PopMusicTransformer(
        checkpoint=chkpt_name,
        is_training=False)
    
    if args.prompt is None:
        # generate from scratch
        for _ in range(int(args.n)):
            model.generate(
                n_target_bar=n_target_bar,
                temperature=1.2,
                topk=5,
                output_path=f"./result/gen({chkpt_name})-{n_target_bar}bar_{datetime.now().strftime('%m-%d_%H%M%S')}.midi",
                prompt_paths=None)
    else:
        # generate continuation
        prompt_paths = {
            'midi_path': args.prompt,
            'melody_annotation_path': None,
            'chord_annotation_path': args.prompt_chord,
        }
        for _ in range(int(args.n)):
            model.generate(
                n_target_bar=n_target_bar,
                temperature=1.2,
                topk=5,
                output_path=f"./result/prompt_gen({chkpt_name})-{n_target_bar}bar_{datetime.now().strftime('%m-%d_%H%M%S')}.midi",
                prompt_paths=prompt_paths)
    
    # close model
    model.close()

if __name__ == '__main__':
    main()
