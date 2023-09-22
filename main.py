from model import PopMusicTransformer
from datetime import datetime
import os
import argparse
import pickle
from finetune import load_split_file

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-melody", action="store_true")
    args = parser.parse_args()

    chkpt_name = 'REMI-chord-melody' if args.only_melody else "REMI-chord"

    # declare model
    model = PopMusicTransformer(
        checkpoint=chkpt_name,
        is_training=False)
    
    # generate from scratch
    model.generate(
        n_target_bar=16,
        temperature=1.2,
        topk=5,
        output_path=f"./result/gen({chkpt_name})_{datetime.now().strftime('%m-%d_%H%M')}.midi",
        prompt=None)
    
    # generate continuation
    train_inds, valid_inds = load_split_file("./split.npz")
    train_inds += 1
    valid_inds += 1
    print(valid_inds)
    num = int(input("choose one from pop909:"))
    prompt_song = valid_inds[num]
    model.generate(
        n_target_bar=16,
        temperature=1.2,
        topk=5,
        output_path=f"./result/prompt_gen({chkpt_name})_{datetime.now().strftime('%m-%d_%H%M')}.midi",
        prompt=f'./POP909-Dataset/POP909/{prompt_song:03}/{prompt_song:03}.mid')
    
    # close model
    model.close()

if __name__ == '__main__':
    main()
