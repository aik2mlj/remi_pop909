from model import PopMusicTransformer
from datetime import datetime
import os
import argparse
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
    model.generate(
        n_target_bar=16,
        temperature=1.2,
        topk=5,
        output_path='./result/continuation.midi',
        prompt='./data/evaluation/000.midi')
    
    # close model
    model.close()

if __name__ == '__main__':
    main()
