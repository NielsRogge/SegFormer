from argparse import ArgumentParser
import torch
from mmseg.apis import init_segmentor, inference_segmentor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')

    args = parser.parse_args()

    # define model based on config file and checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=device)
    # forward pass
    data = dict()
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    print(result.shape)

if __name__ == '__main__':
    main()

