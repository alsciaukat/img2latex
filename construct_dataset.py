# %%
from pandas import read_csv
from torch import Tensor, tensor
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Pad
from torchvision.transforms.functional import invert

from typing import Callable
from os.path import join
from math import floor, ceil

from lexicons import *

import matplotlib.pyplot as plt

INPUT_TENSOR_HEIGHT = 512
INPUT_TENSOR_WIDTH = 1024

class TeXExpressionDataset(Dataset):
    def __init__(self, record_file: str, image_directory: str, transform: Callable[[Tensor], Tensor] = None, embed=True, test=False):
        self.test = test
        if test:
            self.records = read_csv(record_file, sep="\t", header=None, index_col=0, skiprows=50_000, nrows=10_000)
        else:
            self.records = read_csv(record_file, sep="\t", header=None, index_col=0, nrows=50_000)
        self.image_directory = image_directory
        self.transform = transform
        self.embed = embed

    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx) -> tuple[Tensor, tuple[str]]:
        if not isinstance(idx, int):
            raise TypeError("index must be an integer")
        if self.test:
            idx += 50_000
        # the maximum size of the image is 812 x 765
        image = read_image(join(self.image_directory, f"{idx}.png"), ImageReadMode.GRAY)
        if self.transform:
            image = self.transform(image)
        else:
            image = invert(image)
            *_, height, width = image.shape
            # there's no point handling the case where the image is larger than desired output.
            padding_height = INPUT_TENSOR_HEIGHT - height
            padding_width = INPUT_TENSOR_WIDTH - width
            add_padding = Pad( (floor(padding_width/2), ceil(padding_height/2), ceil(padding_width/2), floor(padding_height/2)) )
            image = add_padding(image)
        tex_expression: str = self.records.at[idx, 1]
        
        tokens = []
        tex_expression = tex_expression.lstrip()

        while tex_expression:
            previous_expression = tex_expression
            for i, token in enumerate(TOKENS):
                if tex_expression.startswith(token):
                    tex_expression = tex_expression.removeprefix(token)
                    tex_expression = tex_expression.lstrip()
                    if self.embed:
                        tokens.append(i)
                    else:
                        tokens.append(token)
            if previous_expression == tex_expression:
                raise ValueError("Not supported TeX expression")

        if self.embed:
            tokens.extend([ALL_TOKENS.index("EOL")]*(40-len(tokens)))
        return image, tensor(tokens)

    

# %%
if __name__ == "__main__":
    dataset = TeXExpressionDataset("./generated/record.tsv", "./generated/", test=True)

    print(max(len(dataset[i][1]) for i in range(10_000)))



    # image, tex_tokens = dataset[3029]
    # print(image.shape)
    # print(list(map(lambda index: ALL_TOKENS[index], tex_tokens)))
    # plt.imshow(image[0,:,:])
    # plt.show()

# %%
