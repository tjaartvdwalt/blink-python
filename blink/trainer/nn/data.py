from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DataConfig:
    data_root: str = "."
    num_workers: int = 1
    batch_size: int = 16
    num_classes = 2


class BlinkDataset(torch.utils.data.Dataset):
    def __init__(self, blink_ear_file, non_blink_ear_file):
        super().__init__()

        self.ears = []
        blink_count = 0
        with open(blink_ear_file) as f:
            lines = f.readlines()
            blink_count = len(lines)


        for idx, blink_file in enumerate([non_blink_ear_file, blink_ear_file]):
            with open(blink_file) as f:
                lines = f.readlines()
                for j, line in enumerate(lines):
                    if j > blink_count:
                        break

                    str_line = line.split()
                    elements = torch.tensor([float(i) for i in str_line])
                    classes = torch.nn.functional.one_hot(torch.tensor(idx), num_classes=2)
                    # print(classes)
                    # elements.append(classes)
                    self.ears.append([elements, classes])

        # print(ears)
        # self.ears = ears

    def __getitem__(self, idx):
        ear = self.ears[idx]
        input = ear[0]
        label = ear[1].type(torch.float32)

        return input, label

    def __len__(self):
        return len(self.ears)
