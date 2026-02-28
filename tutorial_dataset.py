import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        # with open('./training/fill50k/prompt.json', 'rt') as f:
        with open('./256x256_sort_train/prompt_train.json', 'rt') as f:
        # with open('./256x256_sort_test/prompt_test.json', 'rt') as f:
        # with open('./dataset/prompt_train.json', 'rt') as f:
        # with open('./test_traffic/prompt_train.json', 'rt') as f:
        # with open('./test_monti/prompt_train.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # source = cv2.imread('./training/fill50k/' + source_filename)
        # target = cv2.imread('./training/fill50k/' + target_filename)
        source = cv2.imread('./256x256_sort_train/' + source_filename)
        target = cv2.imread('./256x256_sort_train/' + target_filename)
        # source = cv2.imread('./256x256_sort_test/' + source_filename)
        # target = cv2.imread('./256x256_sort_test/' + target_filename)
        # source = cv2.imread('./dataset/' + source_filename)
        # target = cv2.imread('./dataset/' + target_filename)
        # source = cv2.imread('./test_traffic/' + source_filename)
        # target = cv2.imread('./test_traffic/' + target_filename)
        # source = cv2.imread('./test_monti/' + source_filename)
        # target = cv2.imread('./test_monti/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = np.repeat(source, 1, axis=0)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

class MyTestDataset(Dataset):
    def __init__(self):
        self.data = []
        # with open('./training/fill50k/prompt.json', 'rt') as f:
        # with open('./256x256_sort_train/prompt_train.json', 'rt') as f:
        with open('./256x256_sort_test/prompt_test.json', 'rt') as f:
        # with open('./dataset/prompt_train.json', 'rt') as f:
        # with open('./test_traffic/prompt_train.json', 'rt') as f:
        # with open('./test_monti/prompt_train.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # source = cv2.imread('./training/fill50k/' + source_filename)
        # target = cv2.imread('./training/fill50k/' + target_filename)
        # source = cv2.imread('./256x256_sort_train/' + source_filename)
        # target = cv2.imread('./256x256_sort_train/' + target_filename)
        source = cv2.imread('./256x256_sort_test/' + source_filename)
        target = cv2.imread('./256x256_sort_test/' + target_filename)
        # source = cv2.imread('./dataset/' + source_filename)
        # target = cv2.imread('./dataset/' + target_filename)
        # source = cv2.imread('./test_traffic/' + source_filename)
        # target = cv2.imread('./test_traffic/' + target_filename)
        # source = cv2.imread('./test_monti/' + source_filename)
        # target = cv2.imread('./test_monti/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = np.repeat(source, 1, axis=0)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)