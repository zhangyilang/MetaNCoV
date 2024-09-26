# Modified from https://github.com/deepmind/leo/blob/master/data.py
# Copyright 2018 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Creates problem instances for LEO."""

import collections
import os
import pickle
import random

import enum
import numpy as np
import six
import torch

NDIM = 640


class StrEnum(enum.Enum):
    """An Enum represented by a string."""

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.__str__()


class MetaDataset(StrEnum):
    """Datasets supported by the DataProvider class."""
    MINI = "miniImageNet"
    TIERED = "tieredImageNet"


class EmbeddingCrop(StrEnum):
    """Embedding types supported by the DataProvider class."""
    CENTER = "center"
    MULTIVIEW = "multiview"


class MetaSplit(StrEnum):
    """Meta-datasets split supported by the DataProvider class."""
    TRAIN = "train"
    VALID = "val"
    TEST = "test"


class DataProvider(object):
    """Creates problem instances from a specific split and dataset."""

    def __init__(self, dataset_split, config):
        self._dataset_split = MetaSplit(dataset_split)
        self._config = config
        self._check_config()

        self._index_data(self._load_data())

    def _check_config(self):
        """Checks configuration arguments of constructor."""
        self._config["dataset_name"] = MetaDataset(self._config["dataset_name"])
        self._config["embedding_crop"] = EmbeddingCrop(self._config["embedding_crop"])
        if self._config["dataset_name"] == MetaDataset.TIERED:
            error_message = "embedding_crop: {} not supported for {}".format(
                self._config["embedding_crop"], self._config["dataset_name"])
            assert self._config["embedding_crop"] == EmbeddingCrop.CENTER, error_message

    def _load_data(self):
        """Loads data into memory and caches ."""
        with open(self._get_full_pickle_path(self._dataset_split), "rb") as f:
            raw_data = self._load(f)
        if self._dataset_split == MetaSplit.TRAIN and self._config["train_on_val"]:
            with open(self._get_full_pickle_path(MetaSplit.VALID), "rb") as f:
                valid_data = self._load(f)
            for key in valid_data:
                raw_data[key] = np.concatenate([raw_data[key],
                                                valid_data[key]], axis=0)

        return raw_data

    @staticmethod
    def _load(opened_file):
        result = pickle.load(opened_file)
        # if six.PY2:
        #     result = pickle.load(opened_file)
        # else:
        #     result = pickle.load(opened_file, encoding="latin1")  # pylint: disable=unexpected-keyword-arg
        return result

    def _index_data(self, raw_data):
        """Builds an index of images embeddings by class."""
        self._all_class_images = collections.OrderedDict()
        self._image_embedding = collections.OrderedDict()
        for i, k in enumerate(raw_data["keys"]):
            _, class_label, image_file = k.split("-") if self._config['embedding_crop'] \
                                                         is EmbeddingCrop.CENTER else k.split(b"-")
            image_file_class_label = image_file.split("_")[0] if self._config['embedding_crop'] \
                                                                 is EmbeddingCrop.CENTER else image_file.split(b"_")[0]
            assert class_label == image_file_class_label
            self._image_embedding[image_file] = raw_data["embeddings"][i]
            if class_label not in self._all_class_images:
                self._all_class_images[class_label] = []
            self._all_class_images[class_label].append(image_file)

        self._check_data_index(raw_data)

        self._all_class_images = collections.OrderedDict([
            (k, np.array(v)) for k, v in six.iteritems(self._all_class_images)
        ])

    def _check_data_index(self, raw_data):
        """Performs checks of the data index and image counts per class."""
        n = raw_data["keys"].shape[0]
        error_message = "{} != {}".format(len(self._image_embedding), n)
        assert len(self._image_embedding) == n, error_message
        error_message = "{} != {}".format(raw_data["embeddings"].shape[0], n)
        assert raw_data["embeddings"].shape[0] == n, error_message

        all_class_folders = list(self._all_class_images.keys())
        error_message = "no duplicate class names"
        assert len(set(all_class_folders)) == len(all_class_folders), error_message
        image_counts = set([len(class_images)
                            for class_images in self._all_class_images.values()])
        error_message = ("len(image_counts) should have at least one element but "
                         "is: {}").format(image_counts)
        assert len(image_counts) >= 1, error_message
        assert min(image_counts) > 0

    def _get_full_pickle_path(self, split_name):
        full_pickle_path = os.path.join(
            self._config["data_path"],
            str(self._config["dataset_name"]),
            str(self._config["embedding_crop"]),
            "{}_embeddings.pkl".format(split_name))

        return full_pickle_path

    def get_instance(self, num_classes, tr_size, val_size):
        """Samples a random N-way K-shot classification problem instance.

    Args:
      num_classes: N in N-way classification.
      tr_size: K in K-shot; number of training examples per class.
      val_size: number of validation examples per class.

    Returns:
      A tuple with 4 Tensors with the following shapes:
      - tr_input: (num_classes, tr_size, NDIM): training image embeddings.
      - tr_output: (num_classes, tr_size, 1): training image labels.
      - val_input: (num_classes, val_size, NDIM): validation image embeddings.
      - val_output: (num_classes, val_size, 1): validation image labels.
    """

        class_list = list(self._all_class_images.keys())
        sample_count = (tr_size + val_size)
        shuffled_folders = class_list[:]
        random.shuffle(shuffled_folders)
        shuffled_folders = shuffled_folders[:num_classes]
        error_message = "len(shuffled_folders) {} is not num_classes: {}".format(
            len(shuffled_folders), num_classes)
        assert len(shuffled_folders) == num_classes, error_message
        image_paths = []
        class_ids = []
        embeddings = self._image_embedding
        for class_id, class_name in enumerate(shuffled_folders):
            all_images = self._all_class_images[class_name]
            all_images = np.random.choice(all_images, sample_count, replace=False)
            error_message = "{} == {} failed".format(len(all_images), sample_count)
            assert len(all_images) == sample_count, error_message
            image_paths.append(all_images)
            class_ids.append([[class_id]] * sample_count)

        label_array = np.array(class_ids, dtype=np.int32)
        path_array = np.array(image_paths)
        embedding_array = np.array([[embeddings[image_path]
                                     for image_path in class_paths]
                                    for class_paths in path_array])

        embedding_array = torch.from_numpy(embedding_array)
        label_array = torch.from_numpy(label_array)
        embedding_array = torch.nn.functional.normalize(embedding_array, dim=-1)

        split_sizes = [tr_size, val_size]
        tr_input, val_input = torch.split(embedding_array, split_sizes, dim=1)
        tr_output, val_output = torch.split(label_array, split_sizes, dim=1)
        tr_input, val_input = tr_input.flatten(end_dim=1), val_input.flatten(end_dim=1)
        tr_output, val_output = tr_output.flatten(), val_output.flatten()

        return tr_input, tr_output, val_input, val_output

    def get_batch(self, batch_size, num_classes, tr_size, val_size):
        """Returns a batch of random N-way K-shot classification problem instances."""
        tr_inputs = []
        tr_labels = []
        val_inputs = []
        val_labels = []
        for _ in range(batch_size):
            tr_input, tr_label, val_input, val_label = self.get_instance(num_classes, tr_size, val_size)
            tr_inputs.append(tr_input), tr_labels.append(tr_label)
            val_inputs.append(val_input), val_labels.append(val_label)

        tr_inputs, tr_labels = torch.stack(tr_inputs), torch.stack(tr_labels).long()
        val_inputs, val_labels = torch.stack(val_inputs), torch.stack(val_labels).long()

        return {"train": (tr_inputs, tr_labels), "test": (val_inputs, val_labels)}


class LEOEmbMetaDataLoader:
    def __init__(self, dataset_split, dataset, batch_size, num_way, num_supp, num_qry, emb_crop='center'):
        config = {'dataset_name': dataset, 'embedding_crop': emb_crop,
                  'data_path': 'datasets/leo_embeddings', 'train_on_val': False}
        self.data_provider = DataProvider(dataset_split, config)
        self.batch_size = batch_size
        self.num_way = num_way
        self.num_supp = num_supp
        self.num_qry = num_qry

    def __iter__(self):
        return self

    def __next__(self):
        return self.data_provider.get_batch(self.batch_size, self.num_way, self.num_supp, self.num_qry)


if __name__ == '__main__':    # randomly shuffle the classes
    np.random.seed(0)

    for dataset, crop in [('miniImageNet', 'center'), ('miniImageNet', 'multiview'), ('tieredImageNet', 'center')]:
        leo_root = os.path.join('../datasets/embeddings', dataset, crop)

        data_in_cls = dict()
        for split in ['train', 'val', 'test']:
            with open(os.path.join(leo_root, f'{split}_embeddings.pkl'), 'rb') as f:
                raw_data = pickle.load(f, encoding='latin1')
                for embedding, file_name in zip(raw_data['embeddings'], raw_data['keys']):
                    cls = file_name.split('-')[1] if crop == 'center' else file_name.split(b'-')[1]
                    if cls not in data_in_cls:
                        data_in_cls[cls] = [[embedding], [file_name]]
                    else:
                        data_in_cls[cls][0].append(embedding)
                        data_in_cls[cls][1].append(file_name)

        classes = list(data_in_cls.keys())
        np.random.shuffle(classes)
        new_root = os.path.join('../datasets/leo_embeddings', dataset, crop)
        os.makedirs(new_root, exist_ok=True)

        counter = 0
        for split, split_percent in zip(['train', 'val', 'test'], [.64, .16, .20]):
            raw_data = {'embeddings': list(), 'keys': list()}
            num_labels_split = round(split_percent * len(classes))
            for cls in classes[counter:counter + num_labels_split]:
                raw_data['embeddings'] += data_in_cls[cls][0]
                raw_data['keys'] += data_in_cls[cls][1]

            raw_data['embeddings'] = np.stack(raw_data['embeddings'], axis=0)
            raw_data['keys'] = np.stack(raw_data['keys'], axis=0)

            with open(os.path.join(new_root, f'{split}_embeddings.pkl'), 'wb') as f:
                pickle.dump(raw_data, f)

            counter += num_labels_split
