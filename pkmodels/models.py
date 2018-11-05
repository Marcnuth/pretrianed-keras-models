import logging
import json
import io
from abc import ABC, abstractmethod
import zipfile
from pathlib import Path
from pkmodels.settings import DIR_MODELS, RAW_MODEL_NAME, RAW_METAS_NAME, ZIP_MODEL_PREFIX
from keras import models
from pkmodels import inputs


logger = logging.getLogger(__name__)


def exists_models():
    return {subdir.name:subdir for subdir in DIR_MODELS.iterdir()}


def exists_versions(model_dir):
    return {subdir.name:subdir for subdir in model_dir.iterdir()}


def unzip_model(model_path):
    # unzip model file and then load
    zips = sorted(list(filter(lambda x: x.name.startswith(ZIP_MODEL_PREFIX), model_path.iterdir())))
    buffer = io.BytesIO()
    for filename in zips:
        with open(filename.as_posix(), 'rb') as fr:
            buffer.write(fr.read())

    zf = zipfile.ZipFile(buffer)
    zf.extractall(model_path)


def load_model(model_path, model_version):

    # load latest model if model_version = None
    if model_version is None:
        model_version = sorted(list(Path(model_path).iterdir()))[-1]

    # fetch the metas
    raw_meta_file = Path(model_path) / model_version / RAW_METAS_NAME
    assert raw_meta_file.exists(), 'file:{} is not found'.format(raw_meta_file.as_posix())
    with open(raw_meta_file.absolute().as_posix(), 'r') as f:
        metas = json.loads(f.read())

    # directly load model if it is h5 file
    raw_model_file = Path(model_path) / model_version / RAW_MODEL_NAME
    if not raw_model_file.exists():
        unzip_model(raw_model_file.parent)

    assert raw_model_file.exists(), 'model does not exists, please check dir: {}'.format(raw_model_file.parent.as_posix())
    model = models.load_model(Path(raw_model_file).absolute().as_posix())
    return metas, model


class PKModel:
    def __init__(self, model_name, **kwargs):
        exists = exists_models()
        assert model_name in exists.keys(), 'model<{}> is not found, available models are {}.'.format(model_name, exists.keys())

        model_path = exists[model_name]
        version = kwargs.get('model_version',  None)
        versions = exists_versions(model_path)
        assert not version or version in versions, 'cannot find version<{}>'.format(version)

        self._model_name = model_name
        self._model_path = model_path
        self._model_version = version

        meta, model = load_model(self._model_path, self._model_version)
        self._model_meta = meta
        self._model = model

    def predict(self, **kwargs):
        kwargs.update(self._model_meta)
        input_tensor = getattr(inputs.Tensors, self._model_meta['input_tensor'])(**kwargs)

        probs = self._model.predict(input_tensor, verbose=2)
        index = probs.argmax(axis=-1)[0]
        return self._model_meta['classes'][index], probs.max()





