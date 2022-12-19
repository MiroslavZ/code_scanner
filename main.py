import re
from os import walk, getenv
from json import loads
from pathlib import Path
from typing import Dict

import joblib
from dotenv import load_dotenv
from logging import basicConfig, getLogger

from numpy import ndarray, nan
from pandas import read_fwf, DataFrame
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from code_file import CodeFileInfo

logger = getLogger(__name__)

DIRS_TO_IGNORE = []
EXTENSIONS_TO_IGNORE = []
TOKENIZER: TfidfVectorizer = None
CLASSIFIER: LogisticRegression = None



def load_envs():
    load_dotenv()
    if getenv('LOG_LEVEL'):
        basicConfig(level=getenv('LOG_LEVEL'))
    global DIRS_TO_IGNORE
    if getenv('IGNORED_DIRS'):
        DIRS_TO_IGNORE = loads(getenv('IGNORED_DIRS'))
    else:
        logger.warning('Unable to load list of ignored dirs')
    global EXTENSIONS_TO_IGNORE
    if getenv('IGNORED_DIRS'):
        EXTENSIONS_TO_IGNORE = loads(getenv('IGNORED_EXTENSIONS'))
    else:
        logger.warning('Unable to load list of ignored extensions')


def load_models():
    global CLASSIFIER
    CLASSIFIER = joblib.load('pretrained/model_clf.pkl')
    global TOKENIZER
    TOKENIZER = joblib.load('pretrained/model_tfidf.pkl')



def setup():
    load_envs()
    load_models()


def find_files_to_scan(scanned_project_path: str) -> list[Path]:
    scan_path = Path(scanned_project_path)
    files_to_scan = []
    for root, dirs, files in walk(scan_path):
        is_ignored_dir = False
        for file in files:
            file_path = Path(root, file)
            # пропускаем лишние директории
            for directory in DIRS_TO_IGNORE:
                if directory in file_path.parts:
                    is_ignored_dir = True
                    break
            if is_ignored_dir:
                break
            # пропускаем файлы с ненужными расширениями
            if file_path.suffixes and file_path.suffixes[-1] in EXTENSIONS_TO_IGNORE:
                continue
            logger.debug(f'Loaded file to scan {file_path}')
            files_to_scan.append(file_path)
        if is_ignored_dir:
            continue
    return files_to_scan


def transform_file_to_df(path: Path) -> DataFrame:
    logger.debug(f'transforming file {path}')
    if path.exists() and path.is_file():
        # нужен более правильный способ читать файлы
        df = read_fwf(path, names=[])
        df.rename(columns={0: 'code'}, inplace=True)
        if df.shape[1] > 1:
            df.drop(df.columns[[1]], axis=1, inplace=True)
        return df
    else:
        logger.error(f'{path} not exists or is not a file')
        raise FileNotFoundError


def get_code_dfs(files: list[Path]) -> [CodeFileInfo]:
    return [CodeFileInfo(path=f, dataframe=transform_file_to_df(f)) for f in files]


def simple_preprocess(df: DataFrame) -> DataFrame:
    df.drop_duplicates(inplace=True)
    #df['code'].replace('', nan, inplace=True)
    #df.dropna(subset=['code'], inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns={'index'}, inplace=True)
    return df


def get_simple_preprocessed_dfs(code_infos: list[CodeFileInfo]) -> list[CodeFileInfo]:
    return [CodeFileInfo(code_info.path, simple_preprocess(code_info.df)) for code_info in code_infos]


def complex_preprocess(df: DataFrame) -> DataFrame:
    df_preprocessed = df.copy()
    df_preprocessed['code'] = df_preprocessed['code'].apply(lambda x: re.sub('[^a-zA-Zа-яА-я\d]', ' ', x))
    df_preprocessed['code'] = df_preprocessed['code'].apply(lambda x: x.lower())
    df_preprocessed['code'] = df_preprocessed['code'].apply(lambda x: re.sub(r'\b\w{1,2}\b', '', x))
    df_preprocessed['code'] = df_preprocessed['code'].apply(lambda x: re.sub(" +", " ", x))
    #df['code'] = df[df['code'].str.len() >= 3].reset_index().drop(columns={'index'})
    # токенизатор не переваривает nan
    df_preprocessed['code'].replace(' ', 'empty', inplace=True)
    #df.dropna(subset=['code'], inplace=True)
    return df_preprocessed


def get_complex_preprocessed_dfs(code_infos: list[CodeFileInfo]) -> list[CodeFileInfo]:
    return [CodeFileInfo(code_info.path, complex_preprocess(code_info.df)) for code_info in code_infos]


def get_sparse_matrices(code_infos: list[CodeFileInfo]) -> list[csr_matrix]:
    return [TOKENIZER.transform(code_info.df['code']) for code_info in code_infos]


def get_predictions(sparse_matrices: list[csr_matrix]) -> list[ndarray]:
    return [CLASSIFIER.predict(sparse_matrix) for sparse_matrix in sparse_matrices]


def combine_results(code_infos: list[CodeFileInfo], predictions: list[ndarray]):
    for code_prediction_pair in zip(code_infos, predictions):
        code_info = code_prediction_pair[0]
        prediction = code_prediction_pair[1]
        code_info.df['is_secret'] = prediction


def export_df_collections(collection: list[CodeFileInfo], prefix: str) -> None:
    for c in collection:
        c.export_df(prefix)


def log_report(code_infos: list[CodeFileInfo]) -> None:
    logger.info('Found secrets:')
    for code_info in code_infos:
        if code_info.df['is_secret'][code_info.df['is_secret'].isin([1])].empty:
            continue
        logger.info(f'file: {code_info.path.absolute()}')
        for i in range(code_info.df.shape[0]):
            series = code_info.df.iloc[i]
            if series['is_secret'] == 1:
                logger.info(f'\t{series["code"]}\t({series["is_secret"]})')


PASSWORD = 'C3RiN592?Ge^.e1'
if __name__ == '__main__':
    setup()
    files_to_scan = find_files_to_scan('.')
    logger.debug(files_to_scan)
    code_infos = get_code_dfs(files_to_scan)
    code_infos_simple = get_simple_preprocessed_dfs(code_infos)
    export_df_collections(code_infos_simple, 'simple_prep')
    code_infos_complex = get_complex_preprocessed_dfs(code_infos_simple)
    export_df_collections(code_infos_complex, 'complex_prep')
    matrices = get_sparse_matrices(code_infos_complex)
    predictions = get_predictions(matrices)
    combine_results(code_infos_simple, predictions)
    export_df_collections(code_infos_simple, 'predicted')
    log_report(code_infos_simple)
