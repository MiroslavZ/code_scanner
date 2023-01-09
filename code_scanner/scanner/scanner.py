import re
from os import walk
from pathlib import Path

import joblib
from logging import basicConfig, getLogger

from numpy import ndarray
from pandas import read_fwf, DataFrame
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from code_scanner.code_file import CodeFileInfo

logger = getLogger(__name__)


class Scanner:
    def __init__(self, ignored_dirs=None, ignored_extensions=None, log_level='INFO'):
        if ignored_extensions is None:
            ignored_extensions = []
        if ignored_dirs is None:
            ignored_dirs = []
        self.ignored_dirs = ignored_dirs
        self.ignored_extensions = ignored_extensions
        self.log_level = log_level

        self._configure_logger(self.log_level)
        self._load_models()

    def _configure_logger(self, log_level='INFO'):
        basicConfig(level=log_level)

    def _load_models(self):
        self._classifier: LogisticRegression = joblib.load('code_scanner/pretrained/model_clf.pkl')
        self._vectorizer: TfidfVectorizer = joblib.load('code_scanner/pretrained/model_tfidf.pkl')

    def find_files_to_scan(self, scanned_project_path: Path) -> list[Path]:
        if not scanned_project_path.exists():
            raise
        logger.info(scanned_project_path.resolve())
        logger.info(Path.cwd())
        files_to_scan = []
        for root, dirs, files in walk(scanned_project_path):
            is_ignored_dir = False
            for file in files:
                file_path = Path(root, file)
                # пропускаем лишние директории
                for directory in self.ignored_dirs:
                    if directory in file_path.parts:
                        is_ignored_dir = True
                        break
                if is_ignored_dir:
                    break
                # пропускаем файлы с ненужными расширениями
                if file_path.suffixes and file_path.suffixes[-1] in self.ignored_extensions:
                    continue
                # пропускаем пустые файлы
                if file_path.stat().st_size == 0:
                    logger.debug(f'Skipping empty file')
                    continue
                logger.debug(f'Loaded file to scan {file_path}')
                files_to_scan.append(file_path)
            if is_ignored_dir:
                continue
        return files_to_scan

    def transform_file_to_df(self, path: Path) -> DataFrame:
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

    def get_code_dfs(self, files: list[Path]) -> [CodeFileInfo]:
        return [CodeFileInfo(path=f, dataframe=self.transform_file_to_df(f)) for f in files]

    def simple_preprocess(self, df: DataFrame) -> DataFrame:
        df.drop_duplicates(inplace=True)
        df.reset_index(inplace=True)
        df.drop(columns={'index'}, inplace=True)
        return df

    def get_simple_preprocessed_dfs(self, code_infos: list[CodeFileInfo]) -> list[CodeFileInfo]:
        return [CodeFileInfo(code_info.path, self.simple_preprocess(code_info.df)) for code_info in code_infos]

    def complex_preprocess(self, df: DataFrame) -> DataFrame:
        df_preprocessed = df.copy()
        df_preprocessed['code'] = df_preprocessed['code'].apply(lambda x: re.sub('[^a-zA-Zа-яА-я\d]', ' ', x))
        df_preprocessed['code'] = df_preprocessed['code'].apply(lambda x: x.lower())
        df_preprocessed['code'] = df_preprocessed['code'].apply(lambda x: re.sub(r'\b\w{1,2}\b', '', x))
        df_preprocessed['code'] = df_preprocessed['code'].apply(lambda x: re.sub(" +", " ", x))
        df_preprocessed['code'].replace(' ', 'empty', inplace=True)
        return df_preprocessed

    def get_complex_preprocessed_dfs(self, code_infos: list[CodeFileInfo]) -> list[CodeFileInfo]:
        return [CodeFileInfo(code_info.path, self.complex_preprocess(code_info.df)) for code_info in code_infos]

    def get_sparse_matrices(self, code_infos: list[CodeFileInfo]) -> list[csr_matrix]:
        return [self._vectorizer.transform(code_info.df['code']) for code_info in code_infos]

    def get_predictions(self, sparse_matrices: list[csr_matrix]) -> list[ndarray]:
        return [self._classifier.predict(sparse_matrix) for sparse_matrix in sparse_matrices]

    def combine_results(self, code_infos: list[CodeFileInfo], predictions: list[ndarray]):
        for code_prediction_pair in zip(code_infos, predictions):
            code_info = code_prediction_pair[0]
            prediction = code_prediction_pair[1]
            code_info.df['is_secret'] = prediction

    def export_df_collections(self, collection: list[CodeFileInfo], prefix: str) -> None:
        for c in collection:
            c.export_df(prefix)

    def log_report(self, code_infos: list[CodeFileInfo]) -> None:
        logger.info('Found secrets:')
        for code_info in code_infos:
            if code_info.df['is_secret'][code_info.df['is_secret'].isin([1])].empty:
                continue
            logger.info(f'file: {code_info.path.absolute()}')
            for i in range(code_info.df.shape[0]):
                series = code_info.df.iloc[i]
                if series['is_secret'] == 1:
                    logger.info(f'\t{series["code"]}\t({series["is_secret"]})')

    def scan(self, directory: Path):
        files_to_scan = self.find_files_to_scan(directory)
        logger.debug(files_to_scan)
        code_infos = self.get_code_dfs(files_to_scan)
        code_infos_simple = self.get_simple_preprocessed_dfs(code_infos)
        # export_df_collections(code_infos_simple, 'simple_prep')
        code_infos_complex = self.get_complex_preprocessed_dfs(code_infos_simple)
        # export_df_collections(code_infos_complex, 'complex_prep')
        matrices = self.get_sparse_matrices(code_infos_complex)
        predictions = self.get_predictions(matrices)
        self.combine_results(code_infos_simple, predictions)
        # export_df_collections(code_infos_simple, 'predicted')
        self.log_report(code_infos_simple)

if __name__ == '__main__':
    Scanner()._load_models()