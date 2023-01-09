from pathlib import Path
from scipy.sparse.csr import csr_matrix
from pandas import DataFrame


class CodeFileInfo:
    '''
    Класс для хранения информации об рассматриваемом файле
    :param path: Путь к сканируемому файлу
    :param dataframe: Датафрейм, полученный из файла после обработки
    '''

    def __init__(self, path: Path, dataframe: DataFrame):
        self.path = path
        self.df = dataframe

    def export_df(self, prefix: str) -> None:
        '''
        Метод для сохранения датафрейма с кодом, для отладки
        :param prefix: префикс перед именем файла
        '''
        if self.df is not None:
            self.df.to_excel(f'{prefix}_{self.path.name}.xlsx')
