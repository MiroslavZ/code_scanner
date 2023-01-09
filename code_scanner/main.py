from os import getenv
from json import loads
from pathlib import Path
from dotenv import load_dotenv
from logging import getLogger, basicConfig
from argparse import ArgumentParser, Namespace
from code_scanner.scanner.scanner import Scanner

logger = getLogger(__name__)

DIRS_TO_IGNORE = []
EXTENSIONS_TO_IGNORE = []
LOG_LEVEL = 'INFO'


def load_envs():
    load_dotenv()
    global LOG_LEVEL
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


def _arg_parse() -> Namespace:
    parser = ArgumentParser(
        prog="Code scanner",
        description="Сканер для поиска секретов в коде",
    )
    parser.add_argument(
        "-s",
        "--scan-dir",
        type=Path,
        help="Путь к директории сканируемого проекта",
    )
    return parser.parse_args()


def main():
    load_envs()
    args = _arg_parse()
    logger.debug(args)
    if args.scan_dir:
        scanner = Scanner(DIRS_TO_IGNORE, EXTENSIONS_TO_IGNORE, LOG_LEVEL)
        scanner.scan(args.scan_dir)
    else:
        logger.info('Не указана директория для сканирования')


if __name__ == '__main__':
    main()
