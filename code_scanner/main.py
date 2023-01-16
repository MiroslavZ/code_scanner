from os import getenv
from json import loads
from pathlib import Path
from dotenv import load_dotenv
from logging import getLogger, basicConfig
from argparse import ArgumentParser, Namespace
from code_scanner.scanner.scanner import Scanner

logger = getLogger(__name__)


def configure_logging():
    if getenv('LOG_LEVEL'):
        basicConfig(level=getenv('LOG_LEVEL'))


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
    load_dotenv()
    configure_logging()
    args = _arg_parse()
    logger.debug(args)
    if args.scan_dir:
        ignored_dirs = loads(getenv('IGNORED_DIRS'))
        ignored_extensions = loads(getenv('IGNORED_EXTENSIONS'))
        scanner = Scanner(ignored_dirs, ignored_extensions, getenv('GITHUB_TOKEN'))
        scanner.scan(args.scan_dir)
    else:
        logger.info('Не указана директория для сканирования')


if __name__ == '__main__':
    main()
