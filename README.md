# Code scanner

Пакет для поиска секретов в исходном коде

Варианты использования сканера:
- Локальный запуск сканера;
- Интеграция сканера в собственный репозиторий через github actions;

## Локальный запуск сканера

1. Загрузка сканера с github `git clone https://github.com/MiroslavZ/code_scanner.git`
2. Установка в качестве пакета `pip install -e PATH/TO/SCANNER/DIR`
3. Запуск сканера с указанием директории для сканирования `code_scanner --scan-dir PATH/TO/SCANNED/PROJECT`

## Интеграция через github action

Скопируйте код данного action в свой репозиторий:

```buildoutcfg
name: Code checks

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  build:
    name: Code scanner
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v2
      - name: Creating scanned area
        run: |
          mkdir scanned_area
          ls | grep -v scanned_area | xargs mv -t scanned_area
      - name: Downloading code_scanner
          git clone https://github.com/MiroslavZ/code_scanner.git
      - name: Installing code_scanner
        run: |
          pip install -e /home/runner/work/${{github.event.repository.name}}/${{github.event.repository.name}}/code_scanner
      - name: Scanning the repo
          code-scanner --scan-dir /home/runner/work/${{github.event.repository.name}}/${{github.event.repository.name}}/scanned_area
```