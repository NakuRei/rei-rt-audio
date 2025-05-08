# rei-rt-audio

sounddeviceを用いたリアルタイム音響信号処理のためのライブラリ。

## Requirements

オーディオデバイスを使う都合上、仮想環境は推奨されない。

Python 3.12で開発。依存関係は`pyproject.toml`および`requirements.txt`を参照のこと。

## Installation

```sh
pip install git+https://github.com/NakuRei/rei-rt-audio@main
```

## Development

Windowsの場合は下記で開発環境を構築できる。

```sh
py -m venv .venv
source ./.venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
```

## Test

テストに必要なパッケージは別途`requirements_test.txt`に書いている。テストを実行する場合はインストールする。

```sh
pip install -r requirements_test.txt
```

テストは`pytest`で実行できる。

```sh
pytest
```

カバレッジも取得する場合は次を実行する。

```sh
pytest --cov=src
```

[Coverage Gutters](https://marketplace.visualstudio.com/items/?itemName=ryanluker.vscode-coverage-gutters)を使う場合は、次も実行する。

```sh
coverage lcov -o lcov.info
```

## Document

ドキュメント生成に必要なパッケージは別途`requirements_docs.txt`に書いている。ローカルでドキュメントを生成する場合はインストールする。

```sh
pip install -r requirements_docstxt
```

Windowsの場合、ドキュメントは下記コマンドで生成できる。

```cmd
del docs\source\modules\*.rst && docs\make.bat html
```

生成された`docs/build/html/index.html`を開く。

## Author

- [NakuRei](https://github.com/NakuRei)

## License

(c) 2025 NakuRei
