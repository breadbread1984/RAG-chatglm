# Introduction

this project implements retrieval augmentation generation with ChatGLM3.

# Usage

## install prerequisite packages

```shell
python3 -m pip install -r requirements.txt
```

## run service

```shell
python3 main.py --doc_dir <path/to/directory/containing/pdfs> [--host 0.0.0.0] [--port 8880]
```

note that **--doc_dir** should provided when running for the first time.
after vectordb directory is created, the **--dic_dir** is not necessary if the documents are not updated.

use web browswer to visite **http://[host]:[port]** to try the service

