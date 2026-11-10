# Controles CLIP/DINOv3

O ambiente `.venv-control` usa Linux x86_64, Python 3.10 e CUDA. Instale com `SIH_PYTHON=python3.10 bash scripts/control.sh install`. O instalador não mistura dependências com o I-MoF legado.

```bash
bash scripts/control.sh download --dinov3-source "$DINOV3_CHECKPOINT"
bash scripts/control.sh check --hashes
bash scripts/control.sh smoke --model llava --run-id control-smoke-llava
bash scripts/control.sh run --model llava --run-id control-pilot-llava
```

Use `--model qwen` e IDs próprios para o Qwen. O checkpoint DINOv3 deve ser obtido com acesso legítimo; o importador verifica formato, tamanho e SHA-256 da versão oficial descrita em `configs/models_control.json`. Identificadores, revisões e parâmetros estão em configurações versionadas; os caminhos resolvidos da instalação ficam em `.cache/control/installed.json` e não são publicados.

O teste smoke é diagnóstico técnico, não estimativa de acurácia. A execução completa guarda `run.json`, `predictions.jsonl`, `complete.json` e `scored/metrics.json`, além das fontes utilizadas. Retome com o mesmo comando e `--resume` apenas quando configuração, entradas e código forem os mesmos.

A arquitetura dos braços está em [architecture.md](architecture.md), os critérios de pontuação em [evaluation.md](evaluation.md), e o treinamento em [experiments.md](experiments.md). O controle de duplicação não foi atribuído ao paper: ele é uma escolha local para controlar a quantidade de tokens.

As configurações definem mínimos conservadores de VRAM: inferência LLaVA/Qwen em 20/16 GiB e treinamento em 48/20 GiB. O executor recusa uma alocação menor ou sem BF16 antes de carregar os pesos. Esses limites são pré-requisitos operacionais, não medições de pico nem garantias para outras resoluções/lotes; configure-os junto ao protocolo e selecione os recursos adequados no scheduler.
