# Configurações

| Arquivo | Conteúdo |
|---|---|
| `data.json` | Saídas relativas à raiz do checkout |
| `baseline_mof.json` | Código, checkpoint, encoders, precisão e geração do I-MoF oficial |
| `checkpoints-mof.json` | Papéis, nomes e SHA-256 dos 17 arquivos esperados |
| `models_control.json` | Revisões de LLaVA/Qwen/CLIP/DINOv3 e identidade do checkpoint importado |
| `control_mof_stage0.json` | Braços, seeds, resolução e protocolo antes do alinhamento |
| `alignment_training_v1.json` | Dados, otimizador, épocas e seleção de checkpoint |
| `requirements-mof.txt`, `constraints-mof.txt` | Ambiente legado Linux/Python 3.10 |
| `requirements-control.txt` | Ambiente independente dos controles |

Não coloque tokens, endereços de máquinas ou caminhos pessoais nestes arquivos. Caches e caminhos resolvidos da instalação são gerados fora do Git. Revisões de modelos, hashes, seeds e parâmetros científicos são deliberadamente explícitos; mudanças exigem uma execução/versionamento próprio.
