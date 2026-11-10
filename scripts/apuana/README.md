# Execução SLURM

Os scripts desta pasta são templates para o Apuana. Partições/QoS refletem esse scheduler; adapte-os ao seu cluster. Não há nó, endereço SSH ou usuário fixado nos comandos. A inferência Python também pode rodar diretamente em outro host Linux/NVIDIA.

Prepare os dados e os logs, entre na raiz da cópia do projeto e confira os comandos antes de submeter:

```bash
mkdir -p logs
python3 scripts/apuana/submit_baseline.py --run-id imof-pilot-01 --dry-run
python3 scripts/apuana/submit_baseline.py --run-id imof-pilot-01
```

A cadeia é instalação → download → smoke → piloto, com dependências `afterok`. `--prepared` pula instalação/download; `--smoke-only` encerra no diagnóstico; `--node "$GPU_NODE"` restringe a alocação somente quando o operador o informar. Sem a flag, o SLURM escolhe o nó. Confirme VRAM e requisitos antes de executar.

Os controles usam outra cadeia e outro ambiente:

```bash
python3 scripts/apuana/submit_control.py \
  --run-id control-smoke-01 --smoke --prepare \
  --dinov3-source "$DINOV3_CHECKPOINT" --dry-run
```

Remova `--dry-run` após verificar a preparação. Os manifestos de submissão ficam em `experiments/submissions/`, com IDs e comandos. Uma falha parcial preserva os jobs já submetidos; consulte o registro antes de tentar novamente.

Para treino, siga primeiro os diagnósticos de [docs/experiments.md](../../docs/experiments.md). Na raiz do checkout:

```bash
sbatch --mem=64G scripts/apuana/check_alignment.sh
sbatch --mem=64G scripts/apuana/train_alignment.sh \
  llava clip_dino alignment-llava-fusion-01 --checks-prefix alignment-check-v1
```

O treino exige o teste de retomada CPU e todos os diagnósticos de gradiente/recarga concluídos. Os comandos de diagnóstico/treino não devem ser enviados simultaneamente sem dependência; confira o sucesso dos pré-requisitos. Nó, memória, partição e tempo podem ser passados ao `sbatch`; os templates GPU não fixam um nó específico.

Submeta a partir da raiz: os templates usam `SLURM_SUBMIT_DIR`. `--chdir` sozinho não substitui o diretório de submissão. Os logs ficam em `logs/`. O launcher de pares completos exige o smoke correspondente e verifica sua fonte; a CLI Python documentada aceita IDs novos para diagnósticos e experimentos próprios.

O monitor configurável está em [scripts/local/watch_mof.py](../local/watch_mof.py). Uma falha de conexão encerra a observação, sem cancelar os jobs.
