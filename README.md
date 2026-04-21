# OpenIE-PT: Extração de Informação Aberta para Português com Dependências Universais e Atenção BERT

Sistema de **Open Information Extraction (OpenIE)** para o português brasileiro que combina análise de dependências universais (UD), validação por regras linguísticas teóricas e mecanismos de atenção do modelo **BERTimbau**. O experimento é avaliado sobre o **BIA (Banco de Interpretações Agregadas)**, um corpus gold de triplas (arg1, relação, arg2) anotadas manualmente.

---

## Visão Geral do Sistema

O sistema extrai triplas relacionais estruturadas `(sujeito, relação, objeto)` de sentenças em português utilizando um pipeline em três camadas:

1. **Parsing UD (Stanza)** — análise morfossintática e de dependências universais
2. **Geração de candidatos** — regras sobre a estrutura UD (sujeito, objeto, cópula, coordenação, voz passiva, modais, orações relativas)
3. **Validação teórica + atenção BERT** — filtragem e reranking por regras explícitas (E1–E9, S1–S5) e por scores de atenção das cabeças do BERTimbau

### Pipeline de extração

```
bia_gold_sentences.jsonl
        │
        ▼
[Stanza: parse UD]
        │
        ▼
[Geração de candidatos via regras UD]
  ├── nsubj / csubj
  ├── obj / iobj / obl+prep
  ├── cópula (ser, estar, ficar, ...)
  ├── coordenação (conj)
  ├── passiva / modais
  └── orações relativas e conjunções
        │
        ▼
[Validação teórica: regras E1–E9 e S1–S5]
  ├── mode=off      → sem filtragem
  ├── mode=annotate → anota sem bloquear
  └── mode=filter   → remove candidatos inválidos
        │
        ▼
[Score de atenção BERTimbau]
  └── rankeia cabeças por papel semântico (subj/obj)
      usando corpus Bosque UD
        │
        ▼
[Deduplicação + saída]
```

---

## Recursos e Dados

| Arquivo | Descrição |
|---|---|
| `bia_gold_sentences.jsonl` | ~800 sentenças com triplas gold (BIA) |
| `pt_bosque-ud-train.conllu` | Treebank Bosque UD para rankeamento de cabeças de atenção |
| `openie_v11_bia_theoretical_experiment_selfcontained_patched.py` | Código principal (self-contained) |
| `openie_pt_experiment_final.ipynb` | Notebook de análise e visualização dos resultados |
| `ablation_experiments/` | Resultados de 27 configurações experimentais |

---

## Instalação

```bash
pip install stanza transformers torch conllu pandas tqdm

# Baixar modelos Stanza para português
python -c "import stanza; stanza.download('pt')"

# O BERTimbau é baixado automaticamente pelo HuggingFace na primeira execução
# neuralmind/bert-base-portuguese-cased
```

---

## Uso

```bash
python openie_v11_bia_theoretical_experiment_selfcontained_patched.py \
  --gold bia_gold_sentences.jsonl \
  --output-dir ablation_experiments/meu_experimento \
  --dataset-name meu_experimento \
  --theory-mode filter \
  --heads-mode rank
```

### Principais argumentos

| Argumento | Opções | Descrição |
|---|---|---|
| `--theory-mode` | `off` / `annotate` / `filter` | Modo de aplicação das regras teóricas |
| `--heads-mode` | `rank` / `all` / `forced` | Seleção de cabeças de atenção BERT |
| `--no-attn` | flag | Desativa atenção BERT (baseline UD puro) |
| `--attn-threshold` | float (ex: `0.15`) | Threshold mínimo de score de atenção |
| `--s2-span-policy` | `minimal` / `extensive` / `both` | Política de span para a regra S2 |
| `--strict-theory` | flag | Aplica regras em modo estrito |
| `--block-reported-belief` | flag | Bloqueia extrações de crença reportada |
| `--no-e8-clausal` | flag | Desativa subrregra E8 clausal |

---

## Regras Teóricas

### Regras de Extração (E1–E9)

| Regra | Descrição |
|---|---|
| E1 | Sujeito deve ter papel gramatical nsubj/csubj |
| E2 | Relação deve ser verbo pleno ou cópula |
| E3 | Argumentos não podem ser pronomes expletivos |
| E4 | Filtro de metadiscurso / expressões metalinguísticas |
| E5 | Compatibilidade de voz (ativa/passiva) |
| E6 | Verificação de escopo modal |
| E7 | Consistência de coordenação |
| E8 | Restrições em orações subordinadas clausais |
| E9 | Coerência de factualidade (asserted/conditional/reported) |

### Regras de Span (S1–S5)

| Regra | Descrição |
|---|---|
| S1 | Span do sujeito não cruza fronteira de oração |
| S2 | Span do objeto obedece política de extensão configurável |
| S3 | Preposição não incluída no span do objeto direto |
| S4 | Auxiliares não incluídos no span da relação |
| S5 | Span mínimo — remove fragmentos triviais |

---

## Experimentos e Resultados

### Questões de Pesquisa (RQs)

| RQ | Foco | Configuração-chave |
|---|---|---|
| RQ1 | Baseline UD puro | sem atenção, sem teoria |
| RQ2 | Impacto do modo cópula | `cop_mode`: off / restricted / full |
| RQ3 | Impacto da atenção BERT | `no_attn` on/off, thresholds |
| RQ4 | Teoria completa (E+S) | `theory_mode=filter`, todos os E+S |
| RQ5 | Apenas E-rules vs apenas S-rules | ablação por família de regra |
| RQ6 | Ablação individual de cada regra | remove uma regra por vez |
| RQ8 | Melhor configuração global | atenção + teoria - E4 |

### Resultados Principais

| Experimento | Precisão | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| RQ1 — Baseline UD puro | 0.450 | 0.637 | **0.527** | 272 | 333 | 155 |
| RQ2 — Cópula full | 0.450 | 0.637 | 0.527 | 272 | 333 | 155 |
| RQ2 — Cópula restricted | 0.452 | 0.501 | 0.476 | 214 | 259 | 213 |
| RQ2 — Cópula off | 0.454 | 0.496 | 0.474 | 212 | 255 | 215 |
| RQ3 — Atenção on (thr=0) | 0.493 | 0.621 | **0.549** | 265 | 273 | 162 |
| RQ3 — Atenção on (thr=0.15) | 0.493 | 0.621 | 0.549 | 265 | 273 | 162 |
| RQ4 — Teoria completa (E+S) | 0.453 | 0.593 | 0.513 | 253 | 306 | 174 |
| RQ5 — Apenas E-rules | 0.452 | 0.593 | 0.513 | 253 | 307 | 174 |
| RQ5 — Apenas S-rules | 0.450 | 0.625 | 0.524 | 267 | 326 | 160 |
| RQ6b — sem E4 | 0.450 | 0.623 | 0.523 | 266 | 325 | 161 |
| **RQ8 — Melhor config (attn + teoria − E4)** | **0.491** | **0.609** | **0.544** | 260 | 269 | 167 |

> **Melhor resultado**: `rq8_best_config_no_E4` — atenção BERT ativa + teoria com filtro + sem regra E4 → **F1 = 0.544**

### Observações-chave

- A atenção BERT (RQ3) melhora F1 de 0.527 → 0.549 ao reduzir falsos positivos sem perder recall
- A teoria completa (RQ4) reduz recall pois descarta alguns candidatos corretos — E4 é a principal responsável
- Remover E4 (RQ6b/RQ8) com atenção ativa produz a melhor combinação F1 = 0.544
- O modo cópula `full` é superior: desativar cópula custa ~5 pontos de F1

---

## Saídas por Experimento

Cada experimento em `ablation_experiments/<nome>/` gera:

| Arquivo | Conteúdo |
|---|---|
| `*_predictions_enriched.jsonl/.csv` | Triplas preditas com metadados linguísticos completos |
| `*_metrics.json` | TP, FP, FN, precisão, recall, F1 |
| `*_summary.json` | Resumo de alto nível |
| `*_selected_heads.json` | Cabeças de atenção BERT selecionadas |
| `*_triples_table.csv` | Gold vs predição lado a lado |
| `*_per_pattern_metrics.csv` | Métricas por padrão UD |
| `*_per_rule_summary.csv` | Efetividade de cada regra E/S |
| `*_per_factuality_summary.csv` | Por factualidade (asserted/conditional/reported) |
| `*_per_variant_type_summary.csv` | Por tipo de variante |

---

## Arquitetura do Código

O sistema é implementado em um único arquivo self-contained (`~3400 linhas`). As principais classes são:

- **`Config`** — todos os parâmetros de extração e experimentação (linhas ~50, ~1880)
- **`OpenIEExtractorBIA`** — motor de extração principal (linhas ~419, ~1999)
- **`CandidateTriple`** — candidato a tripla com metadados (linha ~1930)
- **`TheoryValidation`** — resultados de validação por regra (linha ~1968)
- **`UDTok`** — representação de token UD (linha ~147)

---

## Citação

Se você utilizar este sistema em seu trabalho, por favor cite:

```
@misc{openiept2025,
  author = {Ricardo Gomes Oliveira},
  title  = {OpenIE-PT: Extração de Informação Aberta para Português com UD e Atenção BERT},
  year   = {2025},
  url    = {https://github.com/rgoliveirati/openieptextractor}
}
```

---

## Licença

Este projeto está disponível para fins acadêmicos e de pesquisa.
