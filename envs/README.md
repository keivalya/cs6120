# envs/

One conda env per model — NO shared deps (CLAUDE.md §6). Built in GATE 2 (smolvla,
metrics) and GATE 4 (openvla, oft). Resolved versions recorded to `<name>.lock.txt`
after each build (`pip freeze > envs/<name>.lock.txt`).

| env | model alias(es) | built | lock |
|-----|-----------------|-------|------|
| vla-smolvla | smolvla | GATE 2 | vla-smolvla.lock.txt |
| libero-para-metrics | (analysis, no GPU) | GATE 2 | libero-para-metrics.lock.txt |
| vla-openvla | openvla | GATE 4 | vla-openvla.lock.txt |
| vla-oft | openvla_oft, openvla_oft_film | GATE 4 | vla-oft.lock.txt |

NEVER install LIBERO-Plus into any of these — it shadows the `libero` package (§6).
