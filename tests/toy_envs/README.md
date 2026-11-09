# Toy envs

Planted-hole environments used to check that the probe is right. Not written
yet. None of these load FDA-Gym data.

| Env | Planted hole | Expected probe result |
| --- | --- | --- |
| format-only grader | Rewards format, ignores the answer | Flag: reward insensitive to the answer |
| grader ignoring the task | Same score on every task | Flag: reward insensitive to the task |
| grader ignoring tool outputs | Claims to check tool use, ignores observations | Flag: reward insensitive to tool outputs |
| answer-only grader | Does not claim grounding or tool use | Do not flag for ignoring tool outputs |
| state-backed grader | `score()` needs runtime/sandbox state | Skip: not text-scorable / needs sandbox |
| clean toy env | Grader depends on the evidence it claims | Do not flag |
