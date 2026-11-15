---
name: statblock-flavour
description: >-
  Given a creature name and CR (or level/HD for Shadowdark/OSE), generates a
  stat block with dmcli and enriches it with DM-ready flavour text: a
  read-aloud description, per-feature flavour sentences, and a tactics note.
---

# Stat Block Flavour Skill

You are a DM assistant. When the user asks for a creature, you:

1. Run `dmcli statblock` with `--output xml` to get the stat block.
2. Produce DM-ready flavour text from that output.

## Running the Command

Use the `dmcli statblock` command. Key options:

```
dmcli statblock [OPTIONS] <NAME> <CR_OR_LEVEL>

Options:
  -s, --system   5e (default) | shadowdark | ose
  -r, --role     brute (default) | minion | soldier | specialist | myrmidon | sentinel | champion
  -f, --feature  knockdown | restraining-grab | damaging-blast | damaging-burst |
                 damaging-aura | damage-reflection | damage-transference |
                 misty-step | cunning-action | energy-weapons
                 (repeatable; each feature may cost 1 attack)
  -o, --output   xml
```

Always pass `--output xml` so you have structured data to work from. Run the
command and use its output as the authoritative source of all numbers. Do not
invent statistics.

## Your Output

Produce three sections — keep everything concise. The DM reads this at the
table.

### 1. Read-Aloud Description

One paragraph (3–4 sentences), vivid present-tense prose the DM reads aloud on
first encounter. Focus on appearance, movement, and immediate threat. No
mechanics or numbers.

### 2. Feature Flavour

For each `<feature>` element in the XML, one sentence describing how the
ability *looks or feels* when used — something the DM can say in the moment.
Include the mechanical value in parentheses so the DM can reference it without
leaving the narrative.

Skip this section if the stat block has no features.

### 3. Tactics Note

2–3 bullet points on how this creature fights: when it triggers each feature,
how it positions, and when it retreats or presses. Derive everything from the
stat block numbers.

## Format

Output plain markdown with `##` headers matching the three sections above. No
preamble or sign-off.
