# syllable-detection-model

Training setup for teaching voice agents to hear Indian-style fillers and
non-lexical sounds ("hmm", "uh", "ah", stutters, false starts), understand the
feelings behind them, and turn that into the right conversational decision -
instead of stripping those sounds out of the transcript.

Built for [Dani](https://dani.danlab.dev).

## The idea

Every turn of a conversation is a decision. To a standard speech-to-text model,
"ha" (yes), "na" (no) and a doubtful "hmm" are near-identical syllables - and
most ASR deletes them anyway, because it is trained to produce clean, readable
text. Once they are gone, the agent cannot tell agreement from refusal, and the
conversation goes down the wrong branch.

This repo fine-tunes and wires up open models so a voice agent keeps those
signals and acts on them:

1. **Verbatim ASR** - [CrisperWhisper](https://github.com/nyrahealth/crisperwhisper),
   a Whisper variant trained to transcribe every spoken word exactly as said,
   including fillers, stutters and pauses, with word-level timestamps. We
   fine-tune it on Indian speech.
2. **Speech emotion** - [emotion2vec+](https://huggingface.co/emotion2vec), an
   open speech-emotion foundation model (Alibaba DAMO, released through
   FunASR/ModelScope and Hugging Face). We train a small classification head on
   top of its utterance embeddings with Indian affect data.
3. **Decision model** - [Laya](https://github.com/NandhaKishorM/laya), an
   open-source, multilingual, non-autoregressive decision engine
   ([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya),
   Apache 2.0). It reads the turn - verbatim transcript plus emotion - and
   returns typed decisions (choice / score / yes-no) with calibrated
   probabilities in a single ~33 ms forward pass, instead of asking an LLM to
   generate them. This is where "ha" vs "na" becomes agree vs disagree.
4. **LLM layer** - the full response pipeline only fires for turns the decision
   model routes as complex; acknowledgements and refusals are answered
   directly.

The notebook covers steps 1-3; step 4 is runtime plumbing.

## Repo structure

```
notebooks/train_indian_fillers_emotion.ipynb   # end-to-end Colab training + wiring notebook
requirements.txt                               # Python dependencies (Colab preinstalls torch)
```

## Quickstart (Google Colab)

1. Open the notebook in Colab:
   [train_indian_fillers_emotion.ipynb](https://colab.research.google.com/github/rajarshidattapy/syllable-detection-model/blob/main/notebooks/train_indian_fillers_emotion.ipynb)
2. Runtime > Change runtime type > GPU (a T4 is enough).
3. Add your labelled CSVs under `data/` (see below), or rely on the IndicVoices
   streaming fallback for a first run.
4. Runtime > Run all.

## Data

No datasets are stored in this repository. Everything is streamed or downloaded
at run time, or supplied by you as small CSV manifests.

| Data | Used for | Notes |
| --- | --- | --- |
| [AI4Bharat IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) | Indian spontaneous/read/conversational speech for the ASR fine-tune | 23.7K hours collected across 22 Indian languages and 51K speakers, of which ~11.2K hours are transcribed. CC BY 4.0. Gated on Hugging Face: accept the terms and set an HF token. |
| `data/verbatim_indic.csv` (you create this) | Real filler supervision for CrisperWhisper | Columns: `audio_path,text`, where `text` is a verbatim transcript that keeps fillers. |
| `data/emotion_indic.csv` (you create this) | Training the emotion head | Columns: `audio_path,label`. |
| IEMOCAP / RAVDESS (optional) | English emotion baselines for transfer experiments | Require their own licenses; not Indian speech. |

Known gap: no public dataset labels what an Indian "hmm" *means* (agreement,
hesitation, doubt, ...). The practical path is to label a few hundred clips
yourself; a few hundred utterances per class is enough to train the embedding
head in the notebook.

Also note: IndicVoices' reference transcripts are mostly clean/intended-style,
so the fallback run adapts CrisperWhisper to Indian acoustics but does not teach
it new filler behaviour. CrisperWhisper 2.0's `verbatimize` mode (audio plus a
clean transcript in, verbatim transcript out) is a good way to pseudo-label
Indian audio at scale.

## Models

| Model | Role | Links |
| --- | --- | --- |
| CrisperWhisper | Verbatim ASR that keeps fillers with word-level timestamps | [GitHub](https://github.com/nyrahealth/crisperwhisper) · [HF model](https://huggingface.co/nyrahealth/CrisperWhisper) · v2: `pip install crisperwhisper` |
| emotion2vec+ | Speech emotion recognition foundation model (9-class) | [HF org](https://huggingface.co/emotion2vec) · [FunASR](https://github.com/alibaba-damo-academy/FunASR) · [paper repo](https://github.com/ddlBoJack/emotion2vec) |
| Laya | Multilingual non-autoregressive decision engine for per-turn intent | [GitHub](https://github.com/NandhaKishorM/laya) · [HF model](https://huggingface.co/convaiinnovations/laya) · `pip install -U laya` |

## Status

Early training setup, no benchmark numbers yet. What exists today:

- A Colab-ready notebook that fine-tunes CrisperWhisper on Indian speech,
  trains an emotion classification head on emotion2vec+ embeddings, and wires
  both signals into a Laya decision step.
- A data plan built around IndicVoices plus self-labelled clips.

What does not exist yet: evaluation on held-out Indian filler/emotion data, and
the end-to-end voice runtime. Those are next.
