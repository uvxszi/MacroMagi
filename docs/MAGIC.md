# The Magic tab

The Magic tab builds, trains, tries and packages AI models. It uses a canvas like the Macro tab's.

- **Magi** is MacroMagi's text model: an **agent**. It knows how to work – think, plan, look things up, use tools, check – not facts or exact code. When it needs the exact way to do something it searches **MagiDB**, a library of linked Markdown pages; it reads and changes files, runs commands, opens documents and looks at pictures through its **native tools**, tens of calls for one request. It runs in MacroMagi (the AI Agent window, the AI Agent block) and CodeMagi.
- **Vision models** look at part of the screen and pick an answer. The **AI Watch** and **AI Model** blocks run them.

A `.magic` file is the **workspace**: the canvas and the setup, and as chosen the datasets and pictures – never the model. A trained model is a **Magi model** file (`.szi`): one file with its weights and its native tools. The file formats are in [FILE_FORMAT.md](FILE_FORMAT.md#magic--ai-workspaces).

## Contents

- [Quick start](#quick-start)
- [The Magi architecture](#the-magi-architecture)
- [Magi's tools](#magis-tools)
- [MagiDB](#magidb)
- [Magi's recipe](#magis-recipe)
- [The canvas](#the-canvas)
- [Blocks](#blocks)
- [Datasets](#datasets)
- [Training](#training)
- [Packages: the project folder](#packages-the-project-folder)
- [Running models](#running-models)
- [The AI Agent window](#the-ai-agent-window)
- [Long work and long contexts](#long-work-and-long-contexts)
- [Settings › Runtime and Hardware](#settings--runtime-and-hardware)
- [How text is read and written](#how-text-is-read-and-written)
- [What the architecture blocks are](#what-the-architecture-blocks-are)
- [Limits](#limits)
- [For developers](#for-developers)

## Quick start

1. **File › New text model** (on the Magic tab). It opens as **Magi**: the architecture (about 418 million parameters), its sixteen tools, Thinking and Reasoning Effort, and the whole recipe – pretraining on three billion tokens of web text and code, fine-tuning on agent work and conversations, compaction, reinforcement learning – with its fifteen example datasets. **File › New text model: Magi 150M (a quick test)** is the same design and recipe at about 150 million parameters (64 million at work per token; 512 wide, 24 layers) on 750 million tokens: train it first, a few hours on a G4, to see the whole recipe work before the full size.
2. **File › Export as Python**, and train it on a GPU server: Colab's **G4** (an RTX PRO 6000) or a rented card. A desktop card cannot train a model this size in a reasonable time. See [Training on a GPU server](#training-on-a-gpu-server).
3. When it is done, open the `Magi.szi` it wrote (File › Open, or the AI Agent window's *Load model file…*).
4. **Settings › Runtime › Magi's tools**: download **MagiDB** (or let the model read it online), and install **Document tools** and **Image descriptions** if it is to open documents and look at pictures.
5. In the **AI Agent window** (the robot button), ask it something – *Fix the failing tests in D:/work/shop*, attach a screenshot, or a spreadsheet. Calls that change files or run programs ask you first.

A vision model: **File › New vision model**, give each answer its pictures, choose where Compile saves it, **Train** – that trains here, in minutes.

## The Magi architecture

One architecture, the best this size allows for an agent that reads long transcripts of its own work:

| Part | Magi | Why |
| --- | --- | --- |
| Size | 768 wide, 32 layers; twelve heads of 64 sharing four key/value heads | Deep and narrow is best below a billion parameters; shared key/value heads keep the memory of what it read small. |
| Attention | Queries and keys normalized; three layers in four look only at their own block of 1,024 tokens and the block before (1,024 to 2,048 tokens back), every fourth at everything | A transcript of fifty tool calls is long: local layers cost the same at any length, and the global ones still see it all (Gemma 3, gpt-oss). |
| Feed-forward | Three plain layers, then DeepSeekMoE in the other 29: a shared expert and sixteen small routed ones, two chosen per token by √Softplus affinities, balanced by a bias | The knowledge of a 418M model at the cost of a 148M one for each token – faster to train and to run (DeepSeek-V3/V4). |
| Training | DeepSeek-V3's multi-token prediction (it predicts the token after next too; left out of the finished model); Muon | More learned from every sample; DeepSeek-V4's optimizer. |
| Context | 8,192 tokens in training, YaRN to 128K when it runs | Long enough to train on whole pieces of work. |
| Tokens | 32,768 learned where it trains, from the start of its corpora | About 4.5 bytes of text a token. |
| Thinking | On, and two efforts: *standard* and *superior* | It thinks before each step; at *superior* it plans more and checks more. |

About **418 million parameters**, **148 million** at work for each token. **Presets › Commons › Magi** puts this architecture back in place of the canvas's (after you have changed it). The DeepSeek blocks (latent and compressed attention, hyper-connections) are still in the block list for experiments; the native engine runs Magi's layout.

## Magi's tools

Magi does not keep facts or exact code in its weights: it calls tools, and MacroMagi runs them on this computer and gives the result back, until it answers. Its tools are **native tools** – each a folder with a `tool.json` and a program – and travel inside the `.szi`:

| Tool | What it does |
| --- | --- |
| `db_search(query)`, `db_read(page, part?)` | Searching and reading [MagiDB](#magidb): the exact code, command or way to work. |
| `web_search(query)`, `fetch_page(url)` | The web, for news and what MagiDB does not have. |
| `list_files(folder)`, `read_file(path, from?, lines?)`, `search_files(text, folder)` | Reading folders and text files (numbered lines, 200 at a time), and finding text in a folder's files. |
| `write_file(path, text)`, `edit_file(path, find, replace)` | Writing a file whole, or replacing one exact piece of it. **Asks first.** |
| `run_command(command, folder?)` | Running tests, scripts, git… and reading what they print. **Asks first.** |
| `open_document(path)`, `edit_document(path, edits)` | .docx, .xlsx, .pptx, .pdf and .csv files as numbered parts of plain text (`p3`, `Sheet1!B2`, `s1.2.1`, `page4`), and changing parts – the file is saved in its own format, with its styles, and a copy of it as it was is kept beside it. `edit_document` **asks first**; PDFs are read only. |
| `view_image(path)` | A picture, for a model that never sees pixels: what it shows (a small vision model, SmolVLM-256M) and the text in it (Windows' text recognition, as the Text Scan block uses). |
| `calculator(expression)`, `get_time(location)`, `get_weather(location)` | Exact arithmetic – Magi does no sums in its head – the time and the weather. |

**Calls that change files or run programs ask you first**: a window shows what the model wants to do (the command, the text, the change) with *Allow*, *Allow … until MacroMagi closes* and *Don't allow*. Refused, the model is told so, and says what it wanted to do instead of trying again.

More native tools come with MacroMagi (`transcribe_audio`, `translate`, `wiki_search`) and can be ticked in the Native Tools block. A model's own copy of a tool (under `nt/<tool>/` in its files) takes the place of MacroMagi's.

## MagiDB

MagiDB is the library Magi looks things up in: Markdown pages that link to each other – how Magi works, Python, JavaScript, Rust, Git, the command line, SQL, documents, algorithms, and what error messages mean. A page answers one question, with the exact code or command and the usual mistakes. It lives in its own repository; MacroMagi's starts in [`magidb/`](../magidb), with [how to write pages](../magidb/README.md).

`db_search` and `db_read` look for it in order, and the model does not choose:

1. a folder chosen under *Settings › Runtime › MagiDB folder* (a copy you are writing);
2. the copy downloaded under *Settings › Runtime › MagiDB* (faster, and it works offline);
3. the repository online (*MagiDB online*, the GitHub repository by default), a page at a time – each page kept in MacroMagi's cache once read.

**Case does not matter.** Pages are written as people write – titles and sentences start with capitals – and requests often are not. Searching folds case and accents ("python FILES", "Python files" and "pýthon files" find the same pages), page names are found in any case, and a page can be asked for by its title. The model's own tokens are case-sensitive, as every model's are; it learns that *File* and *file* are one word from the billions of tokens it pretrains on, which are written both ways.

## Magi's recipe

New text model opens with Magi's training pipeline:

1. **Pretraining** – three billion tokens: **FineWeb-Edu** (educational web pages, ODC-By) and **OpenCoder's code corpus** (web pages about programming, MIT), through two **Hugging Face Dataset** blocks, with the example files of plain English wired into the second's *With* input. 22,900 steps of 16 rows of 8,192 tokens, four steps a weight update; Muon, warming up over 1,000 steps. This is the language and the code it can read – not knowledge it is meant to rely on.
2. **Fine-tuning** – 4,000 steps of 8 rows: **smol-smoltalk**'s conversations (Apache-2.0) and the example datasets, above all `work.jsonl`: long pieces of work, tens of tool calls each, MagiDB before code. The answers to avoid make their answers less likely.
3. **Context compaction** – 400 steps: writing a long conversation short, with the other requests mixed in.
4. **Reinforcement learning** – 300 steps, eight answers to each request, a gentle AdamW: rewarded for finishing, calling the tool the dataset's answer calls first, well-formed calls to tools it is offered; punished for broken calls, unknown tools, writing a tool's result itself and going round in circles.

It trains on a GPU server; the console estimates the time after the first steps.

## The canvas

Five kinds of wire connect the blocks. Each block's panel (select the block) lists what is wired into it under **Connections**, and what it would use if nothing were.

| Wire | From → to | What it carries |
| --- | --- | --- |
| **Model** | start → architecture blocks → training stages → Compile Package → Prompt Model / Test Model | The model, left to right. Architecture blocks come before the stages. |
| **Setup** | setup blocks → … → the start's *Setup* input | The model's own settings. Setup blocks chain into each other; the model is what the blocks on the chain say. A setup block that is not on the chain is marked *Not used*. |
| **Feed-forward** | Mixture of Experts or Dual-Persona Router → a layer block's *Feed-forward* input | What replaces that layer block's plain feed-forward layer. One block can feed several layer blocks. |
| **Data** | Text Dataset / Hugging Face Dataset / Vision Dataset → a training stage's data inputs | What the stage learns from. |
| **Optimizer** | Optimizer → a training stage's *Optimizer* input | How that stage learns. Without one, a stage uses AdamW with the defaults. |

Blocks show warnings on their faces, such as *Wire a Text Dataset into it* or *Choose a folder to save into*. The **Train** button says in its tooltip what is still missing. With nothing selected, the panel shows the pipeline from start to end and a short *How it works*.

The banner holds the tab's menus, as on the Macro tab:

| Menu | On the Magic tab |
| --- | --- |
| **File** | New text model (Ctrl+N), New text model: Magi 150M (a quick test), New vision model, Open (Ctrl+O: a `.magic` workspace or a `.szi` model), Open recent, Import (an ONNX project folder, a `.zip` or `.onnx`), Save (Ctrl+S), Save as, **Export as Python** and its experimental twin, which tries compiled layers (see [Exporting the workspace as Python](#exporting-the-workspace-as-python)), Close the model. |
| **Options** | The usual options, plus *Optimize environment when training* and *Notify when training ends* (a sound and a desktop notification when a run of more than a minute finishes or fails; on by default). |
| **Presets** | **Commons** (Magi's architecture), your saved Magic blocks, and *Save selected blocks*. |
| **Add** | The block list (right-clicking the canvas opens it too). |
| **View** | Console, Block settings, Grid, Find, Fit to view, Actual size. |
| Undo / Redo | Every change to the canvas and the pictures (Ctrl+Z, Ctrl+Y). |
| **Train** / **Stop training** | Runs the pipeline. |

## Blocks

### Start

| Block | What it does |
| --- | --- |
| **Start from Zero** | A brand-new model with random weights, text or vision. |
| **Load Base Model** | Starts from a trained model's weights, to train it further: this model's own (empty path), a `.szi` package or a `.magic` file trained in MacroMagi. Keep the base model's architecture blocks. A stage's **Save as base model…** writes one. |
| **Use As It Is** | The model as it is, without training: an imported ONNX model, or one trained before. Change its setup, try it, package it. |

All three take the setup chain in their *Setup* input.

### Setup

| Block | For | Settings |
| --- | --- | --- |
| **Model Info** | all | Name, what it does, author, licence. Also shows the model's size and how it was trained. |
| **Text** | text | The system prompt; its **tokens** – *Learned* (pieces of words, learned when training starts from zero; Magi's 32,768 are learned where it trains, from its corpora) or *Bytes* – see [How text is read and written](#how-text-is-read-and-written); temperature, how many likely tokens to choose among, the longest answer; its personas; its chat tags. |
| **Tools** | text | Tools the model may call that the program running it carries out: in a macro the AI Agent block (*Add MacroMagi's tools* adds press, type, click, move, wait and log), in another app that app's own. |
| **Native Tools** | text | Tools a Magi model runs itself, on this computer – see [Magi's tools](#magis-tools). They go into the model's `nt/` folder. |
| **Thinking** | text | The model can be asked to think or not; *Think unless asked not to* sets what it does by default (on for Magi). Chained in, training learns each answer with reasoning both ways, and the Prompt Model block, the AI Agent block and window get a Thinking switch. Without it, the datasets' reasoning is left out. |
| **Reasoning Effort** | text | Levels of how hard the model reasons – Magi has *standard* and *superior* – and the default one. The model is told its level at the top of every request (“Reasoning effort: superior”) and learns each level from samples written at it. A level with no samples is learned from the nearest level's, and the block says so. |
| **Memory & Cache** | text | Where the model keeps the memory of what it read while it runs. See [Long work and long contexts](#long-work-and-long-contexts). |
| **Screen Area** | vision | A square in the middle of a display (it scales with the display) or a fixed area. |
| **Answers** | vision | Each answer's name, whether it acts, how sure the model must be, and how long to wait. |
| **When It Acts** | vision | The key or mouse button, how long it is held, and how long answers are then ignored. |
| **Sliders** | vision | Sliders that move several values at once, offered again in every AI block that uses the model. |
| **Picture Preparation** | vision | Picture size, layout, color order and normalisation. Training and importing set it. |

### Architecture

| Block | What it adds |
| --- | --- |
| **Context Scaler** | Trains at a short context and runs at a long one: *Train at* (Model Size's context, 2048, 4096 or 8192 tokens), *Run at* (32K to 256K tokens), *Scaling* (YaRN, NTK, or none for compressed attention). See [Long work and long contexts](#long-work-and-long-contexts). |
| **Model Size** | *Width*, *Attention heads*, *Key/value heads*, **Context window** (the longest text the model reads in training, in tokens; for vision, the side of the pictures in pixels), *Positions* (rotary or learned), *Tie input and output* and *Scale embeddings*. |
| **Transformer Layers** | Attention over earlier tokens, then a gated feed-forward layer – or the experts wired into *Feed-forward*. *Finer options*: *Normalize queries and keys*, *Feed-forward gate* (SwiGLU or GeGLU), *Normalization* (before each part, or before and after), *Local window* (a token sees its own block of that many tokens and the block before) with *Every Nth layer sees all*, *Rotary share* and *Biases on queries, keys, values*. |
| **Sliding Window Attention** | Layers that attend only to the last *Window* tokens. |
| **Latent Attention (MLA)** | DeepSeek-V2/V3's multi-head latent attention. |
| **Compressed Sparse Attention**, **Heavily Compressed Attention** | DeepSeek-V4's CSA and HCA, for very long contexts. |
| **Mamba-2 / SSM** | State-space layers. |
| **Mixture of Experts** | DeepSeekMoE, for the layer blocks it is wired into. Shared experts see every token; each token also picks *Experts per token* of the routed experts. *Choosing experts*: softmax, sigmoid or √softplus. *Balancing*: a bias with no extra loss, or a balance loss. *Fixed experts per token*: hash routing. |
| **Dual-Persona Router** | Two personas of the reasoning, each inside its own tag with its own expert. |
| **Hyper-Connections (mHC)** | DeepSeek-V4's manifold-constrained hyper-connections. |
| **Multi-Token Prediction** | While training, an extra layer also predicts the token after next. It is left out of the finished model. |
| **Encoder/Decoder Split** | The request is read in both directions; only the answer is written token by token. |
| **Vision Encoder** | Vision models: convolution layers, then the answers. |

Every layer block has *Layers* (how many) and *Feed-forward width* (used when no Mixture of Experts or Persona Router is wired in).

### Data

| Block | What it holds |
| --- | --- |
| **Text Dataset** | Text files kept in the `.magic` workspace, in two lists: *What to do* and *What not to do*. Add files, start a new `.jsonl` or `.md`, add the examples, or open a file in the dataset editor. |
| **Hugging Face Dataset** | A dataset on huggingface.co (`owner/name`), which of its parquet files to read (a pattern), the column (`text` to pretrain on, `messages` for conversations to fine-tune on) and how many tokens to read at most. Nothing is downloaded into the workspace: the trainer reads it where it runs, a part at a time, cuts it into the model's tokens once and keeps them (`MAGI_CORPUS_DIR`) – about two million tokens a second. A Text Dataset – or another Hugging Face Dataset – wired into its *With* input is read along with it. |
| **Vision Dataset** | Pictures of each answer, and pictures an answer was mistaken for, with thumbnails. |

### Training

| Block | What it does |
| --- | --- |
| **Pretraining** | Next-token prediction on all the text in its data. *Start the context at* grows the context from short to full (a curriculum). |
| **Fine-Tuning** | Requests and answers from *What to do* become likelier, answers from *What not to do* less likely. All weights, or LoRA or DoRA adapters folded back in afterwards. |
| **Context Compaction** | Teaches the model to rewrite a long context into a short one, from `{"context", "compact"}` samples, with the other requests mixed in so it keeps them. |
| **Reinforcement Learning** | Several answers per request are scored by reward rules, and the better-than-average ones become likelier (group-relative policy gradient). |
| **Vision Training** | Teaches a vision model its answers from a Vision Dataset. |
| **Optimizer** | Muon (with AdamW for embeddings and norms, as DeepSeek-V4 does), AdamW or SGD; the learning rates; weight decay; schedule and warm-up; clipping; accumulation; saving graphics memory. |

**Reward rules.** Each answer scores the sum of the rules it meets: *Says…*, *Does not say…*, *Matches the pattern…*, *Gives the dataset's answer*, *Calls tools correctly*, *Writes a broken tool call*, *Calls the tool the dataset's answer calls first*, *Calls a tool it is not offered*, *Writes a tool's result itself*, *Repeats itself (a loop)*, *Is short* and *Finishes its answer*. Magi's: +1 finishing, +2 the dataset's first tool, +1 well-formed calls, −2 broken calls, −2 tools not offered, −3 a made-up result, −3 a loop.

### Package and trying

| Block | What it does |
| --- | --- |
| **Compile Package** | Where the finished model is saved: *Save into* (a folder; training needs one – the project gets a folder of its own in it), *Project name*, and what to save: *.magic workspace* (with or without the datasets and pictures), *Magi Model (.szi)* and *ONNX file*. Its panel has *Save package now*, *Use in Macro* and *Open folder*. Training saves the package by itself when it ends. |
| **Prompt Model** | Text models. **Model**: *Load*/*Unload* – a loaded model stays in memory. **Memory**: how full the hot, warm and disk tiers are. **Answering**: the *Context window*, *Thinking*, *Reasoning effort*, *Native tools* (and at most how many calls per request: 100 by default), the longest answer and the temperature. Write a request, *Send* (Ctrl+Enter), and read the tools it ran, the answer and what it thought. |
| **Test Model** | Vision models: watch the screen live, or try a picture file. |

## Datasets

Text datasets are `.jsonl` files (one JSON object per line), or `.md` and `.txt` files (plain text, split at blank lines, for pretraining):

```json
{"prompt": "What is 15% of 80?", "thinking": "Arithmetic: the calculator.", "response": "12.", "effort": "superior"}
{"messages": [{"role": "user", "content": "List D:/work"}, {"role": "assistant", "content": "", "thinking": "A folder: list_files.", "tool_calls": [{"name": "list_files", "arguments": {"folder": "D:/work"}}]}, {"role": "tool", "content": "{\"folder\": \"D:/work\", \"files\": [\"a.py\"], \"folders\": []}"}, {"role": "assistant", "content": "D:/work holds a.py."}]}
{"prompt": "What is 5 plus 7?", "rejected": "5 plus 7 is 5 plus 7 is"}
{"context": "a long conversation …", "compact": "the short version"}
{"text": "plain text to pretrain on"}
```

- `prompt`/`response` and `messages` teach answers. In a conversation every assistant message is learned, with what came before it as the request.
- `thinking` is the reasoning before an answer (on a sample or an assistant message; `<think>…</think>` at the start of an answer works too). With the Thinking block chained in and on, each answer with reasoning is learned both with it and, after an empty thinking span, without it.
- `effort` names the reasoning effort the sample is written at – one level or several.
- `tools` lists the tools the sample offers. Without it, the model's own tools are listed – as they are whenever the model runs; `"tools": []` lists none. Assistant messages call them with `tool_calls`; `tool` messages carry the results – written as the tool prints them. An assistant message with `"learn": false` is read but not learned: a wrong step the next turns recover from.
- `system` (or a first system message) replaces the model's own system prompt.
- `prompt`/`rejected` (or `bad`) is an answer not to give. `context`/`compact` trains compaction.

### The example datasets

A new Magi model comes with fifteen files (about 26 MB):

| File | What it teaches | Made by |
| --- | --- | --- |
| `work.jsonl` | **Long pieces of work**, 1,400 of them, up to 78 tool calls each: fixing failing tests, adding an option, renaming across up to 30 files, a docstring for every function, listing every TODO, scripts written from a data file (pseudocode first, MagiDB for the exact code, run, fixed when they fail), raising every price in a spreadsheet with the calculator, fixing a Word file's spelling, summarizing PDFs, describing folders of screenshots, reading receipts, answering from MagiDB, explaining error messages, undoing a Git commit. Thinking before every step; on long work, a running list – *Done: … Next: …*. | [`make_work_data.py`](../crates/magi-core/src/nodes/examples/make_work_data.py) |
| `agent.jsonl`, `thinking.jsonl` | Tools one after another, errors read and recovered from, saying what no tool can do; when to think, arithmetic to the calculator. | [`make_agent_data.py`](../crates/magi-core/src/nodes/examples/make_agent_data.py) |
| `small_talk.jsonl`, `explain.jsonl` | Greetings answered in a line; longer answers laid out as people read them. | `make_agent_data.py` |
| `tools.jsonl`, `chat.jsonl`, `titles.jsonl`, `knowing.jsonl`, `simple_qa.jsonl` | General tools, short conversations, naming a chat, saying when it does not know, short answers. | [`make_examples.py`](../crates/magi-core/src/nodes/examples/make_examples.py) |
| `compaction.jsonl` | Writing a conversation short: who the user is, what they want, the facts found. | `make_examples.py` |
| `language.md` | Plain English to pretrain on with the corpora. | `make_examples.py` |
| `conversations.jsonl`, `functions.jsonl` | Human-written conversations (OpenAssistant, Apache-2.0); calling offered tools (glaive-function-calling-v2, Apache-2.0). | [`open_data.py`](../crates/magi-core/src/nodes/examples/open_data.py), credits in [`SOURCES.md`](../crates/magi-core/src/nodes/examples/SOURCES.md) |
| `avoid.jsonl` | Answers not to give: made-up tool results, tools not offered, broken calls, repeating. | `make_examples.py` |

**Every result in `work.jsonl` is real.** Each piece of work happens in a temporary folder standing for a folder on the user's computer: MacroMagi's own native tools read, search and change its files, look things up in the `magidb/` folder, and run its tests and scripts with Python – so the model learns from exactly what the tools give back, error messages included. Only what cannot run there is written out: the web, pictures, Word/Excel/PDF files and Git. `python make_work_data.py` makes it again; more and longer work is a matter of raising the numbers at its end.

The **dataset editor** opens a file in a window on the Magic tab (or a window of its own). It shows the samples as **Chat** – each sample as the chat it teaches, edited where it shows – or **Raw** JSON lines; hovering over a sample's kind shows it exactly as training writes it. **+ Tool call** lists the tools with their arguments ready to fill in. Each sample has its reasoning efforts as chips; tick samples to give them efforts together, duplicate, copy or remove them.

**Edit in bulk** changes many samples at once: **Replace** (text, or a regular expression with `$1` for its groups; match case, whole words) in chosen parts of each sample – any text, requests, answers, thinking, tool calls, tool results, or the whole line – **Delete matching** or **Keep matching**, **Delete long**, **Remove repeats**, **Trim**, and set the **Effort**, the **Tools** or take out the thinking. It reaches the selected samples, this file, or every dataset file of the model; **Preview** says how many samples would change before **Apply**; Undo takes it back.

**Generate** has another AI model write samples: one running on this computer – LM Studio's server, Ollama, llama.cpp's server, vLLM – or any OpenAI-compatible endpoint (a key only if it needs one). Say what the samples are about, the kind (requests and answers, reasoning at the model's efforts, conversations, tool use with the model's own tools – shown with what the native tools really give back) and how many: it asks for them ten at a time, checks each, leaves out repeats, and adds them as they come – one Undo takes them all back. *Thinking* sets how much a thinking model thinks first (`reasoning_effort`, and `/no_think` for Qwen models). Check what a model's terms allow: some services forbid training other models on what they write.

**Scan for issues** looks for what training would skip, misread or learn badly: samples with problems, repeats, samples longer than the context window, efforts without samples, and words the pretraining text never uses. Each sample is checked as you type: valid JSON of one of the shapes above, roles that exist, tool calls that are closed JSON naming an offered tool, efforts that are the model's.

## Training

Training runs in Python with **PyTorch**, installed under *Settings › Runtime* into the *Python for AI*. It runs on the graphics cards chosen under *Settings › Hardware*; several NVIDIA cards share each batch.

When you press **Train**, MacroMagi checks the pipeline, writes a plan and the datasets into `data/training`, and runs the trainer. The console shows each stage, its progress, the loss (as a chart), the mean reward and any warnings, and after a stage's first steps the time a step takes and the time left. When training ends, the trained model goes into the open document and the project is saved into Compile's folder.

**How it trains fast.** bfloat16 with the weights kept in full precision (float16 with a loss scale on cards without bfloat16), PyTorch's fused attention, norms and optimizer; Muon orthogonalizes the matrices of one shape together. Fine-tuning packs several requests into each row (each only sees itself), in rows as long as the step's longest request needs; the next step's rows are packed while the card works. Datasets are mixed by the square root of their size, so small hand-written files are not drowned by big ones. The output head's loss is worked out a slice of positions at a time, with one fused softmax, so the scores of every position are never kept at once. Attention takes cuDNN's fused kernel when the card has it (about twice as fast as the one PyTorch picks by itself, on lengths that come again), local windows run as the causal kernel on overlapping chunks (about four times faster than a masked window), norms hand their output on in half precision and rotary positions keep nothing for the backward pass – on Magi's layers, about twice as fast and a third less memory than before. A causal layer over a length that is not a multiple of 256 (multi-token prediction's) is padded at the end to take the same kernel (twice as fast), and Muon's step runs as fused products over all its matrices at once (30% faster, and closer to the exact result). Reinforcement learning writes its answers with a key/value cache (about fourteen times faster at 3,000-token prompts), and preparing long pieces of work counts their tokens a part at a time.

**When the graphics card runs out of memory**, training does not stop – and it never borrows system memory, which NVIDIA's driver on Windows does silently, about ten times slower (`MAGI_TRAIN_SPILL=1` allows it): it splits each batch into parts (the same work, while each part still has 8,192 tokens or more), and only past that recomputes activations (about a third more work). A batch that fits but leaves so little room that PyTorch has to empty its memory cache at every step – each step then waits, about two thirds longer – is split the same way (never recomputed); on Linux, memory is kept in growing segments (`expandable_segments`), so less of it is lost between blocks. Using all of the card's memory is not faster: four rows of 8,192 tokens already keep it busy.

**A run that stopped goes on with the same tokens**: the tokens learned at its start are read back from `tokenizer.json`, not learned again, and a corpus cut into other tokens is cut again rather than used.

**Long runs go on where they stopped.** The trainer saves where it is every 20 minutes (`MAGI_TRAIN_SAVE_MINUTES`) – training waits only for a copy in system memory, and the file is written beside it – and picks up from there when it runs again with the same plan. The save before is kept too (`resume.prev.pt`), and a model whose weights are not numbers (NaN) is never saved: a step whose gradients are not numbers is skipped, and fifty in a row stop the run. **A long run is rehearsed first**: before a run of more than a thousand steps starts (and not when it goes on from a save), every stage runs three steps on the real model and data and the model is written, in a scratch folder – a few minutes that find what would fail hours in. The tokens it learns and the corpora it cuts are kept for the run. `MAGI_TRAIN_REHEARSE=0` skips it. Everything the console shows also goes into `out/train.log`, with each step's gradient norm, so a run's history outlives the console.

### Training on a GPU server

**File › Export as Python** writes a folder that trains the model without MacroMagi: `run.py`, `plan.json`, `model.json`, `data/`, `nt/`, `runtime/magi_train/` (MacroMagi's trainer), `requirements.txt`, `colab.ipynb` and a `README.md`. It trains as the Train button does, then writes the Magi model as one file, `Magi.szi`.

On **Colab**: zip the folder, open `colab.ipynb` (File › Upload notebook), pick a **G4** GPU (Runtime › Change runtime type), and run the cells. The notebook keeps the run and the corpora, cut into tokens, on **Google Drive**: when a session ends – Colab's do – running the cells again goes on from there. At the end it downloads `Magi.szi`. Elsewhere – a rented card on RunPod, Vast.ai, Lambda – copy the folder, `pip install -r requirements.txt`, `python run.py`, and copy `Magi.szi` back.

| Colab runtime | What it is | For Magi |
| --- | --- | --- |
| *G4 GPU* | An RTX PRO 6000 (96 GB, Blackwell). | The one to use: Magi trains in about 20 hours. |
| *A100 GPU* | 40 or 80 GB. | Works; slower than a G4 for its cost. |
| *L4 GPU* | 24 GB. | Fits with smaller batches; days. |
| *T4 GPU* | 16 GB, no bfloat16. | Too slow for Magi; fine for vision models and experiments. |

### Training from a stage

Each training keeps the weights after every stage in `<project>/checkpoints/`. When some are kept, **Train** asks where to start: everything from the start; the stages after one, from its weights (pretraining once, then fine-tuning as often as the data changes); or one stage again on the trained model. Each stage block's panel says how many tokens the stage reads and how many passes over its data that is, and has **Train from here**, **Only this, on the trained model** and **Save as base model…** (its weights as a `.szi` file, for a Load Base Model block).

**Stop training** ends the Python process; the model stays as it was.

### Exporting the workspace as Python

The exported folder's `README.md` explains each file. `python run.py --package` only writes the model file (after training). `MAGI_CORPUS_DIR` says where the corpora are cut into tokens and kept.

**File › Export as Python (experimental)** writes the same folder with `"compile": true` in `plan.json` (or set `MAGI_TRAIN_COMPILE=1`): before training, the trainer times one step as it is and with each layer compiled (`torch.compile`), and keeps the compiled layers only when they give the same loss and are faster. Compiling needs Linux (Colab is Linux) and takes a few minutes; on Windows, or if anything fails – also later in the run, such as running out of memory while compiling for a new batch size – it trains as it is. Only pretraining uses it (its rows all have one length; fine-tuning's change at every step).

## Packages: the project folder

**Compile Package** writes one folder for the project, `<Save into>/<name>/`:

| Path | When | For |
| --- | --- | --- |
| `<name>.magic` | *.magic workspace* | The **workspace**: the canvas and setup, the datasets and pictures if chosen, and where its model is. It never holds the model: that is the `.szi` beside it, saved with it. |
| `<name>.szi` | *Magi Model (.szi)* | The **Magi model**, in one file (a zip archive): its description, its weights (the native engine runs these), its native tools – and its ONNX model when *ONNX file* is on. Open it in MacroMagi (File › Open, or the AI Agent window), use it as a base model, upload it anywhere, or run it in CodeMagi. |
| `onnx/<name>.onnx` | *ONNX file* | ONNX runtimes. Vision models run from it. |

## Running models

| Model | Runs with | Notes |
| --- | --- | --- |
| A Magi model (Transformer Layers, with or without experts and windows) | MacroMagi's **native engine**, from its weights | A key/value cache: each token is one step, the system text is read once and kept (and on disk), YaRN for long contexts, a three-tier memory. |
| Another text model trained here (latent or compressed attention…) | Its ONNX model, in MacroMagi | Reads the whole context for every token. |
| A vision model | ONNX, in MacroMagi | On the CPU or a graphics card (ONNX Runtime). |

**In macros** (the Macro tab's AI category), *Use in Macro* puts the model in the macro. A block's *Load model file…* takes a `.magic` workspace, a `.szi` Magi model or an `.onnx` model. When a macro stops, every AI model it loaded is unloaded.

| Block | What it does |
| --- | --- |
| **AI Agent** | Asks a text model. *Model*; *Request*; *Carry out its tool calls* (press, type, click, move, wait, log); *Longest answer*, *Temperature*; *Thinking*; *Reasoning effort*; *Context window*; *Run its native tools* and *Tool calls per request* (up to 200). Outputs: *Answer*, *Thinking*, *Tool* and *Arguments*, *Tool calls*, *Everything it wrote*. |
| **AI Load Model** | Loads a text model now, so the AI Agent blocks after it answer at once. |
| **AI Unload Model** | Frees a model's memory until it is asked or loaded again. |

## The AI Agent window

The robot button at the top right opens a chat window – a preview of CodeMagi:

- **Models**: the model open on the Magic tab (kept up to date as it trains), or a `.szi` or `.magic` file (*Load model file…*). *Load* and *Unload*.
- **Chats**: kept in `data/chats/`, each named by the model after its first answer, renamed or deleted at will.
- **Settings** per chat: the system prompt, thinking, reasoning effort, temperature, the longest answer, the context window, native tools and how many calls (100 by default), and compaction. Under *Every chat*, **Thinking shown opened** shows what the model thought in full instead of folded under *Thought* (each one can still be folded).
- **Attachments** (the paperclip): a small text file goes into the message whole; **pictures** and **documents** (.docx, .xlsx, .pptx, .pdf, .csv) – and big text files – are named by their path (`Attached: D:/shots/error.png (picture)`), and the model looks at them with `view_image`, opens them with `open_document` or reads them with `read_file`, as its datasets teach it.
- **Tool calls**: each call shows as a row – its name, its arguments in short, and whether it worked, failed or did not run. Click it for the arguments and the result in full; what the model thought before it shows above it. Calls that change files or run programs ask you first.
- **Compaction**: a model that learned compaction writes the earlier part of a long chat short when it outgrows the context window.

## Long work and long contexts

**A piece of work can take a hundred tool calls.** Each call and its result stay in the model's context, so a long piece of work outgrows it. MacroMagi's agent loop then, as Magi was trained:

1. shortens the results of earlier calls, oldest first, to their first 240 characters and a note (*an earlier result, shortened: call the tool again to see it all*), keeping the latest six whole;
2. if that is not enough, leaves out the oldest steps altogether, keeping the latest twelve, with a note that they were left out.

The system text and the request are never lost, and the model keeps track in its thinking (*Done: … Next: …*), as `work.jsonl` teaches it. The trainer learns every turn from exactly the text the loop would give the model at that point.

**The Context Scaler block** sets the context training uses and the one the finished model runs at. With **YaRN**, the rotary channels that turn many times within the trained context keep their frequency, those that turn less than once are stretched by the factor (target ÷ trained), and those between blend; attention is sharpened by 0.1·ln(factor) + 1. **NTK** raises the rotary base instead. *None* keeps the positions, for compressed attention. The scaling is applied **dynamically**: within the trained context positions are exactly as the model learned them, and only a longer text stretches them.

**The Memory & Cache block** sets where a running model keeps the keys and values of what it has read:

- **Hot tier**: the first block of 256 tokens (the "attention sinks") and the latest blocks, at full precision.
- **Warm tier**: older blocks in system memory, at the *cache precision* (F16, or 8-bit with a scale per vector).
- **Cold tier**: past the warm tier, blocks go to a file in the *disk folder* (an NVMe SSD is best) and are read back when needed.
- **Keep the system text on disk**: the memory of the system text is saved under the SHA-256 of the model and its tokens, and loaded instead of being read again.
- **Blocks read per token**: 0 reads every block; otherwise each token reads the first, the latest and the most relevant ones.

Magi's local-window layers need only their last two blocks of 1,024 tokens; only every fourth layer needs everything.

## Settings › Runtime and Hardware

**Runtime** lists what trains and runs AI models, each with its version, *Install* or *Installed*, and a menu to uninstall or install again:

| Group | Parts |
| --- | --- |
| *Python for AI* | Which Python training and the native tools use (the Python macros use). |
| Training models | PyTorch for NVIDIA graphics cards (CUDA 12.8), PyTorch for the CPU (one replaces the other), onnx (vision models). |
| Magi's tools | **MagiDB** (downloaded from its repository, a few MB), **Document tools** (python-docx, openpyxl, python-pptx, pypdf, about 15 MB, into the Python for AI), **Image descriptions** (transformers and SmolVLM-256M-Instruct, about 600 MB; needs PyTorch). Below them, *MagiDB online* (the address pages are read from when they are not on this computer) and *MagiDB folder* (a folder of pages to read instead of the downloaded copy). |
| Running ONNX models | ONNX Runtime, ONNX Runtime for graphics cards. |

What fits the computer is marked *Recommended here*. *Check again* at the top of the page looks again.

**Hardware** shows the processor, memory and graphics cards, and chooses where training runs (*Automatic*, *The CPU*, or *Chosen cards*) and where ONNX models run.

## How text is read and written

A model's tokens are the 256 byte values of UTF-8 text, then – with **learned tokens** – pieces of words, then the special tags in the order the model lists them, each one token. It is the same in MacroMagi, the trainer and the Python export.

**Learned tokens** are byte pairs learned when training starts from zero: the most frequent pair of neighbouring tokens becomes one token, again and again, up to the tokenizer's size. Text is first cut into chunks – a word with the space before it, a digit, a run of punctuation, line breaks, spaces – and pieces never cross a chunk. Any text still reads – a word the pieces never saw is spelled out in smaller pieces or bytes. A model whose data comes from a Hugging Face Dataset learns its tokens where it trains, from the start of each corpus (`MAGI_TOKENS_SAMPLE_MB`, 200 MB by default), and the trainer writes them back into the model.

A request is written as:

```text
<|system|>the system prompt

Reasoning effort: superior

You can use these tools:
- db_search(query): Searches MagiDB, the library of how things are done: the best pages for a query, with summaries.
- read_file(path, from?: integer, lines?: integer): Reads a text file, numbered lines at a time.
To use one, write <tool_call>{"name": "<tool name>", "arguments": {...}}</tool_call>. Its result comes back in the next turn.<|end|><|user|>the request<|end|><|assistant|>
```

The effort line is there with the Reasoning Effort block, the tools when the model has any – a line each, `name(argument, optional?): what it does`. The model answers until it writes `<|end|>` or reaches the longest answer. It thinks inside `<think>…</think>`; with thinking off, the answer starts after an empty `<think></think>`. A tool's result comes back in its own turn, as the tool printed it – its keys in the tool's order:

```text
<think>A folder: see it first.</think><tool_call>{"name": "list_files", "arguments": {"folder": "D:/work"}}</tool_call><|end|><|tool|>{"folder": "D:/work", "files": ["a.py"], "folders": []}<|end|><|assistant|>
```

Earlier answers are read back without their thinking, except those since the last request.

## What the architecture blocks are

The blocks are clean-room implementations of published designs, in plain PyTorch ([`runtime/python/magi_train/blocks.py`](../runtime/python/magi_train/blocks.py)):

| Block | Implementation |
| --- | --- |
| Transformer Layers | RMSNorm, rotary positions (halves layout), grouped-query attention, SwiGLU. The finer options add RMSNorm per head on queries and keys before they turn, GeGLU, RMSNorm on each part's output, a local window on all but every *N*th layer (a token sees its own block of *window* tokens and the block before; training runs it as the fused causal kernel on overlapping two-block chunks, about four times faster than a masked window, and the native engine reads the same window), rotary positions on a share of each head, query/key/value biases, a tied output head and √width-scaled embeddings. |
| Latent Attention | Queries from an optional normalized latent; keys and values from a normalized latent per token, plus a rotary key part shared by the heads (DeepSeek-V2/V3). |
| Compressed Sparse Attention | Pooled entries scored by a lightning indexer; the top *k* eligible entries chosen by rank; a window of recent tokens; a learned sink per head (DeepSeek-V4). |
| Heavily Compressed Attention | The same, with larger non-overlapping blocks attended to densely. |
| Mixture of Experts | Shared experts plus routed SwiGLU experts. Affinity by softmax, sigmoid or √softplus; choice by affinity plus a balancing bias nudged after each step (or a balance loss); weights normalized over the chosen experts; hash routing by token id. While training, each expert runs only on its tokens. |
| Hyper-Connections | *n* residual streams mixed by input-dependent, Sinkhorn-normalized matrices. |
| Multi-Token Prediction | The final hidden state and the next token's embedding go through one more layer to predict the token after next, weighted into the loss. Left out of the finished model. |
| Muon | Nesterov momentum, orthogonalized by Newton–Schulz, rescaled to an update the size of AdamW's, with decoupled weight decay. AdamW for embeddings, the head, norms and gates, at its own rate. |

Not reproduced: DeepSeek's FP4/FP8 arithmetic and their exact sizes.

## Limits

- **Magi trains on a GPU server**, not on a desktop: about 20 hours on a G4.
- **What it knows of code is what it can read, and the exact way is in MagiDB.** The starter library is small; a page for each library, program and error message the model will meet makes it better at them.
- **ONNX text models recompute the context for every token.** Models with latent or compressed attention, or hyper-connections, do not run on the native engine yet.
- **Image descriptions** need PyTorch, transformers and the vision model (Settings › Runtime); without them `view_image` still gives a picture's text and size. Reading text in pictures uses Windows' text recognition.
- **PDFs are read only.**
- **No torch.compile on Windows.** It needs Triton, which PyTorch does not ship for Windows.

## For developers

The tab turns the canvas into a JSON plan. In [`nodes/studio.rs`](../crates/magi-core/src/nodes/studio.rs), `pipeline` follows the Model wire, `derive` makes the model from the setup chain, and `plan` writes the plan. The trainer, [`runtime/python/magi_train`](../runtime/python/magi_train), reads it:

```json
{
  "kind": "text",
  "out_dir": "…/training/out",
  "devices": [0],
  "text": { "context": 8192, "max_tokens": 2048, "specials": ["<|end|>", "…"], "tools": [], "tool_tag": "<|tool|>",
            "thinking": { "on_by_default": true }, "effort": { "levels": ["standard", "superior"], "default": "standard" },
            "…": "the model's text spec" },
  "architecture": {
    "width": 768, "heads": 12, "kv_heads": 4, "context": 8192, "positions": "rope", "rope_theta": 500000, "tied": true,
    "mtp": { "weight": 0.3 },
    "layers": [ { "attention": { "kind": "full", "qk_norm": true, "window": 1024 }, "ffn": { "kind": "dense", "mult": 2.75 } },
                { "attention": { "kind": "full", "qk_norm": true },
                  "ffn": { "kind": "moe", "experts": 16, "shared": 1, "top_k": 2, "mult": 0.375, "gating": "sqrt_softplus", "balance": "bias" } } ]
  },
  "start": { "mode": "zero", "weights": null },
  "datasets": { "5.positive": [{ "corpus": { "repo": "HuggingFaceFW/fineweb-edu", "files": "sample/10BT/*.parquet", "field": "text", "tokens": 2250000000 } },
                               { "corpus": { "repo": "OpenCoder-LLM/opc-fineweb-code-corpus", "…": "…" } }, "…/data/0-language.md"] },
  "stages": [
    { "type": "pretrain", "data": ["5.positive"], "steps": 22900, "batch": 16,
      "optimizer": { "name": "muon", "muon_lr": 0.002, "embed_lr": 0.006, "warmup": 1000, "accumulate": 4 } },
    { "type": "sft", "data": ["8.positive"], "negatives": ["9.negative"], "adapter": "full", "steps": 4000, "batch": 8 },
    { "type": "compaction", "data": ["9.positive"], "steps": 400, "batch": 8 },
    { "type": "rl", "data": ["9.positive"], "steps": 300, "group": 8, "max_new": 384, "rules": [{ "kind": "ends", "score": 1 }, "…"] }
  ]
}
```

A dataset entry is a file, or a Hugging Face corpus read where the trainer runs and kept, cut into tokens, under `MAGI_CORPUS_DIR`; a `field` of `messages` is conversations, which fine-tuning reads. `"onnx": true` asks for `model.onnx` as well. The trainer saves `resume.pt` every `MAGI_TRAIN_SAVE_MINUTES` and goes on from it; tokens it learned itself are written to `tokenizer.json`. It prints one JSON object per line: `stage`, `progress` (`step`, `steps`, `loss`, `reward`), `log`, `warning`, `error` and `done`, and writes `weights.safetensors` (no pickle) and `report.json`.

Where things are: the Magi package [`magi-core/src/szi.rs`](../crates/magi-core/src/szi.rs); the native tools [`magi-core/src/native_tools.rs`](../crates/magi-core/src/native_tools.rs) and [`src/nt`](../crates/magi-core/src/nt) (a `tool.json` with `"confirm": true` asks the user first; `"run": {"kind": "host"}` is carried out by the program running the model – `view_image`, in [`magi-app/src/magic/host_tools.rs`](../crates/magi-app/src/magic/host_tools.rs)); the native engine [`magi-ml/src/native.rs`](../crates/magi-ml/src/native.rs); the agent loop, approvals and the tool runner [`magi-ml/src/tools.rs`](../crates/magi-ml/src/tools.rs); the long-sample shortening the trainer mirrors in [`magi_train/data.py`](../runtime/python/magi_train/data.py). The native engine's own check against the ONNX model: `MAGI_TEST_MAGIC_DIR=<an unpacked .magic> cargo test --release -p magi-ml native -- --ignored`.

Tests that go the whole way – Magi made tiny on its own example files, trained, packaged and asked (they need Python with PyTorch, given by `MAGI_TEST_PYTHON`; `MAGI_TEST_GPU` picks a card):

```bash
cargo test -p magi-app a_new_text_model_trains_end_to_end -- --ignored
cargo test -p magi-app a_model_trains_and_answers_through_the_interface -- --ignored
cargo test -p magi-app a_vision_model_trains_through_the_interface -- --ignored
```
