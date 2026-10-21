# Changelog

Notable changes to MacroMagi. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

File-format changes are called out explicitly; see [docs/FILE_FORMAT.md](docs/FILE_FORMAT.md) for the compatibility rules.

## Unreleased

### Changed

- **Colab keeps the whole project on Google Drive**: the notebook asks for the zip the first time only and copies the project to *MyDrive/MacroMagi/<name>/project*; when a session ends (which empties `/content`), running the cells again restores it and the run goes on – no upload. `NEW_ZIP = True` takes a new export. The finished `.szi` is copied to Drive too.
- **The terms are an end-user license agreement**: `TERMS.md` is now `EULA.md`, in the form of a classic EULA (grant of license, limitations, termination, no warranties, limitation of liability). What it allows is the same – use, read, build and modify for yourself, contribute; no selling and no distributing builds – and it says outright that what you make with MacroMagi is yours.
- **Each file has one job.** A `.magic` workspace never holds the model: saving it writes a trained model as a `.szi` of the same name beside it (again only when the model changed), and Compile Package does the same. Macros keep their AI models as `.szi` files (`assets/<id>.szi`). **File format:** older `.magic` files with a model inside, and macros with `.magic` model assets, still open.
- The document tools name file types (.docx, .xlsx, .pptx, .pdf, .csv) in their descriptions, messages and MagiDB pages.

### Fixed

- `macromagi FILE.szi` opens the Magi model in the Magic tab (it was read as a macro and failed); the command line's help names the Magic tab.
- **MagiDB downloads again**: *Settings › Runtime* and the `db_search`/`db_read` tools read MagiDB from `MagiGun/MagiDB` (they looked for `MacroMagi/MagiDB`, which does not exist).

## 0.1.7 – 2026-10-21

Everything since 0.1.5. There was no 0.1.6 release: its changes are here.

### Added

- **Magi: one architecture, an agent.** File › New text model is **Magi** – about 418 million parameters, 148 million at work per token: 768 wide and 32 layers deep, twelve heads sharing four key/value heads, queries and keys normalized; three layers in four look only at their own block of 1,024 tokens and the block before, the fourth at everything; after three plain layers, DeepSeekMoE (a shared and sixteen routed experts, two per token, √Softplus, bias balancing); multi-token prediction while training; 8K tokens in training and YaRN to 128K when it runs; 32K learned tokens; thinking on, efforts *standard* and *superior*. It knows how to work, not facts or exact code, and trains on a GPU server (File › Export as Python, Colab's G4).
- **Magi's recipe**: pretraining on three billion tokens of FineWeb-Edu and OpenCoder's code corpus (two Hugging Face Dataset blocks, chained), fine-tuning on smol-smoltalk and the example datasets with compaction mixed in, and reinforcement learning with the agent's reward rules.
- **Native tools**: each a folder (`nt/<tool>/` with `tool.json` and `run.py`) that MacroMagi runs on this computer, giving the result back to the model until it answers – ticked in the **Native Tools** block (the **Tools** block lists the ones the program running the model carries out). Magi's: `db_search`, `db_read`, `web_search`, `fetch_page`, `list_files`, `read_file` (numbered lines, 200 at a time), `search_files`, `write_file`, `edit_file`, `run_command`, `open_document`, `edit_document`, `view_image`, `calculator`, `get_time` and `get_weather`; `wiki_search`, `translate` and `transcribe_audio` are there for other models. Documents are opened as numbered parts of plain text and changed by part, saved in their own format with their styles and a backup. Pictures are described by a small vision model (SmolVLM-256M) and their text read by Windows' text recognition.
- **MagiDB**: Magi's library of how things are done – linked Markdown pages (how Magi works, Python, JavaScript, Rust, Git, the command line, SQL, documents, algorithms, error messages), in the repository's `magidb/` with its search index and how to write pages. `db_search` and `db_read` read a chosen folder, the copy downloaded under Settings › Runtime, or MagiDB online, page by page and cached.
- **Calls that change files or run programs ask first**: a window shows the command, the text or the change, with *Allow*, *Allow until MacroMagi closes* and *Don't allow*; a refused call tells the model so. **File format:** a native tool's `confirm`.
- **Tools MacroMagi carries out itself** (`"run": {"kind": "host"}`), such as `view_image`. **File format.**
- **Long work**: up to 100 tool calls a request by default (200 at most). When the work outgrows the context, earlier results are shortened and then the oldest steps left out, keeping the system text and the request – and the trainer learns every turn from exactly that text.
- **The native engine**: models of Transformer Layers trained here – experts and local windows included – run straight from their weights with a key/value cache: each token is one step, and a request that starts like the last one only reads what is new. Checked against PyTorch to within 0.000005.
- **A Magi model is one `.szi` file**: the description, the weights and the native tools (and an ONNX model only when asked for) in one zip archive – upload it, download it, open it, use it as a base model. **File format:** `.szi`.
- **The AI Agent window** (the robot button, top right): chats with the Magic tab's model or a `.szi` or `.magic` file – chats kept and named by the model; per-chat settings (system prompt, thinking, effort, sampling, context, native tools and how many calls, compaction); pictures and documents attached by their path; sent messages edited, deleted or answered again; every tool call a row that opens to its arguments and result, with what the model thought before it; *Thinking shown opened* for every chat. A long chat's earlier part is written short when the model learned compaction. A preview of CodeMagi.
- **File › Export as Python**: a folder with `run.py`, the plan, the datasets, the native tools, the trainer, a README and `colab.ipynb`, which trains the model without MacroMagi – on Colab, keeping the run and the corpora on Google Drive – and writes its `.szi`. *Export as Python (experimental)* also tries compiled layers (`torch.compile`, Linux only), kept only when faster with the same loss.
- **Long runs go on where they stopped**: the trainer saves every 20 minutes (in the background) and picks up from there with the same tokens when it runs again.
- **Hugging Face Dataset block**: a dataset on huggingface.co read where training runs – a part at a time, cut into tokens once (about two million tokens a second), never kept in the workspace. A column of `messages` is conversations to fine-tune on; a Text Dataset or another Hugging Face Dataset wired into its *With* input is read along with it.
- **Learned tokens**: text cut into pieces of words learned from the data (byte pairs) instead of bytes – about a third as many tokens. The Text block chooses *Learned* or *Bytes*. **File format:** `text.tokenizer`.
- **Long contexts**: the **Context Scaler** block trains at a short context and runs at a long one with YaRN (or NTK); the **Memory & Cache** block keeps a running model's memory in hot, warm and disk tiers.
- **Training from a stage**: each training keeps the weights after every stage (in the project's `checkpoints/`); Train offers everything, the rest from after a stage, or one stage again on the trained model.
- **Base models: pretrain once, fine-tune in minutes.** *Save as base model…* on a stage block writes a `.szi`; a Load Base Model block pointing at it fine-tunes without pretraining again, taking the base's learned tokens and checking its architecture.
- **A new dataset editor**: samples show as the chat they teach and are edited right there; *Raw* shows the JSON lines; undo and redo; reasoning efforts per sample; *+ Tool call* fills one field per argument; **Edit in bulk** (find and replace with regular expressions, delete or keep matches, remove repeats, trim, set effort or tools, take out thinking – with a preview); **Generate** (LM Studio, Ollama, llama.cpp, vLLM or any OpenAI-compatible endpoint writes samples, each checked); **Scan for issues** (problems, repeats, samples longer than the context, efforts without samples, words the pretraining text never uses), also over every file. It pops out into a window of its own.
- **Example datasets for an agent**: `work.jsonl` (1,400 long pieces of work, up to 78 tool calls each, every result from the real tools run on real files: fixing tests, adding options, renaming across up to 30 files, scripts from data, spreadsheets, documents, PDFs, screenshots, MagiDB questions, error messages, Git), `agent.jsonl`, `thinking.jsonl`, `knowing.jsonl`, `small_talk.jsonl`, `explain.jsonl`, `simple_qa.jsonl`, `compaction.jsonl`, and conversations and tool calls from open datasets (credited in `SOURCES.md`). Dataset messages can be marked `"learn": false`: read, not learned.
- **Reward rules for agents**: *Calls the tool the dataset's answer calls first*, *Calls a tool it is not offered*, *Writes a tool's result itself*.
- **Macro blocks AI Load Model and AI Unload Model**, and AI Agent settings for thinking, reasoning effort, context window and native tools. Model blocks load `.magic`, `.szi` and `.onnx` files; when a macro stops, every model it loaded is unloaded.
- **The console** gives each stage's time, estimates the time left, and says when a run would take more than twenty minutes – better done on a GPU server or from a base model. Stage blocks say how many tokens and passes over their data they read.
- **Notify when training ends** (Options, on by default): a sound and a desktop notification when a run of more than a minute finishes or fails.
- **float16 on graphics cards without bfloat16** (Colab's T4, older cards), with a loss scale.

### Changed

- **Training is many times faster and leaner** on a graphics card: bfloat16 with full-precision weights; fused norms, AdamW and Muon; local windows as the fused causal kernel on overlapping chunks; cuDNN's attention; grouped products for the experts; the output head's loss a slice at a time with a fused softmax; fine-tuning requests packed into rows only as long as each step needs; reinforcement learning writing with a key/value cache. When memory runs short, batches are split before activations are recomputed, and training keeps within the card's free memory instead of spilling into system memory (about ten times slower on Windows).
- **Local windows are block-aligned**: a token sees its own block of *window* tokens and the block before, in training and in the native engine alike.
- **Only Magi.** Presets › Commons offers Magi alone; the Qwen3, Gemma 3, Mistral, GLM-4, DeepSeek-V3/V4 and Llama layouts are gone. Their blocks are still in the block list.
- **Compile Package** (was *Compile .magic Package*) has a switch for each thing it saves: the *.magic workspace* (with or without its datasets and pictures), the *Magi Model (.szi)* and an *ONNX file*. One folder per project.
- **Use in Macro** adds Start → AI Load Model → AI Agent → Log → AI Unload Model.
- **Text & Tools** became two blocks, **Text** and **Tools**.
- **Reasoning in datasets is learned only with the Thinking block** chained in and on; with thinking on, answers without reasoning are learned after an empty thinking span, so a model no longer works out a greeting like a math problem.
- **Muon is every Optimizer block's default**, with two learning rates (0.004 for the layers, 0.02 for the embeddings and norms). **File format:** the optimizer's `embed_lr`.
- **Tools are listed a line each** in new models' system text, and tool results read exactly as the tools print them. **File format:** `text.tool_list`.
- Settings › Runtime has *Magi's tools* (MagiDB, Document tools, Image descriptions) and *Uninstall* on installed parts; Settings › Engine calls MacroMagi's own Python *Embedded Python*.
- Less memory when idle: no ONNX copy of a model the native engine runs, and the AI Agent window keeps only what chatting needs.
- Window titles read “*Magi • MacroMagi”.

### Removed

- **GGUF, LM Studio, llama.cpp and Hugging Face exports**: GGUF models no longer import or run, and llama.cpp is no longer offered under Settings › Runtime and Hardware. Old files open; the settings they had for these are ignored.

### Fixed

- On Windows, training that needed more graphics memory than was free ran about ten times slower instead of saving memory.
- Small settings such as learning rates showed as 0.00; they show their digits now and drag in proportion.
- Some notes had runs of spaces in the middle of a sentence.

## 0.1.5 – 2026-10-20

### Added

- **The Magic tab is a node canvas** for building, training, trying and packaging AI models. It has the Macro tab's panning, zooming, wires, groups and searchable block list, a panel for the selected block (its settings, and what is wired into it), and a console with the loss as a chart.
- **Every wire means something.** The *Model* wire runs from a start through the architecture and training stages to Compile .magic Package and Prompt Model / Test Model. Setup blocks (Model Info, Text & Tools, Screen Area, Answers…) chain into the start's *Setup* input, and the model is what they say. *Feed-forward* wires take a Mixture of Experts or Persona Router into layer blocks. *Optimizer* wires take an Optimizer into the stages it drives. Blocks that are not connected where they would be used say so on their faces.
- **The banner works for the Magic tab**:
  - **File**: New text model, New vision model, Open, Open recent, Import (GGUF, ONNX project, `.zip`/`.onnx`), Save, Save as, Close.
  - **Options**: *Optimize environment when training*.
  - **Presets**: *Commons*, saved Magic blocks, *Save selected blocks*.
  - **Add**, **View**, undo/redo, and **Train** / **Stop training**.
- **Presets › Commons**: Magi (MacroMagi's own layout for small agents), Qwen3, Qwen3 Thinking, Gemma 3, Mistral, GLM-4, DeepSeek-V4, DeepSeek-V3 and Llama, each scaled down, put in place of the canvas's architecture (with the setup blocks they need).
- **Thinking** and **Reasoning Effort** setup blocks: training learns answers with and without reasoning and at each effort level; Prompt Model and LM Studio get a Thinking switch and a Reasoning Effort choice.
- **Transformer Layers' finer options**: normalized queries and keys, GeGLU, norms before and after each part, local windows with global layers, partial rotary positions, attention biases; Model Size adds tied and scaled embeddings.
- **Architecture blocks**:
  - attention: Transformer Layers (the Llama layout: RMSNorm, rotary positions, grouped key/value heads, SwiGLU), Sliding Window Attention, Latent Attention (MLA), Compressed Sparse Attention (V4's CSA with its lightning indexer), Heavily Compressed Attention (V4's HCA), Mamba-2 / SSM;
  - feed-forward: Mixture of Experts (DeepSeekMoE with softmax, sigmoid or √softplus gating, bias or loss balancing, hash routing), Dual-Persona Router;
  - whole-model: Hyper-Connections (V4's mHC), Multi-Token Prediction, Encoder/Decoder Split, Vision Encoder.
- **Training on your computer** with PyTorch, on the CPU or one or several NVIDIA graphics cards:
  - stages: Pretraining with a context curriculum, Fine-Tuning (all weights, LoRA or DoRA, with "what not to do" samples), Reinforcement Learning with reward rules, Context Compaction, Vision Training with negative pictures;
  - an Optimizer block per stage: AdamW, Muon as DeepSeek-V4 trains with it, or SGD, with cosine or constant schedules, warm-up, clipping, accumulation, recomputed activations and offloaded optimizer state.
- **A new text model is a working example**: Magi, a 3.5M-parameter assistant with Thinking and Reasoning Effort, four general tools, and small original datasets – plain English, conversation, reasoning at three efforts, tool use with results, chat titles, answers to avoid. It trains in a few minutes on a recent graphics card.
- **Compile .magic Package** saves the model into the folder you choose when training ends: `.magic`, and for Llama, Qwen3 and Gemma 3 layouts a **GGUF** file (F16, Q8_0 or Q4_0, with its size and precision for LM Studio) and a **Hugging Face** folder (Llama, Qwen3). The chat template takes OpenAI-style messages, tools, `enable_thinking` and `reasoning_effort`; llama.cpp reads thinking, answer and tool calls apart. Its panel has *Save package now*, *Add to LM Studio* (the GGUF plus a `model.yaml` with Enable Thinking and Reasoning Effort), *Use in Macro* and *Open folder*.
- **Prompt Model** keeps the model loaded (*Load*, *Unload*), runs it with the built-in engine or llama.cpp, and sets the context window, thinking, effort, longest answer and temperature; it shows the answer, thinking and tool calls. **Test Model** watches the screen or tries a picture.
- **GGUF models**: import them (*File › Import*), prompt them, and run them in macros with llama.cpp, on the graphics cards you choose.
- **Text Dataset** block and **dataset editor**: `.jsonl`, `.md` and `.txt` files in the `.magic` file, searched, filtered, checked (JSON, sample shape, every tool call) and edited in place, with *Add an example* and *Add the examples*. **Vision Dataset** block: pictures per answer and pictures it was mistaken for, with thumbnails.
- **AI Agent** block: asks a text model (ONNX or GGUF) and gives its answer, thinking, first tool call and arguments; optionally carries out press, type, click, move, wait and log calls. *Use in Macro* adds a text model with Start → AI Agent → Log.
- **Settings › Runtime**: what trains and runs models, each with its version, *Install*, *Uninstall* and *Install again*. It covers PyTorch for NVIDIA graphics cards (CUDA 12.8) or for the CPU, onnx, llama.cpp (CUDA 13, CUDA 12, Vulkan, CPU, from its latest release), and ONNX Runtime for the CPU and for graphics cards. It shows which Python training uses.
- **Settings › Hardware**: the processor, memory and graphics cards. It chooses where training runs (Automatic, the CPU, or chosen cards), how GGUF models use cards (which build, how many layers, which cards), and where ONNX models run.
- **Options › Optimize environment when training** (on by default): while a model trains, the Magic tab shows only the training, and the background picture and see-through panels are off.
- If MacroMagi ever fails, it writes `crash.log` into the data folder and says so, instead of closing without a word.
- Settings › Runtime and Hardware are looked at when MacroMagi starts and whenever something that needs them fails; *Check again* looks now.

### Changed

- Opening a `.magic` file from an earlier version builds its canvas from it, with its setup blocks chained into a *Use As It Is* start.
- Settings › AI became **Runtime** and **Hardware**. PyTorch and onnx are listed separately, so an installed PyTorch shows as installed.
- AI Watch and AI Model report an error when given a text model.
- On the Magic tab, Ctrl+N, Ctrl+O, Ctrl+S, Ctrl+Shift+S, Ctrl+Z and Ctrl+Y work on the model.

### Fixed

- The dataset editor closed MacroMagi when a sample was added.
- Installing llama.cpp failed when its latest release had no Windows builds; the newest recent release that has them is used.
- Every prompt reloaded the model.
- llama.cpp servers MacroMagi started could outlive it after a crash; Windows now closes them with it.
- LM Studio showed no size or precision for exported GGUF files, and the example model named every chat "type".

### File format

`.magic` additions, all optional; earlier files load unchanged:
- in `magic.json`: `kind`, `text` (with `tool_tag`, `thinking` and `effort`), `architecture`, `studio` (the canvas), `datasets`, `trained` and `gguf` (a GGUF file the model runs);
- in the archive: `model/weights.safetensors` and `data/text/…`;
- `negative` on pictures, and `weights` and `datasets` under `include`.

New macro block `ai.agent`.

## 0.1.4 – 2026-10-19

### Added

- **Files** blocks: **Read File** (all the text, one line, or a picture), **Write to File** (set the file, add to it, or change one line; a picture is saved as a PNG, replacing one of the same name), **Find in Text** (on which line a word is, what that line says, on how many lines), **Folder Contents** and **Make Folder**. Paths have a Browse button and can be wired in; relative paths go into the `files` folder in the data folder.
- **Screenshot** block: a picture of the display the pointer is on, a chosen display or an area, kept while the macro runs. Pictures from Screenshot or Read File can be wired into **Image Detection** (its new *Picture* input) or **Write to File**.
- **Switch to Display** block: moves the pointer to the middle of a display, and can click there.
- **Current Window** block: the title and program of the window in front, its display, the pointer's display and its middle.
- **Every drop-down can be searched**, the Stop block's *Continue from* included.
- **Auto Typer**: a list of characters to leave out, with Space, Line break and Tab buttons.
- **Recording**: *Skip task switches* (off by default) leaves out Alt+Tab, Win+Tab or a click on another display – the first, the last and those in between, each switchable. The recording editor can add *Move* and *Look* rows, and pick where a movement starts or ends on screen.
- **Recordings** can be deleted from a Play Recording block's drop-down; recordings no block plays are deleted when the macro is saved (*Automatically delete unassigned recordings when saving*, in Settings › Recording and Options › Behaviours; on by default).
- **File › Save as** can save a `.mp` preset – of the selected blocks, or the whole workspace. **File › Export presets** exports the workspace, the selection, or one of your presets.

### Changed

- A Smart recording gets the same conversion as the editor's *Smart* button when it ends, so it is as accurate as converting it by hand.
- The Presets menu lists only your presets, without the ones stored in the open file.
- The AI settings page looks for Python by itself, so PyTorch's status is right without visiting the Engine page; a PyTorch that is found offers *Uninstall* instead of *Install*.
- The program picker of Switch to Window stays open while you type in its search box, and lists programs with windows first.

### File format

All additions are optional; earlier files load unchanged. New blocks `vision.screenshot`, `window.display`, `window.current`, `files.read`, `files.write`, `files.find`, `files.list`, `files.make_folder`; Image Detection's `picture` input; the Auto Typer's `ignore`.

## 0.1.3 – 2026-10-17

### Added

- **Auto Typer** block: types text character by character – its own text, a linked variable, or a value wired in – at a speed in characters per second with random variation, longer pauses after punctuation, optional typos corrected with Backspace, trimming, skipping the first characters, a limit, a choice of what line breaks do, and Enter at the end. Outputs how many characters it typed.
- **Text Scan** block: reads the text in an area of the screen with the text recognition built into Windows (nothing to download). Outputs the text, whether a given text is there, where it is (to click it), and the first number in it. Can enlarge small text and read a chosen language.
- **Nested groups**: grouping blocks that are already in a group puts the new group inside it, drawn inside its box. Clicking a group's title selects the group itself; right-click it for *Select everything in it* and *Move the selection into* it. *Take out of its group* moves a block one level out.
- **AI Watch** has an *Acted* output: the blocks under it run each time the model acts, with *Times acted* and *Last answer* up to date, and AI Watch then goes on watching. *Done* follows when its time is up (or a Stop leaves the loop).
- **Lightweight mode** (the feather at the top right on the Macro tab): only Run, Stop and the console, over the theme and background, without the editor.
- **Key pickers** can be searched and have *Press a key…*, which takes the next key pressed. In the Input block and for a model's key in the Magic tab they also offer mouse buttons, so a model can click instead of pressing a key.
- **Close buttons** on the console and the Variables panel.
- **About**: click the logo at the top left, or see *Settings › General › About*. MacroMagi is developed and created by [MagiGun](https://github.com/MagiGun) and [uvxszi](https://github.com/uvxszi).
- Download progress shows as a small bar.
- The bin next to an image (Image Detection) or a sound (Sound Recognition) also deletes it from the macro when no other block uses it. Files on your computer are never touched: loaded files are copied into the macro.

### Changed

- **Image Detection and Sound Recognition**: groups and items are one list. Each group shows its images or sounds, the ones in no group come last, and an item is dragged by its handle onto a group to put it in. The group drop-downs are gone.
- **Settings › Engine** no longer starts anything when opened. *Check* looks for Pythons and tries the one macros use; Python is also looked for once, the first time a macro runs, and what was found is remembered. The drop-down shows what *(auto)* stands for, with its version and file. Python is downloaded – and uninstalled – here.
- **Settings › AI** lists ONNX Runtime, its graphics card build and PyTorch, with *Check again* at the top and *Uninstall* instead of *Remove*. PyTorch goes into the Python macros use.
- On Windows, keys are sent as scan codes, as a keyboard sends them, so games that ignore other key input receive them.
- The Comment block's color has a color picker.
- Rows of buttons continue on the next line in a narrow window instead of being cut off.
- The interface size has the usual reset arrow instead of a *100%* button; the theme color swatches no longer touch the edge.
- "Colour" is spelled "color" everywhere.
- Releases are built for Windows only, for now.

### Fixed

- AI Watch's *Times acted* and *Last answer* reached other blocks only after it stopped watching – by default never.
- Installing PyTorch asked for Python to be downloaded even when macros were running on an installed one; a Python from the Microsoft Store was not found.
- Opening Settings › Engine or › AI flashed command windows.
- The "→" in some messages showed as a box.
- Links (the credits, "Get Python from python.org") did not open the browser.

### File format

All additions are optional; earlier files load unchanged. Groups may have a `parent` (a group inside another); in files where a block was in several groups, it stays in the smallest and the groups nest. New blocks `input.autotype` and `vision.read_text`. AI Watch has the flow output `acted`. A model's key in `.magic` files may be a mouse button (`"Mouse:Left"`). See [docs/FILE_FORMAT.md](docs/FILE_FORMAT.md).

### Known limitations

- Text Scan uses the text recognition built into Windows; it reads only the languages Windows has text recognition for (usually the display language), and does not work on Linux or macOS yet.
- Linux and macOS packages are not built for now; building from source still works.

## 0.1.2

Not released on its own: its changes are in 0.1.3.

### Added

- **Magic tab**, beside Auto Clicker and Macro, for small vision AI models that macros use.
  - **Import** an ONNX model, or a project folder or ZIP that contains one. The importer reads the project's files – it never runs them – for the class names, which classes call for action, the color normalisation, the screen area, the key and the adjustable settings, and lists what it found and what it guessed.
  - **Configure** the name and description; the screen area (a square in the middle of a display that scales with it, or a fixed area picked on screen); each answer's name, whether it acts, how sure the model must be and how long to wait; the key, how long it is held and the cooldown; adjustable settings that move several values at once; how pictures are prepared; and the pictures to keep for training and fine-tuning.
  - **Try it**: watch the screen live and see what the model sees and how sure it is of each answer, with or without pressing the key, or try it on a picture file.
  - **Save** `.magic` files: the model, its setup, ready-made blocks and its pictures, each optional.
  - **Use in Macro** adds the model to the open macro with blocks that run it. Using it again updates the model and the blocks that use it.
- **Models made for the DBD Auto Skill Check project** run as they do there: importing its repository or its packaged app (folder or ZIP) finds its ten answers, the five that act, the 51% confidence, the 5 ms frontier delay and its screen area, with sliders for the three settings it offers.
- **AI blocks**: **AI Watch** keeps looking and presses the model's key when it should (or only watches), for a time or until stopped, as often as allowed; **AI Model** looks once and gives the answer, how sure the model is, and whether to act. Each block can change the model's adjustable settings for itself, load a `.magic` file, or open its model in the Magic tab.
- **Settings → AI**: run models **automatically** (each way is measured on the model and the fastest kept; the CPU when a graphics card is not faster), on the **CPU**, or on a chosen **graphics card**.
- **Downloads** on the AI settings page (also *Options → Downloads…*): ONNX Runtime (faster on the CPU), ONNX Runtime for graphics cards (DirectX 12, Windows), Python and PyTorch, each with its size, Install and Remove. Copies already on the computer are detected.
- **Python**: the Windows package now comes with Python. *Settings → Engine* lists every Python found, with its version, in a drop-down, and offers *Download Python* when there is none.
- **Image Detection** looks for several images in one block, in order of priority. Images can be put in groups; an image or a group can have its own search area, otherwise the block's is used. *Found when* any or every image is there; **Detail** (Full, Half, A third, A quarter) trades accuracy for speed. New outputs: *Which image*, *Its group*, *How many*. The block shows up to four of its images, and Find (Ctrl+F) finds blocks by the names of their images and groups.
- **Sound Recognition** listens for several sounds in one block, in groups that can have their own accuracy; *Found when* any or every sound has been heard; *Recognise from the first* recognises long sounds while they are still playing. New outputs: *Which sound*, *Its group*, *How many*.
- **Microphone choice** for Sound Detection and Sound Recognition; the system's default when none is chosen.
- **Clear** button in the console.
- Variables can keep their value from one run to the next: *Reset at the start of each run* in the Variables panel. New variables keep their value and start empty; a kept value can be emptied from the panel.
- Blocks that need the Python engine show a warning sign when it is not available; it explains why and opens *Settings → Engine*. The native engine skips them, with a warning in the console, instead of stopping.
- A `.magic` file opened with MacroMagi, or named on the command line, opens in the Magic tab.

### Changed

- The screen picker's display bar can be dragged, opens under the pointer, and stays inside the display.
- *Strict hotkey modifiers* and *Stop on task switch* are under *Behaviours* in the Options menu.
- The Log block's message is empty by default, so it logs the value wired into it.
- The banner logo takes the theme color, like the buttons.
- Sound recognition is cheaper on long sounds: what has been heard is analysed once, not on every look.

### Fixed

- The Presets menu became very wide once it held a preset.

### File format

All additions are optional; files from 0.1.0 and 0.1.1 load unchanged. New `.magic` format for AI models. In `.magi` files: `model` assets; Image Detection has `images`, `groups`, `mode` and `detail` (the single `image` of older files is still read); Sound Recognition has `sounds`, `groups`, `mode` and `match_ms` (likewise for `sound`); sound blocks have `device`; variables have `reset` (`true` when missing). Presets may embed models. See [docs/FILE_FORMAT.md](docs/FILE_FORMAT.md).

### Known limitations

- Training and fine-tuning are not in the Magic tab yet. Pictures can already be collected into `.magic` files by class, and PyTorch can be downloaded for it.
- Graphics cards are used on Windows only; elsewhere models run on the CPU.
- Image Detection runs on the CPU, where its search is fastest.
- A model keeps the licence of the project it came from; the Magic tab shows the licence an import found. Models imported from DBD Auto Skill Check are GPL-3.0, so `.magic` files and macros that contain one carry those terms when shared.

## 0.1.1

Not released on its own: its changes are in 0.1.3.

### Added

- **Variables.** A macro has named variables with starting values. Any block setting can follow a variable through the link button beside it, so several blocks share one value and change together. New blocks: **Variable** (read one), **Set Variable** (set, add, subtract, multiply, divide, append – from the setting or from a wired value) and **Counter** (count passes, with a limit and a *Reached* output). A **Variables** panel lists them with their starting values and their values in the latest run. Python scripts read and change them through `magi.vars`, and a Python Script block can store its result in one.
- **Logic for values**: **Math** (add, subtract, multiply, divide, remainder, power, min, max, rounding, random…), **Compare** (equal, less, greater, contains, starts with, ends with) and **Text** (join, length, upper/lower case, trim, replace, read as a number).
- **Rate** block: a list of inputs done one after another, a set time apart. The time can follow a variable or be wired in from another block. **Variable Rate Editor** window for its timing and steps.
- **Definition** and **Function** blocks: build a piece of a macro once under a Definition and run it from any number of Function blocks, passing a value in through a variable.
- **Repeat until / while** a condition, checked before every round.
- **Stop** can now end only the current round of a loop, leave the loop, or continue from another chosen block.
- **Clipboard** block (the text on the clipboard) and **Notification** block (a sound and a desktop notification). *Options → Notify when a macro finishes* does the same whenever a macro ends by itself.
- **Sound Detection** (loud enough, optionally within a pitch range) and **Sound Recognition** (an uploaded WAV, MP3, OGG or FLAC sound, recognised with an adjustable accuracy at any volume), listening to what the computer plays or to the microphone.
- **Comments**: notes on the canvas that fold up and resize by their corner, in any color.
- **Groups**: put blocks in a named, colored box (Ctrl+G); drag the box to move them together, and turn the whole group off to skip every block in it.
- **Find** (Ctrl+F): search blocks by name, custom name, type or settings, and groups by name.
- **Several displays**: every click position and search area remembers the display it is on, chosen in its settings; Image and Color Detection search a chosen display; the screen picker and Screen targets open on the display MacroMagi is on and switch between displays; recordings note when the pointer moves to another display.
- **Recording Editor** window: drag rows by their grip to reorder them, **Cut** (remove the part between the handles and close the gap) beside **Trim**, and rows for key down/up, button down/up and display changes. Recordings use *down* and *up* rows only when other inputs happen while a key or button is held; otherwise one row with its hold time.
- **Recording options** in a menu beside Record: what to record and how to record movement (Smart Recording by default).
- **Reset buttons** beside every block setting and every Auto Clicker value that differs from its default.
- **View** menu: show or hide the grid, the block activity glow and the variables.
- The block list shows every block. When a wire is dropped on the canvas, the blocks it can connect to come first and the others after them.
- A fourth example, *Variables, functions and groups*.
- A new icon, used for the window, the executable and desktop shortcuts.

### Changed

- **Auto Clicker**: the units after "per" read second, minute, hour, day; Variation sits under High-speed mode; the speed-limit note says to turn on High-speed mode to go faster; Two keys alternate by default, with *Alternating* first; Key and Two keys hide the settings that only matter for the mouse.
- A Play Recording block's information card shows what the recording holds and a **Recording Editor** button; the block's settings show the timeline, handles, Trim, Cut, speed and Smart, without the rows.
- **Screen targets** no longer darkens the screen, redraws without flickering, and a double-click on a target or its line in the list shows its block in the editor.
- The Options menu is narrower; the console, the block settings panel, the settings window and the program rules window keep the size you give them.
- Number settings with a wide range (such as loudness or pitch) change at a sensible pace when dragged.

### Fixed

- Hotkeys could not be assigned, and did not fire, while MacroMagi's own window was in front. The window library's Raw Input keyboard registration made Windows skip MacroMagi's keyboard hook; MacroMagi now takes Raw Input over itself.
- Panels and windows grew wider by themselves.
- The screen picker and Screen targets could crash on very large displays.

### File format

All additions are optional; files from 0.1.0 load unchanged. Projects may have `variables` and graph `groups`; any setting may be a variable binding (`{"$var": "name"}`); positions and areas may have a `display`; recordings may contain `key_down`, `key_up`, `mouse_down`, `mouse_up` and `display` strokes; assets may be `audio` (WAV). See [docs/FILE_FORMAT.md](docs/FILE_FORMAT.md).

### Known limitations

- Listening to what the computer plays needs a loopback or monitor input on Linux and macOS (Windows has one built in).
- Switch to Window, program rules and smart recording are still Windows-only; gamepad input still cannot be sent.

## 0.1.0 – 2026-10-15

The first version. Everything is new.

### Added

- **Auto Clicker** tab: mouse button, key or two keys (together or alternating); single/double/triple clicks with an adjustable gap; speed as clicks per second/minute/hour/day or as an interval from milliseconds to 365 days; speed variation; fixed hold time or duty cycle; course correction (on by default); high-speed mode up to 1000 clicks per second, with a warning and automatic hold shortening; click limit, time limit, and corner and edge stops; click at the pointer or a picked position; toggle or hold-to-run hotkey; separate Start and Stop buttons; undo/redo; presets; "Open as macro".
- **Macro** tab: a node graph editor with pan and zoom, box and multi-selection, copy/paste (carrying images and recordings), duplicate, disable, undo/redo, a searchable, keyboard-driven and resizable block list, a settings panel generated from block definitions, a console, and three built-in examples.
- **Banner menus** per tab: File (both tabs), Options, Presets, Open as macro or Add and View, undo/redo; buttons for settings, keep-on-top and hide-from-screen-captures.
- **Blocks**: Start, Wait, Repeat, Decision, Wait Until, Stop, Log, Input, Auto Click, Play Recording, Image Detection, Color Detection, Switch to Window, Logic, Random Chance, Python Script.
- **Switch to Window**: bring a program to the front by title or program name, aware of windowed, borderless and exclusive fullscreen, with a "Not found" branch.
- **View → Screen targets**: every click position and search area shown over a frozen screenshot, numbered in run order, draggable and resizable.
- **Recording** of keyboard, mouse buttons, mouse movement, scroll wheel and gamepad input, stored as press time plus hold duration per input. **Smart recording** detects games that lock the pointer and records relative motion; Play Recording can replay movement as recorded, as screen positions or as relative motion; older recordings convert from the block's context menu.
- **Recording editor**: a spreadsheet of every input with every field editable for every kind of input, changeable row types, add/duplicate/delete, a timeline with trim handles, speed changes, and a pop-out window.
- **Image Detection** by color-aware template matching with a coarse-to-fine search, an adjustable accuracy threshold (default 80%), an optional search area, and a screen picker for capturing images, areas, points and colors.
- **Two engines** with identical behaviour: Python (standard library only, auto-detected, with Python Script blocks and a `magi` scripting API) and native Rust.
- **Hotkeys** from any keyboard key (Escape and function keys included), mouse button or gamepad button, alone or combined; Auto Clicker, Run, Record, stop-everything and any number of extra stop hotkeys; all listed in Settings with conflict warnings and a reset button. Run and Record show their hotkeys and ask before starting when theirs has been removed.
- **Options** menu: strict hotkey modifiers, stop on task switch (Alt+Tab, Win+Tab), auto-stop hotkeys, program rules (whitelist or blacklist with search and per-program stop or pause), stopping the Auto Clicker while a hotkey is edited, dry run, and creating a desktop shortcut.
- **Appearance**: monochrome by default; a theme color that recolors the whole interface; Light, Dark or System theme (System by default); a background picture or animated GIF/WebP from a file or web address, with opacity, blur, darkness and panel opacity.
- **Settings window**: movable and resizable, in pages (General, Appearance, Hotkeys, Recording, Engine), with "Reset all hotkeys" and "Reset all settings".
- **Export** dialog: choose to include presets, Options and Auto Clicker settings in a `.magi` file; files that carry them offer to apply them on opening.
- **Safety**: keys and buttons held by a macro are released when it stops; dry-run mode; on Windows, input sent by MacroMagi never triggers its own hotkeys or ends up in its own recordings.
- **Memory**: an idle Python process is stopped after a few minutes, and memory is handed back to the system while the window is minimised.
- **Files**: the `.magi` project format (a ZIP with a readable `project.json` and linked assets) and the `.mp` preset format (JSON), both at format version 1. *Format:* projects may carry `options` and `clicker`; recordings may contain `look` strokes and clicks with `here`; Auto Clicker settings gained the fields listed in FILE_FORMAT.md. All additions are optional; earlier files load unchanged.
- **Command line**: `run` (with `--engine` and `--dry-run`), `nodes`, `example`.
- **Releases**: a GitHub Actions workflow builds Windows, Linux and macOS packages and publishes a release only when every build succeeds.
- **Terms**: MacroMagi is source available under [TERMS.md](TERMS.md). [AGENTS.md](AGENTS.md) guides AI coding agents and logs their revisions.

### Fixed

- Hotkeys that never took effect, or only after a restart: Windows silently removes slow keyboard hooks, which left only mouse hotkeys working. The listener now runs at high priority and a watchdog reinstalls the hooks. Any key and combination can be assigned and works immediately.
- The *Add* button and the *Add block* context-menu entry opened the block list and closed it again at once.
- The first entry of the block list was drawn as tall as the whole list.
- Typing a percentage into *Interface size* set the wrong size; the size now changes only after the slider is released.
- *Trim start* in the recording editor appeared to do nothing (recordings already start at the first input); trimming is now done with handles on the timeline.
- Clicks landing on the Auto Clicker's start button no longer stop it.

### Known limitations

- Linux and macOS builds are implemented but untested. Switch to Window, program rules and smart recording are Windows-only for now.
- Gamepad input can trigger macros and be recorded, but cannot be sent.
- Whole-screen searches, the screen picker and the screen-targets view use the primary monitor only (lifted in 0.1.1).
