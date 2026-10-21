# Contributing to MacroMagi

Thanks for wanting to help. This page covers how to get set up, how the project is organised, and what a good change looks like.

MacroMagi is source available under the [MacroMagi EULA](EULA.md). By contributing you agree to the contribution terms in section 3 of that document, and to its code of conduct. If you work with an AI coding agent, point it at [AGENTS.md](AGENTS.md).

- [Ways to help](#ways-to-help)
- [Getting set up](#getting-set-up)
- [Everyday commands](#everyday-commands)
- [Where things live](#where-things-live)
- [Making a change](#making-a-change)
- [Code guidelines](#code-guidelines)
- [Testing](#testing)
- [Reporting bugs](#reporting-bugs)
- [Releasing](#releasing)

## Ways to help

- **Test on Linux and macOS.** Those platforms are implemented but have barely been run. Building, starting the app and telling us what breaks is genuinely valuable.
- **Report bugs** – see [Reporting bugs](#reporting-bugs).
- **Add blocks.** [docs/ADDING_A_NODE.md](docs/ADDING_A_NODE.md) walks through it.
- **Improve the docs**, including this file.
- **Pick up something from the [roadmap](README.md#roadmap).** For anything large, open an issue first so we can agree on the approach before you write it.

## Getting set up

1. Install [Rust](https://rustup.rs). The repository pins its toolchain in `rust-toolchain.toml`; `rustup` fetches the right version, with `rustfmt` and `clippy`, the first time you build.
2. Optional: install Python 3.8 or newer (on Windows, *Settings › Engine* in the app can fetch one). It is needed only for the Python engine and the tests that exercise it; those tests skip themselves without it.
3. On Linux, install the system packages listed in the [README](README.md#building-from-source).

```bash
git clone https://github.com/MagiGun/MacroMagi.git
cd MacroMagi
cargo run -p magi-app
```

The first build takes a few minutes; later ones are quick. Dependencies are optimised even in debug builds (see `Cargo.toml`), so the debug app is smooth enough to work with.

When run from the source tree the app keeps its data in `target/debug/data`. Set `MACROMAGI_DATA_DIR` to put it elsewhere.

## Everyday commands

```bash
cargo run -p magi-app                      # run the app
cargo run -p magi-app -- run FILE.magi --dry-run   # run a macro without the window or any real input
cargo test --workspace                     # all tests
cargo fmt --all                            # format
cargo clippy --workspace --all-targets     # lint; must be warning-free
cargo build --release                      # the optimised single-file build
```

Tests that touch the real machine are ignored by default. The platform tests move the pointer by zero pixels and measure a screenshot without keeping it; the hotkey tests press F13 and F14 (keys with no default action) with the controller in dry-run mode:

```bash
cargo test -p magi-platform --test system -- --ignored --test-threads=1
cargo test -p magi-app system_tests -- --ignored --test-threads=1
```

To edit the Python runtime without rebuilding:

```bash
MACROMAGI_PYTHON_RUNTIME="$PWD/runtime/python" cargo run -p magi-app
```

Documentation screenshots are produced by the app itself and only ever show its own window. Point `MACROMAGI_DATA_DIR` at an empty folder first, so your own settings and theme do not end up in them:

```bash
export MACROMAGI_DATA_DIR="$(mktemp -d)"
cargo run --release -p magi-app -- --screenshot docs/images/auto-clicker.png --tab clicker --size 440x760
cargo run --release -p magi-app -- --screenshot docs/images/macro-editor.png --tab macro --example 0 --size 1280x860 --select vision.find_image
cargo run --release -p magi-app -- --screenshot docs/images/recording.png --tab macro --example 2 --size 1100x680 --cards --select input.recording
echo '{"show_variables": true}' > "$MACROMAGI_DATA_DIR/settings.json"
cargo run --release -p magi-app -- --screenshot docs/images/functions.png --tab macro --example 3 --size 1280x820 --select flow.delay
```

`--tab magic` opens the Magic tab, and a `.magic` file named on the command line opens in it. `--dataset <file>[:view]` opens a dataset file of the Magic tab's model (a new example model when none is open) in the dataset editor, such as `--dataset data/text/tools.jsonl:model` (views: `chat`, `model`, `raw`); `--dataset ""` opens the example model without the editor, for `--select`. `--agent` opens the AI Agent window inside the main window. `--settings <page>` opens the settings window on a page (`general`, `appearance`, `hotkeys`, `recording`, `engine`, `ai`). `--targets` opens *View → Screen targets* at start; it covers the screen, so it is for checking the overlay by hand rather than for `--screenshot`.

## Where things live

```text
crates/magi-core/      data model, block definitions, file formats    (no OS, no UI)
crates/magi-vision/    image matching, color search                  (pure functions on pixels)
crates/magi-audio/     loudness, pitch, sound recognition, sound files (pure functions on samples)
crates/magi-ml/        preparing pictures for AI models, running them  (CPU or graphics card)
crates/magi-platform/  input, listening, gamepads, capture, sound     (the only OS code)
crates/magi-engine/    native interpreter, Python bridge, device operations
crates/magi-app/       the application: UI, controller, command line, the Magic tab, downloads
runtime/python/        the Python engine (standard library only) and the Magic tab's trainer (PyTorch)
magidb/                MagiDB's pages, published as MagiGun/MagiDB
docs/                  architecture, formats, guides
tools/                 scripts that generate documentation
```

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before a change that crosses crates. The short version of its rules:

- Dependencies point downwards: app → engine/platform → core; the engine and the app also use the pure-computation crates (vision, audio, ml).
- Only `magi-platform` talks to the operating system. Engines reach the machine through the `Host` trait.
- Blocks are described by data in `magi-core`. Code that is not about one specific block must not mention one.
- Both engines must behave the same. Behaviour that touches devices is written once, in `magi-engine/src/ops.rs`.
- The app writes nothing outside its data folder.

## Making a change

1. Fork, and branch from `main`.
2. Make the change, with tests.
3. Run `cargo fmt --all`, `cargo clippy --workspace --all-targets` and `cargo test --workspace`, and say in the pull request on which operating system. There is no automatic test run, so this step is yours.
4. Add a line under *Unreleased* in [CHANGELOG.md](CHANGELOG.md) if a user would notice the change.
5. Open a pull request. Say what it changes and why, how you tested it, and on which operating system. Screenshots help for anything visible.

Keep pull requests to one topic. A bug fix and an unrelated refactor are two pull requests.

Commit messages: a short imperative summary line ("Add a Compare block"), then a blank line and any explanation worth keeping.

Changes to the `.magi` or `.mp` formats need extra care: files people already have must keep loading. See *Versioning* in [docs/FILE_FORMAT.md](docs/FILE_FORMAT.md), and update that document in the same pull request.

## Code guidelines

- **Format and lint.** `rustfmt` decides formatting. Clippy must pass without warnings; prefer fixing a lint to silencing it, and when you do silence one, say why.
- **Comments explain why.** The code already says what it does. Write down the reason, the constraint, the thing that would surprise the next reader. Every module starts with a comment saying what it is for.
- **User-facing text is plain language.** Error messages say what went wrong and, where possible, what to do about it. No jargon the user did not type themselves.
- **`unsafe` only in `magi-platform`**, each block with a `// SAFETY:` comment.
- **No panics on user input.** A malformed file, a missing image or an unplugged gamepad is an error to report, not a crash. `expect` is for things that cannot happen, with a message saying why not.
- **Think before adding a dependency.** MacroMagi ships as a single small file. Check the licence, the size and whether a few lines of code would do.
- **Python runtime:** standard library only, Python 3.8-compatible syntax, never write to the real `stdout`.

## Testing

Tests sit next to the code they test (`#[cfg(test)] mod tests`), plus:

- `crates/magi-engine/tests/engines.rs` – macros run end to end on both engines against a simulated machine. Any new block or change to execution belongs here.
- the tests at the bottom of `crates/magi-app/src/ui/graph.rs` – the node editor driven headlessly with a simulated pointer and keyboard (dragging wires, moving and selecting blocks, shortcuts, zooming).
- the tests at the bottom of `crates/magi-app/src/app.rs` – the whole app driven headlessly, with key presses fed to the controller exactly as the system listener would (assigning hotkeys and using them at once, opening the block list, taking a model from the Magic tab into a macro).
- `windows_reads_the_text_in_a_screenshot` in `crates/magi-platform/tests/system.rs` – ignored by default; reads a documentation screenshot with the text recognition built into Windows.
- `real_projects_import` in `crates/magi-app/src/magic/import.rs` – ignored by default; imports the projects named in `MAGI_IMPORT` (separated by `;`) and prints what was found, to check the importer against real models: `MAGI_IMPORT=path/to/project cargo test -p magi-app real_projects_import -- --ignored --nocapture`. With `MAGI_IMPORT_OUT` set to a folder, each import is also saved there as a `.magic` file.
- `crates/magi-platform/tests/system.rs` – ignored by default; checks against the real machine.

Guidelines:

- Test behaviour through public interfaces. A test should still pass after a refactor that did not change behaviour.
- Name tests as statements of fact: `a_flow_output_leads_to_one_place`, not `test_connect_2`.
- Nothing in the default test run may send real input or depend on what is on screen. Use `SimHost`.
- Keep logic out of UI code – in `magi-core`, or in plain functions like `parse_duration` – and test it there. Interactions that only exist in the UI can be tested headlessly with `egui_kittest`, as the node editor's are.

## Reporting bugs

Open an [issue](https://github.com/MagiGun/MacroMagi/issues) with:

- what you did, what you expected, what happened instead;
- your operating system and version (and, on Linux, X11 or Wayland);
- the MacroMagi version (`macromagi --version`) or commit;
- the macro, if it is about one – a `.magi` file, or the steps to build it;
- anything the console showed.

Recordings contain what you typed. Remove them from a `.magi` file, or re-record something harmless, before attaching it.

Security problems – for example a file that makes MacroMagi do something on opening – should be reported privately; see [SECURITY.md](SECURITY.md).

## Releasing

Releases are built and published by [`.github/workflows/release.yml`](.github/workflows/release.yml):

1. Move the *Unreleased* entries in [CHANGELOG.md](CHANGELOG.md) under a new version heading, and set the version in the workspace `Cargo.toml` and in `runtime/python/magi/__init__.py`.
2. Tag the commit and push the tag:

   ```bash
   git tag v0.1.8
   git push origin v0.1.8
   ```

   or run the workflow from *Actions → Release → Run workflow* with the tag name.
3. The workflow builds the Windows package (Linux and macOS packages are not built for now). It also gets Python's embeddable distribution in a `python` folder; its version is `PYTHON_VERSION` in the workflow, kept the same as in `crates/magi-app/src/components.rs`. Only if the build succeeds does it publish a GitHub release with the package and a `SHA256SUMS.txt`. A failed build publishes nothing; fix it and run the workflow again for the same tag.
