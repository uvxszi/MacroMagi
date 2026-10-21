//! The application shell: owns the settings, the open document and the tabs,
//! draws the banner with its menus, and carries out what the UI modules ask for.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use egui::{Align2, RichText, Ui, vec2};
use egui_phosphor::regular as icon;
use magi_core::action::Action;
use magi_core::autoclicker::{AutoClickerConfig, ClickPosition};
use magi_core::geom::Point;
use magi_core::graph::NodeId;
use magi_core::hotkey::{CaptureResult, Hotkey};
use magi_core::nodes::flow::StartParams;
use magi_core::nodes::{Registry, ids};
use magi_core::options::AutomationOptions;
use magi_core::preset::{NodePreset, Preset, PresetBody};
use magi_core::project::Project;
use magi_core::targets::{self, TargetShape, TargetSource};
use magi_core::text::format_hex_color;
use magi_engine::{LogEntry, LogLevel, PythonStatus};

use crate::background::{self, BackgroundLayer};
use crate::components::{Component, Components, RuntimeReport};
use crate::controller::{Controller, ControllerConfig};
use crate::document::Document;
use crate::examples;
use crate::history::History;
use crate::magic::MagicDoc;
use crate::paths::Paths;
use crate::presets::PresetLibrary;
use crate::settings::{ExportChoices, Settings, SettingsPage, Tab};
use crate::theme::{self, Palette, ThemeMode};
use crate::ui::autoclicker::{ClickerEnv, ClickerEvent, ClickerTab};
use crate::ui::inspector::{self, InspectorEnv};
use crate::ui::macro_tab::{MacroEnv, MacroEvent, MacroTab};
use crate::ui::magic_tab::{ImportFrom, MagicEnv, MagicEvent, MagicTab};
use crate::ui::options::{self, OptionsEvent, RulesWindow};
use crate::ui::overlay::{OverlayOutcome, TargetOverlay};
use crate::ui::recording_editor;
use crate::ui::settings_window::{SettingsEnv, SettingsEvent, SettingsWindow, StartHotkey};
use crate::ui::snip::{Snip, SnipOutcome};
use crate::ui::{HotkeyTarget, SnipTarget, UiRequest, widgets};

const TOAST_SECONDS: f32 = 4.0;
const SETTINGS_SAVE_DELAY: Duration = Duration::from_millis(800);
const MACRO_FILTER: (&str, &[&str]) = ("MacroMagi macro", &["magi"]);
const PRESET_FILTER: (&str, &[&str]) = ("MacroMagi preset", &["mp"]);
const MAGIC_FILTER: (&str, &[&str]) = ("MacroMagi workspace", &["magic"]);
/// What File › Open takes on the Magic tab: workspaces and Magi models.
const MAGIC_OPEN_FILTER: (&str, &[&str]) = ("MacroMagi workspace or Magi model", &["magic", "szi"]);
/// Models a macro block can run.
const MODEL_FILTER: (&str, &[&str]) = ("AI model", &["magic", "szi", "onnx"]);

/// A model file as a macro runs it: a `.magic` workspace or `.szi` Magi
/// model, or an ONNX model.
fn model_from_file(path: &std::path::Path) -> Result<magi_core::magic::MagicFile, String> {
    let extension = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    match extension.as_str() {
        "onnx" => crate::magic::import::import(path).map(|imported| imported.file),
        _ => magi_core::szi::open(path).map_err(|e| e.to_string()),
    }
}
/// Below this width the banner's menus show icons only.
const COMPACT_BANNER_BELOW: f32 = 720.0;
/// Below this width the banner's menus are folded into one.
const TINY_BANNER_BELOW: f32 = 520.0;

/// What to do once the user has decided about unsaved changes.
#[derive(Clone)]
enum Pending {
    New,
    Open(Option<PathBuf>),
    Example(usize),
    ClickerGraph,
    Quit,
}

/// What replaces the Magic tab's model once the user has decided about its unsaved changes.
#[derive(Clone)]
enum MagicPending {
    New,
    NewModel {
        text: bool,
    },
    /// Magi 150M: the recipe at a size that trains in a few hours.
    NewTestMagi,
    Open(Option<PathBuf>),
    Import(ImportFrom),
    OpenModel(String),
    Quit,
}

/// What native tools run with: the Python for AI, MagiDB's folder and address, the vision model.
type ToolSettings = (Option<PathBuf>, Option<PathBuf>, String, Option<PathBuf>);

/// Something that was about to start when it turned out to have no hotkey.
#[derive(Clone)]
enum Unguarded {
    Run,
    Record(Option<(NodeId, String)>),
}

struct HotkeyWarning {
    target: HotkeyTarget,
    then: Unguarded,
}

/// Settings found in an opened file, waiting for the user to accept them.
struct FileExtras {
    options: Option<AutomationOptions>,
    clicker: Option<AutoClickerConfig>,
}

#[derive(Clone, Copy, PartialEq)]
enum PresetKind {
    Clicker,
    Blocks,
    /// Blocks of the Magic tab's canvas.
    MagicBlocks,
}

/// How far an install has got, shared with the thread doing it: what it is
/// doing, and the share done when known.
type Progress = Arc<std::sync::Mutex<(String, Option<f32>)>>;

/// Work done off the UI thread, reporting back.
enum Job {
    Toast(String, bool),
    BackgroundReady(String),
    /// A download or removal finished (or failed): the message to show.
    Component(Component, Result<String, String>),
    /// Where the optional parts stand, worked out off the UI thread.
    Runtime(Box<RuntimeReport>),
    /// The Pythons found, and the one "automatic" stands for.
    Pythons(Vec<(PathBuf, String)>, Option<(PathBuf, String)>),
}

struct Toast {
    text: String,
    born: Instant,
    is_error: bool,
}

/// Options for the developer screenshot mode (`--screenshot`).
pub struct Screenshot {
    pub path: PathBuf,
    /// Frames still to draw before the picture is taken, so layouts settle.
    pub frames_left: u32,
    pub requested: bool,
    /// Not before this moment (`--wait`).
    pub not_before: Option<Instant>,
}

pub struct StartupOptions {
    pub open: Option<PathBuf>,
    pub example: Option<usize>,
    pub tab: Option<Tab>,
    pub screenshot: Option<PathBuf>,
    /// Open the first recording block's information card (for screenshots).
    pub show_cards: bool,
    /// Open View › Screen targets (for screenshots).
    pub show_targets: bool,
    /// Select the first block of this type (for screenshots).
    pub select_kind: Option<String>,
    /// Open the settings window on this page (for screenshots).
    pub settings_page: Option<SettingsPage>,
    /// Start training the Magic tab's model (for screenshots).
    pub train: bool,
    /// Open this dataset file of the Magic tab's model (a new example model when none
    /// is open) in the dataset editor, with a view after a colon (for screenshots).
    pub dataset: Option<String>,
    /// Open the AI Agent window, drawn inside the main window (for screenshots).
    pub agent: bool,
    /// Take the screenshot after this many seconds.
    pub wait: Option<f64>,
}

pub struct MagiApp {
    /// The AI Agent window: chats with models (a preview of CodeMagi's).
    pub(crate) agent_chat: crate::ui::agent_chat::AgentChat,
    paths: Paths,
    registry: Arc<Registry>,
    controller: Controller,
    settings: Settings,
    /// The settings as last written to disk, and when they last differed.
    saved_settings: Settings,
    settings_changed: Option<Instant>,
    data_dir_error: Option<String>,

    doc: Document,
    clicker_tab: ClickerTab,
    clicker_history: History<AutoClickerConfig>,
    macro_tab: MacroTab,
    magic_tab: MagicTab,
    presets: PresetLibrary,
    /// Optional downloads, their statuses (when known), and the one being installed.
    components: Components,
    runtime: Option<RuntimeReport>,
    /// This computer, for Settings › Hardware.
    machine: magi_platform::system::Machine,
    installing: Option<(Component, Progress)>,
    /// Python interpreters found on the system (`None`: not looked for yet).
    gpus: Vec<magi_platform::gpu::Gpu>,
    /// Where AI models run, as last applied.
    applied_device: Option<magi_ml::Device>,
    looking_for_python: bool,
    checking_components: bool,

    /// The hotkey being rebound, and the chord pressed so far.
    capture: Option<(HotkeyTarget, Hotkey)>,
    snip: Option<Snip>,
    overlay: Option<TargetOverlay>,
    /// The block a running recording should be linked to when it ends.
    record_into: Option<(NodeId, String)>,
    settings_window: SettingsWindow,
    rules_window: RulesWindow,
    stop_hotkeys_open: bool,
    /// The recording shown in the pop-out editor.
    recording_window: Option<String>,
    /// The Rate block shown in the Variable Rate Editor.
    rate_window: Option<NodeId>,
    pending: Option<Pending>,
    magic_pending: Option<MagicPending>,
    export: Option<ExportChoices>,
    /// The Export Presets dialog, with what is chosen: 0 the workspace, 1 the selection, then the library.
    export_presets: Option<usize>,
    preset_prompt: Option<(PresetKind, String)>,
    hotkey_warning: Option<HotkeyWarning>,
    file_extras: Option<FileExtras>,
    allow_close: bool,
    toasts: Vec<Toast>,
    jobs: Vec<Receiver<Job>>,
    downloading: bool,
    /// A training run was going on at the last frame, and whether it was stopped from here.
    training_seen: Option<bool>,
    backgrounds: Vec<String>,

    logs: Vec<LogEntry>,
    next_log: u64,
    error_node: Option<NodeId>,
    activity: HashMap<NodeId, magi_engine::Activity>,

    background: BackgroundLayer,
    /// The native tools' settings as last given to `magi_ml`: the Python, MagiDB and the vision model.
    applied_tools: Option<ToolSettings>,
    /// The About window, opened from the banner logo.
    about_open: bool,
    /// The banner logo, in the color it was made for.
    logo: Option<(egui::Color32, egui::TextureHandle)>,
    applied_palette: Option<(bool, Option<[u8; 3]>, u32)>,
    applied_window: Option<(bool, bool, f32)>,
    applied_title: String,
    screenshot: Option<Screenshot>,
    /// `--train`: train the Magic tab's model once the window is up.
    train_at_start: bool,
}

impl MagiApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        paths: Paths,
        mut settings: Settings,
        options: StartupOptions,
    ) -> Self {
        theme::install_fonts(&cc.egui_ctx);
        let registry = Arc::new(Registry::builtin());
        let data_dir_error = paths.ensure_data_dir().err();
        if let Some(tab) = options.tab {
            settings.tab = tab;
        }

        let controller =
            Controller::new(Arc::clone(&registry), paths.clone(), ControllerConfig::from_settings(&settings));
        let ctx = cc.egui_ctx.clone();
        controller.monitor().set_waker(move || ctx.request_repaint());
        let ctx = cc.egui_ctx.clone();
        controller.clicker_monitor().set_waker(move || ctx.request_repaint());

        let components = Components::new(&paths.data_dir, &paths.app_dir);
        let mut app = Self {
            agent_chat: Default::default(),
            presets: PresetLibrary::load(paths.presets_dir()),
            backgrounds: background::list(&paths.backgrounds_dir()),
            paths,
            doc: Document::starter(&registry),
            registry,
            controller,
            saved_settings: settings.clone(),
            clicker_history: History::new(&settings.clicker),
            settings,
            settings_changed: None,
            data_dir_error,
            clicker_tab: ClickerTab,
            macro_tab: MacroTab::default(),
            magic_tab: MagicTab::default(),
            components,
            runtime: None,
            machine: magi_platform::system::machine(),
            installing: None,
            gpus: magi_platform::gpu::gpus(),
            applied_device: None,
            looking_for_python: false,
            checking_components: false,
            capture: None,
            snip: None,
            overlay: None,
            record_into: None,
            settings_window: SettingsWindow::default(),
            rules_window: RulesWindow::default(),
            stop_hotkeys_open: false,
            recording_window: None,
            rate_window: None,
            pending: None,
            magic_pending: None,
            export: None,
            export_presets: None,
            preset_prompt: None,
            hotkey_warning: None,
            file_extras: None,
            allow_close: false,
            toasts: Vec::new(),
            jobs: Vec::new(),
            downloading: false,
            training_seen: None,
            logs: Vec::new(),
            next_log: 0,
            error_node: None,
            activity: HashMap::new(),
            background: BackgroundLayer::default(),
            applied_tools: None,
            about_open: false,
            logo: None,
            applied_palette: None,
            applied_window: None,
            applied_title: String::new(),
            screenshot: options.screenshot.map(|path| Screenshot {
                path,
                frames_left: 45,
                requested: false,
                not_before: options.wait.map(|s| Instant::now() + Duration::from_secs_f64(s)),
            }),
            train_at_start: options.train,
        };
        app.macro_tab.view.palette_size = vec2(app.settings.palette_size[0], app.settings.palette_size[1]);
        app.start_onnxruntime();
        // What is installed is looked at once at the start (off the window's thread).
        if app.settings.python_checked || app.settings.python_path.is_some() {
            app.refresh_components(&cc.egui_ctx);
        } else {
            app.find_pythons(&cc.egui_ctx);
        }
        // File blocks' relative paths lead into the data folder.
        magi_engine::set_files_dir(app.paths.data_dir.join("files"));
        // Magi's tools: those MacroMagi carries out itself, and calls the user allows first.
        crate::magic::host_tools::register();
        let ctx = cc.egui_ctx.clone();
        magi_ml::tools::show_approvals(Box::new(move || ctx.request_repaint()));

        let magic = options.open.as_ref().is_some_and(|path| {
            path.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case(magi_core::magic::EXTENSION) || e.eq_ignore_ascii_case("szi"))
        });
        if let Some(path) = options.open.as_ref().filter(|_| magic) {
            // A model: it opens in the Magic tab.
            match magi_core::szi::open(path) {
                Ok(file) => app.magic_tab.doc = Some(MagicDoc::new(file, Some(path.clone()), Vec::new())),
                Err(error) => app.toast_error(format!("Could not open {}: {error}", path.display())),
            }
            app.settings.tab = Tab::Magic;
        } else if let Some(path) = options.open {
            app.open_path(&path);
        } else if let Some(index) = options.example {
            app.replace_document(Document::new(examples::build(index, &app.registry), None));
        }
        if let Some(kind) = &options.select_kind
            && let Some(node) = app.doc.project.graph.nodes().find(|n| n.kind == *kind).map(|n| n.id)
        {
            app.macro_tab.view.focus_node(node);
        }
        if let Some(kind) = &options.select_kind
            && let Some(doc) = &app.magic_tab.doc
            && let Some(node) = doc.studio.graph.nodes().find(|n| n.kind == *kind).map(|n| n.id)
        {
            app.magic_tab.view.selection = [node].into();
        }
        if options.agent {
            app.agent_chat.open = true;
            app.agent_chat.show_settings = true;
            cc.egui_ctx.set_embed_viewports(true);
        }
        if let Some(dataset) = &options.dataset {
            if app.magic_tab.doc.is_none() {
                app.magic_tab.doc = Some(MagicDoc::new(crate::magic::new_model(true), None, Vec::new()));
            }
            let (file, view) = match dataset.split_once(':') {
                Some((file, view)) => (file, Some(view)),
                None => (dataset.as_str(), None),
            };
            if !file.is_empty() {
                app.magic_tab.open_dataset(file, view);
            }
            if let Some(kind) = &options.select_kind
                && let Some(doc) = &app.magic_tab.doc
                && let Some(node) = doc.studio.graph.nodes().find(|n| n.kind == *kind).map(|n| n.id)
            {
                app.magic_tab.view.selection = [node].into();
            }
        }
        if options.show_cards {
            let recordings: Vec<NodeId> =
                app.doc.project.graph.nodes().filter(|n| n.kind == ids::PLAY_RECORDING).map(|n| n.id).collect();
            for node in recordings {
                app.macro_tab.view.show_card(node);
            }
        }
        if options.show_targets {
            app.overlay = TargetOverlay::begin(&cc.egui_ctx).ok();
        }
        if let Some(page) = options.settings_page {
            app.settings.settings_page = page;
            app.settings_window.open = true;
        }
        app
    }

    fn toast(&mut self, text: impl Into<String>) {
        self.toasts.push(Toast { text: text.into(), born: Instant::now(), is_error: false });
    }

    fn toast_error(&mut self, text: impl Into<String>) {
        self.toasts.push(Toast { text: text.into(), born: Instant::now(), is_error: true });
    }

    /// Runs `work` on another thread; its result comes back as a [`Job`].
    fn spawn_job(&mut self, ctx: &egui::Context, work: impl FnOnce() -> Job + Send + 'static) {
        let (sender, receiver) = mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = sender.send(work());
            ctx.request_repaint();
        });
        self.jobs.push(receiver);
    }

    // ---- documents --------------------------------------------------------

    fn replace_document(&mut self, doc: Document) {
        self.controller.stop_macro();
        self.doc = doc;
        let palette_size = self.macro_tab.view.palette_size;
        self.macro_tab = MacroTab::default();
        self.macro_tab.view.palette_size = palette_size;
        self.recording_window = None;
        self.error_node = None;
        self.settings.tab = Tab::Macro;
    }

    fn open_path(&mut self, path: &Path) {
        match Project::load(path) {
            Ok(project) => {
                let extras = FileExtras { options: project.options.clone(), clicker: project.clicker.clone() };
                self.replace_document(Document::new(project, Some(path.to_path_buf())));
                self.settings.remember_file(path);
                let differs = extras.options.as_ref().is_some_and(|o| *o != self.settings.options)
                    || extras.clicker.as_ref().is_some_and(|c| *c != self.settings.clicker);
                if differs {
                    self.file_extras = Some(extras);
                }
            }
            Err(error) => self.toast_error(format!("Could not open {}: {error}", path.display())),
        }
    }

    fn write_document(&mut self, path: PathBuf) -> bool {
        if self.doc.project.meta.name == Project::new().meta.name
            && let Some(stem) = path.file_stem()
        {
            self.doc.project.meta.name = stem.to_string_lossy().into_owned();
        }
        if self.settings.delete_unused_recordings {
            let used = self.doc.project.graph.referenced_assets(&self.registry);
            let unused: Vec<String> = self
                .doc
                .project
                .assets
                .of_kind(magi_core::assets::AssetKind::Recording)
                .filter(|a| !used.contains(&a.id))
                .map(|a| a.id.clone())
                .collect();
            for id in unused {
                self.doc.project.assets.remove(&id);
            }
        }
        match self.doc.project.save(&path) {
            Ok(()) => {
                self.settings.remember_file(&path);
                self.doc.mark_saved(path);
                true
            }
            Err(error) => {
                self.toast_error(format!("Could not save: {error}"));
                false
            }
        }
    }

    fn pick_save_path(&self) -> Option<PathBuf> {
        rfd::FileDialog::new()
            .add_filter(MACRO_FILTER.0, MACRO_FILTER.1)
            .set_file_name(format!("{}.magi", self.doc.title()))
            .save_file()
    }

    /// Saves the document. Returns false if the user cancelled or it failed.
    /// Save As can also write a preset (`.mp`): of the selected blocks, or of
    /// the whole workspace when nothing is selected.
    fn save(&mut self, choose_path: bool) -> bool {
        let path = match (&self.doc.path, choose_path) {
            (Some(path), false) => path.clone(),
            _ => {
                let dialog = rfd::FileDialog::new()
                    .add_filter(MACRO_FILTER.0, MACRO_FILTER.1)
                    .set_file_name(format!("{}.magi", self.doc.title()));
                let dialog = if choose_path { dialog.add_filter(PRESET_FILTER.0, PRESET_FILTER.1) } else { dialog };
                match dialog.save_file() {
                    Some(path) => path,
                    None => return false,
                }
            }
        };
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("mp")) {
            let selected = !self.macro_tab.view.selected_blocks(&self.doc.project).is_empty();
            return match self.workspace_preset(selected) {
                Some(preset) => self.write_preset(&preset, &path),
                None => false,
            };
        }
        self.write_document(path)
    }

    /// A preset of the open tab: its Auto Clicker settings, or the macro's
    /// blocks (only the selected ones when `selection`).
    fn workspace_preset(&self, selection: bool) -> Option<magi_core::preset::Preset> {
        let body = match self.settings.tab {
            Tab::AutoClicker => magi_core::preset::PresetBody::AutoClicker(self.settings.clicker.clone()),
            _ => {
                let ids = if selection {
                    self.macro_tab.view.selected_blocks(&self.doc.project)
                } else {
                    self.doc.project.graph.nodes().map(|n| n.id).collect()
                };
                if ids.is_empty() {
                    return None;
                }
                magi_core::preset::PresetBody::Nodes(NodePreset::capture(
                    &self.doc.project.graph,
                    &ids,
                    &self.doc.project.assets,
                    &self.registry,
                ))
            }
        };
        let name = if selection { format!("{} (selection)", self.doc.title()) } else { self.doc.title() };
        Some(magi_core::preset::Preset { name, description: String::new(), body })
    }

    fn write_preset(&mut self, preset: &magi_core::preset::Preset, path: &Path) -> bool {
        match preset.save(path) {
            Ok(()) => {
                let name = path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                self.toast(format!("Saved the preset {name}"));
                true
            }
            Err(error) => {
                self.toast_error(format!("Could not save the preset: {error}"));
                false
            }
        }
    }

    /// The Export Presets dialog: the workspace, the selection, or one of the user's presets.
    fn export_presets_dialog(&mut self, ctx: &egui::Context) {
        let Some(mut choice) = self.export_presets else { return };
        let library: Vec<magi_core::preset::Preset> = self.presets.presets().cloned().collect();
        let selected = !self.macro_tab.view.selected_blocks(&self.doc.project).is_empty();
        let mut decision = None;
        egui::Modal::new(egui::Id::new("export-presets")).show(ctx, |ui| {
            ui.set_width(320.0);
            ui.label(RichText::new("Export a preset").strong().size(15.0));
            widgets::note(ui, "Saved as a .mp file, to share or to keep.");
            ui.add_space(4.0);
            let workspace = if self.settings.tab == Tab::AutoClicker {
                "Current Auto Clicker settings"
            } else {
                "Current macro workspace"
            };
            ui.radio_value(&mut choice, 0, workspace);
            ui.add_enabled_ui(selected && self.settings.tab != Tab::AutoClicker, |ui| {
                ui.radio_value(&mut choice, 1, "Only selection").on_disabled_hover_text("Select blocks first");
            });
            if !library.is_empty() {
                ui.separator();
                ui.label(RichText::new("MY PRESETS").size(10.5).color(theme::text_faint()));
                egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                    for (index, preset) in library.iter().enumerate() {
                        ui.radio_value(
                            &mut choice,
                            2 + index,
                            format!("{}  ·  {}", preset.name, preset.body.kind_label()),
                        );
                    }
                });
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if widgets::accent_button(ui, "Export…", theme::accent(), vec2(80.0, 26.0)).clicked() {
                    decision = Some(true);
                }
                if ui.button("Cancel").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    decision = Some(false);
                }
            });
        });
        self.export_presets = Some(choice);
        match decision {
            Some(true) => {
                self.export_presets = None;
                let preset = match choice {
                    0 => self.workspace_preset(false),
                    1 => self.workspace_preset(true),
                    n => library.get(n - 2).cloned(),
                };
                let Some(preset) = preset else {
                    return self.toast_error("There is nothing to export.");
                };
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter(PRESET_FILTER.0, PRESET_FILTER.1)
                    .set_file_name(format!("{}.mp", preset.file_stem()))
                    .save_file()
                {
                    self.write_preset(&preset, &path);
                }
            }
            Some(false) => self.export_presets = None,
            None => {}
        }
    }

    /// Saves to a chosen file with the extras picked in the export dialog.
    fn export(&mut self, choices: &ExportChoices) {
        let Some(path) = self.pick_save_path() else { return };
        if choices.presets {
            // Bundle the library, keeping presets that already travel with this file.
            for preset in self.presets.presets() {
                if !self.doc.project.presets.iter().any(|bundled| bundled.name == preset.name) {
                    self.doc.project.presets.push(preset.clone());
                }
            }
        }
        self.doc.project.options = choices.options.then(|| self.settings.options.clone());
        self.doc.project.clicker = choices.clicker.then(|| self.settings.clicker.clone());
        if self.write_document(path.clone()) {
            let name = path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
            self.toast(format!("Exported {name}"));
        }
    }

    /// Runs an action that replaces the document, asking first if there are unsaved changes.
    fn guarded(&mut self, action: Pending) {
        if self.doc.is_dirty() {
            self.pending = Some(action);
        } else {
            self.perform(action);
        }
    }

    fn perform(&mut self, action: Pending) {
        match action {
            Pending::New => self.replace_document(Document::starter(&self.registry)),
            Pending::Open(Some(path)) => self.open_path(&path),
            Pending::Open(None) => {
                if let Some(path) = rfd::FileDialog::new().add_filter(MACRO_FILTER.0, MACRO_FILTER.1).pick_file() {
                    self.open_path(&path);
                }
            }
            Pending::Example(index) => {
                self.replace_document(Document::new(examples::build(index, &self.registry), None));
            }
            Pending::ClickerGraph => {
                let mut project = Project::new();
                project.meta.name = "Auto clicker".into();
                project.graph = self.settings.clicker.to_graph(&self.registry);
                self.replace_document(Document::new(project, None));
            }
            Pending::Quit => self.magic_guarded(MagicPending::Quit),
        }
    }

    /// Runs an action that replaces the Magic tab's model, asking first if it has unsaved changes.
    fn magic_guarded(&mut self, action: MagicPending) {
        if self.magic_tab.doc.as_ref().is_some_and(MagicDoc::would_lose_changes) {
            self.magic_pending = Some(action);
        } else {
            self.magic_perform(action);
        }
    }

    fn magic_perform(&mut self, action: MagicPending) {
        match action {
            MagicPending::New => self.set_magic_doc(None),
            MagicPending::NewModel { text } => {
                let doc = MagicDoc::new(crate::magic::new_model(text), None, Vec::new());
                self.set_magic_doc(Some(doc));
                self.toast(if text {
                    "A new Magi model: an agent of about 400M parameters with its datasets. It trains on a GPU \
                     server: File › Export as Python."
                } else {
                    "A new vision model: give it answers and pictures, choose where Compile saves it, then Train."
                });
            }
            MagicPending::NewTestMagi => {
                let file = crate::magic::new_model_of(true, &magi_core::nodes::studio::MAGI_TEST);
                self.set_magic_doc(Some(MagicDoc::new(file, None, Vec::new())));
                self.toast(
                    "Magi 150M: Magi's design and recipe at about 150M parameters, to try in a few hours on a GPU \
                     server (File › Export as Python) before the full size.",
                );
            }
            MagicPending::Open(path) => {
                let path = path.or_else(|| {
                    rfd::FileDialog::new().add_filter(MAGIC_OPEN_FILTER.0, MAGIC_OPEN_FILTER.1).pick_file()
                });
                let Some(path) = path else { return };
                match magi_core::szi::open(&path) {
                    Ok(file) => {
                        self.remember_model(&path);
                        self.set_magic_doc(Some(MagicDoc::new(file, Some(path), Vec::new())));
                    }
                    Err(error) => self.toast_error(format!("Could not open the model: {error}")),
                }
            }
            MagicPending::Import(from) => {
                let path = if from == ImportFrom::Folder {
                    rfd::FileDialog::new().set_title("Choose the model's project folder").pick_folder()
                } else {
                    rfd::FileDialog::new().add_filter("Model or project", &["onnx", "zip"]).pick_file()
                };
                let Some(path) = path else { return };
                match crate::magic::import::import(&path) {
                    Ok(imported) => {
                        let name = imported.file.model.name.clone();
                        self.set_magic_doc(Some(MagicDoc::new(imported.file, None, imported.notes)));
                        self.toast(format!("Imported “{name}”. Check what the import found, then try it."));
                    }
                    Err(error) => self.toast_error(format!("Could not import: {error}")),
                }
            }
            MagicPending::OpenModel(asset) => {
                let Some(bytes) = self.doc.project.assets.model(&asset).cloned() else { return };
                match magi_core::szi::model_from_bytes(&bytes) {
                    Ok(file) => {
                        let mut doc = MagicDoc::new(file, None, Vec::new());
                        doc.mark_kept();
                        self.set_magic_doc(Some(doc));
                        self.settings.tab = Tab::Magic;
                    }
                    Err(error) => self.toast_error(format!("Could not open the model: {error}")),
                }
            }
            MagicPending::Quit => self.allow_close = true,
        }
    }

    /// Keeps a model among the recent ones.
    fn remember_model(&mut self, path: &std::path::Path) {
        let recent = &mut self.settings.recent_models;
        recent.retain(|p| p != path);
        recent.insert(0, path.to_owned());
        recent.truncate(10);
    }

    /// Puts a model in the Magic tab, stopping anything running on the old one.
    fn set_magic_doc(&mut self, doc: Option<MagicDoc>) {
        self.magic_tab.reset();
        self.magic_tab.doc = doc;
    }

    /// Saves the Magic tab's model, asking where when it has no file yet. True once saved.
    fn save_magic(&mut self, choose_path: bool) -> bool {
        let Some(doc) = &mut self.magic_tab.doc else { return false };
        let path = match (&doc.path, choose_path) {
            (Some(path), false) => Some(path.clone()),
            _ => rfd::FileDialog::new()
                .add_filter(MAGIC_FILTER.0, MAGIC_FILTER.1)
                .set_file_name(format!("{}.magic", doc.title()))
                .save_file(),
        };
        let Some(mut path) = path else { return false };
        if path.extension().is_none() {
            path.set_extension(magi_core::magic::EXTENSION);
        }
        // The blocks it comes with, ready for any macro.
        if doc.file.model.blocks.is_none() {
            doc.file.model.blocks = Some(crate::magic::starter_blocks(&doc.file.model, "", &self.registry));
        }
        // The workspace never holds the model: a trained one goes into a .szi beside it.
        match doc.save_workspace(&path) {
            Ok(szi) => {
                self.toast(match szi {
                    Some(szi) => format!("Saved {} and the model, {}", path.display(), szi.display()),
                    None => format!("Saved {}", path.display()),
                });
                true
            }
            Err(error) => {
                self.toast_error(format!("Could not save: {error}"));
                false
            }
        }
    }

    /// Loads Auto Clicker settings from a `.magi` or `.mp` file.
    fn load_clicker_settings(&mut self) {
        let Some(path) = rfd::FileDialog::new().add_filter("MacroMagi file", &["magi", "mp"]).pick_file() else {
            return;
        };
        let loaded = if path.extension().is_some_and(|e| e == "mp") {
            Preset::load(&path).map_err(|e| e.to_string()).and_then(|preset| match preset.body {
                PresetBody::AutoClicker(config) => Ok(config),
                PresetBody::Nodes(_) => Err("this preset holds blocks, not Auto Clicker settings".to_owned()),
            })
        } else {
            Project::load(&path)
                .map_err(|e| e.to_string())
                .and_then(|project| project.clicker.ok_or_else(|| "this file has no Auto Clicker settings".to_owned()))
        };
        match loaded {
            Ok(config) => {
                self.settings.clicker = config;
                self.toast("Auto Clicker settings loaded");
            }
            Err(error) => self.toast_error(format!("Could not load the settings: {error}")),
        }
    }

    // ---- starting things --------------------------------------------------

    /// Where the user can stop a run from, for warnings.
    fn stop_hint(&self) -> String {
        if self.settings.stop_hotkey.is_none() {
            "from this window".to_owned()
        } else {
            format!("from this window or with the stop-everything hotkey ({})", self.settings.stop_hotkey.label())
        }
    }

    fn run_macro(&mut self, check: bool) {
        if check && self.settings.run_hotkey.is_none() {
            self.hotkey_warning = Some(HotkeyWarning { target: HotkeyTarget::Run, then: Unguarded::Run });
            return;
        }
        self.error_node = None;
        self.sync_controller();
        self.controller.run_macro();
    }

    fn toggle_recording(&mut self, into: Option<(NodeId, String)>, check: bool) {
        let recording = self.controller.recording_status().is_some();
        if !recording && check && self.settings.record_hotkey.is_none() {
            self.hotkey_warning = Some(HotkeyWarning { target: HotkeyTarget::Record, then: Unguarded::Record(into) });
            return;
        }
        if !recording {
            self.record_into = into;
        }
        self.sync_controller();
        self.controller.toggle_recording();
    }

    fn start_capture(&mut self, target: HotkeyTarget) {
        self.controller.begin_capture();
        self.capture = Some((target, Hotkey::none()));
    }

    // ---- requests from the UI ---------------------------------------------

    fn handle_request(&mut self, ctx: &egui::Context, request: UiRequest) {
        match request {
            UiRequest::Snip(mode, target) => match Snip::begin(ctx, mode, target) {
                Ok(snip) => self.snip = Some(snip),
                Err(error) => self.toast_error(error),
            },
            UiRequest::CaptureHotkey(target) => self.start_capture(target),
            UiRequest::CancelCapture => {
                self.controller.cancel_capture();
                self.capture = None;
            }
            UiRequest::LoadImage(node, key) => {
                let dialog = rfd::FileDialog::new().add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp", "gif"]);
                let Some(path) = dialog.pick_file() else { return };
                let loaded = std::fs::read(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|bytes| magi_vision::decode(&bytes).map_err(|e| e.to_string()))
                    .and_then(|bitmap| magi_vision::encode_png(bitmap.as_ref()).map_err(|e| e.to_string()));
                match loaded {
                    Ok(png) => {
                        let name =
                            path.file_stem().map_or_else(|| "Image".to_owned(), |s| s.to_string_lossy().into_owned());
                        let id = self.doc.project.assets.add_image(name, png);
                        if let Some(node) = self.doc.project.graph.node_mut(node) {
                            node.set_param(&key, id);
                        }
                    }
                    Err(error) => self.toast_error(format!("Could not load the image: {error}")),
                }
            }
            UiRequest::RecordInto(node, key) => self.toggle_recording(Some((node, key)), true),
            UiRequest::EditRecording(asset) => self.recording_window = Some(asset),
            UiRequest::LoadSound(node, key) => {
                let dialog = rfd::FileDialog::new().add_filter("Sounds", &["wav", "mp3", "ogg", "flac"]);
                let Some(path) = dialog.pick_file() else { return };
                let loaded = std::fs::read(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|bytes| magi_audio::prepare_upload(&bytes).map_err(|e| e.to_string()));
                match loaded {
                    Ok(wav) => {
                        let name =
                            path.file_stem().map_or_else(|| "Sound".to_owned(), |s| s.to_string_lossy().into_owned());
                        let id = self.doc.project.assets.add_audio(name, wav);
                        if let Some(node) = self.doc.project.graph.node_mut(node) {
                            node.set_param(&key, id);
                        }
                    }
                    Err(error) => self.toast_error(format!("Could not load the sound: {error}")),
                }
            }
            UiRequest::EditRate(node) => self.rate_window = Some(node),
            UiRequest::LoadModel(node, key) => {
                let Some(path) = rfd::FileDialog::new().add_filter(MODEL_FILTER.0, MODEL_FILTER.1).pick_file() else {
                    return;
                };
                let kind = self.doc.project.graph.node(node).map(|n| n.kind.clone()).unwrap_or_default();
                let loaded = model_from_file(&path).and_then(|file| {
                    let text = file.model.kind == magi_core::magic::ModelKind::Text;
                    let wants_text = matches!(kind.as_str(), ids::AI_AGENT | ids::AI_LOAD | ids::AI_UNLOAD);
                    if wants_text && !text {
                        return Err("this is a vision model; this block runs text models".into());
                    }
                    if !wants_text && text {
                        return Err("this is a text model; run it with an AI Agent block".into());
                    }
                    Ok((file.model.name.clone(), file.for_macro().map_err(|e| e.to_string())?))
                });
                match loaded {
                    Ok((name, bytes)) => {
                        let id = self.doc.project.assets.add_model(name, bytes);
                        if let Some(node) = self.doc.project.graph.node_mut(node) {
                            node.set_param(&key, id);
                        }
                    }
                    Err(error) => self.toast_error(format!("Could not open the model: {error}")),
                }
            }
            UiRequest::OpenModel(asset) => self.magic_guarded(MagicPending::OpenModel(asset)),
            UiRequest::AddImageFile(node) => {
                let dialog = rfd::FileDialog::new().add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp", "gif"]);
                let Some(paths) = dialog.pick_files() else { return };
                for path in paths {
                    let loaded = std::fs::read(&path)
                        .map_err(|e| e.to_string())
                        .and_then(|bytes| magi_vision::decode(&bytes).map_err(|e| e.to_string()))
                        .and_then(|bitmap| magi_vision::encode_png(bitmap.as_ref()).map_err(|e| e.to_string()));
                    match loaded {
                        Ok(png) => {
                            let name = path
                                .file_stem()
                                .map_or_else(|| "Image".to_owned(), |s| s.to_string_lossy().into_owned());
                            let id = self.doc.project.assets.add_image(name, png);
                            self.push_image(node, id);
                        }
                        Err(error) => self.toast_error(format!("Could not load {}: {error}", path.display())),
                    }
                }
            }
            UiRequest::AddSoundFile(node) => {
                let dialog = rfd::FileDialog::new().add_filter("Sounds", &["wav", "mp3", "ogg", "flac"]);
                let Some(paths) = dialog.pick_files() else { return };
                for path in paths {
                    let loaded = std::fs::read(&path)
                        .map_err(|e| e.to_string())
                        .and_then(|bytes| magi_audio::prepare_upload(&bytes).map_err(|e| e.to_string()));
                    match loaded {
                        Ok(wav) => {
                            let name = path
                                .file_stem()
                                .map_or_else(|| "Sound".to_owned(), |s| s.to_string_lossy().into_owned());
                            let id = self.doc.project.assets.add_audio(name, wav);
                            if let Some(node) = self.doc.project.graph.node_mut(node) {
                                let mut params: magi_core::nodes::sound::SoundMatchParams = node.params_or_default();
                                let mut sounds = params.targets();
                                if !sounds.iter().any(|t| t.sound == id) {
                                    sounds
                                        .push(magi_core::nodes::sound::SoundTarget { sound: id, group: String::new() });
                                }
                                params.sound.clear();
                                node.set_param("sounds", sounds);
                                node.set_param("sound", "");
                            }
                        }
                        Err(error) => self.toast_error(format!("Could not load {}: {error}", path.display())),
                    }
                }
            }
        }
    }

    fn apply_snip(&mut self, target: SnipTarget, outcome: SnipOutcome) {
        let set = |app: &mut Self, node: NodeId, key: &str, value: serde_json::Value| {
            if let Some(node) = app.doc.project.graph.node_mut(node) {
                node.params.insert(key.to_owned(), value);
            }
        };
        match (target, outcome) {
            (SnipTarget::NodeParam(node, key), SnipOutcome::Point(point)) => {
                set(self, node, &key, serde_json::json!(point))
            }
            (SnipTarget::NodeParam(node, key), SnipOutcome::Region(region)) => {
                set(self, node, &key, serde_json::json!(region))
            }
            (SnipTarget::NodeParam(node, key), SnipOutcome::Color(rgb)) => {
                set(self, node, &key, serde_json::json!(format_hex_color(rgb)));
            }
            (SnipTarget::NodeParam(node, key), SnipOutcome::Image { png }) => {
                let count = self.doc.project.assets.of_kind(magi_core::assets::AssetKind::Image).count();
                let id = self.doc.project.assets.add_image(format!("Image {}", count + 1), png);
                set(self, node, &key, serde_json::json!(id));
            }
            (SnipTarget::Action(node, key, index), SnipOutcome::Point(point)) => {
                let Some(node) = self.doc.project.graph.node_mut(node) else { return };
                let mut actions: Vec<Action> = node.param(&key).unwrap_or_default();
                match actions.get_mut(index) {
                    Some(Action::Mouse { at, .. }) => *at = Some(point),
                    Some(Action::Move { x, y, display, relative, .. }) => {
                        (*x, *y, *display, *relative) = (point.x, point.y, point.display, false)
                    }
                    _ => return,
                }
                node.set_param(&key, actions);
            }
            (SnipTarget::RecordingMove(asset, row, end), SnipOutcome::Point(point)) => {
                let displays = crate::displays::current();
                let Some(recording) = self.doc.project.assets.recording_mut(&asset) else { return };
                // Positions after a display change are relative to that display.
                let shown =
                    recording.strokes[..row.min(recording.strokes.len())].iter().rev().find_map(|s| match s.input {
                        magi_core::recording::StrokeInput::Display { index } => Some(index),
                        _ => None,
                    });
                let global = displays.to_global(point);
                let origin = shown.and_then(|d| displays.get(d)).map_or((0, 0), |r| (r.x, r.y));
                let at = Point::new(global.x - origin.0, global.y - origin.1);
                if let Some(magi_core::recording::StrokeInput::Move { path }) =
                    recording.strokes.get(row).map(|s| &s.input)
                    && let (Some(first), Some(last)) = (path.first(), path.last())
                {
                    let (start, finish) = (Point::new(first.1, first.2), Point::new(last.1, last.2));
                    let (start, finish) = if end { (start, at) } else { (at, finish) };
                    recording_editor::RecordingEdit::MoveEnds(row, start, finish).apply(recording);
                }
            }
            (SnipTarget::ClickerPosition, SnipOutcome::Point(Point { x, y, display })) => {
                self.settings.clicker.position = ClickPosition::Fixed { x, y, display };
            }
            (SnipTarget::AddImage(node), SnipOutcome::Image { png }) => {
                let count = self.doc.project.assets.of_kind(magi_core::assets::AssetKind::Image).count();
                let id = self.doc.project.assets.add_image(format!("Image {}", count + 1), png);
                self.push_image(node, id);
            }
            (SnipTarget::ImageRegion(node, index), SnipOutcome::Region(region)) => {
                let Some(node) = self.doc.project.graph.node_mut(node) else { return };
                let params: magi_core::nodes::vision::FindImageParams = node.params_or_default();
                let mut images = params.targets();
                if let Some(target) = images.get_mut(index) {
                    target.region = Some(region);
                    node.set_param("images", images);
                    node.set_param("image", "");
                }
            }
            (SnipTarget::ImageGroupRegion(node, index), SnipOutcome::Region(region)) => {
                let Some(node) = self.doc.project.graph.node_mut(node) else { return };
                let mut params: magi_core::nodes::vision::FindImageParams = node.params_or_default();
                if let Some(group) = params.groups.get_mut(index) {
                    group.region = Some(region);
                    node.set_param("groups", params.groups);
                }
            }
            (SnipTarget::MagicArea, SnipOutcome::Region(region)) => {
                if let Some(doc) = &mut self.magic_tab.doc {
                    doc.file.model.capture.area = magi_core::magic::CaptureArea::Region { rect: region };
                }
            }
            (SnipTarget::ClickerNode(node), SnipOutcome::Point(point)) => {
                targets::apply(&mut self.doc.project.graph, &TargetSource::Clicker { node }, TargetShape::Point(point));
            }
            _ => {}
        }
    }

    /// Adds an image to an Image Detection block's list.
    fn push_image(&mut self, node: NodeId, id: String) {
        let Some(node) = self.doc.project.graph.node_mut(node) else { return };
        let params: magi_core::nodes::vision::FindImageParams = node.params_or_default();
        let mut images = params.targets();
        if !images.iter().any(|t| t.image == id) {
            images.push(magi_core::nodes::vision::ImageTarget { image: id, ..Default::default() });
        }
        node.set_param("images", images);
        node.set_param("image", "");
    }

    fn apply_hotkey(&mut self, target: HotkeyTarget, hotkey: Hotkey) {
        match target {
            HotkeyTarget::Clicker => self.settings.clicker.hotkey = hotkey,
            HotkeyTarget::StopEverything => self.settings.stop_hotkey = hotkey,
            HotkeyTarget::Record => self.settings.record_hotkey = hotkey,
            HotkeyTarget::Run => self.settings.run_hotkey = hotkey,
            HotkeyTarget::ExtraStop(Some(index)) => {
                if let Some(slot) = self.settings.options.stop_hotkeys.get_mut(index) {
                    *slot = hotkey;
                }
            }
            HotkeyTarget::ExtraStop(None) => {
                if !self.settings.options.stop_hotkeys.contains(&hotkey) {
                    self.settings.options.stop_hotkeys.push(hotkey);
                }
            }
            HotkeyTarget::Node(node, key) => {
                if let Some(node) = self.doc.project.graph.node_mut(node) {
                    node.set_param(&key, hotkey);
                }
            }
        }
    }

    // ---- per-frame bookkeeping (runs even while the window is hidden) -------

    fn tick(&mut self, ctx: &egui::Context) {
        // Recordings that ended (possibly by hotkey, with the window minimised).
        for recording in self.controller.take_recordings() {
            let count = self.doc.project.assets.of_kind(magi_core::assets::AssetKind::Recording).count();
            let name = format!("Recording {}", count + 1);
            let summary = recording.summary();
            let id = self.doc.project.assets.add_recording(name.clone(), recording);
            match self.record_into.take() {
                Some((node, key)) if self.doc.project.graph.contains(node) => {
                    if let Some(node) = self.doc.project.graph.node_mut(node) {
                        node.set_param(&key, &id);
                    }
                }
                _ if self.settings.auto_insert_recording => {
                    let at = self.macro_tab.view.free_spot(&self.doc.project);
                    if let Some(node) =
                        self.macro_tab.view.add_node(&mut self.doc.project, &self.registry, ids::PLAY_RECORDING, at)
                    {
                        if let Some(target) = self.doc.project.graph.node_mut(node) {
                            target.set_param("recording", &id);
                        }
                        self.macro_tab.view.show_card(node);
                    }
                    self.settings.tab = Tab::Macro;
                }
                _ => {}
            }
            self.toast(format!("{name} saved: {summary}"));
        }

        // Hotkey capture.
        if let Some((target, pending)) = &mut self.capture {
            match self.controller.poll_capture() {
                Some(CaptureResult::Pending(chord)) => *pending = chord,
                Some(CaptureResult::Done(hotkey)) => {
                    let target = target.clone();
                    self.capture = None;
                    self.apply_hotkey(target, hotkey);
                }
                Some(CaptureResult::Cancelled) => self.capture = None,
                None => {}
            }
        }

        // Work finished on other threads.
        let mut finished = Vec::new();
        self.jobs.retain(|receiver| match receiver.try_recv() {
            Ok(job) => {
                finished.push(job);
                false
            }
            Err(mpsc::TryRecvError::Empty) => true,
            Err(mpsc::TryRecvError::Disconnected) => false,
        });
        for job in finished {
            match job {
                Job::Toast(text, false) => self.toast(text),
                Job::Toast(text, true) => {
                    self.downloading = false;
                    self.toast_error(text);
                }
                Job::BackgroundReady(name) => {
                    self.downloading = false;
                    self.settings.appearance.background = Some(name);
                    self.backgrounds = background::list(&self.paths.backgrounds_dir());
                }
                Job::Component(component, result) => {
                    self.installing = None;
                    match result {
                        Ok(message) => self.toast(message),
                        Err(error) => self.toast_error(error),
                    }
                    self.refresh_components(ctx);
                    if component == Component::Python {
                        self.find_pythons(ctx);
                    }
                }
                Job::Runtime(report) => {
                    self.checking_components = false;
                    self.runtime = Some(*report);
                }
                Job::Pythons(found, auto) => {
                    self.looking_for_python = false;
                    self.settings.pythons_found = found;
                    self.settings.python_auto = auto;
                    self.settings.python_checked = true;
                    // What is installed in that Python (PyTorch) may now be known.
                    self.refresh_components(ctx);
                }
            }
        }
        // The first time Python is tried (a macro ran), find out which Pythons there are, once.
        if !self.settings.python_checked
            && !self.looking_for_python
            && self.controller.python_status() != PythonStatus::Unknown
        {
            self.find_pythons(ctx);
        }

        // Hand the latest macro and settings to the controller.
        let settled = !ctx.input(|i| i.pointer.any_down()) && !ctx.text_edit_focused();
        self.doc.track(settled);
        self.clicker_history.track(&self.settings.clicker, settled);
        self.sync_controller();
        let minimized = ctx.input(|i| i.viewport().minimized) == Some(true);
        self.controller.set_window_hidden(minimized);

        // Console.
        let fresh = self.controller.monitor().logs_from(self.next_log);
        if let Some(last) = fresh.last() {
            self.next_log = last.seq + 1;
            for entry in &fresh {
                if entry.level == LogLevel::Error {
                    self.error_node = entry.node;
                    self.settings.show_console = true;
                    self.toasts.push(Toast { text: entry.message.clone(), born: Instant::now(), is_error: true });
                }
            }
            self.logs.extend(fresh);
            let excess = self.logs.len().saturating_sub(1000);
            self.logs.drain(..excess);
        }
        self.activity = self.controller.monitor().activity();

        // Macros that ended by themselves.
        let finished = self.controller.monitor().take_finished();
        if self.settings.notify_on_finish
            && let Some(run) = finished.iter().find(|run| run.by_itself)
        {
            let message = if run.failed {
                format!("“{}” stopped because of an error.", self.doc.title())
            } else {
                format!("“{}” has finished.", self.doc.title())
            };
            let notification =
                magi_core::host::Notification { title: "MacroMagi".into(), message, sound: true, popup: true };
            if let Err(error) = magi_platform::notify::notify(&notification) {
                self.toast_error(format!("Could not show a notification: {}", error.0));
            }
        }

        // A training run that ended by itself: it may have taken hours, so it says so.
        if let Some(job) = &self.magic_tab.job {
            let outcome = job.state.lock().unwrap_or_else(|p| p.into_inner()).outcome.clone();
            match outcome {
                None => {
                    self.training_seen.get_or_insert(false);
                }
                Some(result) => {
                    let long = job.started.elapsed() > Duration::from_secs(60);
                    if self.training_seen.take() == Some(false) && long && self.settings.notify_after_training {
                        let message = match result {
                            Ok(()) => format!(
                                "“{}” has finished training.",
                                self.magic_tab.doc.as_ref().map_or_else(|| "The model".into(), |d| d.title())
                            ),
                            Err(error) => error.lines().next().unwrap_or("Training failed.").to_owned(),
                        };
                        let notification = magi_core::host::Notification {
                            title: "MacroMagi".into(),
                            message,
                            sound: true,
                            popup: true,
                        };
                        let _ = magi_platform::notify::notify(&notification);
                    }
                }
            }
        }

        self.apply_device();
        self.apply_tools();

        // Settings are written a moment after they stop changing.
        if self.settings != self.saved_settings {
            let since = *self.settings_changed.get_or_insert_with(Instant::now);
            if since.elapsed() >= SETTINGS_SAVE_DELAY {
                self.save_settings();
            } else {
                ctx.request_repaint_after(SETTINGS_SAVE_DELAY);
            }
        }
    }

    /// Loads ONNX Runtime if it has been downloaded, and applies the AI device setting.
    fn start_onnxruntime(&mut self) {
        self.components.clean_up();
        if let Some((library, gpu)) = self.components.onnxruntime() {
            if let Some(directml) = self.components.directml() {
                magi_platform::gpu::preload(&directml);
            }
            if let Err(error) = magi_ml::use_onnxruntime(&library, gpu) {
                self.toast_error(format!("ONNX Runtime could not be loaded: {}", error.0));
            }
        }
        self.apply_device();
    }

    /// Native tools run with the Python for AI; their files and the native engine's cache go in the
    /// data folder; MagiDB is the chosen folder, the downloaded copy or the online one. Looked at
    /// again when one of them changes.
    fn apply_tools(&mut self) {
        let magidb = self.settings.magidb_dir.clone().filter(|d| d.is_dir()).or_else(|| self.components.magidb());
        let key = (
            self.python_for_packages(),
            magidb.clone(),
            self.settings.magidb_url.trim().to_owned(),
            self.components.vision_model(),
        );
        if self.applied_tools.as_ref() == Some(&key) {
            return;
        }
        magi_ml::tools::set_python(key.0.clone());
        magi_ml::tools::set_cache_dir(self.paths.data_dir.join("cache"));
        let text = |p: &Option<PathBuf>| p.as_ref().map(|p| p.display().to_string());
        magi_ml::tools::set_tool_env("MAGI_DB_DIR", text(&magidb));
        magi_ml::tools::set_tool_env("MAGI_DB_URL", Some(key.2.clone()).filter(|u| !u.is_empty()));
        magi_ml::tools::set_tool_env(
            "MAGI_DB_CACHE",
            Some(self.paths.data_dir.join("cache").join("magidb").display().to_string()),
        );
        magi_ml::tools::set_tool_env("MAGI_VISION_MODEL", text(&key.3));
        self.applied_tools = Some(key);
    }

    /// A tool call the model wants to make that changes files or runs programs: the user allows it or not.
    fn approvals(&mut self, ctx: &egui::Context) {
        let pending = magi_ml::tools::pending_approvals();
        let Some((id, call)) = pending.first().cloned() else { return };
        let arg = |key: &str| call.args.get(key).and_then(|v| v.as_str()).unwrap_or_default().to_owned();
        let what = match call.name.as_str() {
            "write_file" => format!("write the file {}", arg("path")),
            "edit_file" => format!("change the file {}", arg("path")),
            "edit_document" => format!("change the document {}", arg("path")),
            "run_command" => {
                let folder = arg("folder");
                if folder.is_empty() { "run a command".to_owned() } else { format!("run a command in {folder}") }
            }
            other => format!("use {other}"),
        };
        let mut answer: Option<(bool, bool)> = None;
        egui::Window::new(format!("{}  Allow this?", icon::SHIELD_CHECK))
            .id(egui::Id::new("tool-approval"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .default_width(420.0)
            .show(ctx, |ui| {
                ui.label(RichText::new(format!("The model wants to {what}.")).strong());
                if pending.len() > 1 {
                    ui.label(RichText::new(format!("{} more waiting.", pending.len() - 1)).color(theme::text_dim()));
                }
                let shown = match call.name.as_str() {
                    "run_command" => arg("command"),
                    "write_file" => arg("text"),
                    "edit_file" => format!(
                        "{}
→
{}",
                        arg("find"),
                        arg("replace")
                    ),
                    _ => serde_json::to_string_pretty(&call.args).unwrap_or_default(),
                };
                egui::ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
                    ui.add(egui::Label::new(RichText::new(shown).monospace().size(11.5)).wrap());
                });
                ui.add_space(6.0);
                widgets::button_row(ui, |ui| {
                    if widgets::accent_button(ui, "Allow", theme::accent(), vec2(0.0, 26.0)).clicked() {
                        answer = Some((true, false));
                    }
                    if ui.button(format!("Allow {} until MacroMagi closes", call.name)).clicked() {
                        answer = Some((true, true));
                    }
                    if ui.button("Don't allow").clicked() {
                        answer = Some((false, false));
                    }
                });
            });
        if let Some((allow, always)) = answer {
            magi_ml::tools::answer_approval(id, allow, always);
        }
    }

    fn apply_device(&mut self) {
        // Called every frame too, so a changed setting takes effect at once.
        if self.applied_device != Some(self.settings.ai_device) {
            self.applied_device = Some(self.settings.ai_device);
            magi_ml::set_device(self.settings.ai_device);
        }
    }

    /// Where AI models run, in a few words.
    fn ai_engines(&self) -> String {
        let runtime = magi_ml::onnxruntime();
        match self.settings.ai_device {
            magi_ml::Device::Auto => match runtime {
                Some((_, true)) => "the fastest of CPU and graphics card".into(),
                Some(_) => "the faster CPU engine".into(),
                None => "the CPU".into(),
            },
            magi_ml::Device::Cpu => "the CPU".into(),
            magi_ml::Device::Gpu { index } => match (runtime, self.gpus.iter().find(|g| g.index == index)) {
                (Some((_, true)), Some(gpu)) => gpu.name.clone(),
                _ => "a graphics card that is not available".into(),
            },
        }
    }

    /// Works out the optional parts' statuses in the background (it can start Python).
    fn refresh_components(&mut self, ctx: &egui::Context) {
        let components = self.components.clone();
        let python = self.python_for_packages();
        self.checking_components = true;
        self.spawn_job(ctx, move || Job::Runtime(Box::new(components.report(python.as_deref()))));
    }

    /// The graphics cards training uses (CUDA numbers; empty: the CPU).
    fn training_devices(&self) -> Vec<u32> {
        match &self.settings.train_gpus {
            Some(list) => list.clone(),
            None => {
                let cards = self.runtime.as_ref().and_then(|r| r.torch.as_ref()).map(|t| t.devices.len());
                // Not looked at yet: the first card, which the trainer leaves for the CPU if it cannot use it.
                if cards.unwrap_or(1) > 0 { vec![0] } else { Vec::new() }
            }
        }
    }

    /// The Python that packages (PyTorch) go into: the one macros use – the
    /// chosen one, else what "automatic" stood for at the last check.
    fn python_for_packages(&self) -> Option<PathBuf> {
        self.settings.python_path.clone().or_else(|| self.settings.python_auto.as_ref().map(|(p, _)| p.clone()))
    }

    fn python_config(&self) -> magi_engine::python::PythonConfig {
        magi_engine::python::PythonConfig {
            interpreter: self.settings.python_path.clone(),
            app_dir: self.paths.app_dir.clone(),
            data_dir: self.paths.data_dir.clone(),
        }
    }

    /// Looks for Python interpreters, and works out which one "automatic"
    /// stands for, off the UI thread. Starts no visible windows.
    fn find_pythons(&mut self, ctx: &egui::Context) {
        if self.looking_for_python {
            return;
        }
        self.looking_for_python = true;
        let (components, app_dir) = (self.components.clone(), self.paths.app_dir.clone());
        let config = magi_engine::python::PythonConfig { interpreter: None, ..self.python_config() };
        self.spawn_job(ctx, move || {
            let mut found = crate::components::find_pythons(&components, &app_dir);
            let auto = magi_engine::python::resolve_interpreter(&config);
            let same = |a: &std::path::Path, b: &std::path::Path| {
                a.canonicalize().unwrap_or_else(|_| a.to_owned()) == b.canonicalize().unwrap_or_else(|_| b.to_owned())
            };
            if let Some((path, version)) = &auto
                && !found.iter().any(|(p, _)| same(p, path))
            {
                found.push((path.clone(), version.clone()));
            }
            Job::Pythons(found, auto)
        });
    }

    fn install_component(&mut self, ctx: &egui::Context, component: Component, remove: bool) {
        if self.installing.is_some() {
            return;
        }
        let progress = Arc::new(std::sync::Mutex::new((
            if remove { "Uninstalling…".to_owned() } else { "Starting…".to_owned() },
            None,
        )));
        self.installing = Some((component, Arc::clone(&progress)));
        let components = self.components.clone();
        let known = self.python_for_packages();
        let config = self.python_config();
        let repaint = ctx.clone();
        self.spawn_job(ctx, move || {
            let report = |message: &str, share: Option<f32>| {
                *progress.lock().unwrap_or_else(|p| p.into_inner()) = (message.to_owned(), share);
                repaint.request_repaint();
            };
            // Python packages go into the Python macros use, worked out now if it is not known yet.
            let package = matches!(component, Component::PyTorch | Component::PyTorchCuda | Component::Onnx);
            let python = known.or_else(|| {
                package.then(|| magi_engine::python::resolve_interpreter(&config).map(|(path, _)| path)).flatten()
            });
            Job::Component(
                component,
                if remove {
                    components.uninstall(component, python.as_deref())
                } else {
                    components.install(component, python.as_deref(), &report)
                },
            )
        });
    }

    fn save_settings(&mut self) {
        self.settings_changed = None;
        self.saved_settings = self.settings.clone();
        if self.data_dir_error.is_none()
            && let Err(error) = self.settings.save(&self.paths.settings_file())
        {
            self.data_dir_error = Some(format!("Settings could not be saved: {error}"));
        }
    }

    /// Makes sure the controller has this frame's edits before something is started.
    fn sync_controller(&mut self) {
        self.controller.publish(self.doc.revision(), &self.doc.project);
        self.controller.set_config(ControllerConfig::from_settings(&self.settings));
    }

    fn apply_look(&mut self, ctx: &egui::Context) {
        let look = &self.settings.appearance;
        let dark = match look.theme {
            ThemeMode::Dark => true,
            ThemeMode::Light => false,
            ThemeMode::System => ctx.system_theme() != Some(egui::Theme::Light),
        };
        // While training with the environment optimized: no picture, no see-through panels.
        let plain = self.settings.optimize_training && self.magic_tab.training();
        let background = if plain { None } else { look.background.clone() };
        let panel_alpha = if background.is_some() { look.panel_opacity } else { 1.0 };
        let wanted = (dark, look.accent, panel_alpha.to_bits());
        if self.applied_palette != Some(wanted) {
            self.applied_palette = Some(wanted);
            theme::apply(ctx, Palette::build(dark, theme::accent_from(look.accent), panel_alpha));
        }
        let (file, blur) = (background, look.background_blur);
        self.background.sync(ctx, &self.paths.backgrounds_dir(), file.as_deref(), blur);
        if let Some(error) = self.background.take_error() {
            self.toast_error(error);
        }

        let wanted = (self.settings.always_on_top, self.settings.hide_from_capture, self.settings.ui_scale);
        if self.applied_window != Some(wanted) {
            self.applied_window = Some(wanted);
            let level = if wanted.0 { egui::WindowLevel::AlwaysOnTop } else { egui::WindowLevel::Normal };
            ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(level));
            ctx.send_viewport_cmd(egui::ViewportCommand::ContentProtected(wanted.1));
            let range = crate::settings::UI_SCALE_RANGE;
            ctx.set_zoom_factor(wanted.2.clamp(*range.start(), *range.end()));
        }
        let title = match self.settings.tab {
            Tab::AutoClicker => "MacroMagi".to_owned(),
            // A star marks unsaved changes; a dot divides the document from the program.
            Tab::Macro => format!("{}{} • MacroMagi", if self.doc.is_dirty() { "*" } else { "" }, self.doc.title()),
            Tab::Magic => match &self.magic_tab.doc {
                Some(doc) => format!("{}{} • MacroMagi", if doc.is_dirty() { "*" } else { "" }, doc.title()),
                None => "Magic • MacroMagi".to_owned(),
            },
        };
        if title != self.applied_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.applied_title = title;
        }
    }

    // ---- the banner ---------------------------------------------------------

    fn banner(&mut self, ui: &mut Ui) {
        let width = ui.available_width();
        let compact = width < COMPACT_BANNER_BELOW;
        let label = |glyph: &str, word: &str| if compact { glyph.to_owned() } else { format!("{glyph}  {word}") };
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = if compact { 2.0 } else { 6.0 };
            if width > 470.0 {
                let color = theme::accent();
                if self.logo.as_ref().is_none_or(|(made_for, _)| *made_for != color) {
                    self.logo = Some((color, crate::icon::logo(ui.ctx(), color)));
                }
                if let Some((_, logo)) = &self.logo
                    && ui
                        .add(egui::Image::new((logo.id(), vec2(24.0, 24.0))).sense(egui::Sense::click()))
                        .on_hover_text("About MacroMagi")
                        .clicked()
                {
                    self.about_open = true;
                }
            }
            let names = if width < 430.0 { ("Clicker", "Macro") } else { ("Auto Clicker", "Macro") };
            widgets::segmented(
                ui,
                &mut self.settings.tab,
                &[(Tab::AutoClicker, names.0), (Tab::Macro, names.1), (Tab::Magic, "Magic")],
            );
            ui.add_space(if compact { 2.0 } else { 6.0 });

            if width < TINY_BANNER_BELOW {
                // Too narrow for a row of menus: one menu holds them all.
                ui.menu_button(RichText::new(icon::LIST).size(15.0), |ui| {
                    ui.set_min_width(180.0);
                    self.menus(ui, &ctx, |glyph, word| format!("{glyph}  {word}"));
                })
                .response
                .on_hover_text("Menu");
            } else {
                self.menus(ui, &ctx, label);
            }

            let history = match self.settings.tab {
                Tab::AutoClicker => {
                    Some((self.clicker_history.can_undo(&self.settings.clicker), self.clicker_history.can_redo()))
                }
                Tab::Macro => Some((self.doc.can_undo(), self.doc.can_redo())),
                Tab::Magic => self.magic_tab.doc.as_ref().map(|d| (d.can_undo(), d.can_redo())),
            };
            if let Some((can_undo, can_redo)) = history {
                let undo = egui::Button::new(icon::ARROW_COUNTER_CLOCKWISE).frame(false);
                if ui.add_enabled(can_undo, undo).on_hover_text("Undo (Ctrl+Z)").clicked() {
                    self.undo();
                }
                let redo = egui::Button::new(icon::ARROW_CLOCKWISE).frame(false);
                if ui.add_enabled(can_redo, redo).on_hover_text("Redo (Ctrl+Y)").clicked() {
                    self.redo();
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                if widgets::icon_toggle(ui, icon::GEAR, self.settings_window.open, "Settings").clicked() {
                    self.settings_window.open = !self.settings_window.open;
                }
                if widgets::icon_toggle(
                    ui,
                    icon::ROBOT,
                    self.agent_chat.open,
                    "AI Agent: chat with your models in a window of their own",
                )
                .clicked()
                {
                    self.agent_chat.open = !self.agent_chat.open;
                }
                let pinned = self.settings.always_on_top;
                if widgets::icon_toggle(ui, icon::PUSH_PIN, pinned, "Keep on top of other windows").clicked() {
                    self.settings.always_on_top = !pinned;
                }
                let hidden = self.settings.hide_from_capture;
                let (glyph, hint) = if hidden {
                    (
                        icon::EYE_SLASH,
                        "Hidden from screen captures – click to show MacroMagi in screenshots and recordings",
                    )
                } else {
                    (icon::EYE, "Visible in screen captures – click to hide MacroMagi from screenshots and recordings")
                };
                if widgets::icon_toggle(ui, glyph, hidden, hint).clicked() {
                    self.settings.hide_from_capture = !hidden;
                }
                if self.settings.tab == Tab::Macro {
                    let light = self.settings.lightweight;
                    let hint = if light {
                        "Lightweight mode: only Run, Stop and the console – click to show the editor again"
                    } else {
                        "Lightweight mode: hide the editor, keeping only Run, Stop and the console"
                    };
                    if widgets::icon_toggle(ui, icon::FEATHER, light, hint).clicked() {
                        self.settings.lightweight = !light;
                    }
                }
                if self.controller.anything_running() {
                    let text = if compact || self.settings.stop_hotkey.is_none() {
                        icon::STOP.to_owned()
                    } else {
                        format!("{}  Stop all ({})", icon::STOP, self.settings.stop_hotkey.label())
                    };
                    let stop = egui::Button::new(RichText::new(text).size(12.0).color(theme::danger())).frame(false);
                    if ui.add(stop).on_hover_text("Stop the auto clicker and every macro").clicked() {
                        self.controller.stop_everything();
                    }
                }
                if let Some(status) = self.controller.recording_status() {
                    let text = format!("{}  {:.0}s", icon::RECORD, status.seconds);
                    ui.label(RichText::new(text).size(12.0).color(theme::danger()));
                    ui.ctx().request_repaint_after(Duration::from_millis(250));
                }
            });
        });
    }

    /// File, Options, Presets and the tab's own menus.
    fn menus(&mut self, ui: &mut Ui, ctx: &egui::Context, label: impl Fn(&str, &str) -> String) {
        if self.settings.tab == Tab::Magic {
            self.magic_file_menu(ui, &label(icon::FILE, "File"));
        } else {
            self.file_menu(ui, &label(icon::FILE, "File"));
        }
        let mut option_events = Vec::new();
        ui.menu_button(label(icon::SLIDERS, "Options"), |ui| {
            option_events = options::menu(
                ui,
                &mut self.settings.options,
                &mut self.settings.dry_run,
                &mut self.settings.notify_on_finish,
                &mut self.settings.delete_unused_recordings,
            );
            if self.settings.tab == Tab::Magic {
                ui.separator();
                ui.label(RichText::new("MAGIC").size(10.5).color(theme::text_faint()));
                ui.checkbox(&mut self.settings.optimize_training, "Optimize environment when training").on_hover_text(
                    "While a model trains, show only the training – no canvas, background picture or see-through \
                     panels – so the computer spends its time on training",
                );
                ui.checkbox(&mut self.settings.notify_after_training, "Notify when training ends").on_hover_text(
                    "A sound and a desktop notification when a training run of more than a minute finishes or fails",
                );
            }
        })
        .response
        .on_hover_text("Options for the Auto Clicker, macros and training");
        for event in option_events {
            match event {
                OptionsEvent::OpenStopHotkeys => self.stop_hotkeys_open = true,
                OptionsEvent::OpenProcessRules => self.rules_window.open = true,
                OptionsEvent::OpenDownloads => {
                    self.settings.settings_page = SettingsPage::Runtime;
                    self.settings_window.open = true;
                }
                OptionsEvent::CreateShortcut => {
                    let icon_dir = self.paths.data_dir.join("icon");
                    self.spawn_job(ctx, move || {
                        let icons = crate::icon::write_shortcut_icons(&icon_dir).ok();
                        let icon = icons.as_ref().map(|(ico, png)| magi_platform::system::ShortcutIcon { ico, png });
                        match magi_platform::system::create_desktop_shortcut(icon) {
                            Ok(path) => Job::Toast(format!("Shortcut created: {}", path.display()), false),
                            Err(error) => Job::Toast(format!("Could not create the shortcut: {error}"), true),
                        }
                    })
                }
            }
        }
        if self.settings.tab == Tab::Magic {
            self.magic_presets_menu(ui, &label(icon::BOOKMARK_SIMPLE, "Presets"));
        } else {
            self.presets_menu(ui, &label(icon::BOOKMARK_SIMPLE, "Presets"));
        }

        match self.settings.tab {
            Tab::AutoClicker => {
                if ui
                    .button(label(icon::TREE_STRUCTURE, "Open as macro"))
                    .on_hover_text("Turn these settings into blocks you can extend in the Macro tab")
                    .clicked()
                {
                    ui.close();
                    self.guarded(Pending::ClickerGraph);
                }
            }
            Tab::Macro => {
                if ui
                    .button(label(icon::PLUS, "Add"))
                    .on_hover_text("Add a block (or double-click the canvas)")
                    .clicked()
                {
                    ui.close();
                    self.macro_tab.view.open_palette_in_view();
                }
                self.view_menu(ui, &label(icon::MONITOR, "View"));
            }
            Tab::Magic => {
                let has_doc = self.magic_tab.doc.is_some();
                if ui
                    .add_enabled(has_doc, egui::Button::new(label(icon::PLUS, "Add")))
                    .on_hover_text("Add a block (or right-click the canvas)")
                    .clicked()
                {
                    ui.close();
                    self.magic_tab.view.open_palette_in_view();
                }
                self.magic_view_menu(ui, &label(icon::MONITOR, "View"));
                if let Some(doc) = &self.magic_tab.doc {
                    if self.magic_tab.training() {
                        let stop =
                            egui::Button::new(RichText::new(label(icon::STOP, "Stop training")).color(theme::danger()));
                        if ui.add(stop).clicked()
                            && let Some(job) = &self.magic_tab.job
                        {
                            job.stop();
                        }
                    } else {
                        let problem = doc.training_problem();
                        let hint =
                            problem.clone().unwrap_or_else(|| "Run the training stages on the canvas, in order".into());
                        let train =
                            egui::Button::new(RichText::new(label(icon::PLAY, "Train")).color(if problem.is_none() {
                                theme::accent()
                            } else {
                                theme::text_faint()
                            }));
                        let choices = if problem.is_none() { doc.train_choices() } else { Vec::new() };
                        if choices.len() > 1 {
                            // Saved weights to start from: Train asks where to start.
                            let mut chosen = None;
                            let menu = egui::containers::menu::MenuButton::from_button(train).ui(ui, |ui| {
                                ui.set_min_width(260.0);
                                for (label, choice) in &choices {
                                    if ui.button(label).clicked() {
                                        chosen = Some(choice.clone());
                                        ui.close();
                                    }
                                }
                            });
                            menu.0.on_hover_text("Train from the start, or from weights saved after a stage");
                            if let Some(choice) = chosen {
                                self.start_training(ctx, choice);
                            }
                        } else {
                            let response = ui.add(train).on_hover_text(&hint);
                            if response.clicked() {
                                if problem.is_some() {
                                    self.toast_error(hint);
                                } else {
                                    self.start_training(ctx, Default::default());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The Magic tab's File menu: models, not macros.
    fn magic_file_menu(&mut self, ui: &mut Ui, label: &str) {
        let mut chosen: Option<MagicPending> = None;
        let mut save: Option<bool> = None;
        let mut export_python = None;
        let mut close = false;
        let recent = self.settings.recent_models.clone();
        let has_doc = self.magic_tab.doc.is_some();
        ui.menu_button(label, |ui| {
            ui.set_min_width(250.0);
            let mut item = |ui: &mut Ui, text: String, shortcut: &str, action: MagicPending| {
                if ui.add(egui::Button::new(text).shortcut_text(shortcut)).clicked() {
                    chosen = Some(action);
                    ui.close();
                }
            };
            item(ui, format!("{}  New text model", icon::ROBOT), "Ctrl+N", MagicPending::NewModel { text: true });
            item(ui, format!("{}  New text model: Magi 150M (a quick test)", icon::ROBOT), "", MagicPending::NewTestMagi);
            item(ui, format!("{}  New vision model", icon::EYE), "", MagicPending::NewModel { text: false });
            item(ui, format!("{}  Open…", icon::FOLDER_OPEN), "Ctrl+O", MagicPending::Open(None));
            ui.menu_button(format!("{}  Open recent", icon::CLOCK_COUNTER_CLOCKWISE), |ui| {
                if recent.is_empty() {
                    ui.weak("Nothing yet");
                }
                for path in &recent {
                    let name = path
                        .file_name()
                        .map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
                    if ui.button(name).on_hover_text(path.display().to_string()).clicked() {
                        chosen = Some(MagicPending::Open(Some(path.clone())));
                        ui.close();
                    }
                }
            });
            ui.menu_button(format!("{}  Import", icon::DOWNLOAD_SIMPLE), |ui| {
                ui.set_min_width(250.0);
                if ui.button(format!("{}  An ONNX project folder…", icon::FOLDER_OPEN)).clicked() {
                    chosen = Some(MagicPending::Import(ImportFrom::Folder));
                    ui.close();
                }
                if ui.button(format!("{}  A .zip or .onnx file…", icon::FILE_ZIP)).clicked() {
                    chosen = Some(MagicPending::Import(ImportFrom::File));
                    ui.close();
                }
            });
            ui.separator();
            if ui
                .add_enabled(has_doc, egui::Button::new(format!("{}  Save", icon::FLOPPY_DISK)).shortcut_text("Ctrl+S"))
                .clicked()
            {
                save = Some(false);
                ui.close();
            }
            if ui
                .add_enabled(
                    has_doc,
                    egui::Button::new(format!("{}  Save as…", icon::FLOPPY_DISK_BACK)).shortcut_text("Ctrl+Shift+S"),
                )
                .clicked()
            {
                save = Some(true);
                ui.close();
            }
            if ui
                .add_enabled(has_doc, egui::Button::new(format!("{}  Export as Python…", icon::FILE_PY)))
                .on_hover_text("A folder with run.py that trains the model as the canvas does, and a README")
                .clicked()
            {
                export_python = Some(false);
                ui.close();
            }
            if ui
                .add_enabled(has_doc, egui::Button::new(format!("{}  Export as Python (experimental)…", icon::FILE_PY)))
                .on_hover_text(
                    "The same, and the trainer tries compiled layers: on Linux, kept only when faster with the same loss",
                )
                .clicked()
            {
                export_python = Some(true);
                ui.close();
            }
            ui.separator();
            if ui.add_enabled(has_doc, egui::Button::new(format!("{}  Close the model", icon::X))).clicked() {
                close = true;
                ui.close();
            }
        });
        if let Some(action) = chosen {
            self.magic_guarded(action);
        }
        if let Some(choose) = save {
            self.save_magic(choose);
        }
        if let Some(compile) = export_python {
            self.export_magic_python(compile);
        }
        if close {
            self.magic_guarded(MagicPending::New);
        }
    }

    /// The Magic tab's Presets menu: common architectures, and saved blocks.
    fn magic_presets_menu(&mut self, ui: &mut Ui, label: &str) {
        let has_doc = self.magic_tab.doc.is_some();
        let mut common: Option<usize> = None;
        let mut insert: Option<NodePreset> = None;
        let mut delete = None;
        let mut prompt = false;
        let has_selection = !self.magic_tab.view.selection.is_empty();
        let magic_kinds = |preset: &NodePreset| {
            !preset.fragment.nodes.is_empty() && preset.fragment.nodes.iter().all(|n| n.kind.starts_with("magic."))
        };
        ui.menu_button(label, |ui| {
            ui.set_width(260.0);
            ui.add_enabled_ui(has_doc, |ui| {
                ui.menu_button(format!("{}  Commons", icon::STAR), |ui| {
                    ui.set_width(330.0);
                    ui.label(RichText::new("Magi's architectures, from 3.5M to 400M parameters. They replace the architecture blocks on the canvas.").size(11.0).color(theme::text_faint()));
                    // As tall as the window allows; the rest scrolls.
                    let tallest = (ui.ctx().content_rect().height() - 140.0).max(160.0);
                    egui::ScrollArea::vertical().max_height(tallest).auto_shrink([false, true]).show(ui, |ui| {
                        for (index, item) in magi_core::nodes::studio::commons().iter().enumerate() {
                            let response = ui.add(egui::Button::new(RichText::new(item.name).strong()).min_size(vec2(310.0, 24.0)));
                            ui.add(egui::Label::new(RichText::new(item.description).size(11.0).color(theme::text_dim())).wrap());
                            if response.clicked() {
                                common = Some(index);
                                ui.close();
                            }
                            ui.add_space(4.0);
                        }
                    });
                });
            });
            ui.separator();
            let mut any = false;
            for entry in self.presets.entries() {
                let PresetBody::Nodes(nodes) = &entry.preset.body else { continue };
                if !magic_kinds(nodes) {
                    continue;
                }
                any = true;
                ui.horizontal(|ui| {
                    let width = ui.available_width() - 30.0;
                    let button = egui::Button::new(entry.preset.name.clone()).truncate().min_size(vec2(width, 22.0));
                    if ui.add_enabled(has_doc, button).clicked() {
                        insert = Some(nodes.clone());
                        ui.close();
                    }
                    if widgets::icon_button(ui, icon::TRASH, "Delete this preset").clicked() {
                        delete = Some(entry.path.clone());
                    }
                });
            }
            if !any {
                ui.weak("No saved blocks yet.");
            }
            ui.separator();
            let save = egui::Button::new(format!("{}  Save selected blocks…", icon::FLOPPY_DISK));
            if ui.add_enabled(has_selection, save).on_disabled_hover_text("Select one or more blocks first").clicked() {
                prompt = true;
                ui.close();
            }
        });
        if let Some(index) = common
            && let Some(doc) = &mut self.magic_tab.doc
        {
            let commons = magi_core::nodes::studio::commons();
            let at = self.magic_tab.view.view_center();
            let created = magi_core::nodes::studio::apply_common(&mut doc.studio.graph, &commons[index], [at.x, at.y]);
            doc.sync_canvas();
            self.magic_tab.view.selection = created.into_iter().collect();
            self.magic_tab.view.request_fit();
            self.toast(format!("{} put in place of the architecture", commons[index].name));
        }
        if let Some(nodes) = insert
            && let Some(doc) = &mut self.magic_tab.doc
        {
            let at = self.magic_tab.view.view_center();
            self.magic_tab.view.insert_preset(&mut doc.studio, crate::ui::magic_tab::registry(), &nodes, at);
        }
        if let Some(path) = delete {
            self.presets.delete(&path);
        }
        if prompt {
            self.preset_prompt = Some((PresetKind::MagicBlocks, String::new()));
        }
    }

    fn magic_view_menu(&mut self, ui: &mut Ui, label: &str) {
        ui.menu_button(label, |ui| {
            ui.set_min_width(220.0);
            ui.checkbox(&mut self.settings.magic_console, "Console");
            ui.checkbox(&mut self.settings.magic_panel, "Block settings");
            ui.separator();
            ui.checkbox(&mut self.settings.show_grid, "Grid");
            if ui.add(egui::Button::new(format!("{}  Find…", icon::MAGNIFYING_GLASS)).shortcut_text("Ctrl+F")).clicked()
            {
                self.magic_tab.view.open_find();
                ui.close();
            }
            ui.separator();
            if ui.add(egui::Button::new(format!("{}  Fit to view", icon::CORNERS_OUT)).shortcut_text("F")).clicked() {
                self.magic_tab.view.request_fit();
                ui.close();
            }
            if ui.button(format!("{}  Actual size", icon::MAGNIFYING_GLASS)).clicked() {
                self.magic_tab.view.reset_zoom();
                ui.close();
            }
        });
    }

    fn undo(&mut self) {
        match self.settings.tab {
            Tab::AutoClicker => self.clicker_history.undo(&mut self.settings.clicker),
            Tab::Macro => self.doc.undo(),
            Tab::Magic => {
                if let Some(doc) = &mut self.magic_tab.doc {
                    doc.undo();
                }
            }
        }
    }

    fn redo(&mut self) {
        match self.settings.tab {
            Tab::AutoClicker => self.clicker_history.redo(&mut self.settings.clicker),
            Tab::Macro => self.doc.redo(),
            Tab::Magic => {
                if let Some(doc) = &mut self.magic_tab.doc {
                    doc.redo();
                }
            }
        }
    }

    fn file_menu(&mut self, ui: &mut Ui, label: &str) {
        type Action = Box<dyn FnOnce(&mut MagiApp)>;
        let mut chosen: Option<Action> = None;
        let tab = self.settings.tab;
        let recent = self.settings.recent_files.clone();
        let blocks = self.doc.project.graph.len();
        ui.menu_button(label, |ui| {
            ui.set_min_width(240.0);
            let mut item = |ui: &mut Ui, text: String, shortcut: &str, action: Action| {
                if ui.add(egui::Button::new(text).shortcut_text(shortcut)).clicked() {
                    chosen = Some(action);
                    ui.close();
                }
            };
            item(ui, format!("{}  New macro", icon::FILE_PLUS), "Ctrl+N", Box::new(|app| app.guarded(Pending::New)));
            item(
                ui,
                format!("{}  Open…", icon::FOLDER_OPEN),
                "Ctrl+O",
                Box::new(|app| app.guarded(Pending::Open(None))),
            );
            ui.menu_button(format!("{}  Open recent", icon::CLOCK_COUNTER_CLOCKWISE), |ui| {
                if recent.is_empty() {
                    ui.weak("Nothing yet");
                }
                for path in &recent {
                    let name = path
                        .file_name()
                        .map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
                    if ui.button(name).on_hover_text(path.display().to_string()).clicked() {
                        let path = path.clone();
                        chosen = Some(Box::new(move |app: &mut MagiApp| app.guarded(Pending::Open(Some(path)))));
                        ui.close();
                    }
                }
            });
            ui.menu_button(format!("{}  Examples", icon::LIGHTBULB), |ui| {
                for (index, name) in examples::NAMES.iter().enumerate() {
                    if ui.button(*name).clicked() {
                        chosen = Some(Box::new(move |app: &mut MagiApp| app.guarded(Pending::Example(index))));
                        ui.close();
                    }
                }
            });
            ui.separator();
            let mut item = |ui: &mut Ui, text: String, shortcut: &str, action: Action| {
                if ui.add(egui::Button::new(text).shortcut_text(shortcut)).clicked() {
                    chosen = Some(action);
                    ui.close();
                }
            };
            item(
                ui,
                format!("{}  Save", icon::FLOPPY_DISK),
                "Ctrl+S",
                Box::new(|app| {
                    app.save(false);
                }),
            );
            item(
                ui,
                format!("{}  Save as…", icon::FLOPPY_DISK_BACK),
                "Ctrl+Shift+S",
                Box::new(|app| {
                    app.save(true);
                }),
            );
            item(
                ui,
                format!("{}  Export presets…", icon::BOOKMARK_SIMPLE),
                "",
                Box::new(|app| app.export_presets = Some(0)),
            );
            item(
                ui,
                format!("{}  Export…", icon::EXPORT),
                "Ctrl+E",
                Box::new(|app| app.export = Some(app.settings.export.clone())),
            );
            match tab {
                Tab::AutoClicker => {
                    ui.separator();
                    item(
                        ui,
                        format!("{}  Load clicker settings from a file…", icon::DOWNLOAD_SIMPLE),
                        "",
                        Box::new(|app| app.load_clicker_settings()),
                    );
                }
                Tab::Macro | Tab::Magic => {
                    ui.separator();
                    ui.weak(format!("{blocks} blocks"));
                }
            }
        });
        if let Some(action) = chosen {
            action(self);
        }
    }

    /// Presets for the open tab: Auto Clicker settings, or groups of blocks.
    fn presets_menu(&mut self, ui: &mut Ui, label: &str) {
        let tab = self.settings.tab;
        let mut delete = None;
        let mut load: Option<AutoClickerConfig> = None;
        let mut insert: Option<NodePreset> = None;
        let mut prompt = None;
        let mut import = false;
        let has_selection = !self.macro_tab.view.selection.is_empty();
        ui.menu_button(label, |ui| {
            // A fixed width: rows with a button at their right end would otherwise
            // stretch the menu as wide as the screen allows.
            ui.set_width(240.0);
            let library = self.presets.entries().iter().map(|entry| (&entry.preset, Some(&entry.path)));
            let mut any = false;
            for (preset, path) in library {
                let fits = matches!(
                    (&preset.body, tab),
                    (PresetBody::AutoClicker(_), Tab::AutoClicker) | (PresetBody::Nodes(_), Tab::Macro)
                );
                if !fits {
                    continue;
                }
                any = true;
                ui.horizontal(|ui| {
                    let name = preset.name.clone();
                    let width = ui.available_width() - if path.is_some() { 30.0 } else { 0.0 };
                    let button = egui::Button::new(name).truncate().min_size(vec2(width, 22.0));
                    if ui.add_sized([width, 22.0], button).clicked() {
                        match &preset.body {
                            PresetBody::AutoClicker(config) => load = Some(config.clone()),
                            PresetBody::Nodes(nodes) => insert = Some(nodes.clone()),
                        }
                        ui.close();
                    }
                    if let Some(path) = path
                        && widgets::icon_button(ui, icon::TRASH, "Delete this preset").clicked()
                    {
                        delete = Some(path.clone());
                    }
                });
            }
            if !any {
                ui.weak(if tab == Tab::AutoClicker { "No clicker presets yet." } else { "No block presets yet." });
            }
            ui.separator();
            match tab {
                Tab::AutoClicker => {
                    if ui.button(format!("{}  Save current settings…", icon::FLOPPY_DISK)).clicked() {
                        prompt = Some(PresetKind::Clicker);
                        ui.close();
                    }
                }
                Tab::Macro | Tab::Magic => {
                    let save = egui::Button::new(format!("{}  Save selected blocks…", icon::FLOPPY_DISK));
                    if ui
                        .add_enabled(has_selection, save)
                        .on_disabled_hover_text("Select one or more blocks first")
                        .clicked()
                    {
                        prompt = Some(PresetKind::Blocks);
                        ui.close();
                    }
                }
            }
            if ui.button(format!("{}  Import a .mp file…", icon::DOWNLOAD_SIMPLE)).clicked() {
                import = true;
                ui.close();
            }
        });
        if let Some(path) = delete {
            self.presets.delete(&path);
        }
        if let Some(config) = load {
            self.settings.clicker = config;
        }
        if let Some(nodes) = insert {
            let at = self.macro_tab.view.view_center();
            self.macro_tab.view.insert_preset(&mut self.doc.project, &self.registry, &nodes, at);
        }
        if let Some(kind) = prompt {
            self.preset_prompt = Some((kind, String::new()));
        }
        if import && let Some(file) = rfd::FileDialog::new().add_filter(PRESET_FILTER.0, PRESET_FILTER.1).pick_file() {
            match self.presets.import(&file) {
                Ok(preset) => self.toast(format!("Imported “{}”", preset.name)),
                Err(error) => self.toast_error(format!("Could not import: {error}")),
            }
        }
    }

    fn view_menu(&mut self, ui: &mut Ui, label: &str) {
        let mut show_targets = false;
        let width = ui.ctx().content_rect().width();
        ui.menu_button(label, |ui| {
            ui.set_min_width(230.0);
            if ui
                .button(format!("{}  Screen targets…", icon::CROSSHAIR))
                .on_hover_text(
                    "Show every click position and search area on the screen, numbered in run order, to check and \
                     move them",
                )
                .clicked()
            {
                show_targets = true;
                ui.close();
            }
            ui.separator();
            ui.checkbox(&mut self.settings.show_console, "Console");
            let mut inspector = self.macro_tab.inspector_shown(width);
            if ui.checkbox(&mut inspector, "Block settings").changed() {
                self.macro_tab.toggle_inspector(width);
            }
            ui.checkbox(&mut self.settings.show_variables, "Variables");
            ui.separator();
            ui.checkbox(&mut self.settings.show_grid, "Grid").on_hover_text("The grid behind the blocks");
            ui.checkbox(&mut self.settings.show_activity, "Block activity").on_hover_text("Blocks glow while they run");
            if ui.add(egui::Button::new(format!("{}  Find…", icon::MAGNIFYING_GLASS)).shortcut_text("Ctrl+F")).clicked()
            {
                self.macro_tab.view.open_find();
                ui.close();
            }
            ui.separator();
            if ui.add(egui::Button::new(format!("{}  Fit to view", icon::CORNERS_OUT)).shortcut_text("F")).clicked() {
                self.macro_tab.view.request_fit();
                ui.close();
            }
            if ui.button(format!("{}  Actual size", icon::MAGNIFYING_GLASS)).clicked() {
                self.macro_tab.view.reset_zoom();
                ui.close();
            }
        });
        if show_targets {
            match TargetOverlay::begin(ui.ctx()) {
                Ok(overlay) => self.overlay = Some(overlay),
                Err(error) => self.toast_error(error),
            }
        }
    }

    fn status_bar(&mut self, ui: &mut Ui) {
        let problem = self.data_dir_error.clone().or_else(|| {
            self.controller.listener_error().map(|e| format!("Hotkeys and recording are unavailable: {e}"))
        });
        let Some(problem) = problem else { return };
        egui::Panel::bottom("status")
            .frame(
                egui::Frame::new()
                    .fill(theme::mix(theme::bg_panel(), theme::warning(), 0.12))
                    .inner_margin(egui::Margin::symmetric(10, 5)),
            )
            .show(ui, |ui| widgets::notice(ui, icon::WARNING, problem, theme::warning()));
    }

    // ---- the tabs -------------------------------------------------------------

    fn clicker(&mut self, ui: &mut Ui) {
        let monitor = self.controller.clicker_monitor();
        let error = monitor.logs().into_iter().rev().find(|e| e.level == LogLevel::Error).map(|e| e.message);
        let running = self.controller.clicker_running();
        let displays = crate::displays::current();
        let env = ClickerEnv {
            running,
            clicks: monitor.clicks(),
            error,
            capturing: self.capture.as_ref().map(|(target, pending)| (target, pending)),
            displays: &displays,
        };
        let events = self.clicker_tab.show(ui, &mut self.settings.clicker, &env);
        if running {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
        for event in events {
            match event {
                ClickerEvent::Start => {
                    // Publish first, so a setting changed this very frame is used.
                    self.sync_controller();
                    self.controller.start_clicker();
                }
                ClickerEvent::Stop => self.controller.stop_clicker(),
                ClickerEvent::Request(request) => self.handle_request(ui.ctx(), request),
            }
        }
    }

    fn macro_editor(&mut self, ui: &mut Ui) {
        let running = self.controller.running_nodes();
        let python = self.controller.python_status();
        let displays = crate::displays::current();
        let live = self.controller.monitor().variables();
        let mut env = MacroEnv {
            registry: &self.registry,
            activity: &self.activity,
            running: &running,
            error_node: self.error_node,
            logs: &self.logs,
            recording: self.controller.recording_status(),
            run_hotkey: &self.settings.run_hotkey,
            record_hotkey: &self.settings.record_hotkey,
            capturing: self.capture.as_ref().map(|(target, pending)| (target, pending)),
            engine_running: (!running.is_empty()).then(|| self.controller.macro_engine()).flatten(),
            engine_preference: self.controller.preferred_engine(),
            python: &python,
            dry_run: self.settings.dry_run,
            show_console: &mut self.settings.show_console,
            show_variables: &mut self.settings.show_variables,
            lightweight: &mut self.settings.lightweight,
            show_grid: self.settings.show_grid,
            show_activity: self.settings.show_activity,
            record: &mut self.settings.record,
            displays: &displays,
            live: &live,
        };
        let events = self.macro_tab.show(ui, &mut self.doc, &mut env);
        let size = self.macro_tab.view.palette_size;
        self.settings.palette_size = [size.x.round(), size.y.round()];
        for event in events {
            match event {
                MacroEvent::Run => self.run_macro(true),
                MacroEvent::Stop => self.controller.stop_macro(),
                MacroEvent::ToggleStart(node) => {
                    if running.contains(&node) {
                        self.controller.stop_node(node);
                    } else {
                        self.error_node = None;
                        self.sync_controller();
                        self.controller.run_from(node);
                    }
                }
                MacroEvent::RunFrom(node) => {
                    self.error_node = None;
                    self.sync_controller();
                    self.controller.run_from(node);
                }
                MacroEvent::ToggleRecording => self.toggle_recording(None, true),
                MacroEvent::Request(request) => self.handle_request(ui.ctx(), request),
                MacroEvent::Toast(text) => self.toast(text),
                MacroEvent::ClearVariable(name) => self.controller.clear_variable(&name),
                MacroEvent::OpenEngineSettings => {
                    self.settings.settings_page = SettingsPage::Engine;
                    self.settings_window.open = true;
                }
                MacroEvent::ClearConsole => {
                    self.logs.clear();
                    self.error_node = None;
                }
            }
        }
    }

    fn magic(&mut self, ui: &mut Ui) {
        let displays = crate::displays::current();
        let env = MagicEnv {
            displays: &displays,
            engines: self.ai_engines(),
            dry_run: self.settings.dry_run,
            optimize_training: self.settings.optimize_training,
            show_console: self.settings.magic_console,
            show_panel: self.settings.magic_panel,
            show_grid: self.settings.show_grid,
            generator: self.settings.generator.clone(),
        };
        let events = self.magic_tab.show(ui, &env);
        // Something needing PyTorch or onnx failed: look at them again.
        if std::mem::take(&mut self.magic_tab.runtime_failed) && !self.checking_components {
            self.refresh_components(ui.ctx());
        }
        for event in events {
            self.magic_event(ui.ctx(), event, &displays);
        }
    }

    fn magic_event(&mut self, ctx: &egui::Context, event: MagicEvent, displays: &magi_core::geom::Displays) {
        match event {
            MagicEvent::NewModel { text } => self.magic_guarded(MagicPending::NewModel { text }),
            MagicEvent::NewTestMagi => self.magic_guarded(MagicPending::NewTestMagi),
            MagicEvent::Toast(text) => self.toast(text),
            MagicEvent::Train(choice) => self.start_training(ctx, choice),
            MagicEvent::SaveBase(stage) => self.save_base_model(stage),
            MagicEvent::GeneratorSettings(settings) => {
                self.settings.generator = settings;
                self.save_settings();
            }
            MagicEvent::StopTraining => {
                if let Some(seen) = &mut self.training_seen {
                    *seen = true;
                }
                if let Some(job) = &self.magic_tab.job {
                    job.stop();
                }
            }
            MagicEvent::Compile => self.compile_magic(),
            MagicEvent::OpenFolder(folder) => {
                let _ = std::fs::create_dir_all(&folder);
                if let Err(error) = magi_platform::system::open_folder(&folder) {
                    self.toast_error(format!("Could not open the folder: {error}"));
                }
            }
            MagicEvent::OpenRuntime => {
                self.settings.settings_page = SettingsPage::Runtime;
                self.settings_window.open = true;
            }
            MagicEvent::AddDatasetFiles { node, negative } => {
                let dialog = rfd::FileDialog::new().add_filter("Text datasets", &["jsonl", "json", "md", "txt"]);
                let Some(paths) = dialog.pick_files() else { return };
                let Some(doc) = &mut self.magic_tab.doc else { return };
                let mut added = Vec::new();
                for path in paths {
                    let Ok(bytes) = std::fs::read(&path) else { continue };
                    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    let file = std::iter::once(format!("data/text/{name}"))
                        .chain((2..).map(|n| format!("data/text/{n}-{name}")))
                        .find(|f| !doc.file.data.contains_key(f))
                        .expect("a free name");
                    doc.file.data.insert(file.clone(), bytes);
                    added.push(file);
                }
                doc.files_changed();
                if let Some(block) = doc.studio.graph.node_mut(node) {
                    let mut params: magi_core::nodes::studio::TextDataParams = block.params_or_default();
                    let count = added.len();
                    if negative {
                        params.negative.extend(added)
                    } else {
                        params.positive.extend(added)
                    }
                    block.set_params(&params);
                    self.toast(format!("{count} dataset files added"));
                }
            }
            MagicEvent::Open => self.magic_guarded(MagicPending::Open(None)),
            MagicEvent::Import(from) => self.magic_guarded(MagicPending::Import(from)),
            MagicEvent::UseInMacro => self.use_model_in_macro(),
            MagicEvent::ReplaceModel => {
                let Some(path) = rfd::FileDialog::new().add_filter("ONNX model", &["onnx"]).pick_file() else { return };
                let Some(doc) = &mut self.magic_tab.doc else { return };
                match std::fs::read(&path).map_err(|e| e.to_string()).and_then(|bytes| doc.replace_onnx(bytes)) {
                    Ok(()) => {
                        self.magic_tab.tester = None;
                        self.toast("Model replaced");
                    }
                    Err(error) => self.toast_error(format!("Could not use that model: {error}")),
                }
            }
            MagicEvent::PickArea => {
                self.handle_request(ctx, UiRequest::Snip(crate::ui::SnipMode::Region, SnipTarget::MagicArea))
            }
            MagicEvent::AddPictures { class, negative } => {
                let dialog = rfd::FileDialog::new().add_filter("Pictures", &["png", "jpg", "jpeg", "bmp", "webp"]);
                let Some(paths) = dialog.pick_files() else { return };
                let Some(doc) = &mut self.magic_tab.doc else { return };
                let folder = if negative { "not" } else { "train" };
                let mut added = 0;
                for path in paths {
                    let Ok(bytes) = std::fs::read(&path) else { continue };
                    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    let Some(file) = (0..)
                        .map(|n| format!("data/{folder}/{class}/{n:04}-{name}"))
                        .find(|f| !doc.file.data.contains_key(f))
                    else {
                        continue;
                    };
                    doc.file.data.insert(file.clone(), bytes);
                    doc.file.model.training.push(magi_core::magic::DataItem { file, class, negative });
                    added += 1;
                }
                doc.files_changed();
                self.toast(format!("{added} pictures added"));
            }
            MagicEvent::TryPicture => {
                let dialog = rfd::FileDialog::new().add_filter("Pictures", &["png", "jpg", "jpeg", "bmp", "webp"]);
                let Some(path) = dialog.pick_file() else { return };
                let Some(doc) = &self.magic_tab.doc else { return };
                let result = std::fs::read(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|bytes| magi_vision::decode(&bytes).map_err(|e| e.to_string()))
                    .and_then(|bitmap| {
                        let mut model = magi_ml::Model::load(&doc.file.onnx, &doc.file.model.input).map_err(|e| e.0)?;
                        model.classify(&bitmap.rgba, bitmap.width, bitmap.height).map_err(|e| e.0)
                    });
                match result {
                    Ok(probabilities) => {
                        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        self.magic_tab.picture = Some((name, probabilities));
                    }
                    Err(error) => self.toast_error(format!("Could not run the model: {error}")),
                }
            }
            MagicEvent::StartTest => {
                let Some(doc) = &self.magic_tab.doc else { return };
                let host = self.controller.machine(self.settings.dry_run);
                let repaint = ctx.clone();
                self.magic_tab.picture = None;
                self.magic_tab.tester = Some(crate::magic::Tester::start(
                    doc.file.onnx.clone(),
                    doc.file.model.clone(),
                    displays.clone(),
                    host,
                    self.magic_tab.press,
                    move || repaint.request_repaint(),
                ));
            }
            MagicEvent::StopTest => self.magic_tab.tester = None,
        }
    }

    /// Trains the Magic tab's model as its canvas says, in the background: from
    /// where `choice` says (the start, a checkpoint, the trained model), and the
    /// stages it names.
    fn start_training(&mut self, ctx: &egui::Context, choice: crate::magic::TrainChoice) {
        use crate::magic::TrainFrom;
        if self.magic_tab.training() {
            return;
        }
        let Some(doc) = &self.magic_tab.doc else { return };
        if let Some(problem) = doc.training_problem() {
            return self.toast_error(problem);
        }
        let Ok(mut pipeline) = magi_core::nodes::studio::pipeline(&doc.studio.graph) else { return };
        // A learned tokenizer learns its pieces from the datasets when training starts from zero.
        let from_zero = choice.from == TrainFrom::Start && pipeline.entry.kind == magi_core::nodes::studio::kinds::ZERO;
        // With a corpus read where it trains, the trainer learns them there, from it.
        if from_zero
            && let Some(doc) = self.magic_tab.doc.as_mut()
            && doc.reads_corpus()
        {
            doc.forget_tokens();
        } else if from_zero
            && let Some(doc) = self.magic_tab.doc.as_mut()
            && let Some(count) = doc.learn_tokens()
        {
            use crate::magic::train::Level;
            self.magic_tab.console_line(
                Level::Info,
                format!("Learned {count} pieces of words from the datasets for its tokens."),
            );
        }
        // A base model file brings its tokens.
        if let Some(doc) = self.magic_tab.doc.as_mut() {
            match doc.take_base_tokens() {
                Ok(Some(note)) => self.magic_tab.console_line(crate::magic::train::Level::Info, note),
                Ok(None) => {}
                Err(problem) => return self.toast_error(problem),
            }
        }
        let Some(doc) = &self.magic_tab.doc else { return };
        if !choice.stages.is_empty() {
            pipeline.stages.retain(|s| choice.stages.contains(&s.node));
        }
        let from = match choice.from {
            TrainFrom::Start => None,
            TrainFrom::Checkpoint(stage) => match doc.checkpoints().into_iter().find(|c| c.stage == stage) {
                Some(checkpoint) => Some(checkpoint.path),
                None => return self.toast_error("Those saved weights are gone: train from the start."),
            },
            TrainFrom::Trained => {
                // Outside the training folder, which each training empties first.
                let path = self.paths.data_dir.join("training-start.safetensors");
                if let Err(error) = std::fs::write(&path, &doc.file.weights) {
                    return self.toast_error(format!("The trained weights could not be written: {error}"));
                }
                Some(path)
            }
        };
        // The weights after every stage it runs, to start a later training from.
        let key = crate::magic::MagicDoc::architecture_key(&pipeline.architecture, doc.file.model.text.as_ref());
        let checkpoints = doc
            .checkpoint_dir()
            .map(|dir| {
                pipeline
                    .stages
                    .iter()
                    .map(|s| {
                        let path = dir.join(format!("{}.safetensors", s.node));
                        (s.node, (path, serde_json::json!({ "architecture": key })))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let python = self
            .python_for_packages()
            .or_else(|| magi_engine::python::resolve_interpreter(&self.python_config()).map(|(path, _)| path));
        let Some(python) = python else {
            self.settings.settings_page = SettingsPage::Runtime;
            self.settings_window.open = true;
            return self.toast_error("Training needs Python with PyTorch: install them under Settings › Runtime.");
        };
        if let Some(report) = &self.runtime
            && report.torch.is_none()
        {
            self.settings.settings_page = SettingsPage::Runtime;
            self.settings_window.open = true;
            return self.toast_error(format!(
                "PyTorch is not installed in the Python for AI ({}). Install it under Settings › Runtime.",
                python.display()
            ));
        }
        let runtime = match magi_engine::python::runtime_dir(&self.paths.data_dir) {
            Ok(dir) => dir,
            Err(error) => return self.toast_error(format!("The trainer could not be unpacked: {error}")),
        };
        let request = crate::magic::train::Request {
            python,
            runtime,
            work: self.paths.data_dir.join("training"),
            file: doc.file.clone(),
            graph: doc.studio.graph.clone(),
            pipeline,
            devices: self.training_devices(),
            from,
            checkpoints,
        };
        let repaint = ctx.clone();
        self.magic_tab.job = Some(crate::magic::train::TrainJob::start(request, move || repaint.request_repaint()));
    }

    /// A stage's kept weights as a base model file, to fine-tune other models from.
    fn save_base_model(&mut self, stage: magi_core::graph::NodeId) {
        let Some(doc) = &self.magic_tab.doc else { return };
        let name = format!("{} base", doc.title());
        let Some(path) = rfd::FileDialog::new()
            .set_title("Save the base model")
            .add_filter("Magi model", &["szi"])
            .set_file_name(format!("{}.szi", crate::magic::file_stem(&name)))
            .save_file()
        else {
            return;
        };
        match doc.save_base(stage, &path, &name) {
            Ok(()) => {
                self.magic_tab.console_line(crate::magic::train::Level::Done, format!("Saved {}", path.display()));
                self.toast(format!("Saved the base model {}", path.display()));
            }
            Err(error) => self.toast_error(error),
        }
    }

    /// File › Export as Python: the workspace as a folder of Python (`compile`: the experimental one).
    fn export_magic_python(&mut self, compile: bool) {
        // A learned tokenizer's pieces go with it, learned now if they have not been yet.
        if let Some(doc) = self.magic_tab.doc.as_mut()
            && doc.file.model.text.as_ref().is_some_and(|t| t.tokenizer.learned() && t.tokenizer.merges.is_empty())
            && !doc.reads_corpus()
        {
            doc.learn_tokens();
        }
        if let Some(doc) = self.magic_tab.doc.as_mut()
            && let Err(problem) = doc.take_base_tokens()
        {
            return self.toast_error(problem);
        }
        let Some(doc) = &self.magic_tab.doc else { return };
        let Some(folder) = rfd::FileDialog::new().set_title("Where the Python folder goes").pick_folder() else {
            return;
        };
        let runtime = match magi_engine::python::runtime_dir(&self.paths.data_dir) {
            Ok(dir) => dir,
            Err(error) => return self.toast_error(format!("The trainer could not be unpacked: {error}")),
        };
        match crate::magic::export_python::export(doc, &runtime, &folder, compile) {
            Ok(dir) => {
                let _ = magi_platform::system::open_folder(&dir);
                self.toast(format!("Exported to {}", dir.display()));
            }
            Err(error) => self.toast_error(error),
        }
    }

    /// Saves the Magic tab's model as its Compile block says (after training, too).
    fn compile_magic(&mut self) {
        let Some(doc) = &mut self.magic_tab.doc else { return };
        use crate::magic::train::Level;
        match doc.compile() {
            Ok((written, skipped)) => {
                let first = written.first().cloned();
                for path in &written {
                    self.magic_tab.console_line(Level::Done, format!("Saved {}", path.display()));
                }
                for note in &skipped {
                    self.magic_tab.console_line(Level::Warning, note.clone());
                }
                self.magic_tab.compiled = written;
                if let Some(path) = first {
                    self.remember_model(&path);
                    self.toast(format!("Saved {}", path.display()));
                }
            }
            Err(error) => {
                self.magic_tab.console_line(Level::Error, error.clone());
                self.toast_error(error);
            }
        }
    }

    /// Puts the Magic tab's model into the open macro – replacing an earlier
    /// version of it, if the macro has one – with blocks that run it.
    fn use_model_in_macro(&mut self) {
        let Some(doc) = &mut self.magic_tab.doc else { return };
        let bytes = match doc.file.for_macro() {
            Ok(bytes) => bytes,
            Err(error) => return self.toast_error(format!("Could not prepare the model: {error}")),
        };
        doc.mark_kept();
        let name = doc.file.model.name.clone();
        let model = doc.file.model.clone();
        let project = &mut self.doc.project;
        // An earlier version: a model asset of the same name.
        let previous: Vec<String> = project
            .assets
            .of_kind(magi_core::assets::AssetKind::Model)
            .filter(|a| a.name == name)
            .map(|a| a.id.clone())
            .collect();
        let id = project.assets.add_model(name.clone(), bytes);
        let mut updated = 0;
        for node in project.graph.nodes_mut() {
            if matches!(
                node.kind.as_str(),
                ids::AI_WATCH | ids::AI_CLASSIFY | ids::AI_AGENT | ids::AI_LOAD | ids::AI_UNLOAD
            ) && let Some(current) = node.param::<String>("model")
                && previous.contains(&current)
                && current != id
            {
                node.set_param("model", &id);
                updated += 1;
            }
        }
        for old in previous.iter().filter(|old| **old != id) {
            let still_used = project.graph.referenced_assets(&self.registry).contains(old);
            if !still_used {
                project.assets.remove(old);
            }
        }
        let uses = project.graph.nodes().any(|n| {
            matches!(n.kind.as_str(), ids::AI_WATCH | ids::AI_CLASSIFY | ids::AI_AGENT | ids::AI_LOAD | ids::AI_UNLOAD)
                && n.param::<String>("model").as_deref() == Some(id.as_str())
        });
        if !uses {
            // Blocks saved with the model, unless they are the plain set from before AI Load and Unload Model.
            let plain = |blocks: &magi_core::graph::Fragment| {
                let mut kinds: Vec<&str> = blocks.nodes.iter().map(|n| n.kind.as_str()).collect();
                kinds.sort_unstable();
                kinds == [ids::AI_AGENT, ids::LOG, ids::START, ids::COMMENT]
            };
            let mut blocks = model
                .blocks
                .clone()
                .filter(|blocks| !plain(blocks))
                .unwrap_or_else(|| crate::magic::starter_blocks(&model, &id, &self.registry));
            for node in &mut blocks.nodes {
                if matches!(
                    node.kind.as_str(),
                    ids::AI_WATCH | ids::AI_CLASSIFY | ids::AI_AGENT | ids::AI_LOAD | ids::AI_UNLOAD
                ) {
                    node.set_param("model", &id);
                }
            }
            let at = self.macro_tab.view.free_spot(&self.doc.project);
            let new = self.doc.project.graph.insert_fragment(&blocks, [at.x.round(), at.y.round()]);
            if let Some(first) = new.first() {
                self.macro_tab.view.reveal(*first);
            }
            self.toast(format!("“{name}” added to the macro, with blocks that run it"));
        } else {
            self.toast(match updated {
                0 => format!("“{name}” is already in the macro"),
                n => format!("“{name}” updated in the macro ({n} blocks)"),
            });
        }
        self.settings.tab = Tab::Macro;
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.text_edit_focused() || self.capture.is_some() {
            return;
        }
        let (save, save_as, open, new, export, undo, redo) = ctx.input(|i| {
            let cmd = i.modifiers.command;
            (
                cmd && !i.modifiers.shift && i.key_pressed(egui::Key::S),
                cmd && i.modifiers.shift && i.key_pressed(egui::Key::S),
                cmd && i.key_pressed(egui::Key::O),
                cmd && i.key_pressed(egui::Key::N),
                cmd && i.key_pressed(egui::Key::E),
                cmd && !i.modifiers.shift && i.key_pressed(egui::Key::Z),
                cmd && (i.key_pressed(egui::Key::Y) || (i.modifiers.shift && i.key_pressed(egui::Key::Z))),
            )
        });
        // On the Magic tab they are about the model.
        if self.settings.tab == Tab::Magic {
            if (save || save_as) && self.magic_tab.doc.is_some() {
                self.save_magic(save_as);
            }
            if open {
                self.magic_guarded(MagicPending::Open(None));
            }
            if new {
                self.magic_guarded(MagicPending::NewModel { text: true });
            }
            if undo {
                self.undo();
            }
            if redo {
                self.redo();
            }
            return;
        }
        if save {
            self.save(false);
        }
        if save_as {
            self.save(true);
        }
        if open {
            self.guarded(Pending::Open(None));
        }
        if new {
            self.guarded(Pending::New);
        }
        if export {
            self.export = Some(self.settings.export.clone());
        }
        if undo {
            self.undo();
        }
        if redo {
            self.redo();
        }
    }

    // ---- windows and dialogs ----------------------------------------------

    fn windows(&mut self, ctx: &egui::Context) {
        self.settings_window_frame(ctx);
        if self.stop_hotkeys_open {
            let capturing = self.capture.as_ref().map(|(target, pending)| (target, pending));
            let mut open = true;
            let requests = options::stop_hotkeys_window(ctx, &mut open, &mut self.settings.options, capturing);
            self.stop_hotkeys_open = open;
            for request in requests {
                self.handle_request(ctx, request);
            }
        }
        self.rules_window.show(ctx, &mut self.settings.options);
        self.recording_popout(ctx);
        self.rate_editor(ctx);
        if self.agent_chat.open {
            self.agent_chat.set_dir(self.paths.data_dir.join("chats"));
            self.agent_chat.offer_magic_model(self.magic_tab.doc.as_ref().map(|d| &d.file));
            self.agent_chat.open_thoughts = self.settings.agent_open_thoughts;
            self.agent_chat.show(ctx);
            self.settings.agent_open_thoughts = self.agent_chat.open_thoughts;
        } else {
            self.agent_chat.release();
        }
        self.dialogs(ctx);
        self.toasts(ctx);
    }

    fn settings_window_frame(&mut self, ctx: &egui::Context) {
        if !self.settings_window.open {
            return;
        }
        let python = self.controller.python_status();
        let starts: Vec<StartHotkey> = self
            .doc
            .project
            .graph
            .starts()
            .map(|node| StartHotkey {
                node: node.id,
                name: format!("“{}” block", self.registry.title(node)),
                hotkey: node.params_or_default::<StartParams>().hotkey,
            })
            .collect();
        let installing = self.installing.as_ref().map(|(component, progress)| {
            let (message, share) = progress.lock().unwrap_or_else(|p| p.into_inner()).clone();
            (*component, message, share)
        });
        let pythons = self.settings.python_checked.then(|| self.settings.pythons_found.clone());
        let python_auto = self.settings.python_auto.clone();
        let ai_python = self.python_for_packages().map(|path| {
            let version = self
                .settings
                .pythons_found
                .iter()
                .find(|(p, _)| *p == path)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| "?".into());
            (path, version)
        });
        let env = SettingsEnv {
            capturing: self.capture.as_ref().map(|(target, pending)| (target, pending)),
            python: &python,
            starts: &starts,
            backgrounds: &self.backgrounds,
            downloading: self.downloading,
            pythons: pythons.as_deref(),
            python_auto: python_auto.as_ref(),
            looking_for_python: self.looking_for_python,
            python_downloaded: self.components.python(),
            gpus: &self.gpus,
            machine: &self.machine,
            runtime: self.runtime.as_ref(),
            checking_runtime: self.checking_components,
            ai_python: ai_python.as_ref(),
            installing,
        };
        let events = self.settings_window.show(ctx, &mut self.settings, &env);
        for event in events {
            match event {
                SettingsEvent::Request(request) => self.handle_request(ctx, request),
                SettingsEvent::CheckPython => {
                    self.find_pythons(ctx);
                    self.sync_controller();
                    // Starting Python can take a second; do it off the UI thread.
                    let (controller, ctx) = (self.controller.clone(), ctx.clone());
                    std::thread::spawn(move || {
                        controller.probe_python();
                        ctx.request_repaint();
                    });
                }
                SettingsEvent::ChooseBackgroundFile => {
                    let dialog = rfd::FileDialog::new().add_filter("Pictures", background::EXTENSIONS);
                    if let Some(file) = dialog.pick_file() {
                        match background::import_file(&self.paths.backgrounds_dir(), &file) {
                            Ok(name) => {
                                self.settings.appearance.background = Some(name);
                                self.backgrounds = background::list(&self.paths.backgrounds_dir());
                            }
                            Err(error) => self.toast_error(error),
                        }
                    }
                }
                SettingsEvent::DownloadBackground(url) => {
                    self.downloading = true;
                    let dir = self.paths.backgrounds_dir();
                    self.spawn_job(ctx, move || match background::download(&dir, &url) {
                        Ok(name) => Job::BackgroundReady(name),
                        Err(error) => Job::Toast(error, true),
                    });
                }
                SettingsEvent::RemoveBackground(name) => {
                    if self.settings.appearance.background.as_ref() == Some(&name) {
                        self.settings.appearance.background = None;
                    }
                    background::remove(&self.paths.backgrounds_dir(), &name);
                    self.backgrounds = background::list(&self.paths.backgrounds_dir());
                }
                SettingsEvent::ResetHotkeys => {
                    self.settings.reset_hotkeys();
                    self.toast("Hotkeys reset");
                }
                SettingsEvent::ResetAll => {
                    self.settings.reset_all();
                    self.toast("All settings reset");
                }
                SettingsEvent::FindPythons => self.find_pythons(ctx),
                SettingsEvent::CheckComponents => {
                    if !self.checking_components {
                        self.refresh_components(ctx);
                    }
                }
                SettingsEvent::Install(component) => self.install_component(ctx, component, false),
                SettingsEvent::Remove(component) => self.install_component(ctx, component, true),
                SettingsEvent::SetStartHotkey(node, hotkey) => {
                    if let Some(node) = self.doc.project.graph.node_mut(node) {
                        node.set_param("hotkey", hotkey);
                    }
                }
            }
        }
    }

    /// The recording editor in its own window.
    fn recording_popout(&mut self, ctx: &egui::Context) {
        let Some(asset) = self.recording_window.clone() else { return };
        let Some(recording) = self.doc.project.assets.recording(&asset).cloned() else {
            self.recording_window = None;
            return;
        };
        let name = self.doc.project.assets.get(&asset).map_or_else(|| "Recording".to_owned(), |a| a.name.clone());
        let size = self.settings.recording_window_size.unwrap_or([760.0, 560.0]);
        let builder = egui::ViewportBuilder::default()
            .with_title(format!("{name} • MacroMagi"))
            .with_inner_size(size)
            .with_min_inner_size([420.0, 300.0]);
        let mut edit = None;
        let mut closed = false;
        let mut new_size = None;
        ctx.show_viewport_immediate(egui::ViewportId::from_hash_of(("recording", &asset)), builder, |ui, class| {
            let mut contents = |ui: &mut Ui| {
                let height = (ui.available_height() - 150.0).max(120.0);
                edit = recording_editor::full(ui, &recording, height).edit;
            };
            match class {
                egui::ViewportClass::EmbeddedWindow => contents(ui),
                _ => {
                    egui::CentralPanel::default()
                        .frame(egui::Frame::new().fill(theme::bg_panel().to_opaque()).inner_margin(12))
                        .show(ui, |ui| contents(ui));
                }
            }
            let (close, rect) = ui.input(|i| (i.viewport().close_requested(), i.viewport().inner_rect));
            closed = close;
            new_size = rect.map(|r| [r.width().round(), r.height().round()]);
        });
        if let Some(recording_editor::RecordingEdit::PickMoveEnd(row, end)) = edit {
            let target = SnipTarget::RecordingMove(asset.clone(), row, end);
            self.handle_request(ctx, UiRequest::Snip(crate::ui::SnipMode::Point, target));
        } else if let Some(edit) = edit
            && let Some(target) = self.doc.project.assets.recording_mut(&asset)
        {
            edit.apply(target);
        }
        if let Some(size) = new_size.filter(|[w, h]| *w >= 300.0 && *h >= 200.0) {
            self.settings.recording_window_size = Some(size);
        }
        if closed {
            self.recording_window = None;
        }
    }

    /// The Variable Rate Editor: a Rate block's timing and steps in a window
    /// of their own.
    fn rate_editor(&mut self, ctx: &egui::Context) {
        let Some(node_id) = self.rate_window else { return };
        if self.doc.project.graph.node(node_id).is_none_or(|n| n.kind != ids::RATE) {
            self.rate_window = None;
            return;
        }
        let displays = crate::displays::current();
        let live = self.controller.monitor().variables();
        let mut requests = Vec::new();
        let mut open = true;
        let size = self.settings.rate_window_size.unwrap_or([460.0, 560.0]);
        let shown = egui::Window::new(format!("{}  Variable Rate Editor", icon::METRONOME))
            .id(egui::Id::new("rate-editor"))
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size(size)
            .min_size([340.0, 260.0])
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let mut env = InspectorEnv {
                        registry: &self.registry,
                        images: &mut self.macro_tab.view.images,
                        capturing: self.capture.as_ref().map(|(target, pending)| (target, pending)),
                        recording: false,
                        displays: &displays,
                        live: &live,
                    };
                    let project = &mut self.doc.project;
                    if let Some(node) = project.graph.node_mut(node_id) {
                        ui.horizontal(|ui| {
                            widgets::caption(ui, "Name");
                            ui.add(
                                egui::TextEdit::singleline(&mut node.label)
                                    .hint_text("Rate")
                                    .desired_width(f32::INFINITY),
                            );
                        });
                    }
                    widgets::note(
                        ui,
                        "The steps run one after another, the time below apart. To change the time while the macro \
                         runs, link it to a variable (the link button) or wire a number into the block's Rate input.",
                    );
                    let wired = project.graph.data_source(node_id, "rate").is_some();
                    if wired {
                        widgets::notice(
                            ui,
                            icon::PLUG,
                            "A value is wired into Rate: it decides the time while the macro runs.",
                            theme::accent_bright(),
                        );
                    }
                    ui.add_space(4.0);
                    requests.extend(inspector::edit_setting(ui, project, node_id, "rate_ms", &mut env));
                    requests.extend(inspector::edit_setting(ui, project, node_id, "hold_ms", &mut env));
                    ui.add_space(6.0);
                    let params: magi_core::nodes::flow::RateParams =
                        project.graph.node(node_id).map(|n| n.params_or_default()).unwrap_or_default();
                    rate_preview(ui, &params);
                    ui.add_space(6.0);
                    ui.label(RichText::new("STEPS").size(10.5).color(theme::text_faint()));
                    if let Some(node) = project.graph.node_mut(node_id) {
                        inspector::edit_actions(ui, node, "steps", &displays, &mut requests);
                    }
                });
            });
        if let Some(shown) = shown {
            let rect = shown.response.rect;
            self.settings.rate_window_size = Some([rect.width().round(), rect.height().round()]);
        }
        for request in requests {
            self.handle_request(ctx, request);
        }
        if !open {
            self.rate_window = None;
        }
    }

    fn dialogs(&mut self, ctx: &egui::Context) {
        self.export_presets_dialog(ctx);
        if self.about_open {
            let response = egui::Modal::new(egui::Id::new("about")).show(ctx, |ui| {
                ui.set_width(280.0);
                crate::ui::settings_window::about(ui);
                ui.add_space(8.0);
                ui.button("Close").clicked()
            });
            if response.inner || response.should_close() {
                self.about_open = false;
            }
        }
        self.unsaved_dialog(ctx);
        self.unsaved_model_dialog(ctx);
        self.export_dialog(ctx);
        self.preset_dialog(ctx);
        self.hotkey_warning_dialog(ctx);
        self.file_extras_dialog(ctx);
    }

    fn unsaved_dialog(&mut self, ctx: &egui::Context) {
        let Some(action) = self.pending.clone() else { return };
        let mut decision = None;
        egui::Modal::new(egui::Id::new("unsaved-changes")).show(ctx, |ui| {
            ui.set_width(300.0);
            ui.label(RichText::new("Save your changes?").strong().size(15.0));
            ui.label(format!("“{}” has changes that have not been saved.", self.doc.title()));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if widgets::accent_button(ui, "Save", theme::accent(), vec2(70.0, 26.0)).clicked() {
                    decision = Some(Some(true));
                }
                if ui.button("Don't save").clicked() {
                    decision = Some(Some(false));
                }
                if ui.button("Cancel").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    decision = Some(None);
                }
            });
        });
        match decision {
            Some(Some(true)) => {
                self.pending = None;
                if self.save(false) {
                    self.perform(action);
                }
            }
            Some(Some(false)) => {
                self.pending = None;
                self.perform(action);
            }
            Some(None) => self.pending = None,
            None => {}
        }
    }

    fn unsaved_model_dialog(&mut self, ctx: &egui::Context) {
        if self.pending.is_some() {
            return;
        }
        let Some(action) = self.magic_pending.clone() else { return };
        let Some(title) = self.magic_tab.doc.as_ref().map(MagicDoc::title) else {
            self.magic_pending = None;
            return self.magic_perform(action);
        };
        let mut decision = None;
        egui::Modal::new(egui::Id::new("unsaved-model")).show(ctx, |ui| {
            ui.set_width(300.0);
            ui.label(RichText::new("Save the model?").strong().size(15.0));
            ui.label(format!("“{title}” in the Magic tab has changes that have not been saved."));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if widgets::accent_button(ui, "Save", theme::accent(), vec2(70.0, 26.0)).clicked() {
                    decision = Some(Some(true));
                }
                if ui.button("Don't save").clicked() {
                    decision = Some(Some(false));
                }
                if ui.button("Cancel").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    decision = Some(None);
                }
            });
        });
        match decision {
            Some(Some(true)) => {
                self.magic_pending = None;
                if self.save_magic(false) {
                    self.magic_perform(action);
                }
            }
            Some(Some(false)) => {
                self.magic_pending = None;
                self.magic_perform(action);
            }
            Some(None) => self.magic_pending = None,
            None => {}
        }
    }

    fn export_dialog(&mut self, ctx: &egui::Context) {
        let Some(mut choices) = self.export.take() else { return };
        let mut decision = None;
        let library = self.presets.entries().len();
        egui::Modal::new(egui::Id::new("export")).show(ctx, |ui| {
            ui.set_width(320.0);
            ui.label(RichText::new(format!("{}  Export", icon::EXPORT)).strong().size(15.0));
            widgets::note(ui, "Saves the macro as a .magi file. Choose what else goes into it:");
            ui.add_space(4.0);
            widgets::toggle_row(
                ui,
                &format!("My presets ({library})"),
                "Your preset library, so it travels with the file.",
                &mut choices.presets,
            );
            widgets::toggle_row(
                ui,
                "Options",
                "Stop hotkeys, program rules and the other Options.",
                &mut choices.options,
            );
            widgets::toggle_row(ui, "Auto Clicker settings", "", &mut choices.clicker);
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if widgets::accent_button(ui, "Export…", theme::accent(), vec2(90.0, 26.0)).clicked() {
                    decision = Some(true);
                }
                if ui.button("Cancel").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    decision = Some(false);
                }
            });
        });
        match decision {
            Some(true) => self.export(&choices),
            Some(false) => {}
            None => self.export = Some(choices),
        }
    }

    fn preset_dialog(&mut self, ctx: &egui::Context) {
        let Some((kind, mut name)) = self.preset_prompt.take() else { return };
        let mut decision = None;
        egui::Modal::new(egui::Id::new("preset-name")).show(ctx, |ui| {
            ui.set_width(300.0);
            let title = match kind {
                PresetKind::Clicker => "Save Auto Clicker settings as a preset",
                PresetKind::Blocks | PresetKind::MagicBlocks => "Save the selected blocks as a preset",
            };
            ui.label(RichText::new(title).strong());
            let field =
                ui.add(egui::TextEdit::singleline(&mut name).hint_text("Preset name").desired_width(f32::INFINITY));
            field.request_focus();
            let entered = ui.input(|i| i.key_pressed(egui::Key::Enter));
            ui.horizontal(|ui| {
                let ready = !name.trim().is_empty();
                if (ui.add_enabled(ready, egui::Button::new("Save")).clicked() || entered) && ready {
                    decision = Some(true);
                }
                if ui.button("Cancel").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    decision = Some(false);
                }
            });
        });
        match decision {
            Some(true) => {
                let body = match kind {
                    PresetKind::Clicker => PresetBody::AutoClicker(self.settings.clicker.clone()),
                    PresetKind::Blocks => PresetBody::Nodes(NodePreset::capture(
                        &self.doc.project.graph,
                        &self.macro_tab.view.selection,
                        &self.doc.project.assets,
                        &self.registry,
                    )),
                    PresetKind::MagicBlocks => {
                        let Some(doc) = &self.magic_tab.doc else { return };
                        PresetBody::Nodes(NodePreset::capture(
                            &doc.studio.graph,
                            &self.magic_tab.view.selection,
                            &doc.studio.assets,
                            crate::ui::magic_tab::registry(),
                        ))
                    }
                };
                let preset = Preset { name: name.trim().to_owned(), description: String::new(), body };
                match self.presets.save(&preset) {
                    Ok(_) => self.toast(format!("Saved preset “{}”", preset.name)),
                    Err(error) => self.toast_error(format!("Could not save the preset: {error}")),
                }
            }
            Some(false) => {}
            None => self.preset_prompt = Some((kind, name)),
        }
    }

    /// Asks before starting something that has no hotkey to stop it.
    fn hotkey_warning_dialog(&mut self, ctx: &egui::Context) {
        let Some(warning) = self.hotkey_warning.take() else { return };
        let mut decision = None;
        let what = match warning.then {
            Unguarded::Run => "Run",
            Unguarded::Record(_) => "Record",
        };
        let stop_hint = self.stop_hint();
        egui::Modal::new(egui::Id::new("hotkey-warning")).show(ctx, |ui| {
            ui.set_width(340.0);
            ui.label(RichText::new(format!("{}  {what} has no hotkey", icon::KEYBOARD)).strong().size(15.0));
            let text = match warning.then {
                Unguarded::Run => format!("Without one, the macro can only be stopped {stop_hint}."),
                Unguarded::Record(_) => {
                    "Without one, recording has to be stopped with the Stop button, and that click is recorded too."
                        .to_owned()
                }
            };
            widgets::note(ui, text);
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if widgets::accent_button(ui, "Set a hotkey", theme::accent(), vec2(100.0, 26.0)).clicked() {
                    decision = Some(0);
                }
                if ui.button("Start anyway").clicked() {
                    decision = Some(1);
                }
                if ui.button("Cancel").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    decision = Some(2);
                }
            });
        });
        match decision {
            Some(0) => {
                self.settings.settings_page = SettingsPage::Hotkeys;
                self.settings_window.open = true;
                self.start_capture(warning.target);
            }
            Some(1) => match warning.then {
                Unguarded::Run => self.run_macro(false),
                Unguarded::Record(into) => self.toggle_recording(into, false),
            },
            Some(_) => {}
            None => self.hotkey_warning = Some(warning),
        }
    }

    /// Offers the Options and Auto Clicker settings that came with an opened file.
    fn file_extras_dialog(&mut self, ctx: &egui::Context) {
        let Some(extras) = self.file_extras.take() else { return };
        let mut decision = None;
        egui::Modal::new(egui::Id::new("file-extras")).show(ctx, |ui| {
            ui.set_width(330.0);
            ui.label(RichText::new("Use the settings in this file?").strong().size(15.0));
            let mut parts = Vec::new();
            if extras.options.is_some() {
                parts.push("Options (stop hotkeys, program rules…)");
            }
            if extras.clicker.is_some() {
                parts.push("Auto Clicker settings");
            }
            widgets::note(ui, format!("“{}” includes: {}.", self.doc.title(), parts.join(", ")));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if widgets::accent_button(ui, "Use them", theme::accent(), vec2(90.0, 26.0)).clicked() {
                    decision = Some(true);
                }
                if ui.button("Keep mine").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    decision = Some(false);
                }
            });
        });
        match decision {
            Some(true) => {
                if let Some(options) = extras.options {
                    self.settings.options = options;
                }
                if let Some(clicker) = extras.clicker {
                    self.settings.clicker = clicker;
                }
            }
            Some(false) => {}
            None => self.file_extras = Some(extras),
        }
    }

    fn toasts(&mut self, ctx: &egui::Context) {
        self.toasts.retain(|toast| toast.born.elapsed().as_secs_f32() < TOAST_SECONDS);
        if self.toasts.is_empty() {
            return;
        }
        ctx.request_repaint_after(Duration::from_millis(200));
        egui::Area::new(egui::Id::new("toasts"))
            .order(egui::Order::Tooltip)
            .anchor(Align2::CENTER_BOTTOM, vec2(0.0, -18.0))
            .interactable(false)
            .show(ctx, |ui| {
                for toast in self.toasts.iter().rev().take(3).rev() {
                    let age = toast.born.elapsed().as_secs_f32();
                    let fade = ((TOAST_SECONDS - age) / 0.4).clamp(0.0, 1.0);
                    let border = if toast.is_error { theme::danger() } else { theme::stroke_strong() };
                    egui::Frame::new()
                        .fill(theme::with_alpha(theme::bg_raised().to_opaque(), fade))
                        .stroke(egui::Stroke::new(1.0, theme::with_alpha(border, fade)))
                        .corner_radius(8)
                        .inner_margin(egui::Margin::symmetric(12, 7))
                        .show(ui, |ui| {
                            ui.set_max_width(360.0);
                            ui.label(RichText::new(&toast.text).color(theme::with_alpha(theme::text(), fade)));
                        });
                    ui.add_space(4.0);
                }
            });
    }

    /// Developer screenshot mode: draw a few frames, save the picture, quit.
    fn screenshot(&mut self, ctx: &egui::Context) {
        let Some(job) = &mut self.screenshot else { return };
        let image = ctx.input(|i| {
            i.events.iter().find_map(|event| match event {
                egui::Event::Screenshot { image, .. } => Some(Arc::clone(image)),
                _ => None,
            })
        });
        if let Some(image) = image {
            let [width, height] = image.size;
            let rgba: Vec<u8> = image.pixels.iter().flat_map(|pixel| pixel.to_srgba_unmultiplied()).collect();
            let saved = magi_vision::encode_png(magi_vision::ImageRef::new(width as u32, height as u32, &rgba))
                .map_err(|e| e.to_string())
                .and_then(|png| std::fs::write(&job.path, png).map_err(|e| e.to_string()));
            if let Err(error) = saved {
                eprintln!("screenshot failed: {error}");
            }
            self.allow_close = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        if job.frames_left > 0 {
            job.frames_left -= 1;
        } else if job.not_before.is_some_and(|t| Instant::now() < t) {
            // Still waiting (`--wait`).
        } else if !job.requested {
            job.requested = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        }
        ctx.request_repaint();
    }
}

impl eframe::App for MagiApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.tick(ctx);
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.apply_look(&ctx);
        self.approvals(&ctx);
        self.shortcuts(&ctx);

        // Closing with unsaved changes asks first.
        if ctx.input(|i| i.viewport().close_requested()) && !self.allow_close {
            if self.doc.is_dirty() {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.pending = Some(Pending::Quit);
            } else if self.magic_tab.doc.as_ref().is_some_and(MagicDoc::would_lose_changes) {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.magic_pending = Some(MagicPending::Quit);
            }
        }
        if self.allow_close && self.pending.is_none() && self.magic_pending.is_none() && self.screenshot.is_none() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        let (size, minimized) =
            ctx.input(|i| (i.viewport().inner_rect.map(|rect| rect.size()), i.viewport().minimized));
        if let Some(size) = size
            && minimized != Some(true)
            && size.x >= crate::MIN_SIZE[0]
            && size.y >= crate::MIN_SIZE[1]
        {
            self.settings.window_size = Some([size.x.round(), size.y.round()]);
        }

        // The background picture, under everything.
        let look = &self.settings.appearance;
        let painter = ctx.layer_painter(egui::LayerId::background());
        self.background.paint(
            &painter,
            ctx.content_rect(),
            look.background_opacity,
            look.background_darkness,
            theme::palette().dark,
        );

        // The screen picker or the targets overlay, if one is open.
        if let Some(snip) = &mut self.snip
            && let Some(result) = snip.show(&ctx)
        {
            let snip = self.snip.take().expect("checked above");
            if let Some(outcome) = result {
                self.apply_snip(snip.target, outcome);
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
        }
        if let Some(overlay) = &mut self.overlay {
            match overlay.show(&ctx, &mut self.doc.project, &self.registry) {
                OverlayOutcome::Open => {}
                OverlayOutcome::Closed => {
                    self.overlay = None;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                }
                OverlayOutcome::ShowBlock(node) => {
                    self.overlay = None;
                    self.settings.tab = Tab::Macro;
                    self.macro_tab.view.reveal(node);
                    self.macro_tab.docked_inspector = true;
                    self.macro_tab.floating_inspector = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                }
            }
        }

        if self.train_at_start && self.magic_tab.doc.is_some() {
            self.train_at_start = false;
            self.start_training(&ctx, Default::default());
        }
        egui::Panel::top("top-bar")
            .frame(egui::Frame::new().fill(theme::bg_app()).inner_margin(egui::Margin::symmetric(10, 7)))
            .show(ui, |ui| self.banner(ui));
        self.status_bar(ui);
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::bg_panel())).show(ui, |ui| {
            match self.settings.tab {
                Tab::AutoClicker => self.clicker(ui),
                Tab::Macro => self.macro_editor(ui),
                Tab::Magic => self.magic(ui),
            }
        });

        self.windows(&ctx);
        self.screenshot(&ctx);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.controller.shut_down();
        if self.screenshot.is_none() {
            self.save_settings();
        }
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        theme::bg_app().to_opaque().to_normalized_gamma_f32()
    }
}

/// The steps of a Rate block on a line, each at the time it happens.
fn rate_preview(ui: &mut Ui, params: &magi_core::nodes::flow::RateParams) {
    let count = params.steps.len();
    if count == 0 {
        ui.weak("No steps yet.");
        return;
    }
    let total = params.rate_ms * (count.saturating_sub(1)) as f64;
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 40.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 5.0, theme::bg_input());
    let inner = rect.shrink2(vec2(14.0, 8.0));
    painter.hline(inner.x_range(), inner.center().y, egui::Stroke::new(1.0, theme::stroke_strong()));
    for (index, step) in params.steps.iter().enumerate() {
        let t = if count > 1 { index as f32 / (count - 1) as f32 } else { 0.5 };
        let x = inner.left() + t * inner.width();
        painter.circle_filled(egui::pos2(x, inner.center().y), 4.0, theme::accent());
        if count <= 12 {
            painter.text(
                egui::pos2(x, inner.top()),
                Align2::CENTER_TOP,
                format!("{}", index + 1),
                egui::FontId::proportional(10.0),
                theme::text_dim(),
            );
        }
        let _ = step;
    }
    ui.label(
        RichText::new(format!(
            "{count} steps over {} · one every {}",
            magi_core::text::format_ms(total),
            magi_core::text::format_ms(params.rate_ms)
        ))
        .size(11.0)
        .color(theme::text_faint()),
    );
}

#[cfg(test)]
#[path = "app_magic_tests.rs"]
mod magic_tests;

#[cfg(test)]
mod tests {
    //! The whole app, driven headlessly. Input the user presses is fed to the
    //! controller exactly as the system listener would.

    use egui_kittest::Harness;
    use egui_kittest::kittest::Queryable;
    use magi_core::input::{InputEvent, Key};

    use super::*;

    fn app() -> Harness<'static, MagiApp> {
        app_sized(480.0, 1500.0)
    }

    fn app_sized(width: f32, height: f32) -> Harness<'static, MagiApp> {
        let dir = std::env::temp_dir().join(format!(
            "magi-app-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let paths = Paths { app_dir: dir.clone(), data_dir: dir.join("data") };
        let settings = Settings { dry_run: true, ..Settings::default() };
        let options = StartupOptions {
            open: None,
            example: None,
            tab: Some(Tab::AutoClicker),
            screenshot: None,
            show_cards: false,
            show_targets: false,
            select_kind: None,
            settings_page: None,
            train: false,
            wait: None,
            dataset: None,
            agent: false,
        };
        let mut harness = Harness::builder()
            .with_size(vec2(width, height))
            .build_eframe(move |cc| MagiApp::new(cc, paths, settings, options));
        harness.run_steps(4);
        harness
    }

    fn press(harness: &mut Harness<'_, MagiApp>, keys: &[Key]) {
        let controller = harness.state().controller.clone();
        for key in keys {
            controller.inject(InputEvent::KeyDown { key: *key });
        }
        for key in keys.iter().rev() {
            controller.inject(InputEvent::KeyUp { key: *key });
        }
        harness.run_steps(3);
    }

    #[test]
    fn changed_clicker_values_have_a_reset_button() {
        let mut harness = app();
        let undo_and_resets =
            |harness: &Harness<'_, MagiApp>| harness.query_all_by_label(icon::ARROW_COUNTER_CLOCKWISE).count();
        let before = undo_and_resets(&harness);
        harness.state_mut().settings.clicker.hold_ms = 55.0;
        harness.run_steps(3);
        assert_eq!(undo_and_resets(&harness), before + 1, "a reset button appears next to Hold for");
        harness.query_all_by_label(icon::ARROW_COUNTER_CLOCKWISE).last().expect("the reset button").click();
        harness.run_steps(3);
        assert_eq!(
            harness.state().settings.clicker.hold_ms,
            magi_core::autoclicker::ClickerSettings::default().hold_ms
        );
        assert_eq!(undo_and_resets(&harness), before, "and goes away again");
    }

    #[test]
    fn a_block_setting_can_follow_a_new_variable() {
        let mut harness = app_sized(1200.0, 900.0);
        let wait = {
            let state = harness.state_mut();
            state.settings.tab = Tab::Macro;
            let id = state.doc.project.graph.add(&state.registry, ids::DELAY, [0.0, 200.0]).unwrap();
            state.macro_tab.view.focus_node(id);
            id
        };
        harness.run_steps(4);
        // The first link button belongs to the Wait block's Duration.
        harness.query_all_by_label(icon::LINK).next().expect("a link button").click();
        harness.run_steps(3);
        harness.get_by_label(&format!("{}  New variable from this setting", icon::PLUS)).click();
        harness.run_steps(3);

        let project = &harness.state().doc.project;
        assert_eq!(project.variables.len(), 1);
        let name = project.variables[0].name.clone();
        assert_eq!(project.variables[0].value, serde_json::json!(500.0), "it starts with the setting's value");
        let bound = project.graph.node(wait).unwrap().params.get("ms").cloned();
        assert_eq!(bound, Some(magi_core::variables::binding(&name)));
    }

    #[test]
    fn hotkeys_set_in_the_ui_take_effect_at_once() {
        let mut harness = app();
        // The Auto Clicker's hotkey field, showing F6.
        harness.get_by_label(&format!("{}  F6", icon::KEYBOARD)).click();
        harness.run_steps(3);
        assert!(harness.state().capture.is_some(), "the field is listening");

        // Ctrl + Alt + Y, as the user presses it.
        press(&mut harness, &[Key::ControlLeft, Key::AltLeft, Key::KeyY]);
        let expected: Hotkey = "ControlLeft+AltLeft+KeyY".parse().unwrap();
        assert_eq!(harness.state().settings.clicker.hotkey, expected);
        assert!(harness.state().capture.is_none());

        // No restart needed: the new hotkey starts the clicker straight away.
        press(&mut harness, &[Key::ControlLeft, Key::AltLeft, Key::KeyY]);
        std::thread::sleep(Duration::from_millis(100));
        assert!(harness.state().controller.clicker_running(), "the new hotkey works");
        harness.state().controller.stop_everything();
    }

    #[test]
    fn escape_and_function_keys_can_be_hotkeys() {
        let mut harness = app();
        for key in [Key::Escape, Key::F6, Key::F12] {
            let label = format!("{}  {}", icon::KEYBOARD, harness.state().settings.clicker.hotkey.label());
            harness.get_by_label(&label).click();
            harness.run_steps(3);
            press(&mut harness, &[key]);
            assert_eq!(harness.state().settings.clicker.hotkey, Hotkey::key(key));
        }
    }

    #[test]
    fn the_add_button_opens_the_block_list_and_it_stays_open() {
        let mut harness = app_sized(1000.0, 700.0);
        harness.state_mut().settings.tab = Tab::Macro;
        harness.run_steps(3);
        harness.get_by_label(&format!("{}  Add", icon::PLUS)).click();
        harness.run_steps(4);
        assert!(harness.state().macro_tab.view.palette_is_open(), "the click that opened it does not close it");
    }

    #[test]
    fn the_pop_out_recording_editor_edits_the_recording() {
        let mut harness = app_sized(1000.0, 800.0);
        let project = examples::build(2, &harness.state().registry);
        harness.state_mut().replace_document(Document::new(project, None));
        let asset = harness
            .state()
            .doc
            .project
            .assets
            .of_kind(magi_core::assets::AssetKind::Recording)
            .next()
            .expect("the example has a recording")
            .id
            .clone();
        let before = harness.state().doc.project.assets.recording(&asset).unwrap().duration_ms;
        harness.state_mut().handle_request(&egui::Context::default(), UiRequest::EditRecording(asset.clone()));
        harness.run_steps(4);
        harness.get_by_label("×2").click();
        harness.run_steps(3);
        let after = harness.state().doc.project.assets.recording(&asset).unwrap().duration_ms;
        assert!((after - before / 2.0).abs() < 1e-6, "{before} -> {after}");
    }

    #[test]
    fn panels_and_windows_keep_their_size() {
        let mut harness = app_sized(1280.0, 860.0);
        let project = examples::build(0, &harness.state().registry);
        harness.state_mut().replace_document(Document::new(project, None));
        let find = harness.state().doc.project.graph.nodes().find(|n| n.kind == ids::FIND_IMAGE).unwrap().id;
        harness.state_mut().macro_tab.view.focus_node(find);
        harness.state_mut().settings.show_console = true;
        harness.state_mut().settings.show_variables = true;
        harness.state_mut().doc.project.variables.push(magi_core::variables::Variable::new(
            "a rather long variable name that could push the panel wider",
            serde_json::json!("and a long starting value as well, to be sure"),
        ));
        let rate = {
            let state = harness.state_mut();
            state.doc.project.graph.add(&state.registry, ids::RATE, [900.0, 900.0]).unwrap()
        };
        harness.state_mut().rate_window = Some(rate);
        harness.state_mut().rules_window.open = true;
        harness.state_mut().stop_hotkeys_open = true;
        harness.state_mut().settings_window.open = true;
        let measure = |harness: &Harness<'_, MagiApp>| {
            let ctx = &harness.ctx;
            let panel = |id: &str| egui::containers::panel::PanelState::load(ctx, egui::Id::new(id)).map(|p| p.size());
            let area = |id: &str| ctx.memory(|m| m.area_rect(egui::Id::new(id))).map(|r| r.size());
            (
                panel("macro-inspector"),
                panel("macro-console"),
                area("process-rules-window"),
                area("stop-hotkeys-window"),
                area("settings-window"),
                panel("macro-variables"),
                area("rate-editor"),
            )
        };
        harness.run_steps(5);
        let first = measure(&harness);
        harness.run_steps(60);
        assert_eq!(measure(&harness), first, "nothing creeps wider or taller on its own");

        // Whatever block is selected, the settings panel keeps its width.
        let kinds: Vec<String> = harness.state().registry.iter().map(|def| def.id.clone()).collect();
        for kind in kinds {
            let state = harness.state_mut();
            let at = state.macro_tab.view.view_center();
            let node = state.macro_tab.view.add_node(&mut state.doc.project, &state.registry, &kind, at).unwrap();
            state.macro_tab.view.focus_node(node);
            harness.run_steps(6);
            assert_eq!(measure(&harness).0, first.0, "the panel changed width for {kind}");
        }
        for page in SettingsPage::ALL {
            harness.state_mut().settings.settings_page = page;
            harness.run_steps(4);
            assert_eq!(measure(&harness).4, first.4, "the settings window changed size on {page:?}");
        }
    }

    #[test]
    fn hotkeys_can_be_reset_from_the_settings_window() {
        let mut harness = app_sized(900.0, 900.0);
        harness.state_mut().settings.run_hotkey = Hotkey::none();
        harness.state_mut().settings.settings_page = SettingsPage::Hotkeys;
        harness.state_mut().settings_window.open = true;
        harness.run_steps(4);
        harness.get_by_label_contains("Reset all hotkeys").click();
        harness.run_steps(3);
        assert_eq!(harness.state().settings.run_hotkey, Hotkey::key(Key::F7));
    }

    #[test]
    fn the_presets_menu_stays_narrow_with_presets_in_it() {
        let mut harness = app_sized(1280.0, 900.0);
        harness.state_mut().settings.tab = Tab::Macro;
        let preset = magi_core::preset::Preset {
            name: "A preset with a rather long name that would make the menu wide".into(),
            description: String::new(),
            body: magi_core::preset::PresetBody::Nodes(magi_core::preset::NodePreset {
                fragment: magi_core::graph::Fragment::default(),
                assets: Vec::new(),
            }),
        };
        let path = harness.state_mut().presets.save(&preset).unwrap();
        harness.run_steps(2);
        harness.get_by_label_contains("Presets").click();
        harness.run_steps(3);
        let row = harness.get_by_label_contains("rather long name").rect();
        let import = harness.get_by_label_contains("Import a .mp file").rect();
        assert!(row.width() <= 260.0 && import.width() <= 260.0, "{row:?} {import:?}");
        harness.state_mut().presets.delete(&path);
    }

    #[test]
    fn every_block_of_the_magic_canvas_opens_its_settings() {
        let mut harness = app_sized(1280.0, 900.0);
        harness.state_mut().settings.tab = Tab::Magic;
        for text in [true, false] {
            let doc = MagicDoc::new(crate::magic::new_model(text), None, Vec::new());
            harness.state_mut().set_magic_doc(Some(doc));
            harness.run_steps(4);
            // Every kind of block, added to the canvas and selected.
            let kinds: Vec<String> = crate::ui::magic_tab::registry().iter().map(|def| def.id.clone()).collect();
            for kind in kinds {
                let state = harness.state_mut();
                let tab = &mut state.magic_tab;
                let doc = tab.doc.as_mut().unwrap();
                let at = tab.view.view_center();
                let node = tab.view.add_node(&mut doc.studio, crate::ui::magic_tab::registry(), &kind, at).unwrap();
                tab.view.selection = [node].into();
                harness.run_steps(3);
            }
        }
        // A model from before the canvas opens with its setup as blocks.
        let old = magic_doc().file;
        harness.state_mut().set_magic_doc(Some(MagicDoc::new(old, Some("old.magic".into()), Vec::new())));
        harness.run_steps(3);
        let doc = harness.state().magic_tab.doc.as_ref().unwrap();
        assert!(doc.studio.graph.nodes().any(|n| n.kind == magi_core::nodes::studio::kinds::ANSWERS));
        assert!(!doc.is_dirty(), "opening it changes nothing");
        // A change to a dataset's text is a change to the model.
        let doc = harness.state_mut().magic_tab.doc.as_mut().unwrap();
        doc.file.data.insert("data/text/a.jsonl".into(), b"{}".to_vec());
        doc.files_changed();
        assert!(doc.is_dirty() && doc.would_lose_changes());
    }

    /// A tiny model with one adjustable setting, as the Magic tab would hold it.
    fn magic_doc() -> MagicDoc {
        use magi_core::magic::{ClassSpec, MagicFile, MagicModel, ModelSetting, SettingKind, SettingTarget};
        let onnx = include_bytes!("../../magi-engine/tests/data/channel_means.onnx").to_vec();
        let model = MagicModel {
            name: "Watcher".into(),
            classes: ["red", "green", "blue"]
                .iter()
                .map(|label| ClassSpec { label: (*label).into(), act: *label != "red", threshold: 0.5, delay_ms: 0.0 })
                .collect(),
            settings: vec![ModelSetting {
                key: "thresh".into(),
                label: "Confidence".into(),
                kind: SettingKind::Percent,
                min: 1.0,
                max: 100.0,
                default: 60.0,
                value: 60.0,
                targets: vec![SettingTarget::Threshold { class: 1 }, SettingTarget::Threshold { class: 2 }],
                ..ModelSetting::default()
            }],
            ..MagicModel::default()
        };
        MagicDoc::new(MagicFile { model, onnx, ..MagicFile::default() }, None, Vec::new())
    }

    #[test]
    fn a_model_from_the_magic_tab_goes_into_the_macro_with_blocks_that_run_it() {
        let mut harness = app_sized(1280.0, 900.0);
        harness.state_mut().settings.tab = Tab::Magic;
        harness.state_mut().magic_tab.doc = Some(magic_doc());
        harness.run_steps(3);
        harness.state_mut().use_model_in_macro();
        harness.run_steps(3);

        let state = harness.state();
        assert_eq!(state.settings.tab, Tab::Macro);
        let models: Vec<String> =
            state.doc.project.assets.of_kind(magi_core::assets::AssetKind::Model).map(|a| a.id.clone()).collect();
        assert_eq!(models.len(), 1);
        let watch = state.doc.project.graph.nodes().find(|n| n.kind == ids::AI_WATCH).expect("an AI Watch block");
        assert_eq!(watch.param::<String>("model").as_ref(), Some(&models[0]));
        let watch = watch.id;
        let blocks = state.doc.project.graph.nodes().count();

        // Using it again adds nothing new.
        harness.state_mut().use_model_in_macro();
        harness.run_steps(2);
        assert_eq!(harness.state().doc.project.graph.nodes().count(), blocks);

        // The block shows the model's setting, and changing it is kept per block.
        harness.state_mut().macro_tab.view.focus_node(watch);
        harness.run_steps(3);
        harness.get_by_label("Confidence");
        let node = harness.state_mut().doc.project.graph.node_mut(watch).unwrap();
        node.set_param("settings", serde_json::json!({ "thresh": 80.0 }));
        harness.run_steps(2);
        let reset = harness.query_all_by_label(icon::ARROW_COUNTER_CLOCKWISE).count();
        assert!(reset >= 1, "a changed setting has a reset button");

        // And it opens back in the Magic tab.
        harness.get_by_label_contains("Open in Magic").click();
        harness.run_steps(3);
        assert_eq!(harness.state().settings.tab, Tab::Magic);
        assert_eq!(harness.state().magic_tab.doc.as_ref().map(MagicDoc::title).as_deref(), Some("Watcher"));
    }

    #[test]
    fn replacing_an_unsaved_model_asks_first() {
        let mut harness = app_sized(1280.0, 900.0);
        harness.state_mut().settings.tab = Tab::Magic;
        harness.state_mut().magic_tab.doc = Some(magic_doc());
        harness.run_steps(2);
        harness.state_mut().magic_guarded(MagicPending::New);
        harness.run_steps(3);
        assert!(harness.state().magic_tab.doc.is_some(), "nothing is lost before the answer");
        harness.get_by_label("Save the model?");
        harness.get_by_label("Cancel").click();
        harness.run_steps(3);
        assert!(harness.state().magic_tab.doc.is_some());

        harness.state_mut().magic_guarded(MagicPending::New);
        harness.run_steps(3);
        harness.get_by_label("Don't save").click();
        harness.run_steps(3);
        assert!(harness.state().magic_tab.doc.is_none());

        // Once in the macro, nothing would be lost.
        harness.state_mut().magic_tab.doc = Some(magic_doc());
        harness.state_mut().use_model_in_macro();
        harness.state_mut().settings.tab = Tab::Magic;
        harness.run_steps(2);
        harness.state_mut().magic_guarded(MagicPending::New);
        harness.run_steps(2);
        assert!(harness.state().magic_tab.doc.is_none() && harness.state().magic_pending.is_none());
    }
}
