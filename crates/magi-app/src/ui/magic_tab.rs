//! The Magic tab: a canvas for making, training, setting up and trying AI
//! models – text models that answer and call tools, and vision models that
//! watch the screen – and handing them to the Macro tab.
//!
//! The canvas works like the Macro tab's. A model travels along the Model
//! wire: where it starts, its architecture, its training stages fed by
//! datasets, the package, and the block that tries the result. Its own
//! settings – name, tools, answers – are in setup blocks chained into the
//! start. Selecting a block opens its settings in the panel on the right,
//! with what is wired into it. The File, Presets, Add, View and undo buttons
//! are in the banner, as on the Macro tab.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use egui::{DragValue, RichText, Ui, vec2};
use egui_phosphor::regular as icon;
use magi_core::geom::{Displays, Rect};
use magi_core::graph::NodeId;
use magi_core::magic::{
    CaptureArea, ClassSpec, ColorOrder, IMAGENET_MEAN, IMAGENET_STD, Layout, MagicModel, ModelKind, ModelSetting,
    SettingKind, SettingTarget, TextSpec, ToolSpec,
};
use magi_core::nodes::Registry;
use magi_core::nodes::ai::AGENT_TOOLS;
use magi_core::nodes::studio::{self, PromptParams, RewardRule, RlParams, SetupParams, TextDataParams, kinds};

use super::dataset_editor::{self, DatasetEditor, DatasetEvent};
use super::graph::{GraphEnv, GraphEvent, GraphView, node_icon};
use super::inspector::{self, InspectorEnv};
use super::widgets::{self, card};
use crate::magic::train::{Level, TrainJob, stage_name};
use crate::magic::{MagicDoc, Tester};
use crate::theme;

/// Where a model to import comes from.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ImportFrom {
    /// A project folder with an ONNX model in it.
    Folder,
    /// A `.zip` or `.onnx` file.
    File,
}

pub enum MagicEvent {
    NewModel {
        text: bool,
    },
    Open,
    Import(ImportFrom),
    ReplaceModel,
    /// Pick the fixed screen area the model looks at.
    PickArea,
    /// Add pictures of an answer – or, `negative`, of what it is not.
    AddPictures {
        class: usize,
        negative: bool,
    },
    /// Add text files to a Text Dataset block.
    AddDatasetFiles {
        node: NodeId,
        negative: bool,
    },
    /// Run the model on a picture file.
    TryPicture,
    StartTest,
    StopTest,
    /// Train: from where, and which stages.
    Train(crate::magic::TrainChoice),
    /// Save the weights kept after a stage as a base model (`.szi`) to fine-tune other models from.
    SaveBase(NodeId),
    StopTraining,
    /// Save the package as the Compile block says.
    Compile,
    UseInMacro,
    OpenFolder(PathBuf),
    OpenRuntime,
    Toast(String),
    /// The dataset editor's generator endpoint changed: keep it in the settings.
    GeneratorSettings(crate::magic::generate::GeneratorSettings),
}

pub struct MagicEnv<'a> {
    pub displays: &'a Displays,
    /// Where models run, as a sentence.
    pub engines: String,
    pub dry_run: bool,
    /// Options › Optimize environment when training: only the training view while training.
    pub optimize_training: bool,
    pub show_console: bool,
    pub show_panel: bool,
    pub show_grid: bool,
    /// Where the dataset editor generates samples.
    pub generator: crate::magic::generate::GeneratorSettings,
}

/// The Magic tab's blocks.
pub fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(studio::registry)
}

type Pending = Arc<Mutex<Option<Result<magi_ml::tools::AgentAnswer, String>>>>;
type Job = Arc<Mutex<Option<Result<(), String>>>>;

/// Prompting a text model.
#[derive(Default)]
struct Chat {
    request: String,
    pending: Option<(String, Pending)>,
    /// Requests and what the model said (and the native tools it ran), newest last.
    answers: Vec<(String, Result<magi_ml::tools::AgentAnswer, String>)>,
    /// The ONNX model, loaded once and kept, with the fingerprint of its bytes.
    engine: Arc<Mutex<Option<(u64, magi_ml::TextModel)>>>,
    /// Loading it in the background.
    loading: Option<Job>,
    load_error: Option<String>,
}

/// Loads a model for the Prompt Model block: on the native engine (from its
/// weights) when it can, else its ONNX model.
pub(crate) fn load_engine(
    native: bool,
    architecture: &magi_core::magic::Architecture,
    memory: &magi_core::magic::MemorySpec,
    spec: &TextSpec,
    bytes: &[u8],
) -> Result<magi_ml::TextModel, magi_ml::MlError> {
    if native {
        magi_ml::TextModel::load_native(architecture, bytes, memory, &magi_ml::tools::cache_dir())
    } else {
        magi_ml::TextModel::load(bytes, spec.context as usize)
    }
}

/// A cheap fingerprint of a model's bytes: its length and a sample of them.
pub(crate) fn fingerprint(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.len().hash(&mut hasher);
    bytes.iter().step_by(997).for_each(|b| b.hash(&mut hasher));
    bytes.iter().rev().take(4096).for_each(|b| b.hash(&mut hasher));
    hasher.finish()
}

#[derive(Default)]
pub struct MagicTab {
    pub doc: Option<MagicDoc>,
    pub tester: Option<Tester>,
    /// Whether the live test presses the model's key.
    pub press: bool,
    preview: Option<(u64, egui::TextureHandle)>,
    /// The result of "Try a picture": its name and the class probabilities.
    pub picture: Option<(String, Vec<f32>)>,
    pub view: GraphView,
    pub job: Option<TrainJob>,
    /// What the last compile wrote.
    pub compiled: Vec<PathBuf>,
    chat: Chat,
    /// Something that needs the optional parts failed: they are looked at again.
    pub runtime_failed: bool,
    dataset: Option<DatasetEditor>,
    thumbnails: HashMap<String, egui::TextureHandle>,
}

/// Normalisations people know by name.
const NORMALISATIONS: [(&str, [f32; 3], [f32; 3]); 3] = [
    ("ImageNet (most models)", IMAGENET_MEAN, IMAGENET_STD),
    ("0 to 1", [0.0; 3], [1.0; 3]),
    ("−1 to 1", [0.5; 3], [0.5; 3]),
];

impl MagicTab {
    /// Forgets everything about the model that was open.
    pub fn reset(&mut self) {
        self.tester = None;
        self.picture = None;
        self.preview = None;
        self.chat = Chat::default();
        self.dataset = None;
        self.compiled.clear();
        self.thumbnails.clear();
        let palette_size = self.view.palette_size;
        self.view = GraphView::default();
        self.view.palette_size = palette_size;
        if self.training() {
            self.job = None;
        }
    }

    /// Opens a dataset file of the open model in the dataset editor.
    pub fn open_dataset(&mut self, file: &str, view: Option<&str>) {
        let Some(doc) = &self.doc else { return };
        let bytes = doc.file.data.get(file).cloned().unwrap_or_default();
        let mut editor = DatasetEditor::new(file, &bytes);
        if let Some(view) = view {
            editor.set_view(view);
        }
        self.dataset = Some(editor);
    }

    #[cfg(test)]
    pub fn dataset_open(&self) -> bool {
        self.dataset.is_some()
    }

    #[cfg(test)]
    /// A request to the model is still being answered.
    pub fn chat_waiting(&self) -> bool {
        self.chat.pending.is_some()
    }

    #[cfg(test)]
    /// The request being written, and what came of the requests sent.
    pub fn chat_state(&self) -> (String, Vec<Result<String, String>>) {
        let answers = self
            .chat
            .answers
            .iter()
            .map(|(_, a)| a.as_ref().map(|a| a.answer.raw.clone()).map_err(Clone::clone))
            .collect();
        (self.chat.request.clone(), answers)
    }

    #[cfg(test)]
    /// What the model said last.
    pub fn last_answer(&self) -> Option<&magi_ml::Answer> {
        self.chat.answers.last().and_then(|(_, a)| a.as_ref().ok()).map(|a| &a.answer)
    }

    pub fn training(&self) -> bool {
        self.job.as_ref().is_some_and(TrainJob::is_running)
    }

    /// Whether the training view replaces the canvas now.
    pub fn training_view(&self, env_optimize: bool) -> bool {
        env_optimize && self.training()
    }

    pub fn show(&mut self, ui: &mut Ui, env: &MagicEnv<'_>) -> Vec<MagicEvent> {
        let mut events = Vec::new();
        if self.take_trained() {
            events.push(MagicEvent::Compile);
        }
        if self.doc.is_none() {
            egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::bg_panel())).show(ui, |ui| {
                empty_state(ui, &mut events);
            });
            return events;
        }
        if self.training_view(env.optimize_training) {
            egui::CentralPanel::default()
                .frame(egui::Frame::new().fill(theme::bg_panel()).inner_margin(egui::Margin::same(16)))
                .show(ui, |ui| self.training_dashboard(ui, &mut events));
            return events;
        }

        if env.show_console {
            egui::Panel::bottom("magic-console")
                .resizable(true)
                .default_size(170.0)
                .size_range(90.0..=480.0)
                .frame(egui::Frame::new().fill(theme::bg_panel()).inner_margin(egui::Margin::symmetric(10, 6)))
                .show(ui, |ui| self.console(ui));
        }
        if env.show_panel {
            egui::Panel::right("magic-block")
                .resizable(true)
                .default_size(380.0)
                .size_range(280.0..=640.0)
                .frame(egui::Frame::new().fill(theme::bg_panel()).inner_margin(egui::Margin::same(12)))
                .show(ui, |ui| {
                    // Content wider than the panel scrolls; the panel keeps the width it was given.
                    egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                        self.block_panel(ui, env, &mut events);
                    });
                });
        }

        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::bg_canvas())).show(ui, |ui| {
            let doc = self.doc.as_mut().expect("checked above");
            let activity = HashMap::new();
            let graph_env = GraphEnv {
                registry: registry(),
                activity: &activity,
                running: &[],
                error_node: None,
                show_grid: env.show_grid,
                show_activity: false,
                python_missing: false,
            };
            for event in self.view.show(ui, &mut doc.studio, &graph_env) {
                match event {
                    GraphEvent::Inspect(node) => self.view.selection = [node].into(),
                    GraphEvent::Toast(text) => events.push(MagicEvent::Toast(text)),
                    _ => {}
                }
            }
        });

        let settled = ui.ctx().input(|i| !i.pointer.any_down()) && !ui.ctx().egui_wants_keyboard_input();
        let doc = self.doc.as_mut().expect("checked above");
        doc.sync_canvas();
        doc.track_history(settled);

        if let Some(editor) = &mut self.dataset {
            let spec = doc.file.model.text.clone().unwrap_or_default();
            let cx = dataset_editor::DatasetContext {
                spec: &spec,
                data: &doc.file.data,
                stages: studio::stage_files(&doc.studio.graph),
                tools: known_tools(&spec),
                generator: Some(&env.generator),
            };
            let edits = editor.show(ui.ctx(), &cx);
            for event in edits {
                match event {
                    DatasetEvent::Save(file, content) => {
                        doc.file.data.insert(file, content.into_bytes());
                        doc.files_changed();
                    }
                    DatasetEvent::Generator(settings) => events.push(MagicEvent::GeneratorSettings(settings)),
                }
            }
            if !editor.open {
                self.dataset = None;
            }
        }

        // Changes take effect in a running test at once.
        if let (Some(tester), Some(doc)) = (&self.tester, &self.doc) {
            let mut config = tester.config.lock().unwrap_or_else(|p| p.into_inner());
            if *config != doc.file.model {
                *config = doc.file.model.clone();
            }
        }
        events
    }

    /// Puts what a finished training run made into the model. True when it did.
    fn take_trained(&mut self) -> bool {
        let Some(job) = &self.job else { return false };
        let mut state = job.state.lock().unwrap_or_else(|p| p.into_inner());
        if matches!(state.outcome, Some(Err(_))) && !state.reported {
            state.reported = true;
            self.runtime_failed = true;
        }
        let trained = state.trained.take();
        drop(state);
        if let (Some(trained), Some(doc)) = (trained, &mut self.doc) {
            doc.apply_trained(trained);
            self.tester = None;
            self.picture = None;
            self.chat.answers.clear();
            return true;
        }
        false
    }

    /// A line in the console, from the tab (after training: what was saved).
    pub fn console_line(&mut self, level: Level, text: impl Into<String>) {
        if let Some(job) = &self.job {
            job.state.lock().unwrap_or_else(|p| p.into_inner()).push(level, text);
        }
    }

    // ---- the console --------------------------------------------------------

    fn console(&mut self, ui: &mut Ui) {
        let Some(job) = &self.job else {
            ui.label(RichText::new("CONSOLE").size(10.5).color(theme::text_faint()));
            widgets::note(
                ui,
                "Training reports here, line by line: each stage, its progress, the loss and what was saved. Press \
                 Train in the banner when the pipeline is ready – its tooltip says what is missing if it is not.",
            );
            return;
        };
        let state = job.state.lock().unwrap_or_else(|p| p.into_inner());
        let running = state.outcome.is_none();
        let mut clear = false;
        ui.horizontal(|ui| {
            ui.label(RichText::new("CONSOLE").size(10.5).color(theme::text_faint()));
            if running {
                ui.spinner();
                ui.label(RichText::new(progress_text(&state)).color(theme::text_dim()));
                if let Some((step, steps)) = state.progress
                    && steps > 0
                {
                    ui.add(
                        egui::ProgressBar::new(step as f32 / steps as f32).desired_width(160.0).desired_height(10.0),
                    );
                }
                let seconds = job.started.elapsed().as_secs();
                ui.label(RichText::new(format!("{}:{:02}", seconds / 60, seconds % 60)).color(theme::text_faint()));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if !running && widgets::icon_button(ui, icon::BROOM, "Clear the console").clicked() {
                    clear = true;
                }
                if let Some(last) = state.losses.last() {
                    ui.label(RichText::new(format!("loss {:.4}", last[1])).monospace().color(theme::text_dim()));
                }
                if let Some(last) = state.rewards.last() {
                    ui.label(RichText::new(format!("reward {:.2}", last[1])).monospace().color(theme::text_dim()));
                }
            });
        });
        let height = ui.available_height().max(40.0);
        ui.horizontal(|ui| {
            let chart_width = (ui.available_width() * 0.32).clamp(160.0, 360.0);
            let log_width = (ui.available_width() - chart_width - 12.0).max(120.0);
            ui.allocate_ui_with_layout(vec2(log_width, height), egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.set_width(log_width);
                egui::ScrollArea::vertical()
                    .id_salt("console-lines")
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for (level, text) in &state.lines {
                            console_line(ui, *level, text);
                        }
                    });
            });
            loss_chart(ui, &state.losses, vec2(chart_width, height));
        });
        drop(state);
        if clear {
            self.job = None;
        }
    }

    /// While training with the environment optimized: only the run itself.
    fn training_dashboard(&mut self, ui: &mut Ui, events: &mut Vec<MagicEvent>) {
        let Some(job) = &self.job else { return };
        let title = self.doc.as_ref().map(MagicDoc::title).unwrap_or_default();
        let state = job.state.lock().unwrap_or_else(|p| p.into_inner());
        ui.horizontal(|ui| {
            ui.label(RichText::new(icon::MAGIC_WAND).size(26.0).color(theme::accent()));
            ui.vertical(|ui| {
                ui.label(RichText::new(format!("Training “{title}”")).size(18.0).strong());
                ui.label(RichText::new(progress_text(&state)).color(theme::text_dim()));
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if widgets::accent_button(
                    ui,
                    format!("{}  Stop training", icon::STOP),
                    theme::danger(),
                    vec2(0.0, 30.0),
                )
                .clicked()
                {
                    events.push(MagicEvent::StopTraining);
                }
            });
        });
        ui.add_space(6.0);
        let share = state.progress.filter(|(_, n)| *n > 0).map(|(s, n)| s as f32 / n as f32);
        widgets::progress_bar(ui, share);
        let seconds = job.started.elapsed().as_secs_f64();
        let mut facts = vec![format!("{} so far", time_words(seconds))];
        if let Some(last) = state.losses.last() {
            facts.push(format!("loss {:.4}", last[1]));
        }
        if let Some(last) = state.rewards.last() {
            facts.push(format!("reward {:.2}", last[1]));
        }
        if let (Some((step, steps)), Some(share)) = (state.progress, share)
            && step > 5
            && share > 0.0
        {
            let stage_left = (steps - step) as f64 * (state.stage_seconds() / step as f64);
            facts.push(format!("this stage ends in about {}", time_words(stage_left)));
        }
        ui.label(RichText::new(facts.join("  ·  ")).monospace().color(theme::text_dim()));
        ui.add_space(8.0);
        let height = (ui.available_height() * 0.55).max(120.0);
        loss_chart(ui, &state.losses, vec2(ui.available_width(), height));
        ui.add_space(8.0);
        egui::ScrollArea::vertical().id_salt("training-lines").stick_to_bottom(true).auto_shrink([false, false]).show(
            ui,
            |ui| {
                for (level, text) in &state.lines {
                    console_line(ui, *level, text);
                }
            },
        );
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
        widgets::note(
            ui,
            "Effects and the background are off while training. Turn this off under Options › Optimize environment \
             when training.",
        );
    }

    // ---- the panel of the selected block ----------------------------------

    fn block_panel(&mut self, ui: &mut Ui, env: &MagicEnv<'_>, events: &mut Vec<MagicEvent>) {
        let selected = (self.view.selection.len() == 1).then(|| *self.view.selection.first().expect("one"));
        let doc = self.doc.as_mut().expect("checked by the caller");
        let Some(node_id) = selected.filter(|id| doc.studio.graph.contains(*id)) else {
            let training = self.job.as_ref().is_some_and(TrainJob::is_running);
            if let Some(select) = overview(ui, doc, env, training, events) {
                self.view.selection = [select].into();
            }
            return;
        };
        let kind = doc.studio.graph.node(node_id).map(|n| n.kind.clone()).unwrap_or_default();
        let note = doc.studio.graph.node(node_id).and_then(|n| n.param::<String>("note")).unwrap_or_default();
        let custom = kinds::SETUP_BLOCKS.contains(&kind.as_str())
            || [kinds::TEXT_DATA, kinds::IMAGE_DATA, kinds::PROMPT, kinds::TEST].contains(&kind.as_str());
        if custom {
            block_header(ui, &kind);
            if !note.is_empty() {
                widgets::notice(ui, icon::WARNING, note.clone(), theme::warning());
            }
        }
        if kinds::SETUP_BLOCKS.contains(&kind.as_str()) {
            setup_panel(ui, doc, node_id, &kind, env, events);
            return;
        }
        match kind.as_str() {
            kinds::PROMPT => self.chat_panel(ui, node_id, events),
            kinds::TEST => self.test_panel(ui, env, events),
            kinds::TEXT_DATA => {
                if let Some((open, scanned)) = text_data_card(ui, doc, node_id, events) {
                    let bytes = doc.file.data.get(&open).cloned().unwrap_or_default();
                    let mut editor = DatasetEditor::new(&open, &bytes);
                    if scanned {
                        let spec = doc.file.model.text.clone().unwrap_or_default();
                        let cx = dataset_editor::DatasetContext {
                            spec: &spec,
                            data: &doc.file.data,
                            stages: studio::stage_files(&doc.studio.graph),
                            tools: known_tools(&spec),
                            generator: None,
                        };
                        editor = editor.with_scan(&cx);
                    }
                    self.dataset = Some(editor);
                }
            }
            kinds::IMAGE_DATA => pictures_card(ui, doc, &mut self.thumbnails, events),
            _ => {
                let live = serde_json::Map::new();
                let mut inspector_env = InspectorEnv {
                    registry: registry(),
                    images: &mut self.view.images,
                    capturing: None,
                    recording: false,
                    displays: env.displays,
                    live: &live,
                };
                // Their settings are plain values: nothing needs the rest of the app.
                let _ = inspector::show(ui, &mut doc.studio, node_id, &mut inspector_env);
                if !note.is_empty() {
                    widgets::notice(ui, icon::WARNING, note, theme::warning());
                }
                if kind == kinds::RL
                    && let Some(node) = doc.studio.graph.node_mut(node_id)
                {
                    ui.separator();
                    let mut params: RlParams = node.params_or_default();
                    if rules_card(ui, &mut params.rules) {
                        node.set_param("rules", &params.rules);
                    }
                }
                connections(ui, doc, node_id);
                match kind.as_str() {
                    kinds::PRETRAIN | kinds::FINETUNE | kinds::RL | kinds::COMPACTION => {
                        stage_card(ui, doc, node_id, self.job.as_ref().is_some_and(|j| j.is_running()), events)
                    }
                    kinds::COMPILE => compile_card(ui, doc, &self.compiled, events),
                    kinds::BASE => widgets::note(
                        ui,
                        "Training further needs the weights of a model trained in MacroMagi. Models imported from \
                         elsewhere (ONNX) can be set up and used with Use As It Is, but not trained further.",
                    ),
                    kinds::ZERO => widgets::note(
                        ui,
                        "Next: Model Size and layer blocks give it a shape; the training stages after them teach it. \
                         Presets › Commons puts Magi's architectures in place.",
                    ),
                    _ => {}
                }
            }
        }
    }

    fn chat_panel(&mut self, ui: &mut Ui, node_id: NodeId, events: &mut Vec<MagicEvent>) {
        let doc = self.doc.as_mut().expect("checked by the caller");
        if doc.file.model.kind != ModelKind::Text {
            widgets::notice(ui, icon::INFO, "This is a vision model: use a Test Model block.", theme::text_faint());
            return;
        }
        if let Some((request, pending)) = &self.chat.pending {
            let done = pending.lock().unwrap_or_else(|p| p.into_inner()).take();
            if let Some(result) = done {
                if result.is_err() {
                    self.runtime_failed = true;
                }
                self.chat.answers.push((request.clone(), result));
                self.chat.pending = None;
            } else {
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            }
        }
        if let Some(loading) = &self.chat.loading {
            let done = loading.lock().unwrap_or_else(|p| p.into_inner()).take();
            if let Some(result) = done {
                self.chat.loading = None;
                if let Err(error) = result {
                    self.runtime_failed = true;
                    self.chat.load_error = Some(error);
                }
            } else {
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            }
        }
        let Some(node) = doc.studio.graph.node(node_id) else { return };
        let mut params: PromptParams = node.params_or_default();
        let before = params.clone();
        let spec = doc.file.model.text.clone().unwrap_or_default();
        let own = doc.file.model.architecture.is_some();
        let architecture = doc.file.model.architecture.clone().unwrap_or_default();
        let memory = doc.file.model.memory.clone().unwrap_or_default();

        // What runs it: MacroMagi's native engine (with its cache), or the ONNX model.
        let native = own && !doc.file.weights.is_empty() && architecture.native_problem().is_none();
        let has_onnx = !doc.file.onnx.is_empty();
        let ready = native || has_onnx;
        let key = if native {
            fingerprint(&doc.file.weights) ^ fingerprint(serde_json::to_string(&memory).unwrap_or_default().as_bytes())
        } else {
            fingerprint(&doc.file.onnx)
        };
        // The longest text: what it learned, or its Context Scaler's target.
        let longest = architecture.rope_scaling.as_ref().map_or(spec.context, |s| s.target.max(spec.context));
        let native_tools = spec.tools.iter().any(|t| t.native);

        if !ready {
            widgets::notice(
                ui,
                icon::INFO,
                "Train the model first (Train, in the banner): there is nothing to prompt yet.",
                theme::text_faint(),
            );
        }

        // ---- the model: what runs it, and whether it is loaded ----
        card(ui, "Model", |ui| {
            if ready {
                ui.label(
                    RichText::new(if native {
                        "Native engine: a key/value cache, so each token is one step"
                    } else {
                        "ONNX model: reads the whole context for every token"
                    })
                    .size(11.5)
                    .color(theme::text_dim()),
                );
            }
            let busy = self.chat.pending.is_some() || self.chat.loading.is_some();
            let loaded = match self.chat.engine.try_lock() {
                Ok(slot) => slot.as_ref().is_some_and(|(k, _)| *k == key),
                Err(_) => true,
            };
            let (dot, text, color) = if self.chat.loading.is_some() {
                (icon::CIRCLE_NOTCH, "Loading…", theme::text_dim())
            } else if loaded {
                (icon::CIRCLE, "Loaded – requests answer at once", theme::success())
            } else {
                (icon::CIRCLE_DASHED, "Not loaded – it loads with the first request", theme::text_faint())
            };
            ui.label(RichText::new(format!("{dot}  {text}")).size(11.5).color(color));
            widgets::button_row(ui, |ui| {
                ui.add_enabled_ui(ready && !busy, |ui| {
                    if loaded {
                        if ui
                            .small_button(format!("{}  Unload", icon::EJECT))
                            .on_hover_text("Frees its memory")
                            .clicked()
                        {
                            *self.chat.engine.lock().unwrap_or_else(|p| p.into_inner()) = None;
                        }
                    } else if ui.small_button(format!("{}  Load", icon::UPLOAD_SIMPLE)).clicked() {
                        self.chat.load_error = None;
                        let slot: Job = Arc::new(Mutex::new(None));
                        let (done, ctx) = (Arc::clone(&slot), ui.ctx().clone());
                        let engine = Arc::clone(&self.chat.engine);
                        let (spec, architecture, memory) = (spec.clone(), architecture.clone(), memory.clone());
                        let bytes = if native { doc.file.weights.clone() } else { doc.file.onnx.clone() };
                        std::thread::spawn(move || {
                            let result = load_engine(native, &architecture, &memory, &spec, &bytes).map(|model| {
                                *engine.lock().unwrap_or_else(|p| p.into_inner()) = Some((key, model));
                            });
                            *done.lock().unwrap_or_else(|p| p.into_inner()) = Some(result.map_err(|e| e.0));
                            ctx.request_repaint();
                        });
                        self.chat.loading = Some(slot);
                    }
                });
            });
            if let Some(error) = &self.chat.load_error {
                ui.add(egui::Label::new(RichText::new(error).size(11.5).color(theme::danger())).wrap());
            }
            if !native && has_onnx && spec.context > 512 {
                widgets::note(
                    ui,
                    "The ONNX model reads the whole context window again for every token it writes, so long answers \
                     take a while. Models of Transformer Layers trained here run on the native engine instead.",
                );
            }
        });
        ui.add_space(4.0);

        // ---- the memory of what it read (native engine) ----
        if native
            && let Ok(slot) = self.chat.engine.try_lock()
            && let Some(telemetry) = slot.as_ref().filter(|(k, _)| *k == key).and_then(|(_, m)| m.telemetry())
        {
            card(ui, "Memory", |ui| {
                let bar = |ui: &mut Ui, label: &str, used: u64, capacity: u64| {
                    ui.horizontal(|ui| {
                        widgets::fixed_label(ui, RichText::new(label).color(theme::text_dim()), 70.0);
                        let share = if capacity == 0 { 0.0 } else { used as f32 / capacity as f32 };
                        ui.add(
                            egui::ProgressBar::new(share.min(1.0))
                                .desired_width(150.0)
                                .text(format!("{:.1} MB", used as f64 / 1_048_576.0)),
                        );
                    });
                };
                bar(ui, "Hot", telemetry.hot_bytes, telemetry.hot_capacity);
                bar(ui, "Warm", telemetry.warm_bytes, telemetry.warm_capacity);
                bar(ui, "Disk", telemetry.cold_bytes, telemetry.cold_capacity);
                widgets::note(
                    ui,
                    format!(
                        "{} tokens in memory · {:.0}% of what it read came from memory{}",
                        telemetry.tokens,
                        telemetry.hit_rate() * 100.0,
                        if telemetry.from_disk { " · the system text came from disk" } else { "" }
                    ),
                );
            });
            ui.add_space(4.0);
        }

        // ---- how it answers ----
        card(ui, "Answering", |ui| {
            ui.horizontal(|ui| {
                widgets::fixed_label(ui, RichText::new("Context window").color(theme::text_dim()), 110.0);
                if native {
                    let mut own_context = params.context == 0;
                    if ui.checkbox(&mut own_context, format!("The model's ({longest})")).changed() {
                        params.context = if own_context { 0 } else { (longest / 2).max(256) };
                    }
                    if !own_context {
                        ui.add(
                            DragValue::new(&mut params.context)
                                .range(256..=longest.max(256))
                                .speed(64)
                                .suffix(" tokens"),
                        );
                    }
                } else {
                    ui.label(RichText::new(format!("{} tokens", spec.context)).color(theme::text_dim()));
                }
            });
            widgets::note(
                ui,
                if native {
                    format!(
                        "How much text it reads at once – system text, tools, request and answer together. It \
                         learned {} tokens{}; less keeps only the latest part and saves memory.",
                        spec.context,
                        if longest > spec.context {
                            format!(" and runs up to {} (Context Scaler)", studio::tokens_label(longest))
                        } else {
                            String::new()
                        }
                    )
                } else {
                    "The ONNX model reads the context window it was trained with (Model Size).".to_owned()
                },
            );
            if let Some(thinking) = &spec.thinking {
                let mut on = match params.thinking.as_str() {
                    "on" => true,
                    "off" => false,
                    _ => thinking.on_by_default,
                };
                ui.horizontal(|ui| {
                    widgets::fixed_label(ui, RichText::new("Thinking").color(theme::text_dim()), 110.0);
                    if widgets::toggle(ui, &mut on)
                        .on_hover_text("Think before answering (the Thinking block)")
                        .changed()
                    {
                        params.thinking = if on { "on" } else { "off" }.into();
                    }
                });
            }
            if let Some(effort) = &spec.effort {
                ui.horizontal(|ui| {
                    widgets::fixed_label(ui, RichText::new("Reasoning effort").color(theme::text_dim()), 110.0);
                    let mut level = effort.level(&params.effort).to_owned();
                    let options: Vec<(String, &str)> = effort.levels.iter().map(|l| (l.clone(), l.as_str())).collect();
                    if widgets::segmented(ui, &mut level, &options) {
                        params.effort = level;
                    }
                });
            }
            if native_tools {
                ui.horizontal(|ui| {
                    widgets::fixed_label(ui, RichText::new("Native tools").color(theme::text_dim()), 110.0);
                    widgets::toggle(ui, &mut params.native_tools)
                        .on_hover_text("Run the tools it calls on this computer and give it their results");
                    if params.native_tools {
                        ui.add(DragValue::new(&mut params.max_steps).range(1..=200).suffix(" calls at most"));
                    }
                });
            }
            egui::CollapsingHeader::new(RichText::new("More").size(11.0).color(theme::text_faint()))
                .id_salt("prompt-more")
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        widgets::fixed_label(ui, RichText::new("Longest answer").color(theme::text_dim()), 110.0);
                        let mut own_max = params.max_tokens == 0;
                        if ui.checkbox(&mut own_max, format!("The model's ({})", spec.max_tokens)).changed() {
                            params.max_tokens = if own_max { 0 } else { spec.max_tokens.max(16) };
                        }
                        if !own_max {
                            ui.add(DragValue::new(&mut params.max_tokens).range(1..=65536).suffix(" tokens"));
                        }
                    });
                    ui.horizontal(|ui| {
                        widgets::fixed_label(ui, RichText::new("Temperature").color(theme::text_dim()), 110.0);
                        let mut own_temperature = params.temperature < 0.0;
                        if ui.checkbox(&mut own_temperature, format!("The model's ({})", spec.temperature)).changed() {
                            params.temperature = if own_temperature { -1.0 } else { f64::from(spec.temperature) };
                        }
                        if !own_temperature {
                            ui.add(egui::Slider::new(&mut params.temperature, 0.0..=2.0).max_decimals(2));
                        }
                    });
                    widgets::note(ui, "The system prompt comes from the Text block, the tools from the Tools block.");
                });
        });
        if params != before
            && let Some(node) = doc.studio.graph.node_mut(node_id)
            && let serde_json::Value::Object(map) = serde_json::to_value(&params).unwrap_or_default()
        {
            for (key, value) in map {
                node.set_param(&key, value);
            }
        }
        ui.add_space(6.0);

        // ---- the request ----
        let response = ui.add(
            egui::TextEdit::multiline(&mut self.chat.request)
                .hint_text("Ask it something – try “What is 487 times 26?” or “Will it rain in Oslo?”")
                .desired_rows(3)
                .desired_width(f32::INFINITY),
        );
        let enter = response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.command);
        let busy = self.chat.pending.is_some() || self.chat.loading.is_some();
        widgets::button_row(ui, |ui| {
            let can = !busy && ready && !self.chat.request.trim().is_empty();
            ui.add_enabled_ui(can, |ui| {
                let send = widgets::accent_button(
                    ui,
                    format!("{}  Send", icon::PAPER_PLANE_RIGHT),
                    theme::accent(),
                    vec2(0.0, 28.0),
                )
                .on_hover_text("Ctrl+Enter")
                .clicked();
                if send || (enter && can) {
                    let request = self.chat.request.trim().to_owned();
                    let pending: Pending = Arc::new(Mutex::new(None));
                    let (slot, ctx, asked) = (Arc::clone(&pending), ui.ctx().clone(), request.clone());
                    let options = magi_ml::AskOptions {
                        max_tokens: params.max_tokens as usize,
                        temperature: params.temperature as f32,
                        seed: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_or(1, |d| d.as_nanos() as u64),
                        context: params.context as usize,
                        thinking: match params.thinking.as_str() {
                            "on" => Some(true),
                            "off" => Some(false),
                            _ => None,
                        },
                        effort: params.effort.clone(),
                    };
                    let steps = if native_tools && params.native_tools { params.max_steps as usize } else { 0 };
                    let (engine, spec, architecture, memory) =
                        (Arc::clone(&self.chat.engine), spec.clone(), architecture.clone(), memory.clone());
                    // The model's own native tools, if it has any.
                    let tools: std::collections::BTreeMap<String, Vec<u8>> = doc
                        .file
                        .data
                        .iter()
                        .filter(|(path, _)| path.starts_with("nt/"))
                        .map(|(p, b)| (p.clone(), b.clone()))
                        .collect();
                    // The model's bytes go along only when it must be loaded first.
                    let loaded_now =
                        self.chat.engine.try_lock().is_ok_and(|s| s.as_ref().is_some_and(|(k, _)| *k == key));
                    let bytes = match (loaded_now, native) {
                        (true, _) => Vec::new(),
                        (false, true) => doc.file.weights.clone(),
                        (false, false) => doc.file.onnx.clone(),
                    };
                    std::thread::spawn(move || {
                        let mut run = |call: &magi_core::magic::ToolCall| magi_ml::tools::run_call(&spec, &tools, call);
                        let mut generate = |prompt: &str| {
                            let mut slot = engine.lock().unwrap_or_else(|p| p.into_inner());
                            if slot.as_ref().is_none_or(|(k, _)| *k != key) {
                                *slot = None;
                                *slot = Some((key, load_engine(native, &architecture, &memory, &spec, &bytes)?));
                            }
                            let (_, model) = slot.as_mut().expect("loaded above");
                            magi_ml::complete(model, &spec, prompt, &options)
                        };
                        let result = magi_ml::tools::agent(&spec, &asked, &options, steps, &mut generate, &mut run)
                            .map_err(|e| e.0);
                        *slot.lock().unwrap_or_else(|p| p.into_inner()) = Some(result);
                        ctx.request_repaint();
                    });
                    self.chat.pending = Some((request, pending));
                    // The request shows with its answer; the box is ready for the next one.
                    self.chat.request.clear();
                }
            });
            if busy && self.chat.pending.is_some() {
                ui.spinner();
                ui.label(RichText::new("Answering…").color(theme::text_dim()));
            }
            if !self.chat.answers.is_empty() && ui.small_button("Clear").clicked() {
                self.chat.answers.clear();
            }
        });
        ui.add_space(6.0);
        let mut runtime = false;
        for (index, (request, done)) in self.chat.answers.iter().enumerate().rev() {
            ui.push_id(index, |ui| {
                egui::Frame::new().fill(theme::bg_raised()).corner_radius(theme::RADIUS).inner_margin(8).show(
                    ui,
                    |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new(format!("{}  {request}", icon::USER)).color(theme::text_dim()));
                        match done {
                            Err(error) => {
                                ui.add(egui::Label::new(RichText::new(error).color(theme::danger())).wrap());
                                if error.contains("Python") {
                                    runtime |= ui.button(format!("{}  Open Settings › Runtime", icon::GEAR)).clicked();
                                }
                            }
                            Ok(done) => {
                                // The native tools it ran on the way.
                                for step in &done.steps {
                                    ui.add(
                                        egui::Label::new(
                                            RichText::new(format!(
                                                "{}  {}  {}  ›  {}",
                                                icon::WRENCH,
                                                step.call.name,
                                                step.call.args,
                                                step.result
                                            ))
                                            .monospace()
                                            .size(11.0)
                                            .color(theme::text_dim()),
                                        )
                                        .wrap(),
                                    );
                                }
                                let answer = &done.answer;
                                if !answer.thinking.is_empty() {
                                    egui::CollapsingHeader::new(
                                        RichText::new(format!("{}  What it thought", icon::BRAIN))
                                            .size(11.0)
                                            .color(theme::text_faint()),
                                    )
                                    .show(ui, |ui| {
                                        ui.add(
                                            egui::Label::new(
                                                RichText::new(&answer.thinking).size(11.5).color(theme::text_dim()),
                                            )
                                            .wrap(),
                                        );
                                    });
                                }
                                let text = if answer.text.is_empty() {
                                    "(it said nothing besides its tool calls)"
                                } else {
                                    answer.text.as_str()
                                };
                                ui.add(
                                    egui::Label::new(RichText::new(format!("{}  {text}", icon::ROBOT)).strong()).wrap(),
                                );
                                for call in &answer.tools {
                                    ui.add(
                                        egui::Label::new(
                                            RichText::new(format!("{}  {}  {}", icon::WRENCH, call.name, call.args))
                                                .monospace()
                                                .size(11.5)
                                                .color(theme::accent()),
                                        )
                                        .wrap(),
                                    );
                                }
                                egui::CollapsingHeader::new(
                                    RichText::new("Everything it wrote").size(11.0).color(theme::text_faint()),
                                )
                                .show(ui, |ui| {
                                    ui.add(egui::Label::new(RichText::new(&answer.raw).monospace().size(11.0)).wrap());
                                });
                            }
                        }
                    },
                );
            });
            ui.add_space(4.0);
        }
        if runtime {
            events.push(MagicEvent::OpenRuntime);
        }
    }

    fn test_panel(&mut self, ui: &mut Ui, env: &MagicEnv<'_>, events: &mut Vec<MagicEvent>) {
        let doc = self.doc.as_ref().expect("checked by the caller");
        if doc.file.model.kind != ModelKind::Vision {
            widgets::notice(ui, icon::INFO, "This is a text model: use a Prompt Model block.", theme::text_faint());
            return;
        }
        if doc.file.onnx.is_empty() {
            widgets::notice(ui, icon::INFO, "Train the model (or import one) first.", theme::text_faint());
        }
        let running = self.tester.is_some();
        widgets::button_row(ui, |ui| {
            let (text, color) = if running {
                (format!("{}  Stop", icon::STOP), theme::danger())
            } else {
                (format!("{}  Watch the screen", icon::PLAY), theme::accent())
            };
            ui.add_enabled_ui(!doc.file.onnx.is_empty(), |ui| {
                if widgets::accent_button(ui, text, color, vec2(0.0, 28.0)).clicked() {
                    events.push(if running { MagicEvent::StopTest } else { MagicEvent::StartTest });
                }
                if ui
                    .button(format!("{}  Picture…", icon::IMAGE))
                    .on_hover_text("Run the model on a picture file")
                    .clicked()
                {
                    events.push(MagicEvent::TryPicture);
                }
            });
        });
        ui.add_enabled_ui(!running, |ui| {
            widgets::toggle_row(
                ui,
                "Press the key when it acts",
                "Off: the test only shows what the model would do. On: it presses the key, as a macro would.",
                &mut self.press,
            );
        });
        if self.press && env.dry_run {
            widgets::notice(ui, icon::INFO, "Dry run is on (Options): nothing is pressed.", theme::text_faint());
        }

        let mut shown_probabilities: Option<Vec<f32>> = None;
        if let Some(tester) = &self.tester {
            let state = tester.state.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(error) = &state.error {
                widgets::notice(ui, icon::WARNING, error.clone(), theme::danger());
            }
            ui.label(
                RichText::new(format!(
                    "{} · {:.0} looks per second · acted {} times",
                    state.engine, state.looks_per_second, state.acted
                ))
                .size(11.5)
                .color(theme::text_dim()),
            );
            if let Some((rgba, w, h)) = &state.picture
                && self.preview.as_ref().is_none_or(|(frame, _)| *frame != state.frame)
            {
                let image = egui::ColorImage::from_rgba_unmultiplied([*w as usize, *h as usize], rgba);
                let texture = ui.ctx().load_texture("magic-preview", image, egui::TextureOptions::LINEAR);
                self.preview = Some((state.frame, texture));
            }
            if let Some((_, texture)) = &self.preview {
                let side = ui.available_width().min(240.0);
                ui.vertical_centered(|ui| {
                    ui.image((
                        texture.id(),
                        vec2(side, side * texture.size()[1] as f32 / texture.size()[0].max(1) as f32),
                    ));
                });
                ui.label(RichText::new("What the model sees").size(10.5).color(theme::text_faint()));
            }
            if let Some(decision) = &state.decision {
                let (text, color) = if decision.act {
                    (
                        format!("{}  {} – act ({:.0}%)", icon::LIGHTNING, decision.label, decision.confidence * 100.0),
                        theme::success(),
                    )
                } else {
                    (format!("{} ({:.0}%)", decision.label, decision.confidence * 100.0), theme::text())
                };
                ui.label(RichText::new(text).strong().color(color));
            }
            shown_probabilities = Some(state.probabilities.clone());
        } else if let Some((name, probabilities)) = &self.picture {
            ui.label(RichText::new(format!("{}  {name}", icon::IMAGE)).color(theme::text_dim()));
            let decision = doc.file.model.decide(probabilities);
            ui.label(
                RichText::new(format!(
                    "{} ({:.0}%){}",
                    decision.label,
                    decision.confidence * 100.0,
                    if decision.act { " – act" } else { "" }
                ))
                .strong(),
            );
            shown_probabilities = Some(probabilities.clone());
        } else {
            widgets::note(
                ui,
                "Watch the screen to see what the model sees and answers, live – or try it on a picture file.",
            );
        }

        if let Some(probabilities) = shown_probabilities {
            ui.add_space(4.0);
            let best = probabilities.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
            for (class, p) in probabilities.iter().enumerate() {
                let label = doc.file.model.label(class);
                let acts = doc.file.model.classes.get(class).is_some_and(|c| c.act);
                let bar = egui::ProgressBar::new(*p)
                    .desired_height(14.0)
                    .text(RichText::new(format!("{label}  {:.0}%", p * 100.0)).size(11.0))
                    .fill(if Some(class) == best { theme::accent() } else { theme::bg_widget_active() });
                ui.add(bar).on_hover_text(if acts { "Acting on this answer is on" } else { "Not acted on" });
            }
        }
    }
}

fn progress_text(state: &crate::magic::train::Telemetry) -> String {
    let stage = stage_name(&state.stage);
    match state.progress {
        Some((step, steps)) => format!("{stage} · step {step} of {steps}"),
        None if stage.is_empty() => "Starting…".to_owned(),
        None => format!("{stage}…"),
    }
}

fn console_line(ui: &mut Ui, level: Level, text: &str) {
    let (glyph, color) = match level {
        Level::Info => (icon::DOT_OUTLINE, theme::text_dim()),
        Level::Stage => (icon::CARET_RIGHT, theme::text()),
        Level::Warning => (icon::WARNING, theme::warning()),
        Level::Error => (icon::X_CIRCLE, theme::danger()),
        Level::Done => (icon::CHECK_CIRCLE, theme::success()),
    };
    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.label(RichText::new(glyph).size(12.0).color(color));
        ui.add(egui::Label::new(RichText::new(text).size(12.0).color(color)).wrap());
    });
}

/// The name, category and description of a block, above its settings.
fn block_header(ui: &mut Ui, kind: &str) {
    let Some(def) = registry().get(kind) else { return };
    let accent = theme::category_color(def.category);
    ui.horizontal(|ui| {
        ui.label(RichText::new(node_icon(&def.id)).size(20.0).color(accent));
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 1.0;
            ui.label(RichText::new(&def.name).strong().size(15.0));
            ui.label(RichText::new(def.category.label()).size(11.0).color(accent));
        });
    });
    ui.add(egui::Label::new(RichText::new(&def.description).size(12.0).color(theme::text_dim())).wrap());
    ui.separator();
}

/// What is wired into a block's inputs, and where its outputs go.
fn connections(ui: &mut Ui, doc: &MagicDoc, node_id: NodeId) {
    let graph = &doc.studio.graph;
    let Some(node) = graph.node(node_id) else { return };
    let Some(def) = registry().get(&node.kind) else { return };
    let name_of = |id: NodeId| {
        graph.node(id).map_or_else(String::new, |n| {
            if n.label.is_empty() {
                registry().get(&n.kind).map(|d| d.name.clone()).unwrap_or_default()
            } else {
                n.label.clone()
            }
        })
    };
    if def.data_in.is_empty() && def.data_out.is_empty() {
        return;
    }
    ui.add_space(6.0);
    ui.label(RichText::new("CONNECTIONS").size(10.5).color(theme::text_faint()));
    for port in &def.data_in {
        let source = graph.data_source(node_id, &port.id).map(|p| p.node);
        let (text, color) = match source {
            Some(id) => (format!("{}  {}: from {}", icon::ARROW_RIGHT, port.label, name_of(id)), theme::text_dim()),
            None => {
                let missing = match port.id.as_str() {
                    studio::FFN => {
                        "a plain SwiGLU layer (wire a Mixture of Experts or Persona Router here to change it)"
                            .to_owned()
                    }
                    studio::OPTIMIZER => "the default AdamW (wire an Optimizer block here to change it)".to_owned(),
                    studio::SETUP => "nothing chained in".to_owned(),
                    studio::MODEL => "nothing – wire the block before it here".to_owned(),
                    "negative" => "nothing (optional)".to_owned(),
                    _ => "nothing – wire a dataset here".to_owned(),
                };
                (format!("{}  {}: {missing}", icon::ARROW_RIGHT, port.label), theme::text_faint())
            }
        };
        ui.add(egui::Label::new(RichText::new(text).size(11.5).color(color)).wrap());
    }
    for port in &def.data_out {
        let targets: Vec<String> = graph
            .wires()
            .iter()
            .filter(|w| w.from.node == node_id && w.from.port == port.id)
            .map(|w| name_of(w.to.node))
            .collect();
        let text = if targets.is_empty() {
            format!("{}  {}: not wired", icon::ARROW_LEFT, port.label)
        } else {
            format!("{}  {}: to {}", icon::ARROW_LEFT, port.label, targets.join(", "))
        };
        let color = if targets.is_empty() { theme::text_faint() } else { theme::text_dim() };
        ui.add(egui::Label::new(RichText::new(text).size(11.5).color(color)).wrap());
    }
}

/// With no block selected: what the canvas makes, and how to go on.
/// Returns a block to select.
fn overview(
    ui: &mut Ui,
    doc: &MagicDoc,
    env: &MagicEnv<'_>,
    training: bool,
    events: &mut Vec<MagicEvent>,
) -> Option<NodeId> {
    let model = &doc.file.model;
    ui.label(RichText::new(doc.title()).strong().size(15.0));
    ui.label(RichText::new(model.kind.label()).size(11.5).color(theme::text_dim()));
    ui.add_space(6.0);
    match studio::pipeline(&doc.studio.graph) {
        Ok(pipeline) => {
            card(ui, "The pipeline", |ui| {
                for id in &pipeline.path {
                    let Some(node) = doc.studio.graph.node(*id) else { continue };
                    let name = registry().get(&node.kind).map(|d| d.name.clone()).unwrap_or_default();
                    ui.label(format!("{}  {name}", node_icon(&node.kind)));
                }
                let layers = pipeline.architecture.layers.len();
                if layers > 0 {
                    widgets::note(ui, format!("{layers} layers, {} wide.", pipeline.architecture.width));
                }
                if !pipeline.trains() {
                    widgets::note(ui, "Nothing to train: the model is used as it is.");
                }
            });
            match doc.training_problem() {
                _ if training => {
                    widgets::notice(ui, icon::HOURGLASS, "Training… the console shows how it goes.", theme::accent())
                }
                Some(problem) if pipeline.trains() => widgets::notice(ui, icon::INFO, problem, theme::warning()),
                None => {
                    widgets::notice(ui, icon::CHECK_CIRCLE, "Ready to train.", theme::success());
                    if widgets::accent_button(ui, format!("{}  Train", icon::PLAY), theme::accent(), vec2(0.0, 28.0))
                        .clicked()
                    {
                        events.push(MagicEvent::Train(Default::default()));
                    }
                }
                _ => {}
            }
        }
        Err(problem) => widgets::notice(ui, icon::WARNING, problem, theme::warning()),
    }
    if let Some(trained) = &model.trained {
        ui.add_space(6.0);
        card(ui, "Trained", |ui| trained_facts(ui, trained));
    }
    if !doc.notes.is_empty() {
        ui.add_space(6.0);
        card(ui, "What the import found", |ui| {
            for note in &doc.notes {
                ui.add(
                    egui::Label::new(RichText::new(format!("•  {note}")).size(11.5).color(theme::text_dim())).wrap(),
                );
            }
        });
    }
    ui.add_space(8.0);
    card(ui, "How it works", |ui| {
        for line in [
            "The Model wire runs left to right: a start, the architecture, the training stages, Compile Package, \
             then Prompt Model (or Test Model) to try the result.",
            "Setup blocks – Model Info, Text, Tools… – chain into the start's Setup input; what they say is what the \
             model is.",
            "Datasets wire into the stages that learn from them; an Optimizer block wires into each stage it drives.",
            "Select a block to change it and see what is wired into it. Right-click the canvas, or Add in the banner, \
             for more blocks; Presets › Commons has Magi's architectures.",
        ] {
            ui.add(egui::Label::new(RichText::new(format!("•  {line}")).size(11.5).color(theme::text_dim())).wrap());
        }
    });
    ui.label(
        RichText::new(format!("{}  Models run on {}.", icon::CPU, env.engines)).size(11.0).color(theme::text_faint()),
    )
    .on_hover_text("Settings › Hardware");
    if doc.training_problem().is_some_and(|p| p.contains("folder")) {
        ui.add_space(4.0);
        if ui
            .button(format!("{}  Choose where it is saved…", icon::FOLDER_OPEN))
            .on_hover_text("Selects Compile Package, where the folder is chosen")
            .clicked()
        {
            return doc.studio.graph.nodes().find(|n| n.kind == kinds::COMPILE).map(|n| n.id);
        }
    }
    None
}

fn trained_facts(ui: &mut Ui, trained: &magi_core::magic::TrainingRecord) {
    let stages: Vec<&str> = trained.stages.iter().map(|s| stage_name(s)).collect();
    ui.label(RichText::new(stages.join(" › ")).color(theme::text_dim()));
    ui.label(
        RichText::new(format!(
            "{} steps · loss {} · {:.2}M parameters",
            trained.steps,
            if trained.loss.is_finite() { format!("{:.4}", trained.loss) } else { "–".into() },
            trained.parameters as f64 / 1e6
        ))
        .size(11.5)
        .color(theme::text_faint()),
    );
}

/// The loss over the run, as a line.
fn loss_chart(ui: &mut Ui, points: &[[f64; 2]], size: egui::Vec2) {
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, theme::RADIUS, theme::bg_raised());
    if points.len() < 2 {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Loss",
            egui::FontId::proportional(11.0),
            theme::text_faint(),
        );
        return;
    }
    let (x0, x1) = (points[0][0], points[points.len() - 1][0].max(points[0][0] + 1.0));
    let (lo, hi) = points.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| (lo.min(p[1]), hi.max(p[1])));
    let span = (hi - lo).max(1e-9);
    let inner = rect.shrink2(vec2(10.0, 14.0));
    let line: Vec<egui::Pos2> = points
        .iter()
        .map(|p| {
            egui::pos2(
                inner.left() + ((p[0] - x0) / (x1 - x0)) as f32 * inner.width(),
                inner.bottom() - ((p[1] - lo) / span) as f32 * inner.height(),
            )
        })
        .collect();
    painter.add(egui::Shape::line(line, egui::Stroke::new(1.5, theme::accent())));
    let font = egui::FontId::monospace(10.0);
    painter.text(
        rect.left_top() + vec2(6.0, 3.0),
        egui::Align2::LEFT_TOP,
        format!("loss {hi:.3}"),
        font.clone(),
        theme::text_faint(),
    );
    painter.text(
        rect.left_bottom() + vec2(6.0, -3.0),
        egui::Align2::LEFT_BOTTOM,
        format!("{lo:.3}"),
        font,
        theme::text_faint(),
    );
}

fn empty_state(ui: &mut Ui, events: &mut Vec<MagicEvent>) {
    // One narrow column: what Magic is, the two ways to start, and opening one.
    ui.vertical_centered(|ui| {
        ui.add_space(ui.available_height() * 0.16);
        ui.label(RichText::new(icon::MAGIC_WAND).size(40.0).color(theme::accent()));
        ui.label(RichText::new("Magic").size(22.0).strong());
        ui.add_space(2.0);
        ui.label(RichText::new("Build, train and try AI models.").color(theme::text_dim()));
        ui.add_space(16.0);
        let width = ui.available_width().min(340.0);
        let start = |ui: &mut Ui, glyph: &str, title: &str, about: &str| -> bool {
            // Both lines centred in the button; the title is what screen readers hear.
            let (rect, response) = ui.allocate_exact_size(vec2(width, 52.0), egui::Sense::click());
            response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, title));
            let visuals = ui.style().interact(&response);
            ui.painter().rect(
                rect,
                visuals.corner_radius,
                visuals.weak_bg_fill,
                visuals.bg_stroke,
                egui::StrokeKind::Inside,
            );
            let painter = ui.painter();
            let heading = format!("{glyph}  {title}");
            painter.text(
                rect.center_top() + vec2(0.0, 17.0),
                egui::Align2::CENTER_CENTER,
                heading,
                egui::FontId::proportional(14.0),
                theme::text(),
            );
            painter.text(
                rect.center_bottom() - vec2(0.0, 15.0),
                egui::Align2::CENTER_CENTER,
                about,
                egui::FontId::proportional(11.5),
                theme::text_dim(),
            );
            response.clicked()
        };
        if start(ui, icon::ROBOT, "New text model", "An agent that thinks and uses tools") {
            events.push(MagicEvent::NewModel { text: true });
        }
        ui.add_space(6.0);
        if start(ui, icon::EYE, "New vision model", "Tells what is on the screen apart") {
            events.push(MagicEvent::NewModel { text: false });
        }
        ui.add_space(12.0);
        ui.horizontal(|ui| {
            ui.add_space(((ui.available_width() - 230.0) / 2.0).max(0.0));
            if ui.add(egui::Button::new(format!("{}  Open…", icon::FOLDER_OPEN)).frame(false)).clicked() {
                events.push(MagicEvent::Open);
            }
            ui.label(RichText::new("·").color(theme::text_faint()));
            ui.menu_button(format!("{}  Import…", icon::DOWNLOAD_SIMPLE), |ui| {
                for (from, label) in
                    [(ImportFrom::Folder, "An ONNX project folder…"), (ImportFrom::File, "A .zip or .onnx file…")]
                {
                    if ui.button(label).clicked() {
                        events.push(MagicEvent::Import(from));
                        ui.close();
                    }
                }
            });
        });
    });
}

// ---- setup blocks ---------------------------------------------------------------

/// A setup block's settings, edited as the model they make.
fn setup_panel(
    ui: &mut Ui,
    doc: &mut MagicDoc,
    node_id: NodeId,
    kind: &str,
    env: &MagicEnv<'_>,
    events: &mut Vec<MagicEvent>,
) {
    let Some(value) = doc.studio.graph.node(node_id).map(|n| n.params_or_default::<SetupParams>().value) else {
        return;
    };
    let mut model = doc.file.model.clone();
    studio::apply_section(kind, &value, &mut model);
    match kind {
        kinds::INFO => info_card(ui, &mut model, doc, events),
        kinds::TEXT => text_card(ui, &mut model),
        kinds::TOOLS => tools_card(ui, &mut model),
        kinds::NATIVE_TOOLS => native_tools_card(ui, &mut model),
        kinds::MEMORY => memory_card(ui, &mut model),
        kinds::THINKING => thinking_card(ui, &mut model),
        kinds::EFFORT => effort_card(ui, &mut model, doc),
        kinds::INPUT => capture_card(ui, &mut model, env, events),
        kinds::ANSWERS => classes_card(ui, &mut model),
        kinds::ACTION => action_card(ui, &mut model),
        kinds::SETTINGS => settings_card(ui, &mut model),
        kinds::PREP => input_card(ui, &mut model),
        _ => {}
    }
    let changed = studio::section_of(kind, &model);
    if changed != value
        && let Some(node) = doc.studio.graph.node_mut(node_id)
    {
        node.set_param("value", changed);
    }
}

fn thinking_card(ui: &mut Ui, model: &mut MagicModel) {
    let Some(text) = model.text.as_mut() else { return };
    let thinking = text.thinking.get_or_insert_with(Default::default);
    widgets::toggle_row(
        ui,
        "Think unless asked not to",
        "Off: it thinks only when a request turns thinking on",
        &mut thinking.on_by_default,
    );
    widgets::note(
        ui,
        format!(
            "Training learns every answer that has reasoning both ways: thinking first ({}…{}), and straight away after \
             an empty thinking span. Answers without reasoning are learned without it. Requests then choose: the \
             Prompt Model block's Thinking switch, or the AI Agent window's.",
            text.think_tags.0, text.think_tags.1
        ),
    );
    widgets::note(
        ui,
        "Datasets give reasoning as \"thinking\" beside the answer: {\"prompt\": …, \"thinking\": …, \"response\": …}.",
    );
}

fn effort_card(ui: &mut Ui, model: &mut MagicModel, doc: &MagicDoc) {
    let Some(text) = model.text.as_mut() else { return };
    let effort = text.effort.get_or_insert_with(Default::default);
    // How many samples each level is learned from, in the datasets the training stages read.
    // Each file once, though several stages may read it.
    let files: std::collections::BTreeSet<String> =
        studio::stage_files(&doc.studio.graph).into_iter().flat_map(|(_, files)| files).collect();
    let contents: Vec<(&str, String)> = files
        .iter()
        .filter_map(|f| doc.file.data.get(f).map(|bytes| (f.as_str(), String::from_utf8_lossy(bytes).into_owned())))
        .collect();
    let counts = magi_core::dataset::effort_counts(effort, contents.iter().map(|(f, c)| (*f, c.as_str())));
    widgets::caption(ui, "Levels, from least to most effort");
    let mut remove = None;
    let several = effort.levels.len() > 1;
    for (index, level) in effort.levels.iter_mut().enumerate() {
        ui.push_id(("level", index), |ui| {
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(level).desired_width(140.0));
                let count = counts.iter().find(|(l, _)| l == level).map_or(0, |(_, n)| *n);
                let (words, color) = match count {
                    0 => ("no samples".to_owned(), theme::warning()),
                    1 => ("1 sample".to_owned(), theme::text_dim()),
                    n => (format!("{n} samples"), theme::text_dim()),
                };
                ui.label(RichText::new(words).size(11.0).color(color));
                if several && widgets::icon_button(ui, icon::TRASH, "Remove this level").clicked() {
                    remove = Some(index);
                }
            });
        });
    }
    if let Some(index) = remove {
        effort.levels.remove(index);
    }
    if ui.small_button(format!("{}  Add a level", icon::PLUS)).clicked() {
        effort.levels.push(format!("level {}", effort.levels.len() + 1));
    }
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Default").color(theme::text_dim()), 110.0);
        egui::ComboBox::from_id_salt("effort-default").selected_text(effort.default.clone()).show_ui(ui, |ui| {
            for level in effort.levels.clone() {
                ui.selectable_value(&mut effort.default, level.clone(), level);
            }
        });
    });
    // Levels no sample is written at learn from the nearest level that has some, as the trainer does.
    for (level, _) in &counts {
        if let Some(donor) = magi_core::dataset::donor(&counts, level) {
            widgets::notice(
                ui,
                icon::WARNING,
                format!(
                    "No sample is at “{level}”: training uses the “{donor}” samples for it too, so it reasons like \
                     “{donor}” until some samples are written at “{level}”."
                ),
                theme::warning(),
            );
        }
    }
    widgets::note(
        ui,
        format!(
            "The model is told its level at the top of every request – “Reasoning effort: {}” – and learns how much \
             to reason at each level from samples written at it. In the dataset editor, give a sample one or more \
             levels (in its file: \"effort\": \"{}\", or a list for several); samples without one are at the \
             default.",
            effort.levels.last().map_or("high", String::as_str),
            effort.levels.last().map_or("high", String::as_str),
        ),
    );
}

fn info_card(ui: &mut Ui, model: &mut MagicModel, doc: &MagicDoc, events: &mut Vec<MagicEvent>) {
    ui.horizontal(|ui| {
        widgets::caption(ui, "Name");
        ui.add(egui::TextEdit::singleline(&mut model.name).desired_width(f32::INFINITY));
    });
    ui.add(
        egui::TextEdit::multiline(&mut model.description)
            .hint_text("What it does")
            .desired_rows(2)
            .desired_width(f32::INFINITY),
    );
    ui.horizontal(|ui| {
        widgets::caption(ui, "Author");
        ui.add(egui::TextEdit::singleline(&mut model.author).desired_width(f32::INFINITY));
    });
    ui.horizontal(|ui| {
        widgets::caption(ui, "Licence");
        ui.add(
            egui::TextEdit::singleline(&mut model.license)
                .hint_text("Such as MIT or Apache-2.0")
                .desired_width(f32::INFINITY),
        );
    });
    let size_mb = doc.file.onnx.len() as f64 / (1024.0 * 1024.0);
    let facts = if size_mb > 0.0 && doc.file.model.kind == ModelKind::Vision {
        let classes = doc.shape.as_ref().and_then(|s| s.classes());
        format!(
            "Vision · reads {}×{} pictures · {} answers · {size_mb:.1} MB",
            doc.file.model.input.width,
            doc.file.model.input.height,
            classes.map_or_else(|| "?".to_owned(), |n| n.to_string())
        )
    } else if size_mb > 0.0 {
        format!(
            "Text · {} tokens of context · {size_mb:.1} MB{}",
            doc.file.model.text.as_ref().map_or(0, |t| t.context),
            if doc.file.weights.is_empty() { "" } else { " · with its weights" }
        )
    } else {
        format!("{} · not trained yet", doc.file.model.kind.label())
    };
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(facts).size(11.5).color(theme::text_dim()));
        if doc.file.model.kind == ModelKind::Vision
            && ui.small_button(format!("{}  Replace the model file…", icon::ARROWS_CLOCKWISE)).clicked()
        {
            events.push(MagicEvent::ReplaceModel);
        }
    });
    if !model.source.is_empty() {
        ui.label(RichText::new(&model.source).size(11.0).color(theme::text_faint()));
    }
    if let Some(trained) = &doc.file.model.trained {
        ui.add_space(4.0);
        trained_facts(ui, trained);
    }
}

/// A text model's system prompt, tools, way of writing and chat tags.
/// The Memory & Cache block: where the model keeps what it read while it runs.
fn memory_card(ui: &mut Ui, model: &mut MagicModel) {
    let memory = model.memory.get_or_insert_with(Default::default);
    let row = |ui: &mut Ui, label: &str, add: &mut dyn FnMut(&mut Ui)| {
        ui.horizontal(|ui| {
            widgets::fixed_label(ui, RichText::new(label).color(theme::text_dim()), 130.0);
            add(ui);
        });
    };
    card(ui, "Tiers", |ui| {
        row(ui, "Cache precision", &mut |ui| {
            egui::ComboBox::from_id_salt("kv-precision").selected_text(memory.precision.to_uppercase()).show_ui(
                ui,
                |ui| {
                    for (value, label) in [
                        ("f32", "F32 – exact, the largest"),
                        ("f16", "F16 – half the size"),
                        ("q8_0", "Q8_0 – a quarter"),
                    ] {
                        ui.selectable_value(&mut memory.precision, value.to_owned(), label);
                    }
                },
            );
        });
        row(ui, "Hot tier", &mut |ui| {
            ui.add(DragValue::new(&mut memory.hot_mb).range(16..=65536).suffix(" MB"));
        });
        row(ui, "Warm tier (memory)", &mut |ui| {
            ui.add(DragValue::new(&mut memory.warm_mb).range(0..=262_144).suffix(" MB"));
        });
        row(ui, "Cold tier (disk)", &mut |ui| {
            ui.add(DragValue::new(&mut memory.disk_gb).range(0..=4096).suffix(" GB"));
        });
        row(ui, "Disk folder", &mut |ui| {
            ui.add(
                egui::TextEdit::singleline(&mut memory.disk_dir)
                    .hint_text("MacroMagi's data folder")
                    .desired_width(150.0),
            );
            if ui.small_button(icon::FOLDER_OPEN).on_hover_text("Choose a folder (an NVMe SSD is best)").clicked()
                && let Some(folder) = rfd::FileDialog::new().set_title("Where the cache's files go").pick_folder()
            {
                memory.disk_dir = folder.display().to_string();
            }
        });
        widgets::note(
            ui,
            "The hot tier keeps the first block (whose first tokens anchor attention) and the latest blocks at full precision; older blocks move to the warm tier at the cache precision, and past it to files on the disk, read back when needed.",
        );
    });
    ui.add_space(4.0);
    card(ui, "Long texts", |ui| {
        widgets::toggle_row(
            ui,
            "Keep the system text on disk",
            "Its memory is saved under its SHA-256 and loaded next time, so it is never read again",
            &mut memory.persist_prefix,
        );
        row(ui, "Blocks read per token", &mut |ui| {
            ui.add(DragValue::new(&mut memory.read_blocks).range(0..=4096))
                .on_hover_text("0: all of them, exactly. Otherwise the first, the latest and the most relevant ones");
        });
        widgets::note(
            ui,
            "Reading only some blocks makes very long texts fast, at the cost of exactness: each block of 256 tokens is chosen by how well its mean key matches the token's query. The Prompt Model block shows how full each tier is while it runs.",
        );
    });
}

/// The Tools block: the tools the model may call.
/// The tools a Magi model runs itself: MacroMagi runs them and gives the model their results.
fn native_tools_card(ui: &mut Ui, model: &mut MagicModel) {
    let text = model.text.get_or_insert_with(TextSpec::default);
    card(ui, "Native tools", |ui| {
        widgets::note(
            ui,
            "Tools the model runs itself, on this computer: the model only learns which to call and with what, and \
             the tool does the work – exact math, the web, pictures, sound. They go into the model's nt folder \
             when it is compiled.",
        );
        for tool in magi_core::native_tools::built_in() {
            let index = text.tools.iter().position(|t| t.name == tool.name && t.native);
            let mut on = index.is_some();
            ui.horizontal(|ui| {
                if ui.checkbox(&mut on, RichText::new(&tool.name).monospace()).changed() {
                    match (on, index) {
                        (true, None) => {
                            // A regular tool of the same name becomes the native one.
                            text.tools.retain(|t| t.name != tool.name);
                            text.tools.push(tool.spec());
                        }
                        (false, Some(i)) => {
                            text.tools.remove(i);
                        }
                        _ => {}
                    }
                }
                ui.add(
                    egui::Label::new(RichText::new(&tool.description).size(11.0).color(theme::text_dim())).truncate(),
                )
                .on_hover_text(&tool.description);
            });
        }
    });
}

/// Tools the program running the model carries out: the AI Agent block in a macro, or another app.
fn tools_card(ui: &mut Ui, model: &mut MagicModel) {
    let text = model.text.get_or_insert_with(TextSpec::default);
    card(ui, "Tools it may call", |ui| {
        widgets::note(
            ui,
            format!(
                "The model reads these in its system text and calls one by writing {}{{\"name\": …, \"arguments\": \
                 {{…}}}}{}; a tool's result comes back in its own turn. The program running the model carries them \
                 out: in a macro, the AI Agent block does press, type, click, move, wait and log itself, and other \
                 tools come out of its Tool and Arguments outputs. Tools the model runs itself are in the Native \
                 Tools block. Datasets can offer their own tools per sample (\"tools\").",
                text.tool_tags.0, text.tool_tags.1
            ),
        );
        let mut remove = None;
        for (index, tool) in text.tools.iter_mut().enumerate().filter(|(_, t)| !t.native) {
            ui.push_id(("tool", index), |ui| {
                egui::Frame::new().fill(theme::bg_raised()).corner_radius(theme::RADIUS).inner_margin(6).show(
                    ui,
                    |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.add(egui::TextEdit::singleline(&mut tool.name).hint_text("name").desired_width(120.0));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if widgets::icon_button(ui, icon::TRASH, "Remove this tool").clicked() {
                                    remove = Some(index);
                                }
                            });
                        });
                        ui.add(
                            egui::TextEdit::singleline(&mut tool.description)
                                .hint_text("What it does")
                                .desired_width(f32::INFINITY),
                        );
                        json_edit(ui, &mut tool.parameters, "Arguments, as a JSON Schema");
                    },
                );
            });
            ui.add_space(3.0);
        }
        if let Some(index) = remove {
            text.tools.remove(index);
        }
        widgets::button_row(ui, |ui| {
            if ui.small_button(format!("{}  Add a tool", icon::PLUS)).clicked() {
                text.tools.push(ToolSpec {
                    name: format!("tool_{}", text.tools.len() + 1),
                    parameters: serde_json::json!({ "type": "object", "properties": {} }),
                    ..ToolSpec::default()
                });
            }
            let missing: Vec<_> =
                AGENT_TOOLS.iter().filter(|(name, _)| !text.tools.iter().any(|t| t.name == *name)).collect();
            if !missing.is_empty()
                && ui
                    .small_button(format!("{}  Add MacroMagi's tools", icon::WRENCH))
                    .on_hover_text("press, type, click, move, wait and log – the AI Agent block carries them out")
                    .clicked()
            {
                for (name, about) in missing {
                    text.tools.push(crate::magic::agent_tool(name, about));
                }
            }
        });
    });
}

fn text_card(ui: &mut Ui, model: &mut MagicModel) {
    let trained = model.trained.is_some();
    // The prompt it learned with, when that is known (older files: not).
    let learned = model.trained.as_ref().and_then(|t| t.system.clone());
    let text = model.text.get_or_insert_with(TextSpec::default);
    widgets::caption(ui, "System prompt");
    ui.add(
        egui::TextEdit::multiline(&mut text.system)
            .hint_text("What the model is told before every request (optional)")
            .desired_rows(3)
            .desired_width(f32::INFINITY),
    );
    if learned.as_deref().is_some_and(|l| l.trim() != text.system.trim()) {
        widgets::notice(
            ui,
            icon::INFO,
            "The model was trained with another system prompt: it may answer differently until it is trained again.",
            theme::warning(),
        );
    }
    ui.add_space(4.0);
    card(ui, "Tokens", |ui| {
        let mut learned = text.tokenizer.learned();
        ui.horizontal(|ui| {
            widgets::fixed_label(ui, RichText::new("Tokens").color(theme::text_dim()), 120.0);
            if widgets::segmented(ui, &mut learned, &[(true, "Learned"), (false, "Bytes")]) {
                text.tokenizer = if learned {
                    magi_core::magic::TokenizerSpec::learned_of(4096)
                } else {
                    magi_core::magic::TokenizerSpec::default()
                };
            }
        });
        if learned {
            ui.horizontal(|ui| {
                widgets::fixed_label(ui, RichText::new("Size").color(theme::text_dim()), 120.0);
                let mut size = text.tokenizer.size;
                if ui.add(DragValue::new(&mut size).range(512..=65_536).speed(64).suffix(" tokens")).changed() {
                    text.tokenizer.size = size;
                    text.tokenizer.merges.clear();
                }
            });
            let pieces = text.tokenizer.merges.len();
            widgets::note(
                ui,
                if pieces > 0 {
                    format!(
                        "{pieces} pieces learned from the datasets: about a third as many tokens as bytes, so the model reads \
                         three times the text at once and trains three times faster."
                    )
                } else {
                    "Pieces of words learned from the datasets when training starts from zero: about a third as many \
                     tokens as bytes, so the model reads three times the text at once and trains three times faster."
                        .to_owned()
                },
            );
        } else {
            widgets::note(ui, "Every byte is a token: simple, but the model reads a third as much text at once.");
        }
        if trained {
            widgets::note(ui, "Changing the tokens means training again from the start.");
        }
    });
    ui.add_space(4.0);
    card(ui, "How it writes", |ui| {
        ui.horizontal(|ui| {
            widgets::fixed_label(ui, RichText::new("Temperature").color(theme::text_dim()), 120.0);
            ui.add(egui::Slider::new(&mut text.temperature, 0.0..=2.0).max_decimals(2))
                .on_hover_text("0 always writes the likeliest; higher is freer and more varied");
        });
        ui.horizontal(|ui| {
            widgets::fixed_label(ui, RichText::new("Choose among").color(theme::text_dim()), 120.0);
            ui.add(DragValue::new(&mut text.top_k).range(0..=1000).suffix(" likeliest"))
                .on_hover_text("Each token is picked from this many likeliest; 0: from all");
        });
        ui.horizontal(|ui| {
            widgets::fixed_label(ui, RichText::new("Longest answer").color(theme::text_dim()), 120.0);
            ui.add(DragValue::new(&mut text.max_tokens).range(1..=8192).suffix(" tokens"));
        });
        widgets::note(
            ui,
            format!(
                "Context window: {} tokens – set in the Model Size block (a token is a piece of a word or a tag).",
                text.context
            ),
        );
    });
    ui.add_space(4.0);
    card(ui, "Personas", |ui| {
        if text.personas.is_empty() {
            widgets::note(
                ui,
                "None. A Dual-Persona Router block, wired into a layer block, gives the model two, each with its own tag.",
            );
        }
        for persona in &text.personas {
            ui.label(format!("{}  {} – {}", persona.tag, persona.name, persona.role));
        }
    });
    ui.add_space(4.0);
    egui::CollapsingHeader::new(RichText::new("CHAT TAGS").size(10.5).color(theme::text_faint())).show(ui, |ui| {
        widgets::note(ui, "The tags around each turn of a conversation. Most models keep the defaults.");
        if trained {
            widgets::notice(
                ui,
                icon::WARNING,
                "The model learned these tags. Changing them means training it again.",
                theme::warning(),
            );
        }
        let mut changed = false;
        for (label, value) in [
            ("System", &mut text.system_tag),
            ("Request", &mut text.user_tag),
            ("Answer", &mut text.assistant_tag),
            ("End of a turn", &mut text.end_tag),
            ("Thinking opens", &mut text.think_tags.0),
            ("Thinking closes", &mut text.think_tags.1),
            ("Tool call opens", &mut text.tool_tags.0),
            ("Tool call closes", &mut text.tool_tags.1),
            ("Compact opens", &mut text.compact_tags.0),
            ("Compact closes", &mut text.compact_tags.1),
        ] {
            ui.horizontal(|ui| {
                widgets::fixed_label(ui, RichText::new(label).color(theme::text_dim()), 120.0);
                changed |= ui.add(egui::TextEdit::singleline(value).code_editor().desired_width(160.0)).lost_focus();
            });
        }
        if changed {
            text.ensure_specials();
        }
        ui.label(
            RichText::new(format!(
                "{} tokens: 256 bytes, {} learned pieces and {} tags",
                text.vocab_size(),
                text.tokenizer.merges.len(),
                text.specials.len()
            ))
            .size(11.0)
            .color(theme::text_faint()),
        );
    });
}

fn capture_card(ui: &mut Ui, model: &mut MagicModel, env: &MagicEnv<'_>, events: &mut Vec<MagicEvent>) {
    let mut centred = matches!(model.capture.area, CaptureArea::Center { .. });
    if widgets::segmented(ui, &mut centred, &[(true, "Middle of the display"), (false, "A fixed area")]) {
        model.capture.area = if centred {
            CaptureArea::Center { height_share: f64::from(model.input.height) / 1080.0 }
        } else {
            let rect = model.capture.rect(env.displays, None).unwrap_or(Rect::new(
                0,
                0,
                model.input.width,
                model.input.height,
            ));
            CaptureArea::Region { rect: env.displays.rect_to_local(rect) }
        };
    }
    match &mut model.capture.area {
        CaptureArea::Center { height_share } => {
            ui.horizontal(|ui| {
                ui.label(RichText::new("A square of").color(theme::text_dim()));
                let mut pixels = *height_share * 1080.0;
                if ui.add(DragValue::new(&mut pixels).range(8.0..=1080.0).max_decimals(0).suffix(" px")).changed() {
                    *height_share = pixels / 1080.0;
                }
                ui.label(RichText::new("on a 1080-pixel-high display").color(theme::text_dim()));
            });
            ui.horizontal(|ui| {
                ui.label(RichText::new("On").color(theme::text_dim()));
                let mut display = Some(model.capture.display);
                if widgets::display_picker(ui, "magic-display", &mut display, env.displays, false) {
                    model.capture.display = display.unwrap_or(0);
                }
            });
            if let Some(rect) = model.capture.rect(env.displays, None) {
                widgets::note(
                    ui,
                    format!(
                        "Here: {}×{} pixels at {}, {}. It grows and shrinks with the display.",
                        rect.w, rect.h, rect.x, rect.y
                    ),
                );
            }
        }
        CaptureArea::Region { rect } => {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.add(DragValue::new(&mut rect.x).prefix("X "));
                ui.add(DragValue::new(&mut rect.y).prefix("Y "));
                ui.add(DragValue::new(&mut rect.w).range(1..=100_000).prefix("W "));
                ui.add(DragValue::new(&mut rect.h).range(1..=100_000).prefix("H "));
                let mut display = rect.display;
                if widgets::display_picker(ui, "magic-area-display", &mut display, env.displays, true) {
                    *rect = widgets::rect_on(*rect, display, env.displays);
                }
                if ui.button(format!("{}  Select area…", icon::SELECTION)).clicked() {
                    events.push(MagicEvent::PickArea);
                }
            });
        }
    }
}

fn classes_card(ui: &mut Ui, model: &mut MagicModel) {
    widgets::note(ui, "What each answer of the model means, and whether to act on it.");
    if model.classes.is_empty() {
        ui.weak("The model's answers are not known yet.");
    }
    egui::Grid::new("magic-classes").num_columns(4).spacing([8.0, 4.0]).show(ui, |ui| {
        ui.label(RichText::new("#").size(11.0).color(theme::text_faint()));
        ui.label(RichText::new("Name").size(11.0).color(theme::text_faint()));
        ui.label(RichText::new("Act").size(11.0).color(theme::text_faint()));
        ui.label(RichText::new("Sure enough at · then wait").size(11.0).color(theme::text_faint()));
        ui.end_row();
        for (index, class) in model.classes.iter_mut().enumerate() {
            ui.label(RichText::new(index.to_string()).color(theme::text_faint()));
            // Sized outright: a grid cell only offers the width its column had last frame.
            ui.add_sized([130.0, 22.0], egui::TextEdit::singleline(&mut class.label));
            ui.checkbox(&mut class.act, "").on_hover_text("Act when the model gives this answer");
            ui.horizontal(|ui| {
                ui.add_enabled_ui(class.act, |ui| {
                    let mut percent = class.threshold * 100.0;
                    if ui.add(DragValue::new(&mut percent).range(0.0..=100.0).max_decimals(0).suffix("%")).changed() {
                        class.threshold = percent / 100.0;
                    }
                    widgets::duration_edit(ui, &mut class.delay_ms);
                });
            });
            ui.end_row();
        }
    });
    widgets::button_row(ui, |ui| {
        if ui.small_button(format!("{}  Add answer", icon::PLUS)).clicked() {
            let n = model.classes.len();
            model.classes.push(ClassSpec { label: format!("Class {n}"), threshold: 0.5, ..ClassSpec::default() });
        }
        if !model.classes.is_empty() && ui.small_button(format!("{}  Remove the last", icon::MINUS)).clicked() {
            model.classes.pop();
        }
    });
}

fn action_card(ui: &mut Ui, model: &mut MagicModel) {
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Press").color(theme::text_dim()), 120.0);
        if let Some(picked) = widgets::input_picker(ui, "magic-key", model.action.key, true) {
            model.action.key = picked;
        }
    });
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Hold for").color(theme::text_dim()), 120.0);
        widgets::duration_edit(ui, &mut model.action.hold_ms);
    });
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Then ignore it for").color(theme::text_dim()), 120.0)
            .on_hover_text("So one event is not acted on twice");
        widgets::duration_edit(ui, &mut model.action.cooldown_ms);
    });
}

fn settings_card(ui: &mut Ui, model: &mut MagicModel) {
    widgets::note(
        ui,
        "Sliders for the values worth adjusting. Each changes one or more of the model's values, and every \
         block that uses the model can set them its own way.",
    );
    let labels: Vec<String> = model.classes.iter().map(|c| c.label.clone()).collect();
    let mut remove = None;
    for (index, setting) in model.settings.iter_mut().enumerate() {
        ui.push_id(("setting", index), |ui| {
            egui::Frame::new().fill(theme::bg_raised()).corner_radius(theme::RADIUS).inner_margin(6).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&setting.label).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if widgets::icon_button(ui, icon::TRASH, "Remove this setting").clicked() {
                            remove = Some(index);
                        }
                        if widgets::reset_button(ui, setting.value != setting.default) {
                            setting.value = setting.default;
                        }
                    });
                });
                let suffix = match setting.kind {
                    SettingKind::Percent => "%",
                    SettingKind::Milliseconds => " ms",
                };
                let (min, max) = (setting.min.min(setting.max), setting.max.max(setting.min));
                ui.add(egui::Slider::new(&mut setting.value, min..=max).max_decimals(0).suffix(suffix));
                egui::CollapsingHeader::new(RichText::new("Change what it does").size(11.0).color(theme::text_faint()))
                    .id_salt(("edit", index))
                    .show(ui, |ui| edit_setting(ui, setting, &labels));
            });
        });
        ui.add_space(3.0);
    }
    if let Some(index) = remove {
        model.settings.remove(index);
    }
    if ui.button(format!("{}  Add a setting", icon::PLUS)).clicked() {
        let n = model.settings.len() + 1;
        model.settings.push(ModelSetting {
            key: format!("setting_{n}"),
            label: format!("Setting {n}"),
            kind: SettingKind::Percent,
            min: 0.0,
            max: 100.0,
            default: 50.0,
            value: 50.0,
            targets: Vec::new(),
            ..ModelSetting::default()
        });
    }
    // The values follow the sliders.
    *model = model.configured(&serde_json::Map::new());
}

fn edit_setting(ui: &mut Ui, setting: &mut ModelSetting, labels: &[String]) {
    ui.horizontal(|ui| {
        widgets::caption(ui, "Name");
        ui.add(egui::TextEdit::singleline(&mut setting.label).desired_width(f32::INFINITY));
    });
    ui.horizontal(|ui| {
        widgets::caption(ui, "Kind");
        let mut kind = setting.kind;
        if widgets::segmented(
            ui,
            &mut kind,
            &[(SettingKind::Percent, "Confidence %"), (SettingKind::Milliseconds, "Time (ms)")],
        ) {
            setting.kind = kind;
            setting.targets.clear();
        }
    });
    ui.horizontal(|ui| {
        widgets::caption(ui, "From");
        ui.add(DragValue::new(&mut setting.min).max_decimals(0));
        widgets::caption(ui, "to");
        ui.add(DragValue::new(&mut setting.max).max_decimals(0));
        widgets::caption(ui, "default");
        ui.add(DragValue::new(&mut setting.default).max_decimals(0));
    });
    widgets::caption(ui, "Changes");
    let mut toggle = |ui: &mut Ui, target: SettingTarget, text: String| {
        let mut on = setting.targets.contains(&target);
        if ui.checkbox(&mut on, text).changed() {
            if on {
                setting.targets.push(target);
            } else {
                setting.targets.retain(|t| *t != target);
            }
        }
    };
    match setting.kind {
        SettingKind::Percent => {
            for (class, label) in labels.iter().enumerate() {
                toggle(ui, SettingTarget::Threshold { class }, format!("How sure “{label}” must be"));
            }
        }
        SettingKind::Milliseconds => {
            for (class, label) in labels.iter().enumerate() {
                toggle(ui, SettingTarget::Delay { class }, format!("The wait before acting on “{label}”"));
            }
            toggle(ui, SettingTarget::Cooldown, "The time ignored after acting".into());
            toggle(ui, SettingTarget::Hold, "How long the key is held".into());
        }
    }
}

fn input_card(ui: &mut Ui, model: &mut MagicModel) {
    let input = &mut model.input;
    widgets::note(
        ui,
        "How a picture of the screen is turned into what the model reads. Importing and training work this \
         out; change it only if you know the model needs something else.",
    );
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Size").color(theme::text_dim()), 110.0);
        ui.add(DragValue::new(&mut input.width).range(1..=4096).suffix(" px"));
        ui.label("×");
        ui.add(DragValue::new(&mut input.height).range(1..=4096).suffix(" px"));
    });
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Layout").color(theme::text_dim()), 110.0);
        widgets::segmented(ui, &mut input.layout, &[(Layout::Nchw, "Channels first"), (Layout::Nhwc, "Channels last")]);
    });
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Colors").color(theme::text_dim()), 110.0);
        widgets::segmented(ui, &mut input.color, &[(ColorOrder::Rgb, "RGB"), (ColorOrder::Bgr, "BGR")]);
    });
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Normalisation").color(theme::text_dim()), 110.0);
        let current = NORMALISATIONS.iter().find(|(_, m, s)| *m == input.mean && *s == input.std).map(|(n, ..)| *n);
        widgets::combo("normalisation").selected_text(current.unwrap_or("Custom")).show_ui(ui, |ui| {
            let needle = widgets::combo_search(ui);
            for (name, mean, std) in NORMALISATIONS {
                if widgets::option(ui, &needle, current == Some(name), name) {
                    (input.mean, input.std) = (mean, std);
                }
            }
        });
    });
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Mean").color(theme::text_dim()), 110.0);
        for value in &mut input.mean {
            ui.add(DragValue::new(value).speed(0.001).range(-10.0..=10.0).max_decimals(4));
        }
    });
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Spread").color(theme::text_dim()), 110.0);
        for value in &mut input.std {
            ui.add(DragValue::new(value).speed(0.001).range(0.0001..=10.0).max_decimals(4));
        }
    });
    ui.checkbox(&mut input.softmax, "Turn the model's scores into probabilities")
        .on_hover_text("On for most classifiers. Off when the model already gives probabilities.");
}

/// A JSON value edited as text; it changes once the text is valid JSON.
fn json_edit(ui: &mut Ui, value: &mut serde_json::Value, hint: &str) {
    let id = ui.id().with("json");
    let mut text =
        ui.data_mut(|d| d.get_temp::<String>(id)).unwrap_or_else(|| serde_json::to_string(value).unwrap_or_default());
    let edit = ui.add(
        egui::TextEdit::multiline(&mut text).code_editor().hint_text(hint).desired_rows(2).desired_width(f32::INFINITY),
    );
    let parsed = serde_json::from_str::<serde_json::Value>(&text);
    if edit.changed()
        && let Ok(parsed) = &parsed
    {
        *value = parsed.clone();
    }
    if parsed.is_err() && !text.trim().is_empty() {
        ui.label(RichText::new("Not valid JSON yet").size(11.0).color(theme::warning()));
    }
    if edit.has_focus() {
        ui.data_mut(|d| d.insert_temp(id, text));
    } else {
        ui.data_mut(|d| d.remove::<String>(id));
    }
}

/// The reward rules of a Reinforcement Learning block. True when changed.
fn rules_card(ui: &mut Ui, rules: &mut Vec<RewardRule>) -> bool {
    let mut changed = false;
    ui.label(RichText::new("REWARD RULES").size(10.5).color(theme::text_faint()));
    widgets::note(
        ui,
        "Each answer scores the sum of the rules it meets: reward what is right and fast, take points off for \
         loops and broken tool calls.",
    );
    let mut remove = None;
    for (index, rule) in rules.iter_mut().enumerate() {
        ui.push_id(("rule", index), |ui| {
            ui.horizontal_wrapped(|ui| {
                let current = RewardRule::KINDS.iter().find(|(k, _)| *k == rule.kind).map_or("?", |(_, l)| *l);
                widgets::combo("rule-kind").selected_text(current).width(170.0).show_ui(ui, |ui| {
                    let needle = widgets::combo_search(ui);
                    for (kind, label) in RewardRule::KINDS {
                        if widgets::option(ui, &needle, rule.kind == kind, label) {
                            rule.kind = kind.to_owned();
                            changed = true;
                        }
                    }
                });
                if ["contains", "not_contains", "regex"].contains(&rule.kind.as_str()) {
                    changed |= ui.add(egui::TextEdit::singleline(&mut rule.text).desired_width(110.0)).changed();
                }
                if rule.kind == "short" {
                    changed |= ui
                        .add(DragValue::new(&mut rule.tokens).range(1..=4096).prefix("≤ ").suffix(" tokens"))
                        .changed();
                }
                if rule.kind == "loop" {
                    changed |= ui.add(DragValue::new(&mut rule.tokens).range(1..=64).suffix("-token run")).changed();
                    changed |= ui.add(DragValue::new(&mut rule.repeats).range(2..=64).suffix("× repeated")).changed();
                }
                changed |= ui.add(DragValue::new(&mut rule.score).speed(0.1).prefix("score ")).changed();
                if widgets::icon_button(ui, icon::TRASH, "Remove this rule").clicked() {
                    remove = Some(index);
                }
            });
        });
    }
    if let Some(index) = remove {
        rules.remove(index);
        changed = true;
    }
    if ui.small_button(format!("{}  Add a rule", icon::PLUS)).clicked() {
        rules.push(RewardRule::default());
        changed = true;
    }
    changed
}

/// A small picture of a dataset picture, made once.
fn thumbnail(
    ui: &Ui,
    cache: &mut HashMap<String, egui::TextureHandle>,
    file: &str,
    bytes: &[u8],
) -> Option<egui::TextureHandle> {
    if let Some(texture) = cache.get(file) {
        return Some(texture.clone());
    }
    let bitmap = magi_vision::decode(bytes).ok()?;
    let scale = 72.0 / bitmap.width.max(bitmap.height).max(1) as f32;
    let (w, h) =
        (((bitmap.width as f32 * scale).round() as u32).max(1), ((bitmap.height as f32 * scale).round() as u32).max(1));
    let rgb = magi_ml::resize_rgb(&bitmap.rgba, bitmap.width, bitmap.height, w, h);
    let rgba: Vec<u8> = rgb
        .chunks(3)
        .flat_map(|p| [p[0].clamp(0.0, 255.0) as u8, p[1].clamp(0.0, 255.0) as u8, p[2].clamp(0.0, 255.0) as u8, 255])
        .collect();
    let image = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
    let texture = ui.ctx().load_texture(format!("magic-thumb-{file}"), image, egui::TextureOptions::LINEAR);
    cache.insert(file.to_owned(), texture.clone());
    Some(texture)
}

/// The Vision Dataset block: pictures of each answer, and of what it is not.
fn pictures_card(
    ui: &mut Ui,
    doc: &mut MagicDoc,
    thumbnails: &mut HashMap<String, egui::TextureHandle>,
    events: &mut Vec<MagicEvent>,
) {
    if doc.file.model.classes.is_empty() {
        widgets::notice(ui, icon::INFO, "Give the model its answers first (the Answers block).", theme::text_faint());
        return;
    }
    let mut remove: Option<String> = None;
    let labels: Vec<String> = doc.file.model.classes.iter().map(|c| c.label.clone()).collect();
    for (class, label) in labels.iter().enumerate() {
        card(ui, label, |ui| {
            for negative in [false, true] {
                let items: Vec<String> = doc
                    .file
                    .model
                    .training
                    .iter()
                    .chain(&doc.file.model.finetune)
                    .filter(|d| d.class == class && d.negative == negative)
                    .map(|d| d.file.clone())
                    .collect();
                ui.horizontal(|ui| {
                    let title = if negative { format!("Not “{label}”") } else { format!("“{label}”") };
                    ui.label(RichText::new(format!("{title} · {}", items.len())).color(theme::text_dim()));
                    let hint = if negative {
                        "Pictures the model mistook for this answer, to unlearn"
                    } else {
                        "Pictures that show this answer"
                    };
                    if ui.small_button(format!("{}  Add…", icon::PLUS)).on_hover_text(hint).clicked() {
                        events.push(MagicEvent::AddPictures { class, negative });
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
                    for file in items.iter().take(60) {
                        let Some(bytes) = doc.file.data.get(file) else { continue };
                        let Some(texture) = thumbnail(ui, thumbnails, file, bytes) else { continue };
                        let size = texture.size_vec2() * (56.0 / texture.size_vec2().max_elem().max(1.0));
                        let response =
                            ui.add(egui::Image::new((texture.id(), size)).sense(egui::Sense::click())).on_hover_text(
                                format!("{}\nRight-click to remove", file.rsplit('/').next().unwrap_or_default()),
                            );
                        response.context_menu(|ui| {
                            if ui.button(format!("{}  Remove", icon::TRASH)).clicked() {
                                remove = Some(file.clone());
                                ui.close();
                            }
                        });
                    }
                    if items.len() > 60 {
                        ui.label(RichText::new(format!("and {} more", items.len() - 60)).color(theme::text_faint()));
                    }
                });
            }
        });
        ui.add_space(4.0);
    }
    if let Some(file) = remove {
        doc.file.model.training.retain(|d| d.file != file);
        doc.file.model.finetune.retain(|d| d.file != file);
        doc.file.data.remove(&file);
        thumbnails.remove(&file);
    }
}

// ---- datasets -------------------------------------------------------------------

/// A Text Dataset block's files. Returns a file to open in the editor.
/// How many samples a dataset file has, and how many with problems – worked out
/// again only when the file or the model's text setup changes.
fn file_counts(ui: &Ui, file: &str, bytes: &[u8], spec: &TextSpec) -> (usize, usize) {
    type Counts = std::collections::HashMap<String, (usize, usize, String, usize, usize)>;
    let id = egui::Id::new("magic-dataset-counts");
    let key = format!("{:?}{:?}{}", spec.effort, spec.tool_tags, spec.tools.len());
    let (len, at) = (bytes.len(), bytes.as_ptr() as usize);
    if let Some((_, _, _, samples, problems)) = ui
        .data(|d| d.get_temp::<std::sync::Arc<Counts>>(id))
        .and_then(|c| c.get(file).cloned())
        .filter(|(l, a, k, ..)| *l == len && *a == at && *k == key)
    {
        return (samples, problems);
    }
    let jsonl = studio::is_jsonl(file);
    let samples = studio::split_samples(file, &String::from_utf8_lossy(bytes));
    let problems = samples.iter().filter(|s| studio::check_sample(jsonl, s, spec).is_err()).count();
    ui.data_mut(|d| {
        let mut counts: Counts = d.get_temp::<std::sync::Arc<Counts>>(id).map(|c| (*c).clone()).unwrap_or_default();
        counts.insert(file.to_owned(), (len, at, key, samples.len(), problems));
        d.insert_temp(id, std::sync::Arc::new(counts));
    });
    (samples.len(), problems)
}

/// The files of a Text Dataset block, and their scans (Scan all files). Returns
/// a file to open in the dataset editor, and whether to open it with its scan.
fn text_data_card(
    ui: &mut Ui,
    doc: &mut MagicDoc,
    node_id: NodeId,
    events: &mut Vec<MagicEvent>,
) -> Option<(String, bool)> {
    let mut params: TextDataParams = doc.studio.graph.node(node_id)?.params_or_default();
    let spec = doc.file.model.text.clone().unwrap_or_default();
    let mut open = None;
    let mut changed = false;
    // What the last Scan all files found, by file: (bytes scanned, finding titles).
    type Scans = std::collections::HashMap<String, (usize, Vec<String>)>;
    let scans_id = egui::Id::new(("magic-dataset-scans", node_id));
    let scans: Scans =
        ui.data(|d| d.get_temp::<std::sync::Arc<Scans>>(scans_id)).map(|s| (*s).clone()).unwrap_or_default();
    let mut scan_now = false;
    widgets::button_row(ui, |ui| {
        if ui
            .button(format!("{}  Scan all files", icon::MAGNIFYING_GLASS_PLUS))
            .on_hover_text(
                "Scans every file of this block for issues, as the dataset editor's Scan for issues does, and marks \
                 the ones with any: click a mark to see them",
            )
            .clicked()
        {
            scan_now = true;
        }
    });
    for negative in [false, true] {
        let (title, about) = if negative {
            (
                "What not to do",
                "Answers to avoid – wrong tool calls, loops, things it must not say – as {\"prompt\", \"rejected\"} \
                 samples. Fine-Tuning's “What not to do” input makes them less likely.",
            )
        } else {
            (
                "What to do",
                "Requests and answers ({\"prompt\", \"response\"}), conversations ({\"messages\"}) and tool calls, \
                 and .md or .txt text. Pretraining learns the language from all of it; Fine-Tuning learns to answer \
                 from the requests and answers.",
            )
        };
        card(ui, title, |ui| {
            widgets::note(ui, about);
            let list = if negative { &mut params.negative } else { &mut params.positive };
            let mut remove = None;
            if list.is_empty() {
                ui.weak("No files yet.");
            }
            for (index, file) in list.iter().enumerate() {
                let bytes = doc.file.data.get(file);
                let (count, problems) = bytes.map(|b| file_counts(ui, file, b, &spec)).unwrap_or_default();
                ui.horizontal(|ui| {
                    ui.label(format!("{}  {}", icon::FILE_TEXT, file.rsplit('/').next().unwrap_or_default()));
                    let detail = match (bytes.is_some(), problems) {
                        (false, _) => "missing".to_owned(),
                        (true, 0) => format!("{count} samples"),
                        (true, n) => format!("{count} samples · {n} with problems"),
                    };
                    ui.label(RichText::new(detail).size(11.0).color(if problems > 0 {
                        theme::warning()
                    } else {
                        theme::text_faint()
                    }));
                    // Its scan, when Scan all files has run since it last changed.
                    if let Some((_, titles)) = scans.get(file).filter(|(len, _)| Some(*len) == bytes.map(Vec::len)) {
                        if titles.is_empty() {
                            ui.label(RichText::new(icon::CHECK_CIRCLE).color(theme::success()))
                                .on_hover_text("No issues found");
                        } else {
                            let mark = egui::Button::new(
                                RichText::new(format!("{}  {}", icon::WARNING, titles.len())).color(theme::warning()),
                            )
                            .small();
                            if ui
                                .add(mark)
                                .on_hover_text(format!("{}\n\nClick to see them.", titles.join("\n")))
                                .clicked()
                            {
                                open = Some((file.clone(), true));
                            }
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if widgets::icon_button(ui, icon::TRASH, "Take it out of the dataset").clicked() {
                            remove = Some(index);
                        }
                        if bytes.is_some()
                            && widgets::icon_button(ui, icon::PENCIL_SIMPLE, "Open it in the dataset editor").clicked()
                        {
                            open = Some((file.clone(), false));
                        }
                    });
                });
            }
            if let Some(index) = remove {
                list.remove(index);
                changed = true;
            }
            widgets::button_row(ui, |ui| {
                if ui
                    .small_button(format!("{}  Add files…", icon::PLUS))
                    .on_hover_text(".jsonl, .json, .md or .txt")
                    .clicked()
                {
                    events.push(MagicEvent::AddDatasetFiles { node: node_id, negative });
                }
                let stem = if negative { "not-to-do" } else { "to-do" };
                let new_name = |data: &std::collections::BTreeMap<String, Vec<u8>>, ext: &str| {
                    (1..)
                        .map(|n| format!("data/text/{stem}-{n}.{ext}"))
                        .find(|f| !data.contains_key(f))
                        .expect("a free name")
                };
                if ui
                    .small_button(format!("{}  New .jsonl", icon::FILE_PLUS))
                    .on_hover_text("A new file of samples, with an example")
                    .clicked()
                {
                    let file = new_name(&doc.file.data, "jsonl");
                    doc.file.data.insert(file.clone(), dataset_editor::template(&file, negative).into_bytes());
                    list.push(file.clone());
                    open = Some((file, false));
                    changed = true;
                }
                if !negative
                    && ui
                        .small_button(format!("{}  New .md", icon::FILE_PLUS))
                        .on_hover_text("A new file of plain text")
                        .clicked()
                {
                    let file = new_name(&doc.file.data, "md");
                    doc.file.data.insert(file.clone(), dataset_editor::template(&file, false).into_bytes());
                    list.push(file.clone());
                    open = Some((file, false));
                    changed = true;
                }
                let examples: Vec<(String, String, bool)> =
                    studio::example_datasets().into_iter().filter(|(_, _, n)| *n == negative).collect();
                let missing = examples.iter().any(|(f, _, _)| !list.contains(f));
                if missing
                    && ui
                        .small_button(format!("{}  Add the examples", icon::LIGHTBULB))
                        .on_hover_text("The example agent's files: tool calls to press, type, click, wait and log")
                        .clicked()
                {
                    for (file, content, _) in examples {
                        doc.file.data.entry(file.clone()).or_insert_with(|| content.into_bytes());
                        if !list.contains(&file) {
                            list.push(file);
                        }
                    }
                    changed = true;
                }
            });
        });
        ui.add_space(6.0);
    }
    if changed {
        doc.files_changed();
        if let Some(node) = doc.studio.graph.node_mut(node_id) {
            node.set_params(&params);
        }
    }
    if scan_now {
        let cx = dataset_editor::DatasetContext {
            spec: &spec,
            data: &doc.file.data,
            stages: studio::stage_files(&doc.studio.graph),
            tools: known_tools(&spec),
            generator: None,
        };
        let mut found: Scans = Scans::new();
        for file in params.positive.iter().chain(&params.negative) {
            if let Some(bytes) = doc.file.data.get(file) {
                found.insert(file.clone(), (bytes.len(), DatasetEditor::scan_titles(file, bytes, &cx)));
            }
        }
        ui.data_mut(|d| d.insert_temp(scans_id, std::sync::Arc::new(found)));
    }
    connections(ui, doc, node_id);
    open
}

/// Every tool MacroMagi knows, and whether the model has it on: its own tools,
/// the native tools and the AI Agent's.
pub fn known_tools(spec: &TextSpec) -> Vec<(ToolSpec, bool)> {
    let mut tools: Vec<(ToolSpec, bool)> = spec.tools.iter().map(|t| (t.clone(), true)).collect();
    let mut add = |tool: ToolSpec| {
        if !tools.iter().any(|(t, _)| t.name == tool.name) {
            tools.push((tool, false));
        }
    };
    for tool in magi_core::native_tools::built_in() {
        add(tool.spec());
    }
    for (name, about) in AGENT_TOOLS {
        add(crate::magic::agent_tool(name, about));
    }
    tools
}

// ---- training stages ----------------------------------------------------------------

/// A stage's saved weights, and training from it: from here (after the stage
/// before), or this stage again on the trained model.
/// How many tokens a stage's datasets hold, in the model's tokens – counted again only when
/// the files or the tokens change.
fn stage_tokens(ui: &Ui, doc: &MagicDoc, files: &[String]) -> usize {
    let spec = doc.file.model.text.clone().unwrap_or_default();
    let key = format!(
        "{:?}|{}|{}",
        files.iter().map(|f| (f, doc.file.data.get(f).map_or(0, Vec::len))).collect::<Vec<_>>(),
        spec.tokenizer.merges.len(),
        spec.tokenizer.learned()
    );
    let id = egui::Id::new(("magic-stage-tokens", &key));
    if let Some(count) = ui.data(|d| d.get_temp::<usize>(id)) {
        return count;
    }
    // Each file's first 256 KB, read and counted, stands for the whole file: the window stays
    // quick on big datasets.
    const SAMPLE: usize = 256 * 1024;
    let mut count = 0;
    for file in files {
        let Some(bytes) = doc.file.data.get(file) else { continue };
        let mut end = bytes.len().min(SAMPLE);
        while end < bytes.len() && (bytes[end] & 0xC0) == 0x80 {
            end -= 1;
        }
        let content = String::from_utf8_lossy(&bytes[..end]);
        let jsonl = studio::is_jsonl(file);
        let mut part = 0;
        for sample in studio::split_samples(file, &content) {
            for text in magi_core::dataset::texts(&sample, jsonl) {
                // Learned tokens not learned yet: about 3.3 bytes a token on English.
                part += if spec.tokenizer.learned() && spec.tokenizer.merges.is_empty() {
                    text.len() * 10 / 33
                } else {
                    spec.encode(&text).len()
                };
            }
        }
        count += (part as f64 * bytes.len() as f64 / end.max(1) as f64) as usize;
    }
    ui.data_mut(|d| d.insert_temp(id, count));
    count
}

/// A count of tokens, short: 950, 12K, 1.5M, 2.1B.
/// A length of time as people say it: "40 s", "12 min", "3 h 05 min".
fn time_words(seconds: f64) -> String {
    let seconds = seconds.max(0.0).round() as u64;
    match seconds {
        0..90 => format!("{seconds} s"),
        90..5400 => format!("{} min", (seconds + 30) / 60),
        _ => format!("{} h {:02} min", seconds / 3600, seconds % 3600 / 60),
    }
}

fn amount(n: u64) -> String {
    match n {
        0..1_000 => n.to_string(),
        1_000..1_000_000 => format!("{}K", n / 1_000),
        1_000_000..1_000_000_000 => format!("{:.1}M", n as f64 / 1e6),
        _ => format!("{:.1}B", n as f64 / 1e9),
    }
}

/// How much a stage reads, and how many times over its data that is.
fn passes_line(ui: &mut Ui, doc: &MagicDoc, node_id: NodeId) {
    let Ok(pipeline) = studio::pipeline(&doc.studio.graph) else { return };
    let Some(index) = pipeline.stages.iter().position(|s| s.node == node_id) else { return };
    let stage = &pipeline.stages[index];
    if stage.kind == kinds::RL {
        return;
    }
    let files = studio::stage_files(&doc.studio.graph).get(index).map(|(_, f)| f.clone()).unwrap_or_default();
    let data = stage_tokens(ui, doc, &files);
    if data == 0 {
        return;
    }
    let param = |key: &str, default: u64| stage.params.get(key).and_then(serde_json::Value::as_u64).unwrap_or(default);
    let read = param("steps", 300) * param("batch", 16) * u64::from(pipeline.architecture.context.max(1));
    let passes = read as f64 / data as f64;
    let words = amount;
    // Small models gain from many passes over fine-tuning data (33 did better than 13 on the 12M
    // Magi); pretraining text read more than about 16 times adds little (Muennighoff et al., 2023).
    let pretraining = stage.kind == kinds::PRETRAIN;
    let (text, color) = if pretraining && passes > 16.0 {
        (
            format!(
                "Reads about {} tokens: {passes:.0} passes over its {} – past about 16, more steps add little. More data helps more.",
                words(read),
                words(data as u64)
            ),
            theme::warning(),
        )
    } else if !pretraining && passes > 60.0 {
        (
            format!(
                "Reads about {} tokens: {passes:.0} passes over its {} – it may learn them by heart. Fewer steps or more data.",
                words(read),
                words(data as u64)
            ),
            theme::warning(),
        )
    } else if passes < 0.5 {
        (
            format!(
                "Reads about {} tokens: under half of its {} – more steps learn more of it.",
                words(read),
                words(data as u64)
            ),
            theme::warning(),
        )
    } else {
        (
            format!("Reads about {} tokens: about {passes:.1} passes over its {}.", words(read), words(data as u64)),
            theme::text_dim(),
        )
    };
    ui.add(egui::Label::new(RichText::new(format!("{}  {text}", icon::STACK)).size(11.5).color(color)).wrap());
}

fn stage_card(ui: &mut Ui, doc: &MagicDoc, node_id: NodeId, training: bool, events: &mut Vec<MagicEvent>) {
    ui.add_space(6.0);
    passes_line(ui, doc, node_id);
    card(ui, "Training from here", |ui| {
        match doc.checkpoints().into_iter().find(|c| c.stage == node_id) {
            Some(saved) => {
                let loss = saved.loss.map(|l| format!(" · loss {l:.3}")).unwrap_or_default();
                widgets::notice(
                    ui,
                    icon::FLOPPY_DISK,
                    format!("The weights after this stage are kept ({}{loss}).", crate::magic::ago(saved.saved)),
                    theme::text_dim(),
                );
            }
            None if doc.checkpoint_dir().is_some() => widgets::note(
                ui,
                "Each training keeps the weights after this stage, so a later one can start after it instead of \
                 from zero.",
            ),
            None => widgets::note(ui, "Choose Compile Package's folder: the weights after each stage are kept there."),
        }
        widgets::button_row(ui, |ui| {
            let from_here = doc.train_from_here(node_id);
            let button = ui
                .add_enabled(
                    !training && from_here.is_ok(),
                    egui::Button::new(format!("{}  Train from here", icon::PLAY)),
                )
                .on_hover_text(
                    "Runs this stage and the ones after it, starting from the weights saved after the one before",
                );
            let button = match &from_here {
                Err(why) => button.on_disabled_hover_text(why.as_str()),
                Ok(_) => button.on_disabled_hover_text("Training is running"),
            };
            if button.clicked()
                && let Ok(choice) = from_here
            {
                events.push(MagicEvent::Train(choice));
            }
            let again = ui
                .add_enabled(
                    !training && doc.trained_fits(),
                    egui::Button::new(format!("{}  Only this, on the trained model", icon::ARROW_CLOCKWISE)),
                )
                .on_hover_text("Trains the finished model further with this stage alone – another fine-tune or more reinforcement learning")
                .on_disabled_hover_text("Train the model first (with this architecture)");
            if again.clicked() {
                events.push(MagicEvent::Train(crate::magic::TrainChoice {
                    from: crate::magic::TrainFrom::Trained,
                    stages: vec![node_id],
                }));
            }
        });
        let kept = doc.checkpoints().iter().any(|c| c.stage == node_id);
        let base = ui
            .add_enabled(!training && kept, egui::Button::new(format!("{}  Save as base model…", icon::PACKAGE)))
            .on_hover_text(
                "A .szi model of the weights after this stage – after pretraining, a base that knows the language. \
                 New text model from a base starts from it and only fine-tunes: minutes instead of the whole recipe.",
            )
            .on_disabled_hover_text("Train this stage first: its weights are kept, then they can be saved");
        if base.clicked() {
            events.push(MagicEvent::SaveBase(node_id));
        }
    });
}

// ---- packaging --------------------------------------------------------------------

fn compile_card(ui: &mut Ui, doc: &MagicDoc, compiled: &[PathBuf], events: &mut Vec<MagicEvent>) {
    ui.add_space(6.0);
    ui.label(RichText::new("PACKAGE").size(10.5).color(theme::text_faint()));
    let params = doc.compile_settings().unwrap_or_default();
    let has_folder = !params.folder.trim().is_empty();
    let ready = !doc.file.weights.is_empty() || !doc.file.onnx.is_empty();
    if !has_folder {
        widgets::notice(
            ui,
            icon::INFO,
            "Choose a folder above (Save into, then Browse): training saves the finished model there.",
            theme::warning(),
        );
    }
    widgets::button_row(ui, |ui| {
        ui.add_enabled_ui(has_folder && ready, |ui| {
            if widgets::accent_button(
                ui,
                format!("{}  Save package now", icon::PACKAGE),
                theme::accent(),
                vec2(0.0, 28.0),
            )
            .on_hover_text("Saves the model as it is into the folder (training does this by itself when it ends)")
            .on_disabled_hover_text(if ready { "Choose a folder first" } else { "Train the model first" })
            .clicked()
            {
                events.push(MagicEvent::Compile);
            }
        });
        ui.add_enabled_ui(ready, |ui| {
            if ui
                .add(egui::Button::new(format!("{}  Use in Macro", icon::TREE_STRUCTURE)).min_size(vec2(0.0, 28.0)))
                .on_hover_text(if doc.file.model.kind == ModelKind::Text {
                    "Adds the model to the open macro with Start, AI Load Model, AI Agent, Log and AI Unload Model blocks"
                } else {
                    "Adds the model to the open macro with Start and AI Watch blocks"
                })
                .on_disabled_hover_text("Train the model (or import one) first")
                .clicked()
            {
                events.push(MagicEvent::UseInMacro);
            }
        });
        if has_folder && ui.button(format!("{}  Open folder", icon::FOLDER_OPEN)).clicked() {
            events.push(MagicEvent::OpenFolder(PathBuf::from(params.folder.trim())));
        }
    });
    if !compiled.is_empty() {
        ui.label(RichText::new("Saved last time:").size(11.0).color(theme::text_faint()));
        for path in compiled {
            ui.label(RichText::new(format!("{}  {}", icon::CHECK, path.display())).size(11.0).color(theme::text_dim()));
        }
    }
}
