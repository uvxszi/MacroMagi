//! The Magic tab, driven through its interface: the banner's menus, every
//! block's panel and its buttons, the dataset editor, presets and undo – and,
//! with Python and PyTorch, training a model with the Train button and
//! prompting it with the Prompt Model block.

use egui::accesskit::Role;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use magi_core::nodes::studio::{self, kinds};

use super::*;

fn harness(dir_name: &str) -> Harness<'static, MagiApp> {
    let dir = std::env::temp_dir().join(format!("magi-magic-test-{dir_name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let paths = Paths { app_dir: dir.clone(), data_dir: dir.join("data") };
    let settings = Settings { dry_run: true, tab: Tab::Magic, ..Settings::default() };
    let options = StartupOptions {
        open: None,
        example: None,
        tab: Some(Tab::Magic),
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
        .with_size(vec2(1400.0, 900.0))
        .build_eframe(move |cc| MagiApp::new(cc, paths, settings, options));
    harness.run_steps(4);
    harness
}

/// Clicks the first widget whose label contains `text`.
fn click(harness: &mut Harness<'_, MagiApp>, text: &str) {
    harness
        .query_all_by_label_contains(text)
        .next()
        .unwrap_or_else(|| panic!("nothing labelled “{text}”"))
        .scroll_to_me();
    harness.run_steps(2);
    harness
        .query_all_by_label_contains(text)
        .next()
        .unwrap_or_else(|| panic!("nothing labelled “{text}”"))
        .click_accesskit();
    harness.run_steps(3);
}

/// Clicks the widget labelled exactly `text`: the last one, as windows are drawn after the panels.
fn click_exact(harness: &mut Harness<'_, MagiApp>, text: &str) {
    harness.query_all_by_label(text).last().unwrap_or_else(|| panic!("nothing labelled “{text}”")).click_accesskit();
    harness.run_steps(3);
}

fn shows(harness: &Harness<'_, MagiApp>, text: &str) -> bool {
    harness.query_all_by_label_contains(text).next().is_some()
}

/// Types into the first multi-line text box, as a person does: focus it, then type.
fn type_request(harness: &mut Harness<'_, MagiApp>, text: &str) {
    harness.get_by_role(Role::MultilineTextInput).focus();
    harness.run_steps(2);
    harness.get_by_role(Role::MultilineTextInput).type_text(text);
    harness.run_steps(2);
}

fn doc<'a>(harness: &'a Harness<'_, MagiApp>) -> &'a MagicDoc {
    harness.state().magic_tab.doc.as_ref().expect("a model is open")
}

fn node_of(harness: &Harness<'_, MagiApp>, kind: &str) -> magi_core::graph::NodeId {
    doc(harness).studio.graph.nodes().find(|n| n.kind == kind).unwrap_or_else(|| panic!("no {kind}")).id
}

fn select(harness: &mut Harness<'_, MagiApp>, kind: &str) {
    let id = node_of(harness, kind);
    harness.state_mut().magic_tab.view.selection = [id].into();
    harness.run_steps(3);
}

#[test]
fn the_magic_tab_works_through_its_interface() {
    let mut h = harness("ui");
    // The empty tab offers a new model; the example agent opens.
    click(&mut h, "New text model");
    assert_eq!(doc(&h).title(), "Magi");
    let text = doc(&h).file.model.text.clone().unwrap();
    assert_eq!(text.tools.len(), 16, "Magi's native tools come with it");
    assert!(text.thinking.is_some() && text.effort.is_some(), "Thinking and Reasoning Effort are chained in");

    // The banner holds the model's menus.
    click(&mut h, "File");
    assert!(shows(&h, "New vision model") && shows(&h, "Import") && shows(&h, "Save as"));
    h.key_press(egui::Key::Escape);
    h.run_steps(2);
    assert!(shows(&h, "Train"), "the banner has Train");
    assert!(doc(&h).training_problem().unwrap().contains("folder"), "training needs Compile's folder");

    // Every block's panel opens.
    let ids: Vec<_> = doc(&h).studio.graph.nodes().map(|n| n.id).collect();
    for id in ids {
        h.state_mut().magic_tab.view.selection = [id].into();
        h.run_steps(3);
    }
    // And every kind of block, added from the list.
    let all: Vec<String> = crate::ui::magic_tab::registry().iter().map(|d| d.id.clone()).collect();
    for kind in &all {
        let state = h.state_mut();
        let at = state.magic_tab.view.view_center();
        let doc = state.magic_tab.doc.as_mut().unwrap();
        let id = state.magic_tab.view.add_node(&mut doc.studio, crate::ui::magic_tab::registry(), kind, at).unwrap();
        state.magic_tab.view.selection = [id].into();
        h.run_steps(3);
        let state = h.state_mut();
        let doc = state.magic_tab.doc.as_mut().unwrap();
        doc.studio.graph.remove(id);
        h.run_steps(1);
    }

    // Model Info: what the block says is what the model is.
    select(&mut h, kinds::INFO);
    let field =
        h.query_all_by_role(Role::TextInput).find(|n| n.value().as_deref() == Some("Magi")).expect("the name field");
    field.click();
    h.run_steps(1);
    h.query_all_by_role(Role::TextInput).find(|n| n.value().as_deref() == Some("Magi")).unwrap().type_text("!");
    h.run_steps(3);
    assert_eq!(doc(&h).title(), "Magi!");

    // Tools: a tool is added; Native Tools: a native tool is picked.
    select(&mut h, kinds::TOOLS);
    click(&mut h, "Add a tool");
    assert_eq!(doc(&h).file.model.text.as_ref().unwrap().tools.len(), 17);
    click(&mut h, "Add MacroMagi's tools");
    assert!(doc(&h).file.model.text.as_ref().unwrap().tools.len() > 17);
    select(&mut h, kinds::NATIVE_TOOLS);
    click(&mut h, "transcribe_audio");
    assert!(doc(&h).file.model.text.as_ref().unwrap().tools.iter().any(|t| t.name == "transcribe_audio" && t.native));
    // Memory & Cache, and the Context Scaler.
    select(&mut h, kinds::MEMORY);
    assert!(shows(&h, "Keep the system text on disk"));
    select(&mut h, kinds::CONTEXT);
    assert!(shows(&h, "Run at"));

    // Thinking and Reasoning Effort: their blocks change the model.
    select(&mut h, kinds::THINKING);
    click(&mut h, "Think unless asked not to");
    assert!(
        !doc(&h).file.model.text.as_ref().unwrap().thinking.as_ref().unwrap().on_by_default,
        "Magi starts thinking; the switch turns it off"
    );
    select(&mut h, kinds::EFFORT);
    click(&mut h, "Add a level");
    assert_eq!(
        doc(&h).file.model.text.as_ref().unwrap().effort.as_ref().unwrap().levels.len(),
        3,
        "standard, superior, and one more"
    );

    // Prompt Model: its context window and controls, before there is anything to load.
    select(&mut h, kinds::PROMPT);
    assert!(shows(&h, "Context window") && shows(&h, "Not loaded") && shows(&h, "Reasoning effort"));

    // The Text Dataset: new files open in the editor, which adds, filters and keeps samples.
    select(&mut h, kinds::TEXT_DATA);
    click(&mut h, "New .jsonl");
    assert!(h.state().magic_tab.dataset_open(), "the editor opened");
    // The template's sample is learned at "superior" instead of the default.
    click_exact(&mut h, "superior");
    let add = format!("{}  Add", egui_phosphor::regular::PLUS);
    click_exact(&mut h, &add);
    click_exact(&mut h, "Request and answer");
    click(&mut h, "Problems (");
    assert!(shows(&h, "Problem"), "the empty sample is a problem");
    click_exact(&mut h, &add);
    click_exact(&mut h, "Conversation");
    click(&mut h, "Scan for issues");
    assert!(shows(&h, "Issues found") && shows(&h, "1 sample with a problem"));
    // Undo takes the conversation back out; redo puts it back.
    click_exact(&mut h, egui_phosphor::regular::ARROW_COUNTER_CLOCKWISE);
    click_exact(&mut h, "Raw");
    click(&mut h, "Keep changes");
    let undone = String::from_utf8(doc(&h).file.data["data/text/to-do-1.jsonl"].clone()).unwrap();
    assert_eq!(undone.lines().count(), 2, "{undone}");
    click_exact(&mut h, egui_phosphor::regular::ARROW_CLOCKWISE);
    click_exact(&mut h, "Chat");
    click(&mut h, "Keep changes");
    let file = "data/text/to-do-1.jsonl";
    let saved = String::from_utf8(doc(&h).file.data[file].clone()).unwrap();
    assert_eq!(saved.lines().count(), 3, "{saved}");
    assert!(saved.lines().next().unwrap().starts_with(r#"{"effort": "superior", "prompt""#), "{saved}");
    click(&mut h, "New .md");
    click_exact(&mut h, &add);
    click(&mut h, "Paragraph");
    click(&mut h, "Keep changes");
    assert!(doc(&h).file.data.contains_key("data/text/to-do-1.md"));
    assert!(doc(&h).file.model.datasets.iter().any(|f| f == file), "the model keeps the new files");

    // Reinforcement learning's rules.
    {
        let state = h.state_mut();
        let at = state.magic_tab.view.view_center();
        let doc = state.magic_tab.doc.as_mut().unwrap();
        let id =
            state.magic_tab.view.add_node(&mut doc.studio, crate::ui::magic_tab::registry(), kinds::RL, at).unwrap();
        state.magic_tab.view.selection = [id].into();
    }
    h.run_steps(3);
    let added = *h.state().magic_tab.view.selection.first().unwrap();
    let rules = |h: &Harness<'_, MagiApp>| {
        doc(h).studio.graph.node(added).unwrap().params_or_default::<studio::RlParams>().rules.len()
    };
    let before = rules(&h);
    click(&mut h, "Add a rule");
    assert_eq!(rules(&h), before + 1);

    // Presets › Commons puts Magi back in place of a changed architecture; undo takes it back.
    let layers = |h: &Harness<'_, MagiApp>| studio::pipeline(&doc(h).studio.graph).unwrap().architecture.layers.len();
    let experts = doc(&h).studio.graph.nodes().filter(|n| n.kind == kinds::LAYERS).map(|n| n.id).nth(1).unwrap();
    h.state_mut().magic_tab.doc.as_mut().unwrap().studio.graph.node_mut(experts).unwrap().set_param("layers", 2);
    h.run_steps(3);
    assert_eq!(layers(&h), 5);
    click(&mut h, "Presets");
    click(&mut h, "Commons");
    assert!(!shows(&h, "Qwen3 (small)") && !shows(&h, "Magi-F1DS"), "only Magi is offered");
    click_exact(&mut h, "Magi 400M");
    assert_eq!(layers(&h), 32);
    h.state_mut().undo();
    h.run_steps(3);
    assert_eq!(layers(&h), 5, "as it was changed");
    h.state_mut().redo();
    h.run_steps(3);
    assert_eq!(layers(&h), 32);

    // View: the console can be hidden.
    click(&mut h, "View");
    click(&mut h, "Console");
    assert!(!h.state().settings.magic_console);

    // Compile: a folder makes the pipeline ready.
    let compile = node_of(&h, kinds::COMPILE);
    let folder = std::env::temp_dir().join("magi-ui-compile");
    h.state_mut()
        .magic_tab
        .doc
        .as_mut()
        .unwrap()
        .studio
        .graph
        .node_mut(compile)
        .unwrap()
        .set_param("folder", folder.to_string_lossy());
    h.run_steps(3);
    assert_eq!(doc(&h).training_problem(), None);
    select(&mut h, kinds::COMPILE);
    assert!(shows(&h, "Save package now") && shows(&h, "Use in Macro") && shows(&h, "Magi Model (.szi)"));

    // The AI Agent window: a chat of its own, kept in the data folder.
    h.state_mut().agent_chat.open = true;
    h.run_steps(3);
    assert!(shows(&h, "Chats") && shows(&h, "What can I help with?"), "the AI Agent window shows");
    click_exact(&mut h, &format!("{}  New", egui_phosphor::regular::PLUS));
    assert_eq!(h.state().agent_chat.chat_count(), 2, "a new chat");
    h.state_mut().agent_chat.open = false;
    h.run_steps(2);

    // A vision model, after the question about the unsaved one.
    click(&mut h, "File");
    click(&mut h, "New vision model");
    click(&mut h, "Don't save");
    assert_eq!(doc(&h).file.model.kind, magi_core::magic::ModelKind::Vision);
    select(&mut h, kinds::ANSWERS);
    click(&mut h, "Add answer");
    assert_eq!(doc(&h).file.model.classes.len(), 3, "the Answers block makes the model's answers");
    select(&mut h, kinds::IMAGE_DATA);
    assert!(shows(&h, "Pictures an answer was mistaken for") || shows(&h, "Not “"));
    select(&mut h, kinds::TEST);
    assert!(shows(&h, "Watch the screen"));
}

/// The whole way, as a person does it: New text model (made tiny), a folder
/// for Compile, Train, then Prompt Model.
#[test]
#[ignore = "needs Python with PyTorch and onnx (MAGI_TEST_PYTHON)"]
fn a_model_trains_and_answers_through_the_interface() {
    let mut h = harness("train");
    if let Some(python) = std::env::var_os("MAGI_TEST_PYTHON") {
        h.state_mut().settings.python_path = Some(python.into());
    }
    if let Ok(gpu) = std::env::var("MAGI_TEST_GPU") {
        h.state_mut().settings.train_gpus = Some(gpu.parse().ok().into_iter().collect());
    }
    click(&mut h, "New text model");
    crate::magic::tiny(h.state_mut().magic_tab.doc.as_mut().unwrap());
    let folder = std::env::temp_dir().join(format!("magi-ui-train-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&folder);
    {
        let doc = h.state_mut().magic_tab.doc.as_mut().unwrap();
        let ids: Vec<_> = doc.studio.graph.nodes().map(|n| (n.id, n.kind.clone())).collect();
        for (id, kind) in ids {
            let node = doc.studio.graph.node_mut(id).unwrap();
            if kind == kinds::COMPILE {
                node.set_param("folder", folder.to_string_lossy());
            }
        }
    }
    h.run_steps(3);
    click(&mut h, "Train");
    assert!(h.state().magic_tab.training(), "training started");
    assert!(shows(&h, "Training “Magi”"), "the training view replaces the canvas");
    let began = std::time::Instant::now();
    while h.state().magic_tab.training() {
        assert!(began.elapsed().as_secs() < 1800, "training took too long");
        std::thread::sleep(std::time::Duration::from_millis(200));
        h.run_steps(1);
    }
    h.run_steps(5);
    let lines: Vec<String> = h
        .state()
        .magic_tab
        .job
        .as_ref()
        .map(|j| j.state.lock().unwrap().lines.iter().map(|(_, l)| l.clone()).collect())
        .unwrap_or_default();
    assert!(doc(&h).file.model.trained.is_some(), "{lines:#?}");
    // One folder for the project: the workspace and the Magi model, one .szi file with its native tools.
    let project = folder.join("Magi");
    let magic = project.join("Magi.magic");
    let szi = project.join("Magi.szi");
    assert!(magic.is_file() && szi.is_file(), "{lines:#?}");
    assert!(magi_core::szi::read(&szi).unwrap().data.contains_key("nt/calculator/run.py"), "its native tools");
    assert_eq!(doc(&h).path.as_deref(), Some(magic.as_path()), "the open model is the saved one");
    // The workspace holds no model; opening it finds the model in the package.
    let workspace = magi_core::magic::MagicFile::load(&magic).unwrap();
    assert!(workspace.weights.is_empty() && workspace.onnx.is_empty());
    assert!(!magi_core::szi::open(&magic).unwrap().weights.is_empty());

    // Prompt Model on the native engine: it loads once, then answers from memory.
    select(&mut h, kinds::PROMPT);
    let ask = |h: &mut Harness<'_, MagiApp>, request: &str| {
        type_request(h, request);
        click(h, "Send");
        let began = std::time::Instant::now();
        while h.state().magic_tab.chat_waiting() {
            assert!(began.elapsed().as_secs() < 300, "no answer");
            std::thread::sleep(std::time::Duration::from_millis(100));
            h.run_steps(1);
        }
        h.run_steps(3);
        h.state()
            .magic_tab
            .last_answer()
            .cloned()
            .unwrap_or_else(|| panic!("no answer: {:?}", h.state().magic_tab.chat_state()))
    };
    // A tiny model says little that makes sense: it must load once, answer, and stay loaded.
    let answer = ask(&mut h, "List the files in C:/work");
    assert!(!answer.raw.is_empty(), "{answer:?}");
    assert!(shows(&h, "Loaded") && shows(&h, "Native engine"), "the model stays loaded");
    assert!(shows(&h, "Hot"), "the memory's tiers show");
    let began = std::time::Instant::now();
    let _ = ask(&mut h, "Hello!");
    let second = began.elapsed();
    eprintln!("a loaded model answered in {second:?}");
    assert!(second.as_secs() < 20, "the native engine answers quickly: {second:?}");
    let _ = std::fs::remove_dir_all(&folder);
}

/// A vision model, the way a person makes one: New vision model, pictures of
/// each answer in the Vision Dataset, a folder for Compile, Train – and the
/// trained model telling a new picture apart.
#[test]
#[ignore = "needs Python with PyTorch and onnx (MAGI_TEST_PYTHON)"]
fn a_vision_model_trains_through_the_interface() {
    let mut h = harness("vision");
    if let Some(python) = std::env::var_os("MAGI_TEST_PYTHON") {
        h.state_mut().settings.python_path = Some(python.into());
    }
    if let Ok(gpu) = std::env::var("MAGI_TEST_GPU") {
        h.state_mut().settings.train_gpus = Some(gpu.parse().ok().into_iter().collect());
    }
    click(&mut h, "New vision model");
    // Pictures, as Add… puts them in: reddish ones are "Yes", bluish ones "No".
    let picture = |seed: u32, red: bool| {
        let mut image = image::RgbImage::new(48, 48);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            let noise = ((x * 31 + y * 17 + seed * 7) % 40) as u8;
            *pixel = if red {
                image::Rgb([200 + noise / 2, 30 + noise, 30])
            } else {
                image::Rgb([30, 40 + noise, 200 + noise / 2])
            };
        }
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        bytes.into_inner()
    };
    let folder = std::env::temp_dir().join(format!("magi-ui-vision-{}", std::process::id()));
    {
        let doc = h.state_mut().magic_tab.doc.as_mut().unwrap();
        for i in 0..24 {
            for (class, red) in [(0usize, true), (1, false)] {
                let file = format!("data/train/{class}/{i:04}.png");
                doc.file.data.insert(file.clone(), picture(i, red));
                doc.file.model.training.push(magi_core::magic::DataItem { file, class, negative: false });
            }
        }
        doc.files_changed();
        let ids: Vec<_> = doc.studio.graph.nodes().map(|n| (n.id, n.kind.clone())).collect();
        for (id, kind) in ids {
            let node = doc.studio.graph.node_mut(id).unwrap();
            match kind.as_str() {
                kinds::COMPILE => node.set_param("folder", folder.to_string_lossy()),
                kinds::VISION_TRAIN => node.set_param("steps", 120),
                _ => {}
            }
        }
    }
    h.run_steps(3);
    select(&mut h, kinds::IMAGE_DATA);
    assert!(shows(&h, "“Yes” · 24"), "the Vision Dataset block shows the pictures");
    assert_eq!(doc(&h).training_problem(), None);
    click(&mut h, "Train");
    let began = std::time::Instant::now();
    while h.state().magic_tab.training() {
        assert!(began.elapsed().as_secs() < 900, "training took too long");
        std::thread::sleep(std::time::Duration::from_millis(200));
        h.run_steps(1);
    }
    h.run_steps(5);
    let lines: Vec<String> = h
        .state()
        .magic_tab
        .job
        .as_ref()
        .map(|j| j.state.lock().unwrap().lines.iter().map(|(_, l)| l.clone()).collect())
        .unwrap_or_default();
    assert!(doc(&h).file.model.trained.is_some(), "{lines:#?}");
    assert!(folder.join("My vision model.magic").is_file(), "{lines:#?}");
    // A picture it has not seen.
    let fresh = magi_vision::decode(&picture(99, true)).unwrap();
    let mut model = magi_ml::Model::load(&doc(&h).file.onnx, &doc(&h).file.model.input).unwrap();
    let p = model.classify(&fresh.rgba, fresh.width, fresh.height).unwrap();
    assert!(p[0] > 0.8, "{p:?}");
    let _ = std::fs::remove_dir_all(&folder);
}

/// Writes the example text and vision models into MAGI_EXPORT_DIR (for screenshots).
#[test]
#[ignore = "writes files for screenshots"]
fn write_the_example_models() {
    let dir = std::path::PathBuf::from(std::env::var("MAGI_EXPORT_DIR").expect("MAGI_EXPORT_DIR"));
    for (text, name) in [(true, "example-text.magic"), (false, "example-vision.magic")] {
        let mut doc = MagicDoc::new(crate::magic::new_model(text), None, Vec::new());
        doc.sync_canvas();
        doc.file.save(&dir.join(name)).unwrap();
    }
}

/// A workspace (`.magic`) never holds the model: saving a trained one writes the model as a `.szi`
/// beside it, which opening the workspace reads back; saving again leaves an unchanged `.szi` be.
#[test]
fn a_saved_workspace_keeps_its_model_in_a_szi_beside_it() {
    let dir = std::env::temp_dir().join(format!("magi-workspace-szi-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut doc = MagicDoc::new(crate::magic::new_model(true), None, Vec::new());
    doc.file.weights = vec![7; 64];
    let path = dir.join("Model.magic");
    let szi = doc.save_workspace(&path).unwrap().expect("the model is written");
    assert_eq!(szi, dir.join("Model.szi"));
    let workspace = magi_core::magic::MagicFile::load(&path).unwrap();
    assert!(workspace.weights.is_empty() && workspace.onnx.is_empty(), "no model in the .magic");
    assert_eq!(magi_core::szi::open(&path).unwrap().weights, vec![7; 64], "opening it reads the .szi");
    assert!(doc.save_workspace(&path).unwrap().is_none(), "an unchanged model is not written again");
    doc.file.weights = vec![8; 64];
    assert!(doc.save_workspace(&path).unwrap().is_some(), "a model trained again is");
    let _ = std::fs::remove_dir_all(&dir);
}
