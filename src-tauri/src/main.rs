// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! The desktop shell.
//!
//! The application is a Python API plus a web UI. Asking someone to install
//! Python before they can sort their music would defeat the point of shipping a
//! desktop build, so the API is bundled as a sidecar binary and this process
//! owns its lifetime: pick a free port, start it, tell the page where it is,
//! and make sure it dies when the window closes.

use std::net::TcpListener;
use std::sync::Mutex;

use tauri::api::process::{Command, CommandChild, CommandEvent};
use tauri::{Manager, RunEvent, WindowBuilder, WindowUrl};

/// The sidecar process, kept so it can be killed on shutdown.
#[derive(Default)]
struct ApiProcess(Mutex<Option<CommandChild>>);

/// The address the frontend should talk to, also exposed as a command so the
/// UI can ask for it rather than only receiving it through the injected global.
struct ApiBase(String);

#[tauri::command]
fn api_base(state: tauri::State<'_, ApiBase>) -> String {
    state.0.clone()
}

/// A port nobody else is using.
///
/// Binding to port 0 lets the OS choose, and dropping the listener frees it for
/// the sidecar. Two copies of the app therefore never fight over one port — and
/// neither does anything else already sitting on 8000.
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .map(|address| address.port())
        .unwrap_or(8000)
}

fn main() {
    let port = free_port();
    let base = format!("http://127.0.0.1:{port}");

    let injected = format!("window.__MUSIC_CLUSTER_API_BASE__ = {base:?};");

    tauri::Builder::default()
        .manage(ApiProcess::default())
        .manage(ApiBase(base))
        .invoke_handler(tauri::generate_handler![api_base])
        .setup(move |app| {
            start_api(app, port)?;

            // The window is built here rather than declared in tauri.conf.json
            // so the API address can be injected *before* any page script runs.
            // Discovering it afterwards would leave the first request racing
            // the injection.
            WindowBuilder::new(app, "main", WindowUrl::App("index.html".into()))
                .title("Music Cluster")
                .inner_size(1200.0, 800.0)
                .min_inner_size(800.0, 600.0)
                .resizable(true)
                .initialization_script(&injected)
                .build()?;

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the music-cluster application")
        .run(|app, event| {
            if let RunEvent::ExitRequested { .. } | RunEvent::Exit = event {
                shutdown(app);
            }
        });
}

/// Start the bundled API server on `port` and forward its output to our log.
fn start_api(app: &tauri::App, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let (mut rx, child) = Command::new_sidecar("music-cluster-api")?
        .args(["--host", "127.0.0.1", "--port", &port.to_string()])
        .spawn()?;

    app.state::<ApiProcess>().0.lock().unwrap().replace(child);

    // Without this, a server that fails to start leaves a blank window and no
    // explanation anywhere.
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => println!("[api] {line}"),
                CommandEvent::Stderr(line) => eprintln!("[api] {line}"),
                CommandEvent::Error(error) => eprintln!("[api] error: {error}"),
                CommandEvent::Terminated(payload) => {
                    eprintln!("[api] exited with {:?}", payload.code);
                }
                _ => {}
            }
        }
    });

    Ok(())
}

/// Stop the bundled API. Safe to call more than once.
fn shutdown(app: &tauri::AppHandle) {
    if let Some(state) = app.try_state::<ApiProcess>() {
        if let Ok(mut guard) = state.0.lock() {
            if let Some(child) = guard.take() {
                let _ = child.kill();
            }
        }
    }
}
